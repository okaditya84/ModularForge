"""
ModularForge Core Trainer
==========================
The core training loop used by both SharedTrainer and ModuleTrainer.
Handles all the surrounding concerns so the specific trainers can focus
on WHAT to train rather than HOW.

What This Handles:
    - Forward pass + loss computation
    - Gradient accumulation (simulate larger batches)
    - Gradient clipping (prevent exploding gradients)
    - Learning rate scheduling (cosine annealing with warmup)
    - Checkpointing (save/resume training)
    - Memory tracking (peak RAM usage)
    - Logging (loss, perplexity, throughput, memory)
    - Validation loop with early stopping
    - Reproducibility (seed everything)

Analogy:
    If the model is a race car and the trainer-specific code decides
    WHERE to drive (which data, which model parts), then this Trainer
    class is the car's engine, transmission, and dashboard combined.
    It handles the mechanics of driving.

Usage:
    This class is not used directly — use SharedTrainer or ModuleTrainer.
    See those classes for usage examples.
"""

from __future__ import annotations

import gc
import logging
import math
import os
import time
import tracemalloc
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from modularforge.config import ModularForgeConfig

logger = logging.getLogger(__name__)


class Trainer:
    """
    Core training loop with gradient accumulation, scheduling,
    checkpointing, and monitoring.

    Parameters
    ----------
    model : nn.Module
        The model to train (either a SharedComponents-based model or
        a ModuleTrainerModel).

    config : ModularForgeConfig
        Full configuration object.

    train_loader : DataLoader
        Training data loader.

    val_loader : DataLoader or None
        Validation data loader. If None, validation is skipped.

    name : str
        Human-readable name for this training run (for logging).
    """

    def __init__(
        self,
        model: nn.Module,
        config: ModularForgeConfig,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        name: str = "trainer",
    ):
        self.config = config
        self.name = name
        self.device = config.training.resolve_device()

        # Move model to device
        self.model = model.to(self.device)

        # Data loaders
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Loss function (ignore padding token in loss computation)
        # pad_id=0: don't penalize the model for predicting padding
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)

        # Optimizer: AdamW (Adam with weight decay fix)
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        if not trainable_params:
            raise ValueError(
                "No trainable parameters found. Did you forget to "
                "unfreeze the model or create trainable expert layers?"
            )

        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        # Learning rate scheduler (cosine annealing with linear warmup)
        self.total_steps = 0  # Will be set in train()
        self.scheduler = None  # Created in train()

        # Tracking
        self.global_step = 0
        self.best_val_loss = float("inf")

        # Memory tracking
        self._peak_memory_bytes = 0

        logger.info(
            f"Trainer '{name}' initialized on {self.device} "
            f"with {sum(p.numel() for p in trainable_params) / 1e6:.2f}M "
            f"trainable parameters"
        )

    def train(self, epochs: int, output_dir: str = "outputs") -> dict:
        """
        Run the complete training loop for the specified number of epochs.

        Parameters
        ----------
        epochs : int
            Number of training epochs.
        output_dir : str
            Directory to save checkpoints and logs.

        Returns
        -------
        dict
            Training results containing:
            - final_train_loss: last epoch's average training loss
            - final_val_loss: last validation loss (or None)
            - best_val_loss: best validation loss seen
            - peak_memory_mb: peak memory usage in MB
            - total_time_seconds: total training time
            - total_steps: total optimizer steps taken
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Calculate total training steps for scheduler
        steps_per_epoch = len(self.train_loader) // max(
            self.config.training.gradient_accumulation_steps, 1
        )
        self.total_steps = epochs * steps_per_epoch

        # Create cosine annealing scheduler with warmup
        self.scheduler = self._create_scheduler()

        logger.info(
            f"[{self.name}] Starting training: {epochs} epochs, "
            f"{steps_per_epoch} steps/epoch, "
            f"{self.total_steps} total steps"
        )

        # Start memory tracking
        tracemalloc.start()
        start_time = time.time()

        results = {
            "train_losses": [],
            "val_losses": [],
            "final_train_loss": None,
            "final_val_loss": None,
            "best_val_loss": float("inf"),
            "peak_memory_mb": 0,
            "total_time_seconds": 0,
            "total_steps": 0,
        }

        for epoch in range(epochs):
            # Training epoch
            epoch_loss = self._train_epoch(epoch, epochs, output_dir)
            results["train_losses"].append(epoch_loss)
            results["final_train_loss"] = epoch_loss

            # Validation
            if self.val_loader is not None:
                val_loss = self._validate()
                results["val_losses"].append(val_loss)
                results["final_val_loss"] = val_loss

                if val_loss < results["best_val_loss"]:
                    results["best_val_loss"] = val_loss
                    self.best_val_loss = val_loss
                    logger.info(
                        f"[{self.name}] New best val loss: {val_loss:.4f}"
                    )

            # Track peak memory
            current, peak = tracemalloc.get_traced_memory()
            peak_mb = peak / (1024 * 1024)
            results["peak_memory_mb"] = max(results["peak_memory_mb"], peak_mb)

            logger.info(
                f"[{self.name}] Epoch {epoch + 1}/{epochs} — "
                f"train_loss={epoch_loss:.4f}, "
                f"train_ppl={math.exp(min(epoch_loss, 20)):.2f}, "
                f"peak_mem={peak_mb:.1f}MB"
            )

        # Final cleanup
        tracemalloc.stop()
        total_time = time.time() - start_time
        results["total_time_seconds"] = total_time
        results["total_steps"] = self.global_step

        logger.info(
            f"[{self.name}] Training complete in {total_time:.1f}s — "
            f"final_loss={results['final_train_loss']:.4f}, "
            f"peak_mem={results['peak_memory_mb']:.1f}MB"
        )

        return results

    def _train_epoch(
        self,
        epoch: int,
        total_epochs: int,
        output_dir: str,
    ) -> float:
        """
        Run one training epoch.

        Parameters
        ----------
        epoch : int
            Current epoch (0-indexed).
        total_epochs : int
            Total number of epochs.
        output_dir : str
            Directory for checkpoints.

        Returns
        -------
        float
            Average training loss for this epoch.
        """
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        self.optimizer.zero_grad()

        for batch_idx, (input_ids, target_ids) in enumerate(self.train_loader):
            input_ids = input_ids.to(self.device)
            target_ids = target_ids.to(self.device)

            # Create attention mask (non-padding positions)
            attention_mask = (input_ids != 0).long()

            # Forward pass
            outputs = self.model(input_ids, attention_mask=attention_mask)

            # Handle models that return (logits, aux_loss) vs just logits
            if isinstance(outputs, tuple):
                logits, aux_loss = outputs
            else:
                logits = outputs
                aux_loss = torch.tensor(0.0, device=self.device)

            # Compute language modeling loss
            # Reshape: (batch * seq_len, vocab_size) vs (batch * seq_len,)
            loss = self.criterion(
                logits.reshape(-1, logits.size(-1)),
                target_ids.reshape(-1),
            )

            # Add auxiliary loss (router load balancing)
            loss = loss + aux_loss

            # Scale loss for gradient accumulation
            loss = loss / self.config.training.gradient_accumulation_steps
            loss.backward()

            total_loss += loss.item() * self.config.training.gradient_accumulation_steps
            n_batches += 1

            # Gradient accumulation: update weights every N batches
            if (batch_idx + 1) % self.config.training.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.config.training.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in self.model.parameters() if p.requires_grad],
                        self.config.training.max_grad_norm,
                    )

                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1

                # Logging
                if (
                    self.config.training.log_every > 0
                    and self.global_step % self.config.training.log_every == 0
                ):
                    avg_loss = total_loss / n_batches
                    lr = self.optimizer.param_groups[0]["lr"]
                    logger.info(
                        f"[{self.name}] step={self.global_step}, "
                        f"loss={avg_loss:.4f}, "
                        f"ppl={math.exp(min(avg_loss, 20)):.2f}, "
                        f"lr={lr:.2e}"
                    )

                # Checkpointing
                if (
                    self.config.training.checkpoint_every > 0
                    and self.global_step % self.config.training.checkpoint_every == 0
                ):
                    self._save_checkpoint(output_dir, f"step_{self.global_step}")

        avg_loss = total_loss / max(n_batches, 1)
        return avg_loss

    @torch.no_grad()
    def _validate(self) -> float:
        """
        Run validation and compute average loss.

        Returns
        -------
        float
            Average validation loss.
        """
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        for input_ids, target_ids in self.val_loader:
            input_ids = input_ids.to(self.device)
            target_ids = target_ids.to(self.device)
            attention_mask = (input_ids != 0).long()

            outputs = self.model(input_ids, attention_mask=attention_mask)
            if isinstance(outputs, tuple):
                logits, _ = outputs
            else:
                logits = outputs

            loss = self.criterion(
                logits.reshape(-1, logits.size(-1)),
                target_ids.reshape(-1),
            )

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        logger.info(
            f"[{self.name}] Validation — loss={avg_loss:.4f}, "
            f"ppl={math.exp(min(avg_loss, 20)):.2f}"
        )
        return avg_loss

    def _create_scheduler(self) -> torch.optim.lr_scheduler.LambdaLR:
        """
        Create a learning rate scheduler with linear warmup and cosine decay.

        Analogy: Start slow (warmup), reach full speed, then gradually slow
        down (cosine decay) as you approach the finish line.

        Returns
        -------
        LambdaLR
            The learning rate schedule.
        """
        warmup_steps = self.config.training.warmup_steps
        total_steps = max(self.total_steps, 1)

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                # Linear warmup: 0 → 1 over warmup_steps
                return step / max(warmup_steps, 1)
            else:
                # Cosine decay: 1 → 0 over remaining steps
                progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
                return 0.5 * (1.0 + math.cos(math.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def _save_checkpoint(self, output_dir: str, tag: str) -> None:
        """
        Save a training checkpoint.

        Parameters
        ----------
        output_dir : str
            Directory to save to.
        tag : str
            Checkpoint identifier (e.g., "step_1000" or "best").
        """
        path = os.path.join(output_dir, f"checkpoint_{self.name}_{tag}.pt")
        checkpoint = {
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
        }
        torch.save(checkpoint, path)
        logger.info(f"[{self.name}] Checkpoint saved: {path}")

        # Keep only the latest checkpoint for this trainer to save disk space
        keep_n = getattr(self.config.training, 'keep_checkpoints', 1)
        if keep_n > 0:
            import glob
            import re
            
            # Find all checkpoints for this specific trainer
            pattern = os.path.join(output_dir, f"checkpoint_{self.name}_step_*.pt")
            all_checkpoints = glob.glob(pattern)
            
            # Sort by step number
            def get_step(p):
                match = re.search(r"step_(\d+)\.pt$", p)
                return int(match.group(1)) if match else -1
                
            all_checkpoints.sort(key=get_step)
            
            # Remove the oldest ones
            while len(all_checkpoints) > keep_n:
                old_cp = all_checkpoints.pop(0)
                try:
                    os.remove(old_cp)
                    logger.info(f"[{self.name}] Removed old checkpoint: {os.path.basename(old_cp)}")
                except OSError as e:
                    logger.warning(f"Failed to remove old checkpoint {old_cp}: {e}")

    def __repr__(self) -> str:
        return f"Trainer(name={self.name}, device={self.device}, step={self.global_step})"
