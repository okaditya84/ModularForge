#!/usr/bin/env python3
"""
ModularForge — Resume Training Script
========================================
Picks up an interrupted training run from existing checkpoints.

Auto-detects which experts are complete, which have partial checkpoints,
and which need to be trained from scratch. Designed for Kaggle recovery
where a 12-hour session was interrupted mid-training.

Features:
    - Auto-detects training state from output directory
    - Resumes from the latest checkpoint (reconstructs LR scheduler)
    - Trains remaining experts from scratch
    - Kaggle-specific: copies checkpoints from dataset mount

Usage:
    # Resume from local outputs/:
    python scripts/resume_train.py --config configs/default.yaml

    # Resume with checkpoints from a Kaggle dataset mount:
    python scripts/resume_train.py --config configs/default.yaml \
        --checkpoint-dir /kaggle/input/modularforge-checkpoints/outputs

    # Only train specific experts:
    python scripts/resume_train.py --config configs/default.yaml \
        --experts 2 3 4
"""

from __future__ import annotations

import argparse
import gc
import glob
import json
import logging
import math
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader

from modularforge.config import ModularForgeConfig
from modularforge.data.tokenizer import ModularForgeTokenizer
from modularforge.data.dataset import TextDataset
from modularforge.model.shared import SharedComponents
from modularforge.model.module_trainer_model import ModuleTrainerModel
from modularforge.training.trainer import Trainer
from modularforge.evaluation.metrics import MemoryTracker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Training State Detection
# =============================================================================

def detect_training_state(output_dir: str, n_experts: int) -> dict:
    """
    Scan the output directory and determine the training state for each expert.

    Returns a dict mapping expert_idx -> status dict:
        {
            "status": "complete" | "partial" | "not_started",
            "checkpoint_path": str or None,    # latest checkpoint .pt file
            "checkpoint_step": int or 0,        # step number of latest checkpoint
            "final_path": str or None,          # expert_N.pt if complete
        }
    """
    states = {}

    for i in range(n_experts):
        state = {
            "status": "not_started",
            "checkpoint_path": None,
            "checkpoint_step": 0,
            "final_path": None,
        }

        # Check for completed expert (expert_N.pt)
        final_path = os.path.join(output_dir, f"expert_{i}.pt")
        if os.path.exists(final_path):
            state["status"] = "complete"
            state["final_path"] = final_path
            states[i] = state
            continue

        # Check for partial checkpoints (checkpoint_expert_N_step_XXXX.pt)
        pattern = os.path.join(output_dir, f"checkpoint_expert_{i}_step_*.pt")
        checkpoints = glob.glob(pattern)

        if checkpoints:
            # Find the latest checkpoint by step number
            latest_step = 0
            latest_path = None
            for cp in checkpoints:
                match = re.search(r"step_(\d+)\.pt$", cp)
                if match:
                    step = int(match.group(1))
                    if step > latest_step:
                        latest_step = step
                        latest_path = cp

            if latest_path:
                state["status"] = "partial"
                state["checkpoint_path"] = latest_path
                state["checkpoint_step"] = latest_step

        states[i] = state

    return states


def print_training_state(states: dict, n_experts: int) -> None:
    """Print a human-readable summary of the training state."""
    logger.info("=" * 60)
    logger.info("TRAINING STATE DETECTION")
    logger.info("=" * 60)

    for i in range(n_experts):
        state = states[i]
        if state["status"] == "complete":
            logger.info(f"  Expert {i}: ✅ COMPLETE — {state['final_path']}")
        elif state["status"] == "partial":
            logger.info(
                f"  Expert {i}: ⏸️  PARTIAL — "
                f"checkpoint at step {state['checkpoint_step']} "
                f"({state['checkpoint_path']})"
            )
        else:
            logger.info(f"  Expert {i}: ❌ NOT STARTED")

    logger.info("=" * 60)


# =============================================================================
# Checkpoint Copying (Kaggle dataset mount → working directory)
# =============================================================================

def copy_checkpoints(src_dir: str, dst_dir: str) -> int:
    """
    Copy checkpoint files from a source directory (e.g., Kaggle dataset mount)
    to the working output directory.

    Returns the number of files copied.
    """
    Path(dst_dir).mkdir(parents=True, exist_ok=True)

    n_copied = 0
    for pattern in ["shared_components.pt", "expert_*.pt", "checkpoint_*.pt"]:
        for src_path in glob.glob(os.path.join(src_dir, pattern)):
            filename = os.path.basename(src_path)
            dst_path = os.path.join(dst_dir, filename)

            if not os.path.exists(dst_path):
                logger.info(f"  Copying: {filename}")
                shutil.copy2(src_path, dst_path)
                n_copied += 1
            else:
                logger.info(f"  Skipping (exists): {filename}")

    return n_copied


# =============================================================================
# Resume Training for a Single Expert
# =============================================================================

def resume_single_expert(
    expert_idx: int,
    config: ModularForgeConfig,
    shared_path: str,
    texts: list[str],
    val_loader: Optional[DataLoader],
    output_dir: str,
    checkpoint_path: str,
    checkpoint_step: int,
    tokenizer: ModularForgeTokenizer,
) -> dict:
    """
    Resume training for a single expert from a checkpoint.

    This is the critical function — it reconstructs the full training
    state (model, optimizer, LR scheduler) from a checkpoint and
    continues training from where it left off.
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"RESUMING Expert {expert_idx} from step {checkpoint_step}")
    logger.info(f"Checkpoint: {checkpoint_path}")
    logger.info(f"{'='*60}\n")

    # 1. Load shared components (frozen)
    shared = SharedComponents(config.model)
    shared_state = torch.load(shared_path, map_location="cpu", weights_only=False)
    shared.load_state_dict(shared_state)
    shared.freeze()
    del shared_state
    gc.collect()

    # 2. Create model with expert
    model = ModuleTrainerModel(
        shared=shared,
        config=config.model,
        expert_idx=expert_idx,
    )

    # 3. Load checkpoint (model + optimizer state)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    # 4. Create dataset and loader
    train_dataset = TextDataset(
        texts=texts,
        tokenizer=tokenizer,
        max_seq_len=config.model.max_seq_len,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=min(config.training.num_workers, 2),
        pin_memory=False,
        drop_last=True,
    )

    # 5. Create trainer (this sets up the optimizer fresh)
    trainer = Trainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        name=f"expert_{expert_idx}",
    )

    # 6. Restore optimizer state from checkpoint
    trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    trainer.best_val_loss = checkpoint.get("best_val_loss", float("inf"))

    # 7. Calculate total steps and where we are in training
    epochs = config.training.epochs_expert
    steps_per_epoch = len(train_loader) // max(
        config.training.gradient_accumulation_steps, 1
    )
    total_steps = epochs * steps_per_epoch
    trainer.total_steps = total_steps

    # 8. Create scheduler and fast-forward to the checkpoint step
    trainer.scheduler = trainer._create_scheduler()
    for _ in range(checkpoint_step):
        trainer.scheduler.step()

    trainer.global_step = checkpoint_step

    # Log the reconstructed state
    current_lr = trainer.optimizer.param_groups[0]["lr"]
    remaining_steps = total_steps - checkpoint_step
    logger.info(
        f"[expert_{expert_idx}] Resumed at step {checkpoint_step}/{total_steps} "
        f"(lr={current_lr:.2e}, {remaining_steps} steps remaining)"
    )

    # 9. Determine which epoch we're in and how many remain
    #    checkpoint_step tells us how far we've gone; we need to figure
    #    out the current epoch and the position within it.
    completed_epochs = checkpoint_step // steps_per_epoch
    step_in_epoch = checkpoint_step % steps_per_epoch

    logger.info(
        f"[expert_{expert_idx}] Completed epochs: {completed_epochs}/{epochs}, "
        f"position in current epoch: {step_in_epoch}/{steps_per_epoch}"
    )

    # 10. Run remaining training
    #     We'll train for the remaining epochs. The global_step counter
    #     and scheduler are already positioned correctly, so the trainer
    #     will checkpoint and log at the right intervals.
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    import tracemalloc
    tracemalloc.start()
    start_time = time.time()

    results = {
        "train_losses": [],
        "val_losses": [],
        "final_train_loss": None,
        "final_val_loss": None,
        "best_val_loss": trainer.best_val_loss,
        "peak_memory_mb": 0,
        "total_time_seconds": 0,
        "total_steps": 0,
        "resumed_from_step": checkpoint_step,
    }

    # Train remaining epochs (starting from where checkpoint left off)
    for epoch in range(completed_epochs, epochs):
        epoch_loss = trainer._train_epoch(epoch, epochs, output_dir)
        results["train_losses"].append(epoch_loss)
        results["final_train_loss"] = epoch_loss

        if val_loader is not None:
            val_loss = trainer._validate()
            results["val_losses"].append(val_loss)
            results["final_val_loss"] = val_loss
            if val_loss < results["best_val_loss"]:
                results["best_val_loss"] = val_loss

        current, peak = tracemalloc.get_traced_memory()
        peak_mb = peak / (1024 * 1024)
        results["peak_memory_mb"] = max(results["peak_memory_mb"], peak_mb)

        logger.info(
            f"[expert_{expert_idx}] Epoch {epoch + 1}/{epochs} — "
            f"train_loss={epoch_loss:.4f}, "
            f"ppl={math.exp(min(epoch_loss, 20)):.2f}, "
            f"peak_mem={peak_mb:.1f}MB"
        )

    tracemalloc.stop()
    total_time = time.time() - start_time
    results["total_time_seconds"] = total_time
    results["total_steps"] = trainer.global_step

    logger.info(
        f"[expert_{expert_idx}] Training complete in {total_time:.1f}s — "
        f"final_loss={results['final_train_loss']:.4f}, "
        f"peak_mem={results['peak_memory_mb']:.1f}MB"
    )

    # 11. Save the completed expert
    expert_path = str(Path(output_dir) / f"expert_{expert_idx}.pt")
    model.save_expert(expert_path)
    results["expert_path"] = expert_path

    # 12. Cleanup
    del model, trainer, train_dataset, train_loader, shared, checkpoint
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info(f"Expert {expert_idx} complete — memory cleaned up")
    return results


# =============================================================================
# Fresh Training for a Single Expert (identical to module_trainer logic)
# =============================================================================

def train_fresh_expert(
    expert_idx: int,
    config: ModularForgeConfig,
    shared_path: str,
    texts: list[str],
    val_loader: Optional[DataLoader],
    output_dir: str,
    tokenizer: ModularForgeTokenizer,
) -> dict:
    """
    Train a single expert from scratch. This mirrors the logic in
    ModuleTrainer._train_single_expert() but is self-contained here
    to avoid modifying the original codebase.
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"TRAINING Expert {expert_idx} FROM SCRATCH")
    logger.info(f"Partition size: {len(texts):,} articles")
    logger.info(f"{'='*60}\n")

    # 1. Load shared components (frozen)
    shared = SharedComponents(config.model)
    shared_state = torch.load(shared_path, map_location="cpu", weights_only=False)
    shared.load_state_dict(shared_state)
    shared.freeze()
    del shared_state
    gc.collect()

    # 2. Create model with fresh expert
    model = ModuleTrainerModel(
        shared=shared,
        config=config.model,
        expert_idx=expert_idx,
    )

    # 3. Create dataset and loader
    train_dataset = TextDataset(
        texts=texts,
        tokenizer=tokenizer,
        max_seq_len=config.model.max_seq_len,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=min(config.training.num_workers, 2),
        pin_memory=False,
        drop_last=True,
    )

    # 4. Train
    trainer = Trainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        name=f"expert_{expert_idx}",
    )

    results = trainer.train(
        epochs=config.training.epochs_expert,
        output_dir=output_dir,
    )

    # 5. Save expert
    expert_path = str(Path(output_dir) / f"expert_{expert_idx}.pt")
    model.save_expert(expert_path)
    results["expert_path"] = expert_path

    # 6. Cleanup
    del model, trainer, train_dataset, train_loader, shared
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info(f"Expert {expert_idx} complete — memory cleaned up")
    return results


# =============================================================================
# Data Loading Helpers
# =============================================================================

def load_partition_texts(data_dir: Path, n_partitions: int) -> list[list[str]]:
    """Load text partitions from disk."""
    partitions = []
    for i in range(n_partitions):
        path = data_dir / "partitions" / f"partition_{i}.json"
        if not path.exists():
            raise FileNotFoundError(
                f"Partition {i} not found at {path}. "
                f"Run scripts/prepare_data.py first."
            )
        with open(path, "r", encoding="utf-8") as f:
            partitions.append(json.load(f))
    return partitions


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="ModularForge Resume Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Auto-detect and resume:
    python scripts/resume_train.py --config configs/default.yaml

    # Resume with checkpoints from Kaggle dataset mount:
    python scripts/resume_train.py --config configs/default.yaml \\
        --checkpoint-dir /kaggle/input/modularforge-checkpoints/outputs

    # Only train specific experts:
    python scripts/resume_train.py --config configs/default.yaml --experts 2 3 4
        """,
    )
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument(
        "--checkpoint-dir", type=str, default=None,
        help="Directory with existing checkpoints to copy from "
             "(e.g., /kaggle/input/modularforge-checkpoints/outputs)",
    )
    parser.add_argument(
        "--experts", type=int, nargs="+", default=None,
        help="Specific expert indices to train (default: auto-detect)",
    )
    args = parser.parse_args()

    # Load config
    config = ModularForgeConfig.from_yaml(args.config)

    data_dir = Path(args.data_dir or config.data.data_dir)
    output_dir = Path(args.output_dir or config.assembly.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ─── Step 0: Copy checkpoints if --checkpoint-dir is provided ────
    if args.checkpoint_dir:
        logger.info(f"\nCopying checkpoints from: {args.checkpoint_dir}")
        n_copied = copy_checkpoints(args.checkpoint_dir, str(output_dir))
        logger.info(f"Copied {n_copied} checkpoint files\n")

    # ─── Step 1: Detect training state ───────────────────────────────
    states = detect_training_state(str(output_dir), config.model.n_experts)
    print_training_state(states, config.model.n_experts)

    # Verify shared components exist
    shared_path = str(output_dir / "shared_components.pt")
    if not os.path.exists(shared_path):
        logger.error(
            f"Shared components not found at {shared_path}! "
            f"Cannot resume without trained shared components."
        )
        sys.exit(1)

    logger.info(f"✅ Shared components found: {shared_path}")

    # ─── Step 2: Determine which experts to train ────────────────────
    if args.experts is not None:
        experts_to_train = args.experts
    else:
        # Auto-detect: skip completed experts
        experts_to_train = [
            i for i in range(config.model.n_experts)
            if states[i]["status"] != "complete"
        ]

    if not experts_to_train:
        logger.info("\n🎉 All experts are already trained! Nothing to do.")
        return

    logger.info(f"\nExperts to train: {experts_to_train}")

    # Estimate time
    for i in experts_to_train:
        state = states[i]
        if state["status"] == "partial":
            logger.info(
                f"  Expert {i}: RESUME from step {state['checkpoint_step']}"
            )
        else:
            logger.info(f"  Expert {i}: FRESH training")

    # ─── Step 3: Load tokenizer and data ─────────────────────────────
    tokenizer_path = data_dir / "tokenizer.json"
    tokenizer = ModularForgeTokenizer(vocab_size=config.data.tokenizer_vocab_size)
    tokenizer.load(tokenizer_path)

    partitions = load_partition_texts(data_dir, config.model.n_experts)

    # Validation data
    val_path = data_dir / "validation.json"
    val_loader = None
    if val_path.exists():
        with open(val_path, "r", encoding="utf-8") as f:
            val_texts = json.load(f)
        val_dataset = TextDataset(
            texts=val_texts[:500],
            tokenizer=tokenizer,
            max_seq_len=config.model.max_seq_len,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.training.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )

    # ─── Step 4: Train each expert ───────────────────────────────────
    total_start = time.time()
    all_results = []

    for expert_idx in experts_to_train:
        state = states[expert_idx]
        expert_start = time.time()

        if state["status"] == "partial":
            # Resume from checkpoint
            result = resume_single_expert(
                expert_idx=expert_idx,
                config=config,
                shared_path=shared_path,
                texts=partitions[expert_idx],
                val_loader=val_loader,
                output_dir=str(output_dir),
                checkpoint_path=state["checkpoint_path"],
                checkpoint_step=state["checkpoint_step"],
                tokenizer=tokenizer,
            )
        else:
            # Train from scratch
            result = train_fresh_expert(
                expert_idx=expert_idx,
                config=config,
                shared_path=shared_path,
                texts=partitions[expert_idx],
                val_loader=val_loader,
                output_dir=str(output_dir),
                tokenizer=tokenizer,
            )

        expert_time = time.time() - expert_start
        result["wall_time_seconds"] = expert_time
        all_results.append(result)

        # Save results
        result_clean = {
            "expert_idx": expert_idx,
            "final_train_loss": result.get("final_train_loss"),
            "final_val_loss": result.get("final_val_loss"),
            "peak_memory_mb": result.get("peak_memory_mb", 0),
            "time_seconds": result.get("total_time_seconds", 0),
            "resumed_from_step": result.get("resumed_from_step"),
        }
        with open(output_dir / f"expert_{expert_idx}_results.json", "w") as f:
            json.dump(result_clean, f, indent=2)

        # Force GC between experts
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ─── Summary ─────────────────────────────────────────────────────
    total_time = time.time() - total_start

    logger.info("\n" + "=" * 60)
    logger.info("RESUME TRAINING COMPLETE")
    logger.info("=" * 60)

    for i, result in zip(experts_to_train, all_results):
        loss = result.get("final_train_loss", "N/A")
        wall = result.get("wall_time_seconds", 0)
        resumed = result.get("resumed_from_step")
        status = f"(resumed from step {resumed})" if resumed else "(fresh)"
        logger.info(
            f"  Expert {i}: loss={loss:.4f}, time={wall:.0f}s {status}"
        )

    logger.info(f"\nTotal wall time: {total_time:.0f}s ({total_time/3600:.1f}h)")
    logger.info(f"Outputs: {output_dir}/")

    # Verify all experts are now complete
    final_states = detect_training_state(str(output_dir), config.model.n_experts)
    all_complete = all(
        final_states[i]["status"] == "complete"
        for i in range(config.model.n_experts)
    )

    if all_complete:
        logger.info("\n🎉 ALL experts trained — ready for assembly!")
        logger.info(
            f"Next step: python scripts/assemble.py --config {args.config}"
        )
    else:
        incomplete = [
            i for i in range(config.model.n_experts)
            if final_states[i]["status"] != "complete"
        ]
        logger.warning(f"\n⚠️ Experts still incomplete: {incomplete}")


if __name__ == "__main__":
    main()
