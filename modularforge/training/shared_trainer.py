"""
ModularForge Shared Trainer
=============================
Trains the shared components (embedding, attention, norms, LM head)
on the FULL training corpus. This is Phase 1 of the ModularForge
training pipeline.

Why Train Shared Components First:
    The shared components define the "common language" that all experts
    will use. Training them on the full corpus ensures every expert
    starts from the same high-quality foundation, regardless of which
    data partition it will later specialize on.

    Analogy: Before specialists can work in a hospital, the hospital
    itself needs to be built — with common infrastructure like hallways,
    reception, and shared equipment. This trainer builds the hospital.

What Gets Trained:
    - Token embedding (vocabulary → vectors)
    - Positional encoding parameters (if learnable)
    - Self-attention weights (Q, K, V, O projections)
    - Layer norms
    - LM head (tied to embeddings)

    A temporary "dummy" FFN is used during shared training to form a
    complete model. These dummy FFNs are discarded after training.

Output:
    A checkpoint file containing only the shared component weights.
    This checkpoint is then loaded and FROZEN for all expert training.

Usage:
    >>> from modularforge.training.shared_trainer import SharedTrainer
    >>> shared_trainer = SharedTrainer(config)
    >>> shared_trainer.train(full_corpus_loader, val_loader)
    >>> shared_trainer.save("outputs/shared_components.pt")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from modularforge.config import ModularForgeConfig
from modularforge.model.shared import SharedComponents
from modularforge.model.expert import ExpertFFN
from modularforge.training.trainer import Trainer

logger = logging.getLogger(__name__)


class SharedTrainingModel(nn.Module):
    """
    Temporary model for training shared components.

    Since shared components alone can't make predictions (no FFN),
    we add a single "dummy" expert FFN per layer to form a complete
    model. After training, we extract and save only the shared weights.

    Parameters
    ----------
    config : ModelConfig
        Model architecture config.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.shared = SharedComponents(config)

        # Temporary FFN (one per layer) — will be discarded after training
        self.temp_experts = nn.ModuleList([
            ExpertFFN(
                d_model=config.d_model,
                d_ff=config.d_ff,
                dropout=config.dropout,
                layer_idx=i,
                expert_idx=-1,  # -1 = temporary
            )
            for i in range(config.n_layers)
        ])

    def forward(self, input_ids, attention_mask=None):
        """Forward pass through the complete temporary model."""
        x = self.shared.embed(input_ids)

        for layer_idx in range(self.config.n_layers):
            x = self.shared.apply_attention(x, layer_idx, attention_mask)
            x = self.temp_experts[layer_idx](x)

        logits = self.shared.predict(x)
        return logits


class SharedTrainer:
    """
    Phase 1 trainer: trains shared components on the full corpus.

    Parameters
    ----------
    config : ModularForgeConfig
        Full configuration.
    """

    def __init__(self, config: ModularForgeConfig):
        self.config = config
        self.model = SharedTrainingModel(config.model)

        # Log parameter breakdown
        shared_params = self.model.shared.n_params
        temp_params = sum(p.numel() for p in self.model.temp_experts.parameters())
        logger.info(
            f"SharedTrainer: {shared_params / 1e6:.2f}M shared params + "
            f"{temp_params / 1e6:.2f}M temporary FFN params"
        )

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        output_dir: str = "outputs",
    ) -> dict:
        """
        Train the shared components.

        Parameters
        ----------
        train_loader : DataLoader
            Full corpus training data.
        val_loader : DataLoader or None
            Validation data.
        output_dir : str
            Directory to save checkpoints.

        Returns
        -------
        dict
            Training results (loss, memory, time, etc.)
        """
        trainer = Trainer(
            model=self.model,
            config=self.config,
            train_loader=train_loader,
            val_loader=val_loader,
            name="shared",
        )

        results = trainer.train(
            epochs=self.config.training.epochs_shared,
            output_dir=output_dir,
        )

        return results

    def save(self, path: str) -> None:
        """
        Save only the shared component weights (not the temporary FFN).

        Parameters
        ----------
        path : str
            Output file path.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.shared.state_dict(), path)
        logger.info(f"Shared components saved to {path}")

    def get_shared_components(self) -> SharedComponents:
        """
        Extract the trained shared components.

        Returns
        -------
        SharedComponents
            The trained (but not yet frozen) shared components.
        """
        return self.model.shared
