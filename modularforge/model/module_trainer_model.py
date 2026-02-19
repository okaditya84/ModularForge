"""
ModularForge Module Trainer Model
===================================
The "training wrapper" that combines frozen shared components with a
single trainable expert. This is what actually gets trained during the
sequential module training phase.

Architecture During Training:
    Input token IDs
      → [FROZEN] Token Embedding
      → [FROZEN] Positional Encoding
      → For each layer:
          → [FROZEN] Self-Attention
          → [FROZEN] FFN Norm
          → [TRAINABLE] Expert FFN
      → [FROZEN] Output Norm
      → [FROZEN] LM Head
    Output: next-token logits

Analogy:
    Imagine training a new employee in a company. The office building
    (shared components) is already built and operational. The new
    employee (expert) learns their job using the company's existing
    infrastructure, but only THEIR skills are being developed — the
    building doesn't change.

Memory Budget:
    Since shared components are frozen (no gradients), memory =
    shared_params (inference only) + expert_params + optimizer + gradients
    ≈ 18M × 4B (no grad) + 8M × 4B (params) + 8M × 4B × 2 (Adam states) + 8M × 4B (grad)
    ≈ 72MB + 32MB + 64MB + 32MB ≈ ~200MB total

Usage:
    >>> model = ModuleTrainerModel(shared_components, expert_idx=0, config=config)
    >>> logits = model(input_ids, attention_mask)
    >>> loss = criterion(logits, target_ids)
    >>> loss.backward()  # Only expert parameters get gradients
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn

from modularforge.config import ModelConfig
from modularforge.model.shared import SharedComponents
from modularforge.model.expert import ExpertFFN

logger = logging.getLogger(__name__)


class ModuleTrainerModel(nn.Module):
    """
    Training wrapper: frozen shared components + one trainable expert.

    This model is used during Phase 2 (sequential expert training).
    For each expert module, we:
    1. Load the frozen shared checkpoint
    2. Create a fresh ExpertFFN
    3. Wrap them in this model
    4. Train only the ExpertFFN on its data partition

    Parameters
    ----------
    shared : SharedComponents
        Pre-trained, frozen shared components.
        Must have .freeze() called before wrapping.

    config : ModelConfig
        Model configuration.

    expert_idx : int
        Index of the expert being trained (for logging).
    """

    def __init__(
        self,
        shared: SharedComponents,
        config: ModelConfig,
        expert_idx: int = 0,
    ):
        super().__init__()
        self.config = config
        self.expert_idx = expert_idx

        # Shared components (should be frozen)
        self.shared = shared

        # One expert FFN per layer (all trainable)
        self.experts = nn.ModuleList([
            ExpertFFN(
                d_model=config.d_model,
                d_ff=config.d_ff,
                dropout=config.expert_dropout,
                layer_idx=layer_idx,
                expert_idx=expert_idx,
            )
            for layer_idx in range(config.n_layers)
        ])

        # Verify shared is frozen
        shared_trainable = sum(
            1 for p in self.shared.parameters() if p.requires_grad
        )
        if shared_trainable > 0:
            logger.warning(
                f"SharedComponents has {shared_trainable} trainable parameters! "
                f"Call shared.freeze() before wrapping in ModuleTrainerModel."
            )

        # Log parameter counts
        expert_params = sum(p.numel() for p in self.experts.parameters())
        logger.info(
            f"ModuleTrainerModel[expert={expert_idx}]: "
            f"{expert_params / 1e6:.2f}M trainable expert params, "
            f"{self.shared.n_params / 1e6:.2f}M frozen shared params"
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through the training model.

        Data flow:
            input_ids → embed → [attention → expert] × n_layers → predict

        Parameters
        ----------
        input_ids : torch.Tensor
            Token IDs, shape (batch_size, seq_len).

        attention_mask : torch.Tensor or None
            Padding mask, shape (batch_size, seq_len).
            1 = real token, 0 = padding.

        Returns
        -------
        torch.Tensor
            Next-token logits, shape (batch_size, seq_len, vocab_size).
        """
        # Step 1: Embed tokens (frozen)
        x = self.shared.embed(input_ids)

        # Step 2: Apply transformer layers
        for layer_idx in range(self.config.n_layers):
            # Attention (frozen)
            x = self.shared.apply_attention(x, layer_idx, attention_mask)

            # Expert FFN (trainable)
            x = self.experts[layer_idx](x)

        # Step 3: Predict next token (frozen)
        logits = self.shared.predict(x)

        return logits

    def get_trainable_params(self) -> list[nn.Parameter]:
        """
        Return only the trainable (expert) parameters.

        Used to create the optimizer — we only optimize expert weights,
        not the frozen shared components.

        Returns
        -------
        list[nn.Parameter]
            List of trainable parameters (expert FFN weights/biases/norms).
        """
        return [p for p in self.experts.parameters() if p.requires_grad]

    @property
    def n_trainable_params(self) -> int:
        """Number of trainable parameters."""
        return sum(p.numel() for p in self.get_trainable_params())

    @property
    def n_total_params(self) -> int:
        """Total parameters including frozen shared components."""
        return sum(p.numel() for p in self.parameters())

    def save_expert(self, path: str) -> None:
        """
        Save only the expert parameters to disk.

        We save ONLY the expert weights (not frozen shared components),
        keeping the file small (~32MB for 8M params in float32).

        Parameters
        ----------
        path : str
            File path to save the expert checkpoint.
        """
        from pathlib import Path
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        expert_state = {
            "expert_idx": self.expert_idx,
            "experts_state_dict": self.experts.state_dict(),
            "config": {
                "d_model": self.config.d_model,
                "d_ff": self.config.d_ff,
                "n_layers": self.config.n_layers,
                "expert_dropout": self.config.expert_dropout,
            },
        }
        torch.save(expert_state, path)
        logger.info(f"Expert {self.expert_idx} saved to {path}")

    @classmethod
    def load_expert_state(cls, path: str) -> dict:
        """
        Load expert checkpoint from disk.

        Parameters
        ----------
        path : str
            File path to the expert checkpoint.

        Returns
        -------
        dict
            Expert state dictionary.
        """
        if not __import__("pathlib").Path(path).exists():
            raise FileNotFoundError(f"Expert checkpoint not found: {path}")

        state = torch.load(path, map_location="cpu", weights_only=False)
        logger.info(f"Expert {state.get('expert_idx', '?')} loaded from {path}")
        return state

    def __repr__(self) -> str:
        return (
            f"ModuleTrainerModel("
            f"expert={self.expert_idx}, "
            f"trainable={self.n_trainable_params / 1e6:.2f}M, "
            f"frozen={self.shared.n_params / 1e6:.2f}M)"
        )
