"""
ModularForge Expert FFN Module
================================
The Expert FFN (Feed-Forward Network) is the independently trainable
unit — the "specialist" in our team. Each expert is trained on a
different data partition and learns different knowledge.

Architecture:
    Input (d_model)
      → LayerNorm (normalize activations)
      → Linear (d_model → d_ff)     "Expand"
      → GELU activation              "Non-linear transformation"
      → Dropout                       "Regularization"
      → Linear (d_ff → d_model)      "Compress"
      → Dropout                       "Regularization"
      + Residual connection           "Skip connection"
    Output (d_model)

Analogy:
    Each expert is like a specialist doctor in a hospital:
    - The cardiologist (expert 0) trained on heart-related data
    - The neurologist (expert 1) trained on brain-related data
    - The orthopedist (expert 2) trained on bone-related data

    When a patient (token) arrives, the router decides which specialist
    to consult. The specialist examines the patient (applies FFN),
    writes a diagnosis (output), and sends them back.

Why This Design:
    - Pre-norm (LayerNorm before FFN): More stable training than post-norm
    - GELU activation: Smoother than ReLU, standard in modern transformers
    - Residual connection: Allows information to flow through unchanged
      if the expert has nothing useful to add
    - Relatively small: ~8M params at d_model=512, d_ff=2048

Usage:
    >>> expert = ExpertFFN(d_model=512, d_ff=2048)
    >>> x = torch.randn(4, 128, 512)  # (batch, seq_len, d_model)
    >>> out = expert(x)                # Same shape: (4, 128, 512)
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ExpertFFN(nn.Module):
    """
    A single expert feed-forward network module.

    This is the fundamental building block that gets trained independently
    on a specific data partition. After training, it becomes one "expert"
    in the assembled Mixture-of-Experts model.

    Parameters
    ----------
    d_model : int
        Input and output dimensionality. Must match the shared components'
        d_model so the expert can "plug in" to the transformer.
        Analogy: The standard connector size — all experts must use the
        same plug format.

    d_ff : int
        Hidden layer dimensionality (the "expanded" internal representation).
        Usually 4× d_model. Bigger = more capacity but more parameters.
        Analogy: How much "scratch paper" the specialist has to work with.

    dropout : float
        Dropout probability for regularization.
        Analogy: Randomly hiding some notes during training so the
        specialist learns to be robust even with incomplete information.

    layer_idx : int
        Which transformer layer this expert belongs to.
        Used for logging and identification.

    expert_idx : int
        Which expert this is within the MoE layer.
        Used for logging and identification.
    """

    def __init__(
        self,
        d_model: int = 512,
        d_ff: int = 2048,
        dropout: float = 0.1,
        layer_idx: int = 0,
        expert_idx: int = 0,
    ):
        super().__init__()

        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")
        if d_ff <= 0:
            raise ValueError(f"d_ff must be positive, got {d_ff}")

        self.d_model = d_model
        self.d_ff = d_ff
        self.layer_idx = layer_idx
        self.expert_idx = expert_idx

        # Pre-norm: normalize input before the FFN
        self.norm = nn.LayerNorm(d_model)

        # Two-layer FFN with GELU activation
        # "Expand" phase: d_model → d_ff (4× larger)
        self.fc1 = nn.Linear(d_model, d_ff)
        # "Compress" phase: d_ff → d_model (back to original size)
        self.fc2 = nn.Linear(d_ff, d_model)

        # Activation function (GELU = Gaussian Error Linear Unit)
        # Smoother than ReLU, avoids "dead neurons"
        self.activation = nn.GELU()

        # Dropout for regularization
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """
        Initialize expert weights using Kaiming He initialization.

        Why Kaiming:
            - Designed for networks with ReLU-like activations
            - Keeps variance stable across layers
            - Prevents vanishing/exploding gradients at initialization
        """
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity="relu")
        nn.init.zeros_(self.fc1.bias)
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity="relu")
        nn.init.zeros_(self.fc2.bias)
        nn.init.ones_(self.norm.weight)
        nn.init.zeros_(self.norm.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the expert FFN with residual connection.

        Data flow:
            x → LayerNorm → Linear → GELU → Dropout → Linear → Dropout
                                                                    ↓
            output = x + ffn_output    (residual connection)

        Parameters
        ----------
        x : torch.Tensor
            Input tensor, shape (..., d_model).
            Can be any number of leading dimensions (batch, seq_len, etc.)

        Returns
        -------
        torch.Tensor
            Output tensor, same shape as input.
        """
        # Save input for residual connection
        residual = x

        # Pre-norm → FFN
        h = self.norm(x)
        h = self.fc1(h)           # Expand: d_model → d_ff
        h = self.activation(h)    # Non-linearity
        h = self.dropout1(h)      # Regularization
        h = self.fc2(h)           # Compress: d_ff → d_model
        h = self.dropout2(h)      # Regularization

        # Residual connection: add original input
        output = residual + h

        return output

    @property
    def n_params(self) -> int:
        """Total number of parameters in this expert."""
        return sum(p.numel() for p in self.parameters())

    @property
    def memory_bytes(self) -> int:
        """Approximate memory usage in bytes (float32)."""
        return self.n_params * 4  # 4 bytes per float32 parameter

    def __repr__(self) -> str:
        return (
            f"ExpertFFN("
            f"d_model={self.d_model}, d_ff={self.d_ff}, "
            f"params={self.n_params / 1e6:.2f}M, "
            f"layer={self.layer_idx}, expert={self.expert_idx})"
        )
