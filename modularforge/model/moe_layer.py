"""
ModularForge MoE Transformer Layer
====================================
A complete Mixture-of-Experts transformer layer that combines the shared
attention mechanism with the expert pool and router.

Architecture:
    Input (batch, seq_len, d_model)
      ├─ SharedAttention → attention output
      ├─ FFN Norm → normalized hidden states
      ├─ Router → selects top-K experts + weights
      ├─ Selected Experts → expert outputs
      ├─ Weighted Combine → merged expert output
      └─ Residual → final output
    Output (batch, seq_len, d_model)

Analogy:
    One "floor" of the ModularForge building:
    1. Everyone reads the document together (attention)
    2. The receptionist routes to specialists (router)
    3. The selected specialists provide their analysis (experts)
    4. The analyses are weighted-averaged (combine)
    5. Add back the original understanding (residual)

Usage:
    >>> layer = MoETransformerLayer(config)
    >>> x = torch.randn(4, 128, 512)
    >>> output, aux_loss = layer(x)
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn

from modularforge.config import ModelConfig
from modularforge.model.expert import ExpertFFN
from modularforge.model.router import MoERouter

logger = logging.getLogger(__name__)


class MoETransformerLayer(nn.Module):
    """
    A transformer layer with Mixture-of-Experts feed-forward network.

    This combines:
    - Multi-head self-attention (from shared components)
    - MoE routing (router selects top-K experts per token)
    - Expert FFN processing (selected experts apply their FFN)
    - Weighted combination (expert outputs are weighted-summed)

    Parameters
    ----------
    config : ModelConfig
        Model configuration.
    layer_idx : int
        Index of this layer in the full model (for logging/identification).
    """

    def __init__(self, config: ModelConfig, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        # Pre-norm for the FFN (MoE) part
        self.ffn_norm = nn.LayerNorm(config.d_model)

        # Router: decides which experts handle each token
        self.router = MoERouter(
            d_model=config.d_model,
            n_experts=config.n_experts,
            top_k=config.top_k,
        )

        # Expert pool: N independent FFN modules
        self.experts = nn.ModuleList([
            ExpertFFN(
                d_model=config.d_model,
                d_ff=config.d_ff,
                dropout=config.expert_dropout,
                layer_idx=layer_idx,
                expert_idx=i,
            )
            for i in range(config.n_experts)
        ])

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply the MoE transformer layer.

        Parameters
        ----------
        x : torch.Tensor
            Input after attention, shape (batch_size, seq_len, d_model).
        attention_mask : torch.Tensor or None
            Padding mask (not used in FFN, kept for API consistency).

        Returns
        -------
        tuple containing:
            output : torch.Tensor
                Layer output, same shape as input.
            aux_loss : torch.Tensor
                Router's load-balancing loss (scalar).
        """
        # Normalize before MoE (pre-norm)
        normed = self.ffn_norm(x)

        # Route tokens to experts
        weights, indices, aux_loss = self.router(normed)
        # weights: (batch, seq_len, top_k) — how much to weight each expert
        # indices: (batch, seq_len, top_k) — which experts are selected

        # Compute expert outputs (sparse — only selected experts run)
        moe_output = self._sparse_expert_forward(normed, weights, indices)

        # Residual connection (add back the original input)
        output = x + moe_output

        return output, aux_loss

    def _sparse_expert_forward(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute sparse expert outputs — only activated experts process tokens.

        Implementation Strategy:
            For each expert, we find all tokens routed to it, batch them,
            run the expert once, then scatter the results back. This is
            much more efficient than running each token through selected
            experts individually.

        Parameters
        ----------
        x : torch.Tensor
            Normalized input, shape (batch, seq_len, d_model).
        weights : torch.Tensor
            Expert weights, shape (batch, seq_len, top_k).
        indices : torch.Tensor
            Expert indices, shape (batch, seq_len, top_k).

        Returns
        -------
        torch.Tensor
            Weighted expert output, shape (batch, seq_len, d_model).
        """
        batch_size, seq_len, d_model = x.shape
        output = torch.zeros_like(x)

        # Flatten batch and sequence dimensions for easier indexing
        flat_x = x.reshape(-1, d_model)           # (B*S, d_model)
        flat_weights = weights.reshape(-1, self.config.top_k)  # (B*S, top_k)
        flat_indices = indices.reshape(-1, self.config.top_k)  # (B*S, top_k)
        flat_output = torch.zeros_like(flat_x)

        # For each expert, gather its assigned tokens and process them
        for expert_idx, expert in enumerate(self.experts):
            # Find all (token, k) pairs where this expert is selected
            # expert_mask: (B*S, top_k) — True where this expert is selected
            expert_mask = (flat_indices == expert_idx)

            if not expert_mask.any():
                continue  # This expert has no assigned tokens

            # For each k-slot where this expert is selected
            for k in range(self.config.top_k):
                slot_mask = expert_mask[:, k]  # (B*S,)

                if not slot_mask.any():
                    continue

                # Gather assigned tokens
                token_indices = slot_mask.nonzero(as_tuple=True)[0]
                expert_input = flat_x[token_indices]  # (n_tokens, d_model)

                # Run the expert (the ExpertFFN includes its own residual,
                # so we subtract input to get just the expert's contribution)
                expert_out = expert(expert_input) - expert_input

                # Get the routing weight for this expert
                slot_weights = flat_weights[token_indices, k].unsqueeze(-1)
                # Shape: (n_tokens, 1)

                # Weighted output contribution
                flat_output[token_indices] += slot_weights * expert_out

        output = flat_output.reshape(batch_size, seq_len, d_model)
        return output

    @property
    def n_params(self) -> int:
        """Total parameters in this MoE layer."""
        return sum(p.numel() for p in self.parameters())

    def __repr__(self) -> str:
        expert_params = self.experts[0].n_params if self.experts else 0
        return (
            f"MoETransformerLayer(layer={self.layer_idx}, "
            f"experts={len(self.experts)}, "
            f"expert_params={expert_params / 1e6:.2f}M each, "
            f"total={self.n_params / 1e6:.2f}M)"
        )
