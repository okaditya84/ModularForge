"""
ModularForge Assembled MoE Model
==================================
The fully assembled Mixture-of-Experts model for inference. This is the
final output of the ModularForge pipeline — a working language model
built from independently trained expert modules.

Architecture:
    ┌───────────────────────────────────────────────────────────┐
    │  Input Token IDs                                         │
    │    → Token Embedding (shared)                            │
    │    → Positional Encoding (shared)                        │
    │    → For each layer:                                     │
    │        → Self-Attention (shared)                         │
    │        → MoE Layer:                                      │
    │            → Router → selects top-K experts              │
    │            → Expert FFNs → sparse computation            │
    │            → Weighted combine                            │
    │    → Output LayerNorm (shared)                           │
    │    → LM Head (shared, weight-tied)                       │
    │  Output: next-token logits                               │
    └───────────────────────────────────────────────────────────┘

Loading Process:
    The assembled model is loaded from a single checkpoint file that
    contains: shared weights + all expert weights + router weights.
    This file is produced by the StreamingAssembler.

Usage:
    >>> model = AssembledMoEModel(config)
    >>> model.load_from_checkpoint("outputs/assembled_model.pt")
    >>> logits = model(input_ids)
    >>> generated = model.generate(prompt_ids, max_new_tokens=100)
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from modularforge.config import ModelConfig
from modularforge.model.shared import SharedComponents
from modularforge.model.moe_layer import MoETransformerLayer

logger = logging.getLogger(__name__)


class AssembledMoEModel(nn.Module):
    """
    The fully assembled Mixture-of-Experts language model.

    This combines the shared components with ALL expert modules and
    the router into a single model capable of inference and text
    generation.

    Parameters
    ----------
    config : ModelConfig
        Model configuration.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Shared components (embedding, attention, norms, LM head)
        self.shared = SharedComponents(config)

        # MoE layers (one per transformer block)
        self.moe_layers = nn.ModuleList([
            MoETransformerLayer(config, layer_idx=i)
            for i in range(config.n_layers)
        ])

        # Log total parameter count
        total = sum(p.numel() for p in self.parameters())
        n_expert = sum(
            sum(p.numel() for p in layer.experts.parameters())
            for layer in self.moe_layers
        )
        n_shared = self.shared.n_params
        n_router = sum(
            sum(p.numel() for p in layer.router.parameters())
            for layer in self.moe_layers
        )

        logger.info(
            f"AssembledMoEModel: {total / 1e6:.2f}M total params "
            f"({n_shared / 1e6:.2f}M shared, "
            f"{n_expert / 1e6:.2f}M expert, "
            f"{n_router / 1e6:.2f}M router)"
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the full assembled model.

        Parameters
        ----------
        input_ids : torch.Tensor
            Token IDs, shape (batch_size, seq_len).
        attention_mask : torch.Tensor or None
            Padding mask, shape (batch_size, seq_len).

        Returns
        -------
        tuple containing:
            logits : torch.Tensor
                Next-token predictions, shape (batch_size, seq_len, vocab_size).
            total_aux_loss : torch.Tensor
                Sum of router load-balancing losses from all layers.
        """
        # Embed tokens
        x = self.shared.embed(input_ids)

        # Apply transformer layers with MoE
        total_aux_loss = torch.tensor(0.0, device=x.device)

        for layer_idx in range(self.config.n_layers):
            # Shared attention
            x = self.shared.apply_attention(x, layer_idx, attention_mask)

            # MoE expert processing
            x, aux_loss = self.moe_layers[layer_idx](x, attention_mask)
            total_aux_loss = total_aux_loss + aux_loss

        # Predict next token
        logits = self.shared.predict(x)

        return logits, total_aux_loss

    # ─── Text Generation ────────────────────────────────────────────────

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate text autoregressively.

        Starting from the input_ids (prompt), generates one token at a
        time by:
        1. Running the model on the current sequence
        2. Taking the logits for the LAST position
        3. Applying temperature, top-k, and top-p filtering
        4. Sampling a token from the filtered distribution
        5. Appending the token and repeating

        Parameters
        ----------
        input_ids : torch.Tensor
            Prompt token IDs, shape (1, prompt_len) or (prompt_len,).
        max_new_tokens : int
            Maximum number of tokens to generate.
        temperature : float
            Sampling temperature. Higher = more random/creative.
            Lower = more focused/deterministic. 0 = greedy (argmax).
        top_k : int
            Only consider the top-k most likely tokens. 0 = no filtering.
        top_p : float
            Nucleus sampling threshold. Only consider tokens whose
            cumulative probability mass >= top_p. 1.0 = no filtering.
        eos_token_id : int or None
            If provided, stop generation when this token is produced.

        Returns
        -------
        torch.Tensor
            Generated token IDs, shape (1, prompt_len + generated_len).
        """
        self.eval()

        # Ensure input is 2D
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        generated = input_ids

        for _ in range(max_new_tokens):
            # Truncate if exceeding max_seq_len
            if generated.shape[1] > self.config.max_seq_len:
                context = generated[:, -self.config.max_seq_len:]
            else:
                context = generated

            # Forward pass
            logits, _ = self.forward(context)

            # Take logits for the last position
            next_logits = logits[:, -1, :]  # (1, vocab_size)

            # Apply temperature
            if temperature > 0:
                next_logits = next_logits / temperature
            else:
                # Greedy: take argmax
                next_token = next_logits.argmax(dim=-1, keepdim=True)
                generated = torch.cat([generated, next_token], dim=-1)
                if eos_token_id is not None and next_token.item() == eos_token_id:
                    break
                continue

            # Apply top-k filtering
            if top_k > 0:
                next_logits = self._top_k_filter(next_logits, top_k)

            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                next_logits = self._top_p_filter(next_logits, top_p)

            # Sample from filtered distribution
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append to generated sequence
            generated = torch.cat([generated, next_token], dim=-1)

            # Check for EOS
            if eos_token_id is not None and next_token.item() == eos_token_id:
                break

        return generated

    @staticmethod
    def _top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
        """
        Keep only the top-k highest logits, set rest to -infinity.

        Analogy: From all possible next words, only consider the k most
        likely ones. This prevents the model from choosing very unlikely
        words that might produce gibberish.

        Parameters
        ----------
        logits : torch.Tensor
            Raw logits, shape (..., vocab_size).
        k : int
            Number of top entries to keep.

        Returns
        -------
        torch.Tensor
            Filtered logits.
        """
        if k <= 0 or k >= logits.shape[-1]:
            return logits

        values, _ = logits.topk(k, dim=-1)
        min_value = values[..., -1:]
        return logits.where(logits >= min_value, torch.full_like(logits, float("-inf")))

    @staticmethod
    def _top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
        """
        Nucleus sampling: keep the smallest set of tokens whose
        cumulative probability >= p.

        Analogy: Instead of a fixed number of candidates (top-k), keep
        enough candidates to cover p% of the probability mass. This
        adapts automatically — when the model is confident, it considers
        fewer words; when uncertain, it considers more.

        Parameters
        ----------
        logits : torch.Tensor
            Raw logits, shape (..., vocab_size).
        p : float
            Cumulative probability threshold.

        Returns
        -------
        torch.Tensor
            Filtered logits.
        """
        sorted_logits, sorted_indices = logits.sort(descending=True, dim=-1)
        cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)

        # Find the cutoff point
        sorted_mask = cumulative_probs - sorted_logits.softmax(dim=-1) >= p
        sorted_logits[sorted_mask] = float("-inf")

        # Unsort
        return sorted_logits.scatter(-1, sorted_indices.argsort(-1), sorted_logits)

    # ─── Checkpoint Loading ─────────────────────────────────────────────

    def load_from_checkpoint(self, path: str) -> None:
        """
        Load model weights from a checkpoint file.

        Supports both PyTorch (.pt) and safetensors (.safetensors) formats.

        Parameters
        ----------
        path : str
            Path to the checkpoint file.

        Raises
        ------
        FileNotFoundError
            If the checkpoint file doesn't exist.
        """
        from pathlib import Path as PathLib

        path_obj = PathLib(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        if path.endswith(".safetensors"):
            from safetensors.torch import load_file
            state_dict = load_file(path, device="cpu")
        else:
            state_dict = torch.load(path, map_location="cpu", weights_only=False)
            if "model_state_dict" in state_dict:
                state_dict = state_dict["model_state_dict"]

        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        if missing:
            logger.warning(f"Missing keys in checkpoint: {missing}")
        if unexpected:
            logger.warning(f"Unexpected keys in checkpoint: {unexpected}")

        logger.info(f"Model loaded from {path}")

    @property
    def n_params(self) -> int:
        """Total number of parameters."""
        return sum(p.numel() for p in self.parameters())

    @property
    def n_active_params_per_token(self) -> int:
        """
        Number of parameters actively used per token (sparse computation).

        In an MoE model, only top_k experts are active for each token,
        so the effective compute is much less than total params.
        """
        shared_params = self.shared.n_params
        expert_params_per_layer = (
            self.moe_layers[0].experts[0].n_params * self.config.top_k
            if self.moe_layers
            else 0
        )
        router_params_per_layer = (
            sum(p.numel() for p in self.moe_layers[0].router.parameters())
            if self.moe_layers
            else 0
        )
        return shared_params + self.config.n_layers * (
            expert_params_per_layer + router_params_per_layer
        )

    def __repr__(self) -> str:
        return (
            f"AssembledMoEModel("
            f"total_params={self.n_params / 1e6:.2f}M, "
            f"active_per_token={self.n_active_params_per_token / 1e6:.2f}M, "
            f"experts={self.config.n_experts}, "
            f"layers={self.config.n_layers})"
        )
