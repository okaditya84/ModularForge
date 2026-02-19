"""
ModularForge Shared Components
===============================
The shared components are the "common infrastructure" that all expert
modules build upon. They are trained ONCE on the full corpus, then
FROZEN and distributed to every expert during module training.

Architecture Analogy:
    Think of a team of specialists working in the same building:
    - Token Embedding = The shared dictionary everyone uses
    - Positional Encoding = Page numbers so everyone knows word order
    - Self-Attention = The shared "reading comprehension" skill
    - Layer Norms = Quality control checkpoints
    - LM Head = The shared "writing" skill (predicts next word)

    The building (shared components) stays the same — only the
    specialists (expert FFNs) are different.

Why Share These Components:
    1. EMBEDDINGS must be shared so all experts operate in the same
       vector space. If each expert had its own embeddings, their
       internal representations would be incompatible.
    2. ATTENTION is shared because it learns general language patterns
       (syntax, co-reference) that are domain-independent.
    3. LM HEAD is tied to the embedding weights (standard practice)
       to reduce parameters and improve coherence.

Usage:
    >>> shared = SharedComponents(config.model)
    >>> x = shared.embed(token_ids)           # Tokens → vectors
    >>> x = shared.attention_block(x, layer=0) # Self-attention
    >>> logits = shared.lm_head(x)             # Vectors → predictions
"""

from __future__ import annotations

import math
import logging
from typing import Optional

import torch
import torch.nn as nn

from modularforge.config import ModelConfig

logger = logging.getLogger(__name__)


class SinusoidalPositionalEncoding(nn.Module):
    """
    Fixed sinusoidal positional encoding (non-learnable).

    Analogy: Like page numbers in a book — each position gets a unique
    "stamp" so the model knows the ORDER of words. Position 1 always
    gets the same stamp, position 2 always gets the same stamp, etc.

    Why sinusoidal instead of learnable:
        - Works for any sequence length (even longer than training data)
        - No extra parameters to train
        - Proven effective in the original "Attention is All You Need"

    Parameters
    ----------
    d_model : int
        Dimensionality of the encoding (must match model dimension).
    max_seq_len : int
        Maximum sequence length to precompute encodings for.
    dropout : float
        Dropout probability applied after adding positional encoding.
    """

    def __init__(self, d_model: int, max_seq_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Precompute the positional encoding matrix
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        # Even dimensions get sin, odd dimensions get cos
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register as buffer (saved with model, but not a parameter)
        pe = pe.unsqueeze(0)  # Shape: (1, max_seq_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input embeddings.

        Parameters
        ----------
        x : torch.Tensor
            Input embeddings, shape (batch_size, seq_len, d_model).

        Returns
        -------
        torch.Tensor
            Embeddings with positional information added.
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention mechanism.

    Analogy: Imagine reading a sentence with 8 different colored
    highlighters simultaneously. Each highlighter (head) focuses on a
    different relationship:
        - Head 1: highlights subject-verb agreement
        - Head 2: highlights pronoun references
        - Head 3: highlights adjective-noun pairs
        - Head 4: highlights temporal relationships
        ... and so on.

    Then all the highlighted information is combined into a single
    understanding of the sentence.

    Parameters
    ----------
    d_model : int
        Total model dimension.
    n_heads : int
        Number of attention heads. Must divide d_model evenly.
    dropout : float
        Attention dropout probability.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()

        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
            )

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = math.sqrt(self.head_dim)

        # Q, K, V projections (combined into one linear for efficiency)
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)

        # Output projection
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal: bool = True,
    ) -> torch.Tensor:
        """
        Compute multi-head self-attention.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor, shape (batch_size, seq_len, d_model).
        attention_mask : torch.Tensor or None
            Padding mask, shape (batch_size, seq_len). 1 = attend, 0 = ignore.
        causal : bool
            If True, apply causal (autoregressive) masking so each position
            can only attend to earlier positions. Required for language
            modeling (can't peek at the future!).

        Returns
        -------
        torch.Tensor
            Output tensor, same shape as input.
        """
        batch_size, seq_len, _ = x.shape

        # Compute Q, K, V in one shot, then split
        qkv = self.qkv_proj(x)  # (B, S, 3*D)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, S, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each: (B, H, S, head_dim)

        # Scaled dot-product attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        # Shape: (B, H, S, S)

        # Apply causal mask (prevent attending to future tokens)
        if causal:
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
                diagonal=1,
            )
            attn_weights = attn_weights.masked_fill(causal_mask, float("-inf"))

        # Apply padding mask (ignore padded positions)
        if attention_mask is not None:
            # attention_mask: (B, S) → (B, 1, 1, S) for broadcasting
            pad_mask = (attention_mask == 0).unsqueeze(1).unsqueeze(2)
            attn_weights = attn_weights.masked_fill(pad_mask, float("-inf"))

        # Softmax + dropout
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Weighted sum of values
        attn_output = torch.matmul(attn_weights, v)  # (B, H, S, head_dim)

        # Concatenate heads and project
        attn_output = attn_output.transpose(1, 2).reshape(
            batch_size, seq_len, self.d_model
        )
        output = self.out_proj(attn_output)
        output = self.out_dropout(output)

        return output


class TransformerBlock(nn.Module):
    """
    A single shared transformer block containing attention + placeholder
    for the expert FFN.

    Architecture (Pre-Norm style):
        x → LayerNorm → Self-Attention → + (residual)
        → LayerNorm → [Expert FFN goes here] → + (residual)

    During TRAINING: The expert FFN slot is filled by the specific expert
    being trained.
    During INFERENCE: The expert FFN slot is replaced by MoE routing
    across all experts.

    Parameters
    ----------
    d_model : int
        Model dimension.
    n_heads : int
        Number of attention heads.
    dropout : float
        Dropout probability.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()

        self.attn_norm = nn.LayerNorm(d_model)
        self.attention = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.ffn_norm = nn.LayerNorm(d_model)
        # Note: FFN is NOT included here — it's provided by the expert
        # or MoE layer at runtime.

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply attention with pre-norm residual connection.

        This only applies the ATTENTION half of the transformer block.
        The FFN half is handled by the expert or MoE layer.

        Parameters
        ----------
        x : torch.Tensor
            Input, shape (batch_size, seq_len, d_model).
        attention_mask : torch.Tensor or None
            Padding mask.

        Returns
        -------
        torch.Tensor
            Output after attention + residual, shape unchanged.
        """
        # Pre-norm attention with residual connection
        normed = self.attn_norm(x)
        attn_out = self.attention(normed, attention_mask=attention_mask)
        x = x + attn_out

        return x


class SharedComponents(nn.Module):
    """
    All shared (non-expert) components of the ModularForge model.

    These are trained once on the full corpus and then frozen. Every
    expert module uses the same shared components, ensuring they all
    operate in the same representational space.

    Contains:
        - Token embedding (vocabulary → vectors)
        - Positional encoding (adds position information)
        - Transformer blocks (attention + layer norms, no FFN)
        - Output layer norm
        - LM head (vectors → vocabulary logits, weight-tied)

    Parameters
    ----------
    config : ModelConfig
        Model architecture configuration.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Token embedding: maps token IDs to dense vectors
        # Analogy: Looking up each word in a dictionary to get its meaning
        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.d_model,
            padding_idx=0,  # PAD token (id=0) always maps to zero vector
        )

        # Positional encoding: adds position information
        self.pos_encoding = SinusoidalPositionalEncoding(
            config.d_model,
            config.max_seq_len,
            config.dropout,
        )

        # Transformer blocks (attention only — FFN comes from experts)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(config.d_model, config.n_heads, config.dropout)
            for _ in range(config.n_layers)
        ])

        # Final layer norm before LM head
        self.output_norm = nn.LayerNorm(config.d_model)

        # LM head: predicts next token probabilities
        # Weight-tied with token embedding (standard practice)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

        # Initialize weights
        self._init_weights()

        # Log parameter count
        n_params = sum(p.numel() for p in self.parameters())
        logger.info(f"SharedComponents: {n_params / 1e6:.2f}M parameters")

    def _init_weights(self) -> None:
        """
        Initialize weights using Xavier uniform for linear layers and
        normal distribution for embeddings.

        These initialization schemes are standard for transformers and
        help training converge faster.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)
                # Keep padding embedding at zero
                if module.padding_idx is not None:
                    nn.init.zeros_(module.weight[module.padding_idx])
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Convert token IDs to embedded vectors with positional encoding.

        Parameters
        ----------
        token_ids : torch.Tensor
            Token IDs, shape (batch_size, seq_len).

        Returns
        -------
        torch.Tensor
            Embedded vectors, shape (batch_size, seq_len, d_model).
        """
        x = self.token_embedding(token_ids)
        x = self.pos_encoding(x)
        return x

    def apply_attention(
        self,
        x: torch.Tensor,
        layer_idx: int,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply one transformer block's attention to the input.

        Parameters
        ----------
        x : torch.Tensor
            Input, shape (batch_size, seq_len, d_model).
        layer_idx : int
            Which transformer block to use (0 to n_layers-1).
        attention_mask : torch.Tensor or None
            Padding mask.

        Returns
        -------
        torch.Tensor
            Output after attention, same shape.
        """
        if layer_idx < 0 or layer_idx >= len(self.transformer_blocks):
            raise IndexError(
                f"layer_idx {layer_idx} out of range "
                f"[0, {len(self.transformer_blocks) - 1}]"
            )
        return self.transformer_blocks[layer_idx](x, attention_mask)

    def get_ffn_norm(self, layer_idx: int) -> nn.LayerNorm:
        """
        Get the FFN layer norm for a specific transformer block.

        This norm is applied BEFORE the expert FFN (pre-norm architecture).

        Parameters
        ----------
        layer_idx : int
            Which transformer block.

        Returns
        -------
        nn.LayerNorm
            The FFN normalization layer.
        """
        return self.transformer_blocks[layer_idx].ffn_norm

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply output norm and LM head to get next-token logits.

        Parameters
        ----------
        x : torch.Tensor
            Hidden states, shape (batch_size, seq_len, d_model).

        Returns
        -------
        torch.Tensor
            Logits over vocabulary, shape (batch_size, seq_len, vocab_size).
        """
        x = self.output_norm(x)
        logits = self.lm_head(x)
        return logits

    def freeze(self) -> None:
        """
        Freeze all parameters (stop gradient computation).

        Called after shared component training is complete. Once frozen,
        these parameters will not be updated during expert training.
        """
        for param in self.parameters():
            param.requires_grad = False
        logger.info("SharedComponents frozen — no gradients will be computed")

    def unfreeze(self) -> None:
        """Unfreeze all parameters (allow gradient computation)."""
        for param in self.parameters():
            param.requires_grad = True
        logger.info("SharedComponents unfrozen — gradients enabled")

    @property
    def n_params(self) -> int:
        """Total number of parameters (including tied weights)."""
        return sum(p.numel() for p in self.parameters())

    @property
    def n_trainable_params(self) -> int:
        """Number of trainable (non-frozen) parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
