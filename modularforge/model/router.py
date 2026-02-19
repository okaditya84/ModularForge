"""
ModularForge MoE Router
========================
The router is the "traffic controller" of the Mixture-of-Experts model.
For each input token, it decides which experts should process it and
how much weight to give each expert's output.

How It Works (Analogy):
    Imagine a hospital reception desk. When a patient arrives:
    1. The receptionist (router) looks at the patient's symptoms (token)
    2. Decides which specialists to consult (top-k experts)
    3. Assigns importance weights (e.g., 70% cardiologist, 30% pulmonologist)
    4. The specialists independently examine the patient
    5. Their diagnoses are combined using the assigned weights

Gating Mechanism:
    token (d_model) → Linear → logits (n_experts) → Top-K → softmax → weights

    Only the top-K experts (default K=2) are activated per token.
    This is what makes MoE "sparse" — most experts are idle for any
    given token, saving compute while maintaining high capacity.

Load Balancing:
    Without intervention, the router might send all tokens to just one
    or two "favorite" experts, leaving others unused. The load balancing
    loss penalizes this: it encourages the router to distribute tokens
    evenly across all experts.

Usage:
    >>> router = MoERouter(d_model=512, n_experts=5, top_k=2)
    >>> x = torch.randn(4, 128, 512)  # (batch, seq_len, d_model)
    >>> weights, indices, aux_loss = router(x)
    # weights: (4, 128, 2) — gating weights for top-2 experts
    # indices: (4, 128, 2) — which 2 experts are selected
    # aux_loss: scalar — load balancing penalty
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class MoERouter(nn.Module):
    """
    Sparse Top-K gating network for Mixture-of-Experts routing.

    For each token, this module:
    1. Computes a score for each expert
    2. Selects the top-K experts with the highest scores
    3. Normalizes scores into probability weights (summing to 1)
    4. Optionally adds noise for exploration during training
    5. Computes an auxiliary load-balancing loss

    Parameters
    ----------
    d_model : int
        Input dimension (must match the transformer model dimension).
        Analogy: The "language" the router reads to make its decision.

    n_experts : int
        Total number of experts available.
        Analogy: Number of specialist offices in the hospital.

    top_k : int
        Number of experts to activate per token (the "K" in top-K).
        Higher K = more compute but potentially better quality.
        K=1: each token goes to exactly one expert (most sparse)
        K=2: each token consults two experts (good balance)
        K=n_experts: every expert processes every token (dense, no savings)

    noise_std : float
        Standard deviation of Gaussian noise added to router logits
        during training. This encourages exploration and prevents the
        router from always picking the same experts.
        Analogy: Adding a bit of randomness to help discover new experts.
        Set to 0 to disable noise.

    load_balance_weight : float
        Weight of the auxiliary load-balancing loss. This loss penalizes
        the router when tokens are distributed unevenly across experts.
        Typical values: 0.01-0.1.
        Higher = more balanced but router has less freedom to specialize.
    """

    def __init__(
        self,
        d_model: int = 512,
        n_experts: int = 5,
        top_k: int = 2,
        noise_std: float = 0.1,
        load_balance_weight: float = 0.01,
    ):
        super().__init__()

        if top_k > n_experts:
            raise ValueError(
                f"top_k ({top_k}) cannot exceed n_experts ({n_experts})"
            )
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")

        self.d_model = d_model
        self.n_experts = n_experts
        self.top_k = top_k
        self.noise_std = noise_std
        self.load_balance_weight = load_balance_weight

        # The gate: a simple linear layer that scores each expert
        # Input: token representation (d_model)
        # Output: one score per expert (n_experts)
        self.gate = nn.Linear(d_model, n_experts, bias=False)

        # Learnable noise parameter (only used during training)
        self.noise_weight = nn.Linear(d_model, n_experts, bias=False)

        self._init_weights()

    def _init_weights(self) -> None:
        """
        Initialize router weights.

        The gate is initialized with small values so that initial routing
        is approximately uniform (each expert gets ~1/n_experts of tokens).
        This prevents early commitment to a subset of experts.
        """
        nn.init.kaiming_uniform_(self.gate.weight, a=0.01)
        nn.init.zeros_(self.noise_weight.weight)

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Route tokens to top-K experts.

        Parameters
        ----------
        x : torch.Tensor
            Input token representations, shape (batch_size, seq_len, d_model).

        Returns
        -------
        tuple containing:
            weights : torch.Tensor
                Gating weights for selected experts.
                Shape: (batch_size, seq_len, top_k).
                These sum to 1 along the last dimension.

            indices : torch.Tensor
                Indices of selected experts.
                Shape: (batch_size, seq_len, top_k).
                Values in range [0, n_experts).

            aux_loss : torch.Tensor
                Auxiliary load-balancing loss (scalar).
                Should be added to the main loss during training.
        """
        # Compute raw scores for each expert
        logits = self.gate(x)  # (batch, seq_len, n_experts)

        # Add noise during training for exploration
        if self.training and self.noise_std > 0:
            noise_logits = self.noise_weight(x)
            noise = torch.randn_like(logits) * F.softplus(noise_logits) * self.noise_std
            logits = logits + noise

        # Select top-K experts per token
        top_k_logits, top_k_indices = logits.topk(self.top_k, dim=-1)
        # top_k_logits: (batch, seq_len, top_k)
        # top_k_indices: (batch, seq_len, top_k)

        # Normalize weights to sum to 1 (softmax over selected experts only)
        top_k_weights = F.softmax(top_k_logits, dim=-1)

        # Compute load-balancing auxiliary loss
        aux_loss = self._compute_load_balance_loss(logits)

        return top_k_weights, top_k_indices, aux_loss

    def _compute_load_balance_loss(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Compute the load-balancing auxiliary loss.

        This loss encourages the router to distribute tokens evenly
        across experts. Without it, the router tends to collapse to
        using only a few "favorite" experts.

        The loss is computed as:
            L_balance = n_experts * sum_i(f_i * P_i)

        Where:
            f_i = fraction of tokens routed to expert i
            P_i = mean routing probability for expert i

        If routing is perfectly balanced: f_i = P_i = 1/n_experts for all i,
        and the loss equals 1. If routing collapses to one expert:
        the loss approaches n_experts.

        Parameters
        ----------
        logits : torch.Tensor
            Raw router logits, shape (batch, seq_len, n_experts).

        Returns
        -------
        torch.Tensor
            Scalar load-balancing loss.
        """
        # Flatten batch and sequence dimensions
        flat_logits = logits.reshape(-1, self.n_experts)
        n_tokens = flat_logits.shape[0]

        if n_tokens == 0:
            return torch.tensor(0.0, device=logits.device)

        # f_i: fraction of tokens where expert i is in top-k
        top_k_indices = flat_logits.topk(self.top_k, dim=-1).indices
        # Create a one-hot mask for selected experts
        expert_mask = torch.zeros_like(flat_logits).scatter_(
            1, top_k_indices, 1.0
        )
        f = expert_mask.mean(dim=0)  # (n_experts,)

        # P_i: mean routing probability for expert i
        probs = F.softmax(flat_logits, dim=-1)
        P = probs.mean(dim=0)  # (n_experts,)

        # Load balance loss
        loss = self.n_experts * (f * P).sum()

        return loss * self.load_balance_weight

    def __repr__(self) -> str:
        return (
            f"MoERouter(d_model={self.d_model}, n_experts={self.n_experts}, "
            f"top_k={self.top_k}, noise={self.noise_std})"
        )
