"""
ModularForge Evaluation Metrics
=================================
Quantitative metrics for measuring the quality and efficiency of the
assembled ModularForge model.

Metrics Explained:

1. PERPLEXITY (PPL)
   How "surprised" the model is by test text. Formally:
       PPL = exp(average negative log-likelihood per token)

   Interpretation:
   - PPL=1: the model perfectly predicts every token (impossible)
   - PPL=30: on average, the model is as uncertain as choosing from
     30 equally likely words
   - PPL=100: the model is quite uncertain
   - PPL>1000: the model is basically guessing

   For our 50M MoE model on WikiText-103:
   - A monolithic baseline might achieve PPL ≈ 40-60
   - Our target: PPL < 100 (within 2× of monolithic)

2. MEMORY TRACKING
   Peak RAM usage at each pipeline stage, measured via tracemalloc.
   This is the proof that our approach is truly memory-bounded.

3. TIMING
   Wall-clock time for training, assembly, and inference.

Usage:
    >>> from modularforge.evaluation.metrics import compute_perplexity
    >>> ppl = compute_perplexity(model, test_loader, device)
    >>> print(f"Perplexity: {ppl:.2f}")
"""

from __future__ import annotations

import gc
import logging
import math
import time
import tracemalloc
from contextlib import contextmanager
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


@torch.no_grad()
def compute_perplexity(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    pad_id: int = 0,
) -> float:
    """
    Compute perplexity of a language model on a dataset.

    Perplexity is the standard metric for language model quality.
    Lower is better. It's computed as:
        PPL = exp( (1/N) × Σ -log P(token_i | context) )

    where N is the total number of non-padding tokens.

    Parameters
    ----------
    model : nn.Module
        The language model to evaluate.
    data_loader : DataLoader
        Test data loader yielding (input_ids, target_ids) pairs.
    device : torch.device
        Device to run evaluation on.
    pad_id : int
        Padding token ID (excluded from perplexity computation).

    Returns
    -------
    float
        Perplexity score. Lower = better.
        Returns float('inf') if computation fails.

    Example
    -------
    >>> ppl = compute_perplexity(model, test_loader, torch.device("cpu"))
    >>> print(f"Perplexity: {ppl:.2f}")
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    criterion = nn.CrossEntropyLoss(ignore_index=pad_id, reduction="sum")

    for input_ids, target_ids in data_loader:
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)

        # Forward pass
        outputs = model(input_ids, attention_mask=(input_ids != pad_id).long())
        if isinstance(outputs, tuple):
            logits = outputs[0]
        else:
            logits = outputs

        # Compute loss (summed, not averaged)
        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            target_ids.reshape(-1),
        )

        # Count non-padding tokens
        n_tokens = (target_ids != pad_id).sum().item()

        total_loss += loss.item()
        total_tokens += n_tokens

    if total_tokens == 0:
        logger.warning("No non-padding tokens found. Returning inf perplexity.")
        return float("inf")

    avg_loss = total_loss / total_tokens

    # Cap the loss to prevent overflow in exp()
    if avg_loss > 100:
        logger.warning(
            f"Average loss ({avg_loss:.2f}) is very high. "
            f"Perplexity will be astronomical."
        )
        return float("inf")

    perplexity = math.exp(avg_loss)
    logger.info(
        f"Perplexity: {perplexity:.2f} "
        f"(avg_loss={avg_loss:.4f}, {total_tokens:,} tokens)"
    )

    return perplexity


class MemoryTracker:
    """
    Context manager for tracking peak memory usage during a block of code.

    Uses Python's tracemalloc for accurate tracking of Python-level
    memory allocations. Note: this does NOT track GPU memory (use
    torch.cuda.max_memory_allocated for that).

    Usage:
        >>> with MemoryTracker("Assembly") as tracker:
        ...     # do assembly
        >>> print(f"Peak: {tracker.peak_mb:.1f} MB")

    Analogy:
        Like a water meter that records the highest water level during
        a flood — even after the water recedes, you know the peak.
    """

    def __init__(self, label: str = "operation"):
        """
        Parameters
        ----------
        label : str
            Human-readable label for this tracked operation.
        """
        self.label = label
        self.peak_mb: float = 0.0
        self.current_mb: float = 0.0
        self.duration_seconds: float = 0.0
        self._start_time: float = 0.0

    def __enter__(self):
        """Start tracking."""
        tracemalloc.start()
        self._start_time = time.time()
        return self

    def __exit__(self, *args):
        """Stop tracking and record results."""
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        self.current_mb = current / (1024 * 1024)
        self.peak_mb = peak / (1024 * 1024)
        self.duration_seconds = time.time() - self._start_time

        logger.info(
            f"[{self.label}] Memory: peak={self.peak_mb:.1f}MB, "
            f"current={self.current_mb:.1f}MB, "
            f"time={self.duration_seconds:.2f}s"
        )

    def __repr__(self) -> str:
        return (
            f"MemoryTracker({self.label}: "
            f"peak={self.peak_mb:.1f}MB, "
            f"time={self.duration_seconds:.2f}s)"
        )


class Timer:
    """
    Simple context manager for timing operations.

    Usage:
        >>> with Timer("Training") as t:
        ...     train()
        >>> print(f"Took: {t.elapsed:.2f}s")
    """

    def __init__(self, label: str = "operation"):
        self.label = label
        self.elapsed: float = 0.0
        self._start: float = 0.0

    def __enter__(self):
        self._start = time.time()
        return self

    def __exit__(self, *args):
        self.elapsed = time.time() - self._start
        logger.info(f"[{self.label}] Time: {self.elapsed:.2f}s")
