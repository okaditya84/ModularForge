"""
ModularForge Dataset
====================
PyTorch Dataset class that converts raw text articles into fixed-length
token sequences suitable for language model training.

How It Works (Analogy):
    Imagine you have a long book and you want to read it in pages of
    exactly 512 words each. You'd read the first 512 words as page 1,
    then start page 2 from word 513, and so on. If the last page is
    shorter than 512 words, you pad it with blank space.

    That's exactly what this dataset does with tokens instead of words.

Sequence Construction:
    Article text → Tokenize → Split into fixed-length chunks → Tensor

    Example (max_seq_len=8, simplified):
        "The cat sat on the mat and slept peacefully"
        → tokens: [BOS, The, cat, sat, on, the, mat, and, slept, peacefully, EOS]
        → chunk 1: [BOS, The, cat, sat, on, the, mat, and]     (input)
                    [The, cat, sat, on, the, mat, and, slept]    (target)
        → chunk 2: [slept, peacefully, EOS, PAD, PAD, PAD, PAD, PAD]  (input)
                    [peacefully, EOS, PAD, PAD, PAD, PAD, PAD, PAD]    (target)

    The target is always the input shifted by 1 position (next-token prediction).

Usage:
    >>> from modularforge.data.dataset import TextDataset
    >>> dataset = TextDataset(texts=["Hello world"], tokenizer=tok, max_seq_len=64)
    >>> input_ids, target_ids = dataset[0]
"""

from __future__ import annotations

import logging
from typing import Optional, Union

import torch
from torch.utils.data import Dataset

from modularforge.data.tokenizer import ModularForgeTokenizer

logger = logging.getLogger(__name__)


class TextDataset(Dataset):
    """
    Converts a list of text articles into fixed-length token sequences
    for language model training.

    Each item returned is a pair (input_ids, target_ids) where target_ids
    is input_ids shifted right by 1 — the standard next-token prediction
    setup for autoregressive language models.

    Attributes
    ----------
    sequences : list[torch.Tensor]
        Pre-tokenized and chunked sequences.
    max_seq_len : int
        Length of each sequence.
    pad_id : int
        Token ID used for padding.

    Design Decision:
        We pre-tokenize everything and store it in memory as tensors.
        For WikiText-103 (~103M tokens), this uses ~400MB at int32.
        If memory is tight, you could switch to on-the-fly tokenization,
        but pre-tokenization is much faster for repeated epochs.
    """

    def __init__(
        self,
        texts: list[str],
        tokenizer: ModularForgeTokenizer,
        max_seq_len: int = 512,
        stride: Optional[int] = None,
    ):
        """
        Initialize the dataset by tokenizing and chunking all texts.

        Parameters
        ----------
        texts : list[str]
            Raw text articles/documents.
        tokenizer : ModularForgeTokenizer
            Trained tokenizer instance.
        max_seq_len : int
            Fixed sequence length for all chunks. Longer articles are
            split into multiple chunks; shorter ones are padded.
            Analogy: The "page size" of your book.
        stride : int or None
            Step size between consecutive chunks (for overlapping windows).
            None defaults to max_seq_len (no overlap — most memory-efficient).
            Using max_seq_len // 2 gives 50% overlap (better coverage but
            2× more sequences).
            Analogy: How far you slide the reading window between pages.

        Raises
        ------
        ValueError
            If texts is empty or max_seq_len is too small.
        """
        if not texts:
            raise ValueError("Cannot create dataset from empty text list.")
        if max_seq_len < 4:
            raise ValueError(
                f"max_seq_len must be >= 4 (need room for BOS + token + token + EOS), "
                f"got {max_seq_len}"
            )

        self.max_seq_len = max_seq_len
        self.pad_id = tokenizer.pad_id
        self.stride = stride if stride is not None else max_seq_len

        if self.stride < 1:
            raise ValueError(f"stride must be >= 1, got {self.stride}")

        # Tokenize all texts and chunk into fixed-length sequences
        self.sequences: list[torch.Tensor] = []
        self._build_sequences(texts, tokenizer)

        logger.info(
            f"Dataset created: {len(self.sequences):,} sequences "
            f"(seq_len={max_seq_len}, stride={self.stride})"
        )

    def _build_sequences(
        self,
        texts: list[str],
        tokenizer: ModularForgeTokenizer,
    ) -> None:
        """
        Tokenize all texts and split into fixed-length chunks.

        We concatenate all articles into one long token stream (with EOS
        between articles), then slice it into non-overlapping windows.
        This maximizes data utilization — no tokens are wasted in short
        articles.

        Parameters
        ----------
        texts : list[str]
            Raw text articles.
        tokenizer : ModularForgeTokenizer
            Trained tokenizer.
        """
        # Concatenate all tokens into one long stream
        all_tokens: list[int] = []

        for text in texts:
            if not text or not text.strip():
                continue

            tokens = tokenizer.encode(text, add_special_tokens=True)
            all_tokens.extend(tokens)

        if not all_tokens:
            logger.warning("All texts were empty after tokenization.")
            return

        total_tokens = len(all_tokens)
        logger.info(f"Total tokens: {total_tokens:,}")

        # Slice into fixed-length chunks
        # We need seq_len + 1 tokens per chunk (input = first seq_len,
        # target = last seq_len, shifted by 1)
        chunk_len = self.max_seq_len + 1

        for start in range(0, total_tokens - chunk_len + 1, self.stride):
            chunk = all_tokens[start: start + chunk_len]
            self.sequences.append(torch.tensor(chunk, dtype=torch.long))

        # Handle the last partial chunk (pad if necessary)
        remainder_start = len(self.sequences) * self.stride if self.sequences else 0
        if remainder_start < total_tokens:
            remaining = all_tokens[remainder_start:]
            if len(remaining) >= 2:  # Need at least 2 tokens for input/target pair
                padded = remaining + [self.pad_id] * (chunk_len - len(remaining))
                self.sequences.append(torch.tensor(padded, dtype=torch.long))

    def __len__(self) -> int:
        """Number of sequences in the dataset."""
        return len(self.sequences)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single (input, target) pair.

        Parameters
        ----------
        idx : int
            Sequence index.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            input_ids: shape (max_seq_len,) — the input sequence
            target_ids: shape (max_seq_len,) — the target (shifted by 1)

        Example
        -------
        If the stored chunk is [BOS, A, B, C, D, EOS, PAD, PAD, PAD]:
            input_ids  = [BOS, A, B, C, D, EOS, PAD, PAD]
            target_ids = [A, B, C, D, EOS, PAD, PAD, PAD]
        """
        if idx < 0 or idx >= len(self.sequences):
            raise IndexError(
                f"Index {idx} out of range for dataset of size "
                f"{len(self.sequences)}"
            )

        chunk = self.sequences[idx]
        input_ids = chunk[:-1]   # All but last token
        target_ids = chunk[1:]   # All but first token (shifted by 1)

        return input_ids, target_ids

    @property
    def total_tokens(self) -> int:
        """Total number of tokens across all sequences (including padding)."""
        return len(self.sequences) * self.max_seq_len

    def get_attention_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Create an attention mask for the given input IDs.

        Tokens that are NOT padding get mask=1, padding tokens get mask=0.
        This tells the attention mechanism to ignore padding.

        Parameters
        ----------
        input_ids : torch.Tensor
            Token IDs, shape (seq_len,) or (batch_size, seq_len).

        Returns
        -------
        torch.Tensor
            Attention mask with same shape as input_ids.
        """
        return (input_ids != self.pad_id).long()

    def __repr__(self) -> str:
        return (
            f"TextDataset(sequences={len(self.sequences):,}, "
            f"seq_len={self.max_seq_len}, "
            f"total_tokens=~{self.total_tokens:,})"
        )
