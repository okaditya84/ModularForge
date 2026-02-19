"""
ModularForge Tokenizer
=======================
Trains and manages a Byte-Pair Encoding (BPE) tokenizer for the ModularForge
pipeline. This tokenizer is SHARED across all expert modules — it defines
the common "vocabulary" that every module understands.

How BPE Works (Simple Analogy):
    Imagine you're creating a shorthand for writing. You start by knowing
    every individual letter. Then you notice "th" appears very often, so
    you create a single symbol for "th". Then "the" is common, so that
    becomes one symbol too. You keep merging frequent pairs until you
    have a vocabulary of, say, 16,384 symbols. Now you can represent
    any text using these symbols, and common words are just 1-2 symbols
    while rare words get broken into smaller pieces.

Why BPE Instead of Word-Level:
    - Handles any word, even ones never seen during training (by breaking
      them into subword pieces)
    - Compact vocabulary (16K vs 100K+ for word-level)
    - Better for morphologically rich text (e.g., "running" → "run" + "ning")

Usage:
    >>> from modularforge.data.tokenizer import ModularForgeTokenizer
    >>> tok = ModularForgeTokenizer()
    >>> tok.train(texts=["Hello world", "Machine learning is great"])
    >>> ids = tok.encode("Hello world")
    >>> text = tok.decode(ids)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional, Union

from tokenizers import Tokenizer, pre_tokenizers, models, trainers, decoders
from tokenizers.processors import TemplateProcessing

logger = logging.getLogger(__name__)

# Special token constants — used consistently across the entire pipeline
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"

SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]


class ModularForgeTokenizer:
    """
    BPE tokenizer for ModularForge.

    This tokenizer is the "lingua franca" of the entire system. Every text
    that enters the pipeline (for training, evaluation, or generation) passes
    through this tokenizer first.

    Attributes
    ----------
    tokenizer : tokenizers.Tokenizer or None
        The underlying HuggingFace tokenizer. None until train() or load().
    vocab_size : int
        Target vocabulary size.

    Important Design Decisions:
        1. We use ByteLevelBPE, which operates on bytes instead of Unicode
           characters. This means it can tokenize ANY text (including code,
           math, foreign languages) without unknown tokens.
        2. We add BOS (beginning-of-sequence) and EOS (end-of-sequence)
           tokens automatically during encoding. This helps the model
           understand where sequences start and end.
        3. Padding is always to the RIGHT of the sequence (left-aligned text).
    """

    def __init__(self, vocab_size: int = 16384):
        """
        Initialize the tokenizer.

        Parameters
        ----------
        vocab_size : int
            Target vocabulary size for BPE training.
            Larger = better coverage but larger embedding layer.
            Recommended: 8192-32768 for models under 100M params.
        """
        if vocab_size < len(SPECIAL_TOKENS):
            raise ValueError(
                f"vocab_size ({vocab_size}) must be at least "
                f"{len(SPECIAL_TOKENS)} to fit special tokens."
            )

        self.vocab_size = vocab_size
        self._tokenizer: Optional[Tokenizer] = None

    @property
    def tokenizer(self) -> Tokenizer:
        """Access the underlying tokenizer, raising if not initialized."""
        if self._tokenizer is None:
            raise RuntimeError(
                "Tokenizer not initialized. Call train() or load() first."
            )
        return self._tokenizer

    # ─── Training ───────────────────────────────────────────────────────

    def train(self, texts: list[str], min_frequency: int = 2) -> None:
        """
        Train the BPE tokenizer on a list of texts.

        This is typically called once on the FULL training corpus, before
        any data partitioning. The resulting tokenizer is then shared
        across all expert modules.

        Parameters
        ----------
        texts : list[str]
            List of text strings to train on. Each string can be an
            article, paragraph, or document.
        min_frequency : int
            Minimum frequency for a character pair to be merged.
            Higher = smaller effective vocabulary but potentially
            more unknown tokens.

        Raises
        ------
        ValueError
            If texts is empty or contains only whitespace.

        Example
        -------
        >>> tok = ModularForgeTokenizer(vocab_size=16384)
        >>> tok.train(["Hello world!", "Machine learning is great."])
        >>> print(tok.encode("Hello"))
        [2, 435, 3]  # [BOS, "Hello", EOS]
        """
        # Validate input
        if not texts:
            raise ValueError("Cannot train tokenizer on empty text list.")

        non_empty = [t for t in texts if t and t.strip()]
        if not non_empty:
            raise ValueError(
                "All provided texts are empty or whitespace-only."
            )

        logger.info(
            f"Training BPE tokenizer (vocab_size={self.vocab_size}) "
            f"on {len(non_empty):,} texts..."
        )

        # Create the BPE tokenizer
        self._tokenizer = Tokenizer(models.BPE())

        # Pre-tokenizer: split on whitespace + punctuation before BPE
        # This ensures subword tokens don't span across word boundaries
        self._tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
            add_prefix_space=False
        )

        # Decoder: converts byte-level tokens back to readable text
        self._tokenizer.decoder = decoders.ByteLevel()

        # BPE trainer configuration
        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            min_frequency=min_frequency,
            special_tokens=SPECIAL_TOKENS,
            show_progress=True,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        )

        # Train on the provided texts
        self._tokenizer.train_from_iterator(non_empty, trainer=trainer)

        # Add post-processing to automatically add BOS/EOS tokens
        bos_id = self._tokenizer.token_to_id(BOS_TOKEN)
        eos_id = self._tokenizer.token_to_id(EOS_TOKEN)

        if bos_id is None or eos_id is None:
            raise RuntimeError(
                "Failed to add special tokens to vocabulary. "
                "This is a bug — please report it."
            )

        self._tokenizer.post_processor = TemplateProcessing(
            single=f"{BOS_TOKEN} $A {EOS_TOKEN}",
            special_tokens=[
                (BOS_TOKEN, bos_id),
                (EOS_TOKEN, eos_id),
            ],
        )

        # Enable padding (right-side, to max length)
        pad_id = self._tokenizer.token_to_id(PAD_TOKEN)
        self._tokenizer.enable_padding(pad_id=pad_id, pad_token=PAD_TOKEN)

        actual_size = self._tokenizer.get_vocab_size()
        logger.info(
            f"Tokenizer trained. Actual vocab size: {actual_size} "
            f"(target: {self.vocab_size})"
        )

    # ─── Encoding / Decoding ────────────────────────────────────────────

    def encode(
        self,
        text: str,
        max_length: Optional[int] = None,
        add_special_tokens: bool = True,
    ) -> list[int]:
        """
        Encode a text string into a list of token IDs.

        Parameters
        ----------
        text : str
            The text to encode.
        max_length : int or None
            If provided, truncate or pad to this length.
            None = return natural length.
        add_special_tokens : bool
            Whether to add BOS/EOS tokens (default True).

        Returns
        -------
        list[int]
            List of integer token IDs.

        Example
        -------
        >>> tok.encode("Hello world")
        [2, 435, 1024, 3]  # [BOS, Hello, world, EOS]
        """
        if not isinstance(text, str):
            raise TypeError(f"Expected str, got {type(text).__name__}")

        # Handle empty/whitespace text gracefully
        if not text or not text.strip():
            if add_special_tokens:
                bos = self.tokenizer.token_to_id(BOS_TOKEN)
                eos = self.tokenizer.token_to_id(EOS_TOKEN)
                ids = [bos, eos]
            else:
                ids = []
        else:
            # Temporarily disable post-processing if no special tokens wanted
            if not add_special_tokens:
                saved_processor = self.tokenizer.post_processor
                self.tokenizer.post_processor = None

            encoding = self.tokenizer.encode(text)
            ids = encoding.ids

            if not add_special_tokens:
                self.tokenizer.post_processor = saved_processor

        # Truncate if needed
        if max_length is not None and len(ids) > max_length:
            ids = ids[:max_length]
            # Ensure EOS is at the end (replace last token)
            if add_special_tokens:
                eos = self.tokenizer.token_to_id(EOS_TOKEN)
                ids[-1] = eos

        # Pad if needed
        if max_length is not None and len(ids) < max_length:
            pad = self.tokenizer.token_to_id(PAD_TOKEN)
            ids = ids + [pad] * (max_length - len(ids))

        return ids

    def encode_batch(
        self,
        texts: list[str],
        max_length: Optional[int] = None,
    ) -> list[list[int]]:
        """
        Encode multiple texts in batch (more efficient than encoding one
        at a time).

        Parameters
        ----------
        texts : list[str]
            List of texts to encode.
        max_length : int or None
            If provided, truncate/pad all sequences to this length.

        Returns
        -------
        list[list[int]]
            List of token ID lists.
        """
        if not texts:
            return []

        # Use batch encoding for speed
        encodings = self.tokenizer.encode_batch(texts)
        results = []

        for enc in encodings:
            ids = enc.ids

            if max_length is not None:
                if len(ids) > max_length:
                    ids = ids[:max_length]
                    eos = self.tokenizer.token_to_id(EOS_TOKEN)
                    ids[-1] = eos
                elif len(ids) < max_length:
                    pad = self.tokenizer.token_to_id(PAD_TOKEN)
                    ids = ids + [pad] * (max_length - len(ids))

            results.append(ids)

        return results

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        """
        Decode a list of token IDs back into text.

        Parameters
        ----------
        ids : list[int]
            Token IDs to decode.
        skip_special_tokens : bool
            Whether to remove special tokens (PAD, BOS, EOS, UNK) from output.

        Returns
        -------
        str
            Decoded text string.
        """
        if not ids:
            return ""
        return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    # ─── Token ID Lookups ───────────────────────────────────────────────

    @property
    def pad_id(self) -> int:
        """ID of the padding token."""
        return self.tokenizer.token_to_id(PAD_TOKEN)

    @property
    def bos_id(self) -> int:
        """ID of the beginning-of-sequence token."""
        return self.tokenizer.token_to_id(BOS_TOKEN)

    @property
    def eos_id(self) -> int:
        """ID of the end-of-sequence token."""
        return self.tokenizer.token_to_id(EOS_TOKEN)

    @property
    def unk_id(self) -> int:
        """ID of the unknown token."""
        return self.tokenizer.token_to_id(UNK_TOKEN)

    @property
    def actual_vocab_size(self) -> int:
        """Actual vocabulary size after training (may differ from target)."""
        return self.tokenizer.get_vocab_size()

    # ─── Save / Load ────────────────────────────────────────────────────

    def save(self, path: Union[str, Path]) -> None:
        """
        Save the tokenizer to disk.

        Parameters
        ----------
        path : str or Path
            File path to save the tokenizer (e.g., "data/tokenizer.json").
            Parent directories are created automatically.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save(str(path))
        logger.info(f"Tokenizer saved to {path}")

    def load(self, path: Union[str, Path]) -> None:
        """
        Load a previously trained tokenizer from disk.

        Parameters
        ----------
        path : str or Path
            File path to load the tokenizer from.

        Raises
        ------
        FileNotFoundError
            If the tokenizer file does not exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Tokenizer file not found: {path}")

        self._tokenizer = Tokenizer.from_file(str(path))
        logger.info(
            f"Tokenizer loaded from {path} "
            f"(vocab_size={self.actual_vocab_size})"
        )

    def __repr__(self) -> str:
        status = "trained" if self._tokenizer is not None else "not trained"
        size = self.actual_vocab_size if self._tokenizer else "N/A"
        return f"ModularForgeTokenizer(vocab_size={size}, status={status})"
