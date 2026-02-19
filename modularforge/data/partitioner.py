"""
ModularForge Data Partitioner
==============================
Splits the training corpus into N disjoint partitions — one for each
expert module. This is a critical design decision because HOW you split
the data determines WHAT each expert learns.

Three Strategies (with analogies):

1. RANDOM — "Shuffled Deck"
   Shuffle all articles and deal them into N equal piles.
   + Simple, unbiased
   - Experts don't specialize; all learn similar things

2. CLUSTERED — "Specialist Departments"
   Use sentence embeddings to group similar articles, then assign
   each cluster to an expert.
   + Experts become domain specialists (code, math, history, etc.)
   + Best for diverse corpora
   - Requires sentence-transformers model (one-time cost)

3. CURRICULUM — "Easy to Hard"
   Sort articles by complexity (length, vocab rarity), then split
   into N groups from easiest to hardest.
   + Each expert handles a different difficulty level
   - Harder experts may have insufficient data for simpler patterns

Overlap:
   All strategies support a configurable overlap ratio (e.g., 10%).
   Overlapping data acts as a "bridge" between partitions, ensuring
   that experts share some common knowledge for better coherence
   when assembled.

Usage:
    >>> partitioner = DataPartitioner(n_partitions=5, strategy="clustered")
    >>> partitions = partitioner.partition(articles)
    >>> len(partitions)  # 5 lists of articles
    5
"""

from __future__ import annotations

import logging
import random
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class DataPartitioner:
    """
    Splits a list of text articles into N partitions using a specified
    strategy.

    Each partition will be used to train one expert module. The
    partitioner ensures that data is allocated fairly and that optional
    overlap is applied consistently.

    Parameters
    ----------
    n_partitions : int
        Number of partitions (= number of expert modules).
    strategy : str
        Partitioning strategy: "random", "clustered", or "curriculum".
    overlap_ratio : float
        Fraction of data to share between partitions (0.0 = no overlap).
        Each partition gets its core data + overlap_ratio × core_size
        samples randomly drawn from other partitions.
    seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        n_partitions: int = 5,
        strategy: str = "clustered",
        overlap_ratio: float = 0.1,
        seed: int = 42,
    ):
        if n_partitions < 1:
            raise ValueError(
                f"n_partitions must be >= 1, got {n_partitions}"
            )
        if strategy not in ("random", "clustered", "curriculum"):
            raise ValueError(
                f"Unknown strategy: '{strategy}'. "
                f"Choose from: random, clustered, curriculum"
            )
        if not 0.0 <= overlap_ratio <= 0.5:
            raise ValueError(
                f"overlap_ratio must be in [0, 0.5], got {overlap_ratio}. "
                f"Values above 0.5 would make partitions mostly overlapping."
            )

        self.n_partitions = n_partitions
        self.strategy = strategy
        self.overlap_ratio = overlap_ratio
        self.seed = seed

    def partition(self, articles: list[str]) -> list[list[str]]:
        """
        Partition articles into N groups.

        Parameters
        ----------
        articles : list[str]
            List of text articles to partition.

        Returns
        -------
        list[list[str]]
            N lists of articles, one per expert module.

        Raises
        ------
        ValueError
            If too few articles for the requested number of partitions.
        """
        if not articles:
            raise ValueError("Cannot partition empty article list.")

        # Filter out empty/whitespace articles
        valid_articles = [a for a in articles if a and a.strip()]
        if len(valid_articles) < self.n_partitions:
            raise ValueError(
                f"Need at least {self.n_partitions} valid articles for "
                f"{self.n_partitions} partitions, but only got "
                f"{len(valid_articles)} non-empty articles."
            )

        logger.info(
            f"Partitioning {len(valid_articles):,} articles into "
            f"{self.n_partitions} partitions (strategy={self.strategy}, "
            f"overlap={self.overlap_ratio:.0%})"
        )

        # Apply the chosen strategy
        if self.strategy == "random":
            core_partitions = self._random_partition(valid_articles)
        elif self.strategy == "clustered":
            core_partitions = self._clustered_partition(valid_articles)
        elif self.strategy == "curriculum":
            core_partitions = self._curriculum_partition(valid_articles)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        # Apply overlap (add shared samples between partitions)
        if self.overlap_ratio > 0:
            partitions = self._apply_overlap(core_partitions, valid_articles)
        else:
            partitions = core_partitions

        # Log partition statistics
        for i, part in enumerate(partitions):
            logger.info(f"  Partition {i}: {len(part):,} articles")

        return partitions

    # ─── Strategy: Random ───────────────────────────────────────────────

    def _random_partition(self, articles: list[str]) -> list[list[str]]:
        """
        Random partitioning: shuffle and split evenly.

        Analogy: Shuffling a deck of cards and dealing them into N piles.
        Each pile gets roughly the same number of cards, but the content
        of each pile is random.

        Parameters
        ----------
        articles : list[str]
            Articles to partition.

        Returns
        -------
        list[list[str]]
            N lists of articles.
        """
        rng = random.Random(self.seed)
        shuffled = articles.copy()
        rng.shuffle(shuffled)

        # Split as evenly as possible (last partition may be slightly larger)
        partitions: list[list[str]] = []
        chunk_size = len(shuffled) // self.n_partitions

        for i in range(self.n_partitions):
            start = i * chunk_size
            if i == self.n_partitions - 1:
                # Last partition gets all remaining articles
                partitions.append(shuffled[start:])
            else:
                partitions.append(shuffled[start: start + chunk_size])

        return partitions

    # ─── Strategy: Clustered ────────────────────────────────────────────

    def _clustered_partition(self, articles: list[str]) -> list[list[str]]:
        """
        Clustered partitioning: group semantically similar articles.

        Uses sentence-transformers to embed articles, then KMeans to
        cluster them. Each cluster becomes one partition.

        Analogy: Sorting books by genre — all the science books go to
        one shelf, all the history books to another, etc.

        Parameters
        ----------
        articles : list[str]
            Articles to partition.

        Returns
        -------
        list[list[str]]
            N lists of articles, each containing topically similar content.
        """
        try:
            from sentence_transformers import SentenceTransformer
            from sklearn.cluster import KMeans
        except ImportError:
            logger.warning(
                "sentence-transformers or scikit-learn not installed. "
                "Falling back to random partitioning. Install with: "
                "pip install sentence-transformers scikit-learn"
            )
            return self._random_partition(articles)

        logger.info("Computing article embeddings for clustering...")

        # Truncate articles for embedding (first 512 chars is enough for
        # topic classification, and much faster than full articles)
        truncated = [a[:512] for a in articles]

        # Use a lightweight embedding model that runs fast on CPU
        encoder = SentenceTransformer(
            "all-MiniLM-L6-v2",
            device="cpu",  # Embedding model is small, CPU is fine
        )

        embeddings = encoder.encode(
            truncated,
            batch_size=64,
            show_progress_bar=True,
            normalize_embeddings=True,
        )

        # Free the encoder to save memory
        del encoder

        logger.info(
            f"Clustering {len(embeddings)} embeddings into "
            f"{self.n_partitions} clusters..."
        )

        # KMeans clustering
        kmeans = KMeans(
            n_clusters=self.n_partitions,
            random_state=self.seed,
            n_init=10,           # Run 10 times, keep best
            max_iter=300,
        )
        labels = kmeans.fit_predict(embeddings)

        # Group articles by cluster label
        partitions: list[list[str]] = [[] for _ in range(self.n_partitions)]
        for article, label in zip(articles, labels):
            partitions[label].append(article)

        # Handle empty clusters (rare but possible with extreme data)
        empty_clusters = [i for i, p in enumerate(partitions) if len(p) == 0]
        if empty_clusters:
            logger.warning(
                f"Clusters {empty_clusters} are empty. Redistributing "
                f"articles from the largest clusters..."
            )
            partitions = self._rebalance_partitions(partitions)

        return partitions

    # ─── Strategy: Curriculum ───────────────────────────────────────────

    def _curriculum_partition(self, articles: list[str]) -> list[list[str]]:
        """
        Curriculum partitioning: sort by complexity, then split.

        Articles are scored by a complexity heuristic and sorted from
        easiest to hardest. The first partition gets the simplest
        articles, the last partition gets the most complex.

        Analogy: Organizing textbooks by grade level — 1st grade books
        in one pile, 2nd grade in another, etc.

        Complexity heuristic:
            score = avg_word_length × log(unique_words) × sqrt(total_words)
            - Simple texts: short words, small vocabulary, short articles
            - Complex texts: long words, large vocabulary, long articles

        Parameters
        ----------
        articles : list[str]
            Articles to partition.

        Returns
        -------
        list[list[str]]
            N lists of articles, ordered from simplest to most complex.
        """
        import math

        # Compute complexity score for each article
        scored_articles: list[tuple[float, str]] = []

        for article in articles:
            words = article.split()
            if not words:
                scored_articles.append((0.0, article))
                continue

            n_words = len(words)
            n_unique = len(set(words))
            avg_word_len = sum(len(w) for w in words) / n_words

            # Complexity formula: longer words × richer vocab × longer text
            score = avg_word_len * math.log(max(n_unique, 2)) * math.sqrt(n_words)
            scored_articles.append((score, article))

        # Sort by complexity (ascending = simplest first)
        scored_articles.sort(key=lambda x: x[0])

        # Split into N equal groups
        just_articles = [article for _, article in scored_articles]
        chunk_size = len(just_articles) // self.n_partitions
        partitions: list[list[str]] = []

        for i in range(self.n_partitions):
            start = i * chunk_size
            if i == self.n_partitions - 1:
                partitions.append(just_articles[start:])
            else:
                partitions.append(just_articles[start: start + chunk_size])

        return partitions

    # ─── Overlap Application ────────────────────────────────────────────

    def _apply_overlap(
        self,
        core_partitions: list[list[str]],
        all_articles: list[str],
    ) -> list[list[str]]:
        """
        Add overlapping samples to each partition.

        For each partition, we add overlap_ratio × partition_size samples
        randomly drawn from OTHER partitions. This ensures experts share
        some common knowledge for coherence.

        Analogy: Each specialist on a team has their own expertise, but
        they all attend some common meetings so they can communicate.

        Parameters
        ----------
        core_partitions : list[list[str]]
            The base (non-overlapping) partitions.
        all_articles : list[str]
            Full article list for sampling overlap.

        Returns
        -------
        list[list[str]]
            Partitions with overlap added.
        """
        rng = random.Random(self.seed + 1)  # Different seed from partitioning
        result: list[list[str]] = []

        for i, partition in enumerate(core_partitions):
            # Collect articles from OTHER partitions
            other_articles = []
            for j, other_part in enumerate(core_partitions):
                if j != i:
                    other_articles.extend(other_part)

            # Sample overlap
            n_overlap = int(len(partition) * self.overlap_ratio)
            if n_overlap > 0 and other_articles:
                n_overlap = min(n_overlap, len(other_articles))
                overlap_samples = rng.sample(other_articles, n_overlap)
                augmented = partition + overlap_samples
            else:
                augmented = partition.copy()

            result.append(augmented)

        return result

    # ─── Utilities ──────────────────────────────────────────────────────

    def _rebalance_partitions(
        self,
        partitions: list[list[str]],
    ) -> list[list[str]]:
        """
        Rebalance partitions by moving articles from the largest to
        empty partitions.

        This handles the edge case where KMeans produces empty clusters
        (can happen with very small datasets or highly uniform data).

        Parameters
        ----------
        partitions : list[list[str]]
            Potentially unbalanced partitions.

        Returns
        -------
        list[list[str]]
            Rebalanced partitions (no empty ones).
        """
        while any(len(p) == 0 for p in partitions):
            # Find the largest and an empty partition
            largest_idx = max(range(len(partitions)), key=lambda i: len(partitions[i]))
            empty_idx = next(i for i, p in enumerate(partitions) if len(p) == 0)

            # Move half of the largest partition's articles to the empty one
            half = len(partitions[largest_idx]) // 2
            if half == 0:
                # Can't split further — just duplicate
                partitions[empty_idx] = partitions[largest_idx].copy()
            else:
                partitions[empty_idx] = partitions[largest_idx][half:]
                partitions[largest_idx] = partitions[largest_idx][:half]

        return partitions

    def __repr__(self) -> str:
        return (
            f"DataPartitioner(n={self.n_partitions}, "
            f"strategy={self.strategy}, overlap={self.overlap_ratio:.0%})"
        )
