#!/usr/bin/env python3
"""
ModularForge — Data Preparation Script
========================================
Downloads WikiText-103, trains a BPE tokenizer, partitions the data,
and saves everything to disk.

This is Step 1 of the ModularForge pipeline.

Usage:
    python scripts/prepare_data.py --config configs/default.yaml
    python scripts/prepare_data.py --smoke-test  # Quick validation
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modularforge.config import ModularForgeConfig
from modularforge.data.tokenizer import ModularForgeTokenizer
from modularforge.data.partitioner import DataPartitioner
from modularforge.evaluation.metrics import MemoryTracker, Timer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_wikitext(config: ModularForgeConfig) -> dict[str, list[str]]:
    """
    Download and preprocess WikiText-103.

    Returns articles split into train/validation/test sets.

    Parameters
    ----------
    config : ModularForgeConfig
        Configuration with data settings.

    Returns
    -------
    dict with keys "train", "validation", "test", each containing
    a list of article strings.
    """
    from datasets import load_dataset

    logger.info("Downloading WikiText-103...")
    dataset = load_dataset(
        config.data.dataset_name,
        config.data.dataset_config,
        trust_remote_code=True,
    )

    result = {}
    for split in ["train", "validation", "test"]:
        raw_texts = dataset[split]["text"]

        # WikiText-103 format: articles separated by "= Title =" lines
        # We group consecutive non-empty lines into articles
        articles = _extract_articles(raw_texts, config.data.min_article_length)

        # Limit articles if configured (for smoke testing)
        if config.data.max_articles is not None:
            articles = articles[: config.data.max_articles]

        result[split] = articles
        logger.info(f"  {split}: {len(articles):,} articles")

    return result


def _extract_articles(raw_texts: list[str], min_length: int) -> list[str]:
    """
    Extract clean articles from WikiText raw format.

    WikiText-103 stores text line-by-line with article headers like
    "= Title =" and empty lines between sections. We concatenate
    consecutive content lines into articles.

    Parameters
    ----------
    raw_texts : list[str]
        Raw lines from the dataset.
    min_length : int
        Minimum character length for an article.

    Returns
    -------
    list[str]
        Clean articles.
    """
    articles = []
    current_article = []

    for line in raw_texts:
        line = line.strip()

        if not line:
            # Empty line — end of article section
            if current_article:
                article_text = " ".join(current_article)
                if len(article_text) >= min_length:
                    articles.append(article_text)
                current_article = []
        elif line.startswith("=") and line.endswith("="):
            # Article header — start new article
            if current_article:
                article_text = " ".join(current_article)
                if len(article_text) >= min_length:
                    articles.append(article_text)
                current_article = []
        else:
            current_article.append(line)

    # Don't forget the last article
    if current_article:
        article_text = " ".join(current_article)
        if len(article_text) >= min_length:
            articles.append(article_text)

    return articles


def main():
    parser = argparse.ArgumentParser(
        description="ModularForge Data Preparation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Full data preparation:
    python scripts/prepare_data.py --config configs/default.yaml

    # Quick smoke test:
    python scripts/prepare_data.py --smoke-test

    # Custom output directory:
    python scripts/prepare_data.py --config configs/default.yaml --data-dir data/my_experiment
        """,
    )
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="Run with minimal data for quick validation",
    )
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="Override data output directory",
    )
    args = parser.parse_args()

    # Load config
    if args.smoke_test:
        config = ModularForgeConfig.for_smoke_test()
        logger.info("Running in SMOKE TEST mode (minimal data)")
    else:
        config = ModularForgeConfig.from_yaml(args.config)

    if args.data_dir:
        config.data.data_dir = args.data_dir

    data_dir = Path(config.data.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    with MemoryTracker("Data preparation") as mem:
        # Step 1: Download and extract articles
        logger.info("=" * 60)
        logger.info("Step 1: Loading WikiText-103")
        logger.info("=" * 60)

        data = load_wikitext(config)

        # Step 2: Train tokenizer on full training corpus
        logger.info("=" * 60)
        logger.info("Step 2: Training BPE Tokenizer")
        logger.info("=" * 60)

        tokenizer = ModularForgeTokenizer(vocab_size=config.data.tokenizer_vocab_size)
        tokenizer.train(data["train"])

        tokenizer_path = data_dir / "tokenizer.json"
        tokenizer.save(tokenizer_path)

        # Step 3: Partition training data
        logger.info("=" * 60)
        logger.info("Step 3: Partitioning Training Data")
        logger.info("=" * 60)

        partitioner = DataPartitioner(
            n_partitions=config.model.n_experts,
            strategy=config.data.partition_strategy,
            overlap_ratio=config.data.overlap_ratio,
            seed=config.training.seed,
        )

        partitions = partitioner.partition(data["train"])

        # Save partitions
        partitions_dir = data_dir / "partitions"
        partitions_dir.mkdir(exist_ok=True)

        for i, partition in enumerate(partitions):
            partition_path = partitions_dir / f"partition_{i}.json"
            with open(partition_path, "w", encoding="utf-8") as f:
                json.dump(partition, f, ensure_ascii=False)
            logger.info(
                f"  Partition {i}: {len(partition):,} articles → {partition_path}"
            )

        # Save validation and test data
        for split in ["validation", "test"]:
            split_path = data_dir / f"{split}.json"
            with open(split_path, "w", encoding="utf-8") as f:
                json.dump(data[split], f, ensure_ascii=False)
            logger.info(f"  {split}: {len(data[split]):,} articles → {split_path}")

        # Save metadata
        metadata = {
            "dataset": config.data.dataset_name,
            "n_train_articles": len(data["train"]),
            "n_val_articles": len(data["validation"]),
            "n_test_articles": len(data["test"]),
            "n_partitions": config.model.n_experts,
            "partition_strategy": config.data.partition_strategy,
            "overlap_ratio": config.data.overlap_ratio,
            "tokenizer_vocab_size": tokenizer.actual_vocab_size,
            "partition_sizes": [len(p) for p in partitions],
        }

        meta_path = data_dir / "metadata.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

    logger.info(f"\nData preparation complete!")
    logger.info(f"  Peak memory: {mem.peak_mb:.1f} MB")
    logger.info(f"  Time: {mem.duration_seconds:.1f}s")
    logger.info(f"  Output: {data_dir}/")


if __name__ == "__main__":
    main()
