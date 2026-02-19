#!/usr/bin/env python3
"""
ModularForge — Training Script
================================
Trains shared components on the full corpus, then sequentially trains
each expert module on its data partition.

This is Step 2 of the ModularForge pipeline.

Usage:
    python scripts/train.py --config configs/default.yaml --data-dir data
    python scripts/train.py --smoke-test
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modularforge.config import ModularForgeConfig
from modularforge.data.tokenizer import ModularForgeTokenizer
from modularforge.data.dataset import TextDataset
from modularforge.training.shared_trainer import SharedTrainer
from modularforge.training.module_trainer import ModuleTrainer
from modularforge.evaluation.metrics import MemoryTracker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_partition_texts(data_dir: Path, n_partitions: int) -> list[list[str]]:
    """Load text partitions from disk."""
    partitions = []
    for i in range(n_partitions):
        path = data_dir / "partitions" / f"partition_{i}.json"
        if not path.exists():
            raise FileNotFoundError(
                f"Partition {i} not found at {path}. "
                f"Run scripts/prepare_data.py first."
            )
        with open(path, "r", encoding="utf-8") as f:
            partitions.append(json.load(f))
    return partitions


def main():
    parser = argparse.ArgumentParser(
        description="ModularForge Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Full training:
    python scripts/train.py --config configs/default.yaml

    # Quick smoke test:
    python scripts/train.py --smoke-test

    # Resume from existing shared checkpoint:
    python scripts/train.py --config configs/default.yaml --skip-shared
        """,
    )
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
    )
    parser.add_argument(
        "--smoke-test", action="store_true",
    )
    parser.add_argument(
        "--data-dir", type=str, default=None,
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
    )
    parser.add_argument(
        "--skip-shared", action="store_true",
        help="Skip shared component training (use existing checkpoint)",
    )
    args = parser.parse_args()

    # Load config
    if args.smoke_test:
        config = ModularForgeConfig.for_smoke_test()
    else:
        config = ModularForgeConfig.from_yaml(args.config)

    data_dir = Path(args.data_dir or config.data.data_dir)
    output_dir = Path(args.output_dir or config.assembly.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load tokenizer
    tokenizer_path = data_dir / "tokenizer.json"
    tokenizer = ModularForgeTokenizer(vocab_size=config.data.tokenizer_vocab_size)
    tokenizer.load(tokenizer_path)

    # Load data
    val_path = data_dir / "validation.json"
    val_texts = []
    if val_path.exists():
        with open(val_path, "r", encoding="utf-8") as f:
            val_texts = json.load(f)

    shared_path = str(output_dir / "shared_components.pt")

    # ─── Phase 1: Train Shared Components ───────────────────────────
    if not args.skip_shared:
        logger.info("=" * 60)
        logger.info("Phase 1: Training Shared Components")
        logger.info("=" * 60)

        # Load full training corpus
        partitions = load_partition_texts(data_dir, config.model.n_experts)
        all_train_texts = []
        for partition in partitions:
            all_train_texts.extend(partition)

        # Remove duplicates (from overlap)
        all_train_texts = list(set(all_train_texts))
        logger.info(f"Full training corpus: {len(all_train_texts):,} unique articles")

        # Create datasets
        train_dataset = TextDataset(
            texts=all_train_texts,
            tokenizer=tokenizer,
            max_seq_len=config.model.max_seq_len,
        )

        from torch.utils.data import DataLoader

        train_loader = DataLoader(
            train_dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=min(config.training.num_workers, 4),
            pin_memory=False,
            drop_last=True,
        )

        val_loader = None
        if val_texts:
            val_dataset = TextDataset(
                texts=val_texts[:500],  # Use subset for faster validation
                tokenizer=tokenizer,
                max_seq_len=config.model.max_seq_len,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=config.training.batch_size,
                shuffle=False,
                num_workers=0,
            )

        # Train
        shared_trainer = SharedTrainer(config)

        with MemoryTracker("Shared training") as mem:
            shared_results = shared_trainer.train(
                train_loader=train_loader,
                val_loader=val_loader,
                output_dir=str(output_dir),
            )

        shared_trainer.save(shared_path)

        logger.info(
            f"Shared training complete: "
            f"loss={shared_results['final_train_loss']:.4f}, "
            f"peak_mem={mem.peak_mb:.1f}MB, "
            f"time={mem.duration_seconds:.1f}s"
        )

        # Save results
        with open(output_dir / "shared_results.json", "w") as f:
            json.dump({
                "final_train_loss": shared_results["final_train_loss"],
                "final_val_loss": shared_results.get("final_val_loss"),
                "peak_memory_mb": mem.peak_mb,
                "time_seconds": mem.duration_seconds,
            }, f, indent=2)

        # Free memory
        del shared_trainer, train_loader, train_dataset
        if val_loader:
            del val_loader, val_dataset
        import gc
        gc.collect()

    else:
        logger.info(f"Skipping shared training, using: {shared_path}")

    # ─── Phase 2: Train Expert Modules ──────────────────────────────
    logger.info("=" * 60)
    logger.info("Phase 2: Training Expert Modules (Sequential)")
    logger.info("=" * 60)

    partitions = load_partition_texts(data_dir, config.model.n_experts)

    module_trainer = ModuleTrainer(
        config=config,
        shared_path=shared_path,
        tokenizer=tokenizer,
    )

    with MemoryTracker("Expert training") as mem:
        expert_results = module_trainer.train_all(
            partitions=partitions,
            val_texts=val_texts[:500] if val_texts else None,
            output_dir=str(output_dir),
        )

    # Save expert training results
    for i, result in enumerate(expert_results):
        result_clean = {
            "expert_idx": i,
            "final_train_loss": result.get("final_train_loss"),
            "final_val_loss": result.get("final_val_loss"),
            "peak_memory_mb": result.get("peak_memory_mb", 0),
            "time_seconds": result.get("total_time_seconds", 0),
        }
        with open(output_dir / f"expert_{i}_results.json", "w") as f:
            json.dump(result_clean, f, indent=2)

    logger.info(
        f"\nAll training complete!"
        f"\n  Peak memory: {mem.peak_mb:.1f}MB"
        f"\n  Total time: {mem.duration_seconds:.1f}s"
        f"\n  Outputs: {output_dir}/"
    )


if __name__ == "__main__":
    main()
