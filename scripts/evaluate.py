#!/usr/bin/env python3
"""
ModularForge — Evaluation Script
===================================
Evaluates the assembled model: computes perplexity, generates text
samples, analyzes router statistics, and produces a results report.

This is Step 4 of the ModularForge pipeline.

Usage:
    python scripts/evaluate.py --config configs/default.yaml
    python scripts/evaluate.py --smoke-test
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader

from modularforge.config import ModularForgeConfig
from modularforge.data.tokenizer import ModularForgeTokenizer
from modularforge.data.dataset import TextDataset
from modularforge.model.assembled_model import AssembledMoEModel
from modularforge.evaluation.evaluator import Evaluator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="ModularForge Evaluation")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--model-path", type=str, default=None,
                        help="Path to assembled model (overrides default)")
    args = parser.parse_args()

    if args.smoke_test:
        config = ModularForgeConfig.for_smoke_test()
    else:
        config = ModularForgeConfig.from_yaml(args.config)

    data_dir = Path(args.data_dir or config.data.data_dir)
    output_dir = Path(args.output_dir or config.assembly.output_dir)

    # Device
    device = config.training.resolve_device()

    # Load tokenizer
    tokenizer = ModularForgeTokenizer(config.data.tokenizer_vocab_size)
    tokenizer.load(data_dir / "tokenizer.json")

    # Load test data
    test_path = data_dir / "test.json"
    if test_path.exists():
        with open(test_path, "r", encoding="utf-8") as f:
            test_texts = json.load(f)
    else:
        logger.warning("No test data found, using validation data")
        val_path = data_dir / "validation.json"
        with open(val_path, "r", encoding="utf-8") as f:
            test_texts = json.load(f)

    test_dataset = TextDataset(
        texts=test_texts,
        tokenizer=tokenizer,
        max_seq_len=config.model.max_seq_len,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.evaluation.eval_batch_size,
        shuffle=False,
        num_workers=0,
    )

    # Load model
    if args.model_path:
        model_path = args.model_path
    else:
        # Try calibrated first, then assembled
        ext = ".safetensors" if config.assembly.output_format == "safetensors" else ".pt"
        calibrated = output_dir / f"calibrated_model{ext}"
        assembled = output_dir / f"assembled_model{ext}"
        model_path = str(calibrated if calibrated.exists() else assembled)

    logger.info(f"Loading model from {model_path}")
    model = AssembledMoEModel(config.model)
    model.load_from_checkpoint(model_path)

    # Load assembly results
    assembly_results = None
    assembly_path = output_dir / "assembly_results.json"
    if assembly_path.exists():
        with open(assembly_path) as f:
            assembly_results = json.load(f)

    # Run evaluation
    evaluator = Evaluator(config, tokenizer, device)
    eval_dir = str(output_dir / "eval")

    results = evaluator.evaluate(
        model=model,
        test_loader=test_loader,
        output_dir=eval_dir,
        assembly_results=assembly_results,
    )

    logger.info(f"\nEvaluation complete! Results saved to {eval_dir}/")


if __name__ == "__main__":
    main()
