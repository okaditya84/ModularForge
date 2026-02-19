#!/usr/bin/env python3
"""
ModularForge — Assembly Script
================================
Assembles trained expert modules into a single MoE model using
streaming O(M) memory. Optionally applies post-assembly calibration.

This is Step 3 of the ModularForge pipeline.

Usage:
    python scripts/assemble.py --config configs/default.yaml
    python scripts/assemble.py --smoke-test
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modularforge.config import ModularForgeConfig
from modularforge.assembly.assembler import StreamingAssembler
from modularforge.assembly.calibration import LayerNormCalibrator
from modularforge.data.tokenizer import ModularForgeTokenizer
from modularforge.data.dataset import TextDataset
from modularforge.model.assembled_model import AssembledMoEModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="ModularForge Assembly")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--no-calibrate", action="store_true",
                        help="Skip post-assembly calibration")
    args = parser.parse_args()

    if args.smoke_test:
        config = ModularForgeConfig.for_smoke_test()
    else:
        config = ModularForgeConfig.from_yaml(args.config)

    data_dir = Path(args.data_dir or config.data.data_dir)
    output_dir = Path(args.output_dir or config.assembly.output_dir)

    # Locate checkpoints
    shared_path = str(output_dir / "shared_components.pt")
    expert_paths = [
        str(output_dir / f"expert_{i}.pt")
        for i in range(config.model.n_experts)
    ]

    # Output path
    ext = ".safetensors" if config.assembly.output_format == "safetensors" else ".pt"
    assembled_path = str(output_dir / f"assembled_model{ext}")

    # ─── Assembly ───────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Streaming Assembly")
    logger.info("=" * 60)

    assembler = StreamingAssembler(config)
    assembly_results = assembler.assemble(
        shared_path=shared_path,
        expert_paths=expert_paths,
        output_path=assembled_path,
    )

    # Save assembly results
    with open(output_dir / "assembly_results.json", "w") as f:
        json.dump(assembly_results, f, indent=2)

    # ─── Calibration ────────────────────────────────────────────────
    if config.assembly.calibrate and not args.no_calibrate:
        logger.info("=" * 60)
        logger.info("Post-Assembly LayerNorm Calibration")
        logger.info("=" * 60)

        # Load assembled model
        model = AssembledMoEModel(config.model)
        model.load_from_checkpoint(assembled_path)

        # Load calibration data
        tokenizer = ModularForgeTokenizer(config.data.tokenizer_vocab_size)
        tokenizer.load(data_dir / "tokenizer.json")

        val_path = data_dir / "validation.json"
        if val_path.exists():
            with open(val_path, "r", encoding="utf-8") as f:
                cal_texts = json.load(f)[:config.assembly.calibration_samples]
        else:
            logger.warning("No validation data found for calibration. Skipping.")
            cal_texts = []

        if cal_texts:
            cal_dataset = TextDataset(
                texts=cal_texts,
                tokenizer=tokenizer,
                max_seq_len=config.model.max_seq_len,
            )

            from torch.utils.data import DataLoader
            cal_loader = DataLoader(
                cal_dataset,
                batch_size=config.training.batch_size,
                shuffle=False,
                num_workers=0,
            )

            calibrator = LayerNormCalibrator(config)
            cal_results = calibrator.calibrate(model, cal_loader)

            # Save calibrated model
            calibrated_path = str(output_dir / f"calibrated_model{ext}")
            import torch
            if ext == ".safetensors":
                from safetensors.torch import save_file
                save_file(model.state_dict(), calibrated_path)
            else:
                torch.save(
                    {"model_state_dict": model.state_dict(), "config": config.to_dict()},
                    calibrated_path,
                )

            logger.info(f"Calibrated model saved to {calibrated_path}")

            # Update assembly results
            assembly_results["calibration"] = cal_results
            with open(output_dir / "assembly_results.json", "w") as f:
                json.dump(assembly_results, f, indent=2, default=str)

    logger.info(f"\nAssembly complete! Output: {assembled_path}")


if __name__ == "__main__":
    main()
