#!/usr/bin/env python3
"""
ModularForge — Full Pipeline Script
======================================
Runs the complete ModularForge pipeline end-to-end:
    1. Data preparation (download, tokenize, partition)
    2. Training (shared components → expert modules)
    3. Assembly (streaming O(M) memory)
    4. Calibration (optional LayerNorm recalibration)
    5. Evaluation (perplexity, generation, routing analysis)

Usage:
    # Full pipeline:
    python scripts/run_all.py --config configs/default.yaml

    # Smoke test (5 minutes on CPU):
    python scripts/run_all.py --smoke-test

    # Skip steps:
    python scripts/run_all.py --config configs/default.yaml --skip-data --skip-shared
"""

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def run_step(name: str, cmd: list[str], cwd: str) -> float:
    """
    Run a pipeline step as a subprocess.

    Parameters
    ----------
    name : str
        Step name for logging.
    cmd : list[str]
        Command to run.
    cwd : str
        Working directory.

    Returns
    -------
    float
        Time taken in seconds.
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"STEP: {name}")
    logger.info(f"CMD: {' '.join(cmd)}")
    logger.info(f"{'='*60}\n")

    start = time.time()

    result = subprocess.run(
        cmd,
        cwd=cwd,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )

    elapsed = time.time() - start

    if result.returncode != 0:
        logger.error(f"Step '{name}' failed! (exit code {result.returncode})")
        sys.exit(1)

    logger.info(f"\n✓ {name} complete ({elapsed:.1f}s)\n")
    return elapsed


def main():
    parser = argparse.ArgumentParser(
        description="ModularForge Full Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script runs the complete ModularForge pipeline:

  1. prepare_data.py  — Download WikiText-103, tokenize, partition
  2. train.py         — Train shared components + expert modules
  3. assemble.py      — Stream-assemble into a single MoE model
  4. evaluate.py      — Compute perplexity, generate text, analyze routing

Smoke Test Mode (--smoke-test):
  Uses tiny data and minimal epochs for quick pipeline validation.
  Completes in < 5 minutes on any hardware.
  NOT representative of final quality — just checks that code works.
        """,
    )
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--skip-data", action="store_true",
                        help="Skip data preparation")
    parser.add_argument("--skip-shared", action="store_true",
                        help="Skip shared component training")
    parser.add_argument("--skip-training", action="store_true",
                        help="Skip all training")
    parser.add_argument("--skip-assembly", action="store_true",
                        help="Skip assembly")
    parser.add_argument("--skip-eval", action="store_true",
                        help="Skip evaluation")
    args = parser.parse_args()

    project_root = str(Path(__file__).resolve().parent.parent)
    python = sys.executable  # Use the same Python interpreter

    base_args = ["--smoke-test"] if args.smoke_test else ["--config", args.config]

    total_start = time.time()
    step_times = {}

    # Step 1: Data Preparation
    if not args.skip_data:
        t = run_step(
            "Data Preparation",
            [python, "scripts/prepare_data.py"] + base_args,
            cwd=project_root,
        )
        step_times["data_preparation"] = t

    # Step 2: Training
    if not args.skip_training:
        train_args = base_args.copy()
        if args.skip_shared:
            train_args.append("--skip-shared")

        t = run_step(
            "Training (Shared + Expert Modules)",
            [python, "scripts/train.py"] + train_args,
            cwd=project_root,
        )
        step_times["training"] = t

    # Step 3: Assembly
    if not args.skip_assembly:
        t = run_step(
            "Streaming Assembly + Calibration",
            [python, "scripts/assemble.py"] + base_args,
            cwd=project_root,
        )
        step_times["assembly"] = t

    # Step 4: Evaluation
    if not args.skip_eval:
        t = run_step(
            "Evaluation",
            [python, "scripts/evaluate.py"] + base_args,
            cwd=project_root,
        )
        step_times["evaluation"] = t

    # Summary
    total_time = time.time() - total_start
    step_times["total"] = total_time

    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)
    for step, t in step_times.items():
        logger.info(f"  {step}: {t:.1f}s")
    logger.info(f"\nTotal time: {total_time:.1f}s")

    if args.smoke_test:
        logger.info("\n🎉 Smoke test passed! The pipeline works end-to-end.")
        logger.info("For real training, run without --smoke-test.")


if __name__ == "__main__":
    main()
