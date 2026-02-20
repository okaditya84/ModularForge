#!/usr/bin/env python3
"""
ModularForge — Kaggle Resume Wrapper
=======================================
Copy-paste this into a Kaggle notebook cell, or run it as the main
script in a "Save & Run All" notebook to resume interrupted training.

This script:
    1. Clones/updates the ModularForge repo
    2. Installs requirements
    3. Copies checkpoints from the Kaggle dataset mount
    4. Runs resume_train.py to complete training
    5. Packages outputs for saving

Kaggle Setup:
    1. Create a new notebook with GPU (T4)
    2. Add your "modularforge-checkpoints" dataset as input
    3. Add your "modularforge-data" dataset as input (if separate)
    4. Paste this script into a code cell
    5. Click "Save & Run All" to run in background

    Alternatively, set this as the notebook's main script and
    use "Save & Run All (Commit)" for unattended execution.
"""

import os
import subprocess
import sys
import time

# =============================================================================
# Configuration — EDIT THESE
# =============================================================================

# Your GitHub repo URL
REPO_URL = "https://github.com/YOUR_USERNAME/ModularForge.git"

# Kaggle paths (edit if your dataset names differ)
KAGGLE_CHECKPOINT_DIR = "/kaggle/input/modularforge-checkpoints/outputs"
KAGGLE_DATA_DIR = "/kaggle/input/modularforge-data/data"

# Where to clone the repo on Kaggle
WORK_DIR = "/kaggle/working/ModularForge"

# Config file (relative to repo root)
CONFIG = "configs/default.yaml"

# =============================================================================
# Helper
# =============================================================================

def run(cmd: str, cwd: str = None) -> None:
    """Run a shell command, raising on failure."""
    print(f"\n>>> {cmd}")
    result = subprocess.run(
        cmd, shell=True, cwd=cwd,
        stdout=sys.stdout, stderr=sys.stderr,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Command failed (exit {result.returncode}): {cmd}")


# =============================================================================
# Main
# =============================================================================

def main():
    start = time.time()

    # Step 1: Clone or update repo
    if os.path.exists(WORK_DIR):
        print("Repository already cloned, pulling latest...")
        run("git pull", cwd=WORK_DIR)
    else:
        print("Cloning repository...")
        run(f"git clone {REPO_URL} {WORK_DIR}")

    # Step 2: Install requirements
    print("\nInstalling requirements...")
    req_file = os.path.join(WORK_DIR, "requirements.txt")
    if os.path.exists(req_file):
        run(f"pip install -q -r {req_file}")
    else:
        run("pip install -q torch pyyaml tokenizers datasets safetensors")

    # Step 3: Determine data directory
    # Check if data exists in checkpoint dataset or separate dataset
    data_dir = None
    possible_data_dirs = [
        KAGGLE_DATA_DIR,
        os.path.join(os.path.dirname(KAGGLE_CHECKPOINT_DIR), "data"),
        "/kaggle/input/modularforge-checkpoints/data",
        os.path.join(WORK_DIR, "data"),
    ]
    for d in possible_data_dirs:
        if os.path.exists(d) and os.path.exists(os.path.join(d, "tokenizer.json")):
            data_dir = d
            print(f"Found data directory: {data_dir}")
            break

    if data_dir is None:
        print("ERROR: Could not find data directory with tokenizer.json!")
        print(f"Searched: {possible_data_dirs}")
        sys.exit(1)

    # Step 4: Build the resume command
    resume_cmd = [
        sys.executable,
        "scripts/resume_train.py",
        "--config", CONFIG,
        "--data-dir", data_dir,
        "--output-dir", os.path.join(WORK_DIR, "outputs"),
    ]

    # Add checkpoint directory if it exists
    if os.path.exists(KAGGLE_CHECKPOINT_DIR):
        resume_cmd.extend(["--checkpoint-dir", KAGGLE_CHECKPOINT_DIR])
        print(f"Will copy checkpoints from: {KAGGLE_CHECKPOINT_DIR}")
    else:
        print(f"WARNING: Checkpoint dir not found: {KAGGLE_CHECKPOINT_DIR}")
        print("Training will start from scratch for all experts.")

    # Step 5: Run resume training
    print("\n" + "=" * 60)
    print("STARTING RESUME TRAINING")
    print("=" * 60 + "\n")

    run(" ".join(resume_cmd), cwd=WORK_DIR)

    # Step 6: Summary
    elapsed = time.time() - start
    print("\n" + "=" * 60)
    print(f"KAGGLE RESUME COMPLETE — {elapsed:.0f}s ({elapsed/3600:.1f}h)")
    print("=" * 60)
    print(f"Outputs saved in: {os.path.join(WORK_DIR, 'outputs')}")
    print("\nTo save outputs as a Kaggle dataset:")
    print("  1. Click 'Save Version' in the notebook")
    print("  2. Select 'Save & Run All (Commit)'")
    print("  3. Your outputs will be available at:")
    print(f"     /kaggle/working/ModularForge/outputs/")


if __name__ == "__main__":
    main()
