# Scripts — CLI Entry Points

> **Source:** [scripts/](file:///Users/aditya/Documents/Aditya/Python/LLM_In_Parts/scripts/) · **Prereqs:** All library docs

---

## Overview

These scripts are the user-facing CLI commands that drive the 4-step pipeline. Each wraps the corresponding library module with argument parsing, logging, and file I/O.

```bash
# Step 1: Download WikiText-103, train tokenizer, partition data
python scripts/prepare_data.py --config configs/default.yaml

# Step 2: Train shared components + all expert modules
python scripts/train.py --config configs/default.yaml

# Step 3: Assemble experts into one MoE model + calibrate
python scripts/assemble.py --config configs/default.yaml

# Step 4: Evaluate: perplexity, text generation, router stats
python scripts/evaluate.py --config configs/default.yaml

# Or run everything at once:
python scripts/run_all.py --config configs/default.yaml
```

---

## `prepare_data.py` — Step 1

[Source](file:///Users/aditya/Documents/Aditya/Python/LLM_In_Parts/scripts/prepare_data.py) · 252 lines

### What It Does

1. Downloads WikiText-103 via HuggingFace `datasets`
2. Extracts clean articles from WikiText's raw format
3. Trains a BPE tokenizer on all training articles
4. Partitions training data into N expert-specific sets
5. Saves tokenizer, partitions, validation, and test data to `data/`

### Article Extraction (Lines 80-129)

WikiText-103's raw format has articles separated by `= Title =` headers. The `_extract_articles()` function groups consecutive content lines between headers:

```python
for line in raw_texts:
    if not line.strip():
        # Empty line → save current article, start new one
    elif line.startswith("=") and line.endswith("="):
        # Article header → save current, start new
    else:
        current_article.append(line)  # Content line → accumulate
```

**`min_length` filter:** Articles shorter than 100 characters are discarded (too short to learn from).

### Output Structure

```
data/
├── tokenizer.json          ← Trained BPE tokenizer
├── metadata.json           ← Dataset statistics
├── validation.json         ← Validation articles
├── test.json               ← Test articles
└── partitions/
    ├── partition_0.json    ← Expert 0's training data
    ├── partition_1.json    ← Expert 1's training data
    └── ...
```

---

## `train.py` — Step 2

[Source](file:///Users/aditya/Documents/Aditya/Python/LLM_In_Parts/scripts/train.py) · ~200 lines

### What It Does

1. Phase 1: Creates `SharedTrainer`, trains on full corpus, saves `shared_components.pt`
2. Phase 2: Creates `ModuleTrainer`, trains each expert on its partition, saves `expert_0.pt` through `expert_4.pt`

### Key Pattern

```python
# Phase 1: Train shared components on FULL corpus
shared_trainer = SharedTrainer(config)
shared_trainer.train(full_train_loader, val_loader, output_dir)
shared_trainer.save(shared_path)

# Phase 2: Train experts SEQUENTIALLY on partitions
module_trainer = ModuleTrainer(config, shared_path, tokenizer)
module_trainer.train_all(partitions, val_texts, output_dir)
```

---

## `assemble.py` — Step 3

[Source](file:///Users/aditya/Documents/Aditya/Python/LLM_In_Parts/scripts/assemble.py) · 147 lines

### What It Does

1. Streams expert checkpoints into a single MoE model via `StreamingAssembler`
2. Optionally runs `LayerNormCalibrator` on validation data
3. Saves both `assembled_model.pt` and `calibrated_model.pt`

### `--no-calibrate` Flag

Skips calibration if you want to test the raw assembled model. Useful for A/B comparison (calibrated vs uncalibrated).

---

## `evaluate.py` — Step 4

[Source](file:///Users/aditya/Documents/Aditya/Python/LLM_In_Parts/scripts/evaluate.py) · 126 lines

### What It Does

1. Loads the best available model (calibrated > assembled)
2. Runs full evaluation via `Evaluator`
3. Saves results to `outputs/eval/`

### Smart Model Selection (Lines 90-97)

```python
calibrated = output_dir / f"calibrated_model{ext}"
assembled = output_dir / f"assembled_model{ext}"
model_path = str(calibrated if calibrated.exists() else assembled)
```

Prefers the calibrated model if it exists; falls back to the uncalibrated assembled model.

---

## Common Pattern: All Scripts

Every script follows the same structure:

1. `argparse` for CLI arguments
2. `--smoke-test` flag for quick validation
3. `--config` for YAML config path
4. `--data-dir` and `--output-dir` overrides
5. `sys.path.insert(0, ...)` for import resolution
6. Logging setup
7. Memory/time tracking

---

## Q&A

**Q: Why `sys.path.insert(0, ...)` in every script?**
A: Scripts are run from the project root (`python scripts/prepare_data.py`), but Python doesn't automatically add the parent directory to the import path. The `sys.path.insert` adds the project root so `from modularforge import ...` works.
**Alternative:** `pip install -e .` and remove `sys.path` hacks. Both approaches work; the `sys.path` approach is more self-contained.

**Q: Why is `run_all.py` a separate script?**
A: Convenience — runs all 4 steps in sequence with a single command. Internally, it imports and calls each step's `main()` function, or invokes them as subprocesses.
