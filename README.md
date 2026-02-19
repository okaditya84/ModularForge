# 🏗️ ModularForge

**Memory-Bounded Modular Training and Zero-Retraining Assembly of Large Language Models on Resource-Constrained Hardware**

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-green.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org)

---

## The Big Idea

**What if you could train a billion-parameter language model on a laptop with 8GB of RAM?**

ModularForge makes this possible by:

1. **Decomposing** the model into small, independent expert modules (~8M params each)
2. **Training** each module sequentially on different data partitions (never more than one in memory)
3. **Assembling** all modules into a Mixture-of-Experts model via streaming (O(M) memory — never loading more than one module at a time)
4. **Achieving** coherent outputs without any retraining after assembly

> **No pre-trained base model. No retraining. No GPU required.**

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│              ModularForge Pipeline                   │
│                                                     │
│  📚 Data (WikiText-103)                             │
│   ├─ BPE Tokenizer (shared vocabulary)              │
│   └─ Partitioner (N topic-clustered splits)         │
│                                                     │
│  🏗️ Phase 1: Train Shared Components                │
│   └─ Embedding + Attention + LM Head → frozen .pt   │
│                                                     │
│  🔬 Phase 2: Train Expert Modules (sequential)      │
│   ├─ Expert 0 (frozen shared + trainable FFN)       │
│   ├─ Expert 1 (frozen shared + trainable FFN)       │
│   └─ ...Expert N (each saved, then freed)           │
│                                                     │
│  ⚡ Phase 3: Streaming Assembly                      │
│   └─ Stream experts → assembled MoE (.safetensors)  │
│                                                     │
│  📊 Phase 4: Evaluation                             │
│   └─ Perplexity + Generation + Router Analysis      │
└─────────────────────────────────────────────────────┘
```

---

## Quick Start

### 1. Setup

```bash
# Clone the repository
git clone https://github.com/your-username/ModularForge.git
cd ModularForge

# Create virtual environment
python3 -m venv env
source env/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### 2. Smoke Test (< 5 minutes, any hardware)

```bash
# Run the full pipeline with tiny data
python scripts/run_all.py --smoke-test
```

### 3. Full Training (WikiText-103)

```bash
# Step 1: Download data, train tokenizer, partition
python scripts/prepare_data.py --config configs/default.yaml

# Step 2: Train shared components + expert modules
python scripts/train.py --config configs/default.yaml

# Step 3: Assemble into MoE model
python scripts/assemble.py --config configs/default.yaml

# Step 4: Evaluate
python scripts/evaluate.py --config configs/default.yaml
```

Or run everything at once:

```bash
python scripts/run_all.py --config configs/default.yaml
```

### 4. Running on Kaggle GPU

```python
# In a Kaggle notebook:
!pip install safetensors tokenizers pyyaml sentence-transformers
!git clone https://github.com/your-username/ModularForge.git
%cd ModularForge
!pip install -e .

# Update config for GPU
!python scripts/run_all.py --config configs/default.yaml
```

---

## Running Tests

```bash
# All tests
python -m pytest tests/ -v --tb=short

# With coverage
python -m pytest tests/ -v --tb=short --cov=modularforge
```

---

## Project Structure

```
ModularForge/
├── modularforge/                  # Core library
│   ├── __init__.py                # Package overview
│   ├── config.py                  # Configuration system
│   ├── data/                      # Data pipeline
│   │   ├── __init__.py            # Flow: text → tokenize → partition → dataset
│   │   ├── tokenizer.py           # BPE tokenizer (shared vocabulary)
│   │   ├── dataset.py             # PyTorch Dataset (fixed-length sequences)
│   │   └── partitioner.py         # Data splitting (random/clustered/curriculum)
│   ├── model/                     # Neural network components
│   │   ├── __init__.py            # Architecture diagram + flow
│   │   ├── shared.py              # Shared components (embedding, attention)
│   │   ├── expert.py              # Expert FFN module (trainable unit)
│   │   ├── router.py              # MoE gating network (top-k routing)
│   │   ├── moe_layer.py           # Full MoE transformer layer
│   │   ├── module_trainer_model.py # Training wrapper (frozen + trainable)
│   │   └── assembled_model.py     # Assembled model (inference + generation)
│   ├── training/                  # Training engine
│   │   ├── __init__.py            # Two-phase training flow
│   │   ├── trainer.py             # Core training loop
│   │   ├── shared_trainer.py      # Phase 1: shared component training
│   │   └── module_trainer.py      # Phase 2: sequential expert training
│   ├── assembly/                  # Model assembly
│   │   ├── __init__.py            # Streaming assembly explanation
│   │   ├── assembler.py           # O(M) memory streaming assembly
│   │   └── calibration.py         # Post-assembly LayerNorm calibration
│   └── evaluation/                # Metrics and evaluation
│       ├── __init__.py            # What we measure and why
│       ├── metrics.py             # Perplexity, memory tracking, timing
│       ├── evaluator.py           # Full evaluation pipeline
│       └── generate.py            # Text generation utilities
├── scripts/                       # CLI entry points
│   ├── prepare_data.py            # Step 1: data download + tokenize + partition
│   ├── train.py                   # Step 2: train shared + experts
│   ├── assemble.py                # Step 3: stream-assemble + calibrate
│   ├── evaluate.py                # Step 4: evaluate + generate samples
│   └── run_all.py                 # Full pipeline (steps 1-4)
├── configs/                       # Configuration files
│   └── default.yaml               # Default 50M PoC configuration
├── tests/                         # Test suite
│   └── test_modularforge.py       # Comprehensive tests
├── requirements.txt               # Dependencies
├── setup.py                       # Package setup
├── LICENSE                        # Apache 2.0
└── README.md                      # This file
```

---

## Key Hyperparameters (Default PoC — 50M params)

| Parameter       | Value  | Description                 |
| --------------- | ------ | --------------------------- |
| `d_model`       | 512    | Model dimension             |
| `d_ff`          | 2048   | Expert FFN hidden dim       |
| `n_layers`      | 4      | Transformer blocks          |
| `n_experts`     | 5      | Number of expert modules    |
| `top_k`         | 2      | Experts activated per token |
| `vocab_size`    | 16,384 | BPE vocabulary size         |
| `max_seq_len`   | 512    | Maximum context length      |
| `batch_size`    | 16     | Training batch size         |
| `learning_rate` | 3e-4   | AdamW learning rate         |

---

## Memory Budget

| Pipeline Stage  | Peak Memory | What's Loaded                                |
| --------------- | ----------- | -------------------------------------------- |
| Shared Training | ~160 MB     | Shared model + optimizer + batch             |
| Expert Training | ~200 MB     | Frozen shared + 1 expert + optimizer + batch |
| Assembly        | ~80 MB      | Model structure + 1 expert at a time         |
| Inference       | ~200 MB     | Full assembled model (sparse computation)    |

---

## Novelty

This work introduces three technical contributions:

1. **From-scratch independent module training** — Expert modules are initialized and trained independently (no shared pre-trained base model)
2. **Streaming O(M) memory assembly** — Modules are assembled into a working model by loading one at a time (peak memory = size of one module)
3. **Zero-retraining coherence** — The assembled model produces coherent outputs without any post-assembly fine-tuning

No existing work combines all three. See the [research analysis](Idea.md) for detailed comparison with prior work (BTX, TIES-Merging, DARE, Git Re-Basin, etc.)

---

## Citation

```bibtex
@article{modularforge2025,
  title={ModularForge: Memory-Bounded Modular Training and Zero-Retraining Assembly of Large Language Models},
  author={Aditya},
  year={2025}
}
```

---

## License

Apache 2.0. See [LICENSE](LICENSE).
