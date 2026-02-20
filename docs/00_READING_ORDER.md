# ModularForge — Complete Codebase Reading Order

## How to Use This Guide

This document gives you a **sequential learning path** through every source file in ModularForge. Each file is numbered. Read them in order — each builds on the previous ones.

> **Start from #1 and work your way down. By the end, you'll understand every line of code.**

---

## 🗺️ The Big Picture First

ModularForge trains a language model by **splitting it into small independent pieces**, training each piece separately, and then **assembling them back together**. The entire pipeline has 4 steps:

```
Step 1: Prepare Data     → Download text, build vocabulary, split into partitions
Step 2: Train            → Train shared components, then each expert one-at-a-time
Step 3: Assemble         → Stream-combine all pieces into one model (O(M) memory)
Step 4: Evaluate         → Measure quality, generate text, analyze routing
```

The key innovation: **at no point does the system need more RAM than the size of ONE expert module.**

---

## 📖 Reading Order

### Phase 1: Foundation (Read First)

| #   | File                                                                                                         | What You'll Learn                                              |
| --- | ------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------- |
| 1   | [`setup.py`](file:///Users/aditya/Documents/Aditya/Python/LLM_In_Parts/setup.py)                             | How Python packages work, what dependencies ModularForge needs |
| 2   | [`configs/default.yaml`](file:///Users/aditya/Documents/Aditya/Python/LLM_In_Parts/configs/default.yaml)     | Every hyperparameter and what it controls                      |
| 3   | [`modularforge/config.py`](file:///Users/aditya/Documents/Aditya/Python/LLM_In_Parts/modularforge/config.py) | How configs are loaded, validated, and used everywhere         |

**→ Docs:** [setup_py.md](01_setup_and_config/setup_py.md) · [default_yaml.md](01_setup_and_config/default_yaml.md) · [config_py.md](01_setup_and_config/config_py.md)

---

### Phase 2: Data Pipeline (How Text Becomes Numbers)

| #   | File                                                                                                                             | What You'll Learn                                      |
| --- | -------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------ |
| 4   | [`modularforge/data/tokenizer.py`](file:///Users/aditya/Documents/Aditya/Python/LLM_In_Parts/modularforge/data/tokenizer.py)     | BPE tokenization — turning text into integer sequences |
| 5   | [`modularforge/data/dataset.py`](file:///Users/aditya/Documents/Aditya/Python/LLM_In_Parts/modularforge/data/dataset.py)         | PyTorch Dataset — fixed-length chunks for training     |
| 6   | [`modularforge/data/partitioner.py`](file:///Users/aditya/Documents/Aditya/Python/LLM_In_Parts/modularforge/data/partitioner.py) | Splitting data into expert-specific partitions         |

**→ Docs:** [tokenizer_py.md](02_data_pipeline/tokenizer_py.md) · [dataset_py.md](02_data_pipeline/dataset_py.md) · [partitioner_py.md](02_data_pipeline/partitioner_py.md)

---

### Phase 3: Model Architecture (The Neural Network)

| #   | File                                                                                                                                                 | What You'll Learn                                                 |
| --- | ---------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------- |
| 7   | [`modularforge/model/shared.py`](file:///Users/aditya/Documents/Aditya/Python/LLM_In_Parts/modularforge/model/shared.py)                             | Embedding, positional encoding, attention — the shared foundation |
| 8   | [`modularforge/model/expert.py`](file:///Users/aditya/Documents/Aditya/Python/LLM_In_Parts/modularforge/model/expert.py)                             | The FFN module — the independently trainable "specialist"         |
| 9   | [`modularforge/model/router.py`](file:///Users/aditya/Documents/Aditya/Python/LLM_In_Parts/modularforge/model/router.py)                             | MoE gating — how tokens get sent to the right expert              |
| 10  | [`modularforge/model/moe_layer.py`](file:///Users/aditya/Documents/Aditya/Python/LLM_In_Parts/modularforge/model/moe_layer.py)                       | Complete MoE transformer layer — putting it all together          |
| 11  | [`modularforge/model/module_trainer_model.py`](file:///Users/aditya/Documents/Aditya/Python/LLM_In_Parts/modularforge/model/module_trainer_model.py) | Training wrapper (frozen shared + 1 trainable expert)             |
| 12  | [`modularforge/model/assembled_model.py`](file:///Users/aditya/Documents/Aditya/Python/LLM_In_Parts/modularforge/model/assembled_model.py)           | Fully assembled MoE model for inference                           |

**→ Docs:** [shared_py.md](03_model_architecture/shared_py.md) · [expert_py.md](03_model_architecture/expert_py.md) · [router_py.md](03_model_architecture/router_py.md) · [moe_layer_py.md](03_model_architecture/moe_layer_py.md) · [module_trainer_model_py.md](03_model_architecture/module_trainer_model_py.md) · [assembled_model_py.md](03_model_architecture/assembled_model_py.md)

---

### Phase 4: Training Engine (Making the Model Learn)

| #   | File                                                                                                                                           | What You'll Learn                                                     |
| --- | ---------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------- |
| 13  | [`modularforge/training/trainer.py`](file:///Users/aditya/Documents/Aditya/Python/LLM_In_Parts/modularforge/training/trainer.py)               | Core training loop — gradient accumulation, scheduling, checkpointing |
| 14  | [`modularforge/training/shared_trainer.py`](file:///Users/aditya/Documents/Aditya/Python/LLM_In_Parts/modularforge/training/shared_trainer.py) | Phase 1: training the shared foundation                               |
| 15  | [`modularforge/training/module_trainer.py`](file:///Users/aditya/Documents/Aditya/Python/LLM_In_Parts/modularforge/training/module_trainer.py) | Phase 2: sequential expert training with memory cleanup               |

**→ Docs:** [trainer_py.md](04_training_engine/trainer_py.md) · [shared_trainer_py.md](04_training_engine/shared_trainer_py.md) · [module_trainer_py.md](04_training_engine/module_trainer_py.md)

---

### Phase 5: Assembly & Calibration (Combining the Pieces)

| #   | File                                                                                                                                     | What You'll Learn                            |
| --- | ---------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------- |
| 16  | [`modularforge/assembly/assembler.py`](file:///Users/aditya/Documents/Aditya/Python/LLM_In_Parts/modularforge/assembly/assembler.py)     | O(M) streaming assembly — the key innovation |
| 17  | [`modularforge/assembly/calibration.py`](file:///Users/aditya/Documents/Aditya/Python/LLM_In_Parts/modularforge/assembly/calibration.py) | LayerNorm recalibration after assembly       |

**→ Docs:** [assembler_py.md](05_assembly/assembler_py.md) · [calibration_py.md](05_assembly/calibration_py.md)

---

### Phase 6: Evaluation (Measuring Success)

| #   | File                                                                                                                                     | What You'll Learn                                  |
| --- | ---------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------- |
| 18  | [`modularforge/evaluation/metrics.py`](file:///Users/aditya/Documents/Aditya/Python/LLM_In_Parts/modularforge/evaluation/metrics.py)     | Perplexity, memory tracking, timing utilities      |
| 19  | [`modularforge/evaluation/generate.py`](file:///Users/aditya/Documents/Aditya/Python/LLM_In_Parts/modularforge/evaluation/generate.py)   | Text generation with sampling strategies           |
| 20  | [`modularforge/evaluation/evaluator.py`](file:///Users/aditya/Documents/Aditya/Python/LLM_In_Parts/modularforge/evaluation/evaluator.py) | Full evaluation pipeline — orchestrates everything |

**→ Docs:** [metrics_py.md](06_evaluation/metrics_py.md) · [generate_py.md](06_evaluation/generate_py.md) · [evaluator_py.md](06_evaluation/evaluator_py.md)

---

### Phase 7: CLI Scripts (Running the Pipeline)

| #   | File                                                                                                           | What You'll Learn                                         |
| --- | -------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------- |
| 21  | [`scripts/prepare_data.py`](file:///Users/aditya/Documents/Aditya/Python/LLM_In_Parts/scripts/prepare_data.py) | Step 1 CLI — downloads data, trains tokenizer, partitions |
| 22  | [`scripts/train.py`](file:///Users/aditya/Documents/Aditya/Python/LLM_In_Parts/scripts/train.py)               | Step 2 CLI — shared + expert training                     |
| 23  | [`scripts/assemble.py`](file:///Users/aditya/Documents/Aditya/Python/LLM_In_Parts/scripts/assemble.py)         | Step 3 CLI — assembly + calibration                       |
| 24  | [`scripts/evaluate.py`](file:///Users/aditya/Documents/Aditya/Python/LLM_In_Parts/scripts/evaluate.py)         | Step 4 CLI — evaluation + report                          |
| 25  | [`scripts/run_all.py`](file:///Users/aditya/Documents/Aditya/Python/LLM_In_Parts/scripts/run_all.py)           | Master script — runs the full pipeline end-to-end         |

**→ Docs:** [prepare_data_py.md](07_scripts/prepare_data_py.md) · [train_py.md](07_scripts/train_py.md) · [assemble_py.md](07_scripts/assemble_py.md) · [evaluate_py.md](07_scripts/evaluate_py.md) · [run_all_py.md](07_scripts/run_all_py.md)

---

## 🔑 Key Concepts to Know Before Starting

1. **Tensor**: A multi-dimensional array (like a matrix but can have more dimensions). PyTorch's `torch.Tensor` is the fundamental data type.
2. **Module (`nn.Module`)**: PyTorch's building block. Any neural network layer or model is a Module.
3. **Forward Pass**: Running data through the model to get predictions.
4. **Backward Pass**: Computing gradients to update model weights.
5. **Dataclass**: Python's way of creating configuration objects — like a struct with type hints.
6. **State Dict**: A dictionary mapping parameter names to their tensor values. Used to save/load models.

## 📐 Architecture at a Glance

```
                    ┌─────────────────────────────────────┐
 "The cat sat"  →   │  Token Embedding (shared)           │  → vectors
                    │  + Positional Encoding (shared)      │
                    ├─────────────────────────────────────┤
                    │  🔄 For each of 4 layers:           │
                    │    ├─ Self-Attention (shared)        │  → context-aware
                    │    ├─ Router → picks top-2 experts   │  → routing decision
                    │    └─ Expert FFN × 2 (weighted sum)  │  → specialized knowledge
                    ├─────────────────────────────────────┤
                    │  Output LayerNorm (shared)           │
                    │  LM Head (shared, weight-tied)       │
                    └─────────────────────────────────────┘
                                      ↓
                              "on" (predicted next word)
```
