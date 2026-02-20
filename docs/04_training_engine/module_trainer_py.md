# `module_trainer.py` — Phase 2: Sequential Expert Training

> **Source:** [module_trainer.py](file:///Users/aditya/Documents/Aditya/Python/LLM_In_Parts/modularforge/training/module_trainer.py) · **Lines:** 269 · **Prereqs:** `shared_trainer.py`, `module_trainer_model.py`

---

## What This File Does

Trains ALL expert modules **one at a time**, each on its own data partition. This is the core of ModularForge's memory-bounded guarantee.

---

## The Sequential Loop (Lines 87-172)

```python
def train_all(self, partitions, val_texts, output_dir):
    for expert_idx in range(self.config.model.n_experts):
        result = self._train_single_expert(expert_idx, partitions[expert_idx], ...)

        # Force memory cleanup between experts
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
```

**Why sequential?** Memory. Loading ALL experts simultaneously would use `5 × 8.4M × 4B ≈ 168 MB` just for expert params, plus optimizer states. Sequential training holds only ONE expert at a time.

### Single Expert Training (Lines 174-268)

Each call to `_train_single_expert` is **fully self-contained**:

```python
def _train_single_expert(self, expert_idx, texts, val_loader, output_dir):
    # 1. Load FRESH shared components
    shared = SharedComponents(self.config.model)
    shared.load_state_dict(torch.load(self.shared_path, ...))
    shared.freeze()
    del shared_state; gc.collect()

    # 2. Create model with new expert
    model = ModuleTrainerModel(shared=shared, config=..., expert_idx=expert_idx)

    # 3. Create dataset from this partition
    train_dataset = TextDataset(texts=texts, tokenizer=self.tokenizer, ...)
    train_loader = DataLoader(train_dataset, ...)

    # 4. Train
    trainer = Trainer(model=model, ..., name=f"expert_{expert_idx}")
    results = trainer.train(epochs=self.config.training.epochs_expert)

    # 5. Save expert checkpoint (ONLY expert weights)
    model.save_expert(expert_path)

    # 6. Clean up EVERYTHING
    del model, trainer, train_dataset, train_loader, shared
    gc.collect()
    torch.cuda.empty_cache()
```

**Why load fresh shared components for EACH expert?** Guarantees every expert trains on the exact same frozen foundation. If you reused the same `SharedComponents` object, PyTorch might accumulate hidden state or cached computations between experts.

**Why `gc.collect()` + `torch.cuda.empty_cache()`?** Python's garbage collector doesn't always free memory immediately. `gc.collect()` forces it. `empty_cache()` tells CUDA to release unused GPU memory back to the OS. Without both, memory usage would slowly creep up across experts.

**Why `del model, trainer, ...` before `gc.collect()`?** Explicitly dropping references ensures Python's garbage collector CAN free these objects. If any variable still holds a reference, `gc.collect()` can't free it.

### Memory Timeline

```
Expert 0: Load shared(72MB) + Expert(33MB) + Optimizer(67MB) = 172MB
          → Train → Save → Delete all → gc.collect → 0MB

Expert 1: Load shared(72MB) + Expert(33MB) + Optimizer(67MB) = 172MB
          → Train → Save → Delete all → gc.collect → 0MB

...peak never exceeds ~172MB for expert training...
```

### Validation Loader (Lines 128-141)

```python
val_loader = None
if val_texts:
    val_dataset = TextDataset(texts=val_texts, ...)
    val_loader = DataLoader(val_dataset, ...)
```

**Why shared validation?** All experts are evaluated on the SAME validation set for fair comparison. This lets you see which expert has the lowest loss — useful for understanding what each partition teaches.

**`pin_memory=False`:** Usually `pin_memory=True` speeds up CPU→GPU transfer, but it pins RAM that the garbage collector can't free. Since we're carefully managing memory, we disable it.

---

## Q&A

**Q: Could experts be trained in parallel on multiple GPUs?**
A: Yes! Each expert's training is independent. You could launch 5 separate processes, each on its own GPU, training different experts simultaneously. You'd need to modify the loop to use `multiprocessing` and ensure each process loads its own data partition.

**Q: What happens if training crashes mid-expert?**
A: Completed expert checkpoints are safe on disk. You'd need to modify `train_all` to skip already-completed experts (check if `expert_i.pt` exists). Currently, it would retrain from expert 0.

**Q: Does the order of expert training matter?**
A: Technically no — each expert trains independently on its own partition with freshly loaded shared components. The order doesn't affect the final model quality.
