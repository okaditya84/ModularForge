# `shared_trainer.py` — Phase 1: Training the Foundation

> **Source:** [shared_trainer.py](file:///Users/aditya/Documents/Aditya/Python/LLM_In_Parts/modularforge/training/shared_trainer.py) · **Lines:** 183 · **Prereqs:** `trainer.py`, `shared.py`, `expert.py`

---

## What This File Does

Trains the shared components (embedding, attention, norms, LM head) on the **full training corpus**. This is Phase 1 — building the foundation that all experts will stand on.

---

## The Dummy Expert Trick (Lines 57-98)

```python
class SharedTrainingModel(nn.Module):
    def __init__(self, config):
        self.shared = SharedComponents(config)
        self.temp_experts = nn.ModuleList([
            ExpertFFN(d_model=..., expert_idx=-1)   # -1 = temporary
            for i in range(config.n_layers)
        ])
```

**Problem:** Shared components alone can't make predictions — attention without FFN produces garbage because the information never gets "processed" into useful representations.

**Solution:** Add temporary "dummy" expert FFNs to complete the model. These are trained alongside the shared components, then **discarded** when saving.

**`expert_idx=-1`:** Convention for "this is not a real expert." Used for logging clarity.

### Save Only Shared Weights (Lines 160-171)

```python
def save(self, path):
    torch.save(self.model.shared.state_dict(), path)
```

**What:** Saves ONLY `self.model.shared` — the embedding, attention, norms, and LM head. The temporary FFNs are NOT saved.
**Why:** The dummy FFNs served their purpose (making training possible) but are not part of the final model. Each real expert will be trained separately.

---

## Training Flow

```python
class SharedTrainer:
    def train(self, train_loader, val_loader, output_dir):
        trainer = Trainer(
            model=self.model,      # SharedTrainingModel (shared + dummy FFNs)
            config=self.config,
            train_loader=train_loader,  # FULL corpus
            name="shared",
        )
        results = trainer.train(
            epochs=self.config.training.epochs_shared,  # default: 3 epochs
        )
```

**`epochs_shared=3`:** 3 full passes over the entire corpus. More epochs give better shared representations but take longer. 3 is a good balance for WikiText-103.

---

## Q&A

**Q: Why not skip shared training and start directly with expert training?**
A: Without pre-trained shared components, each expert would need to learn attention + embeddings from scratch — duplicating work across 5 experts. Pre-training shared components once saves time and ensures all experts use the same high-quality foundation.

**Q: Do the dummy FFN weights affect the final shared components?**
A: Yes, during shared training the dummy FFNs interact with the shared attention and embeddings. Different dummy FFN initialization could produce slightly different shared weights. In practice, this variance is small because the shared components dominate the model behavior.
