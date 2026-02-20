# `trainer.py` — Core Training Loop

> **Source:** [trainer.py](file:///Users/aditya/Documents/Aditya/Python/LLM_In_Parts/modularforge/training/trainer.py) · **Lines:** 420 · **Prereqs:** All model files

---

## What This File Does

The engine that makes models learn. Handles all "mechanical" training concerns so that `SharedTrainer` and `ModuleTrainer` can focus on WHAT to train rather than HOW.

**Responsibilities:** Forward pass → loss → gradient accumulation → gradient clipping → optimizer step → LR scheduling → checkpointing → memory tracking → logging → validation.

---

## Key Concepts

### Gradient Accumulation (Lines 293-304)

```python
loss = loss / self.config.training.gradient_accumulation_steps
loss.backward()

if (batch_idx + 1) % self.config.training.gradient_accumulation_steps == 0:
    torch.nn.utils.clip_grad_norm_(...)
    self.optimizer.step()
    self.scheduler.step()
    self.optimizer.zero_grad()
```

**Problem:** You want effective batch size 64 but only have GPU memory for 16.
**Solution:** Process 4 mini-batches of 16, accumulate gradients, then update. The `/` ensures the averaged gradient magnitude matches a true batch of 64.

### Learning Rate Schedule (Lines 371-395)

```python
def lr_lambda(step):
    if step < warmup_steps:
        return step / max(warmup_steps, 1)     # Linear warmup: 0 → 1
    else:
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))  # Cosine: 1 → 0
```

```
LR
 ↑
1.0│    ╭──────╮
   │   ╱        ╲
   │  ╱ warmup   ╲ cosine decay
   │ ╱             ╲
0.0│╱               ╲──
   └──────────────────→ steps
```

**Why warmup?** Early in training, the model's representations are random. Large learning rates cause destructive updates. Warming up gradually lets the model find a stable region before ramping up.

**Why cosine decay?** Gradual slowdown near the end allows fine-tuning of already-good weights. Linear decay drops LR too aggressively in the middle; step decay is too abrupt.

### Gradient Clipping (Lines 295-299)

```python
torch.nn.utils.clip_grad_norm_(params, self.config.training.max_grad_norm)
```

**What:** If the total gradient norm exceeds `max_grad_norm` (default: 1.0), scale ALL gradients down proportionally.
**Why:** Prevents "gradient explosions" where a single bad batch produces enormous gradients that destroy the model.

### Loss Computation (Lines 275-283)

```python
self.criterion = nn.CrossEntropyLoss(ignore_index=0)
...
loss = self.criterion(logits.reshape(-1, vocab_size), target_ids.reshape(-1))
loss = loss + aux_loss  # Router load balancing
```

**`ignore_index=0`:** Token ID 0 = PAD. Don't penalize the model for predictions at padding positions — they're meaningless.
**`reshape(-1, ...)`:** Flattens batch × seq_len into one long sequence. CrossEntropyLoss doesn't care about the 2D structure.

### Validation (Lines 331-369)

```python
@torch.no_grad()
def _validate(self):
```

**`@torch.no_grad()`:** Disables gradient computation. Saves memory and time — we're only measuring performance, not learning.

### Checkpointing (Lines 397-416)

```python
checkpoint = {
    "global_step": self.global_step,
    "model_state_dict": self.model.state_dict(),
    "optimizer_state_dict": self.optimizer.state_dict(),
    "best_val_loss": self.best_val_loss,
}
```

Saves everything needed to RESUME training: model weights, optimizer states (momentum, variance for Adam), and training progress. Without optimizer states, resuming restarts the optimizer from scratch, potentially causing instability.

---

## Q&A

**Q: Why AdamW instead of SGD?**
A: AdamW has per-parameter adaptive learning rates (via momentum and RMS estimates). Transformers train poorly with SGD because different parameters need very different learning rates.

**Q: What's the difference between Adam and AdamW?**
A: Adam applies weight decay to the gradient before the Adam update; AdamW applies it directly to the weights after the update. AdamW decouples learning rate from weight decay, giving better generalization. All modern transformers use AdamW.
