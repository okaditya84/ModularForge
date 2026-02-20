# `module_trainer_model.py` — Training Wrapper (Frozen Shared + 1 Expert)

> **Source:** [module_trainer_model.py](file:///Users/aditya/Documents/Aditya/Python/LLM_In_Parts/modularforge/model/module_trainer_model.py) · **Lines:** 244 · **Prereqs:** `shared.py`, `expert.py`

---

## What This File Does

The "training harness" used during Phase 2 — wraps frozen shared components with ONE trainable expert to create a complete model that can compute loss and gradients. Only the expert parameters receive gradient updates.

```
                    ┌─────────────────────────────────────────────┐
 Input token IDs →  │ [FROZEN] Embedding + Positions              │
                    │ For each layer:                              │
                    │   [FROZEN] Self-Attention                    │
                    │   [TRAINABLE] Expert FFN ← only this learns │
                    │ [FROZEN] Output Norm + LM Head              │
                    └─────────────────────────────────────────────┘
                                        ↓
                                  Next-token logits
```

---

## Memory Budget Proof

```
Component                    Params    Memory (float32)    Gradients?
────────────────────────────────────────────────────────────────────
Shared (frozen inference)    18M       72 MB              NO  (no grad storage)
Expert (trainable)           8.4M      33 MB              YES (+ 33 MB grads)
AdamW states (2 per param)   8.4M×2    67 MB              N/A (optimizer buffers)
────────────────────────────────────────────────────────────────────
Total                        ~35M      ~205 MB
```

Without freezing, shared parameters would also need gradients (72 MB) and optimizer states (144 MB), pushing total to 516 MB — 2.5× more.

---

## Key Code

### Forward Pass (Lines 122-161)

```python
def forward(self, input_ids, attention_mask=None):
    x = self.shared.embed(input_ids)          # [FROZEN] embed tokens

    for layer_idx in range(self.config.n_layers):
        x = self.shared.apply_attention(x, layer_idx, attention_mask)  # [FROZEN]
        x = self.experts[layer_idx](x)        # [TRAINABLE] — gradients computed here

    logits = self.shared.predict(x)            # [FROZEN] predict next token
    return logits
```

**Why interleave shared attention and trainable expert?** Each layer needs BOTH attention (context-aware, shared) and FFN (knowledge, expert-specific). You can't do all attention first then all FFN — the outputs of attention layer 2 depend on the FFN output of layer 1.

### Saving Only Expert Weights (Lines 187-213)

```python
def save_expert(self, path):
    expert_state = {
        "expert_idx": self.expert_idx,
        "experts_state_dict": self.experts.state_dict(),
        "config": {"d_model": ..., "d_ff": ..., "n_layers": ..., "expert_dropout": ...},
    }
    torch.save(expert_state, path)
```

**Why save ONLY expert weights?** The shared components are already saved separately (from Phase 1). Saving them again per expert would waste disk space. Expert checkpoint ≈ 33 MB vs full model ≈ 100+ MB.

**Why include config in the checkpoint?** When loading expert weights during assembly, we need to verify that the expert was trained with the same architecture config. A mismatch (e.g., different d_ff) would cause silent shape errors.

---

## Q&A

**Q: What if someone forgets to call `shared.freeze()` before wrapping?**
A: The constructor logs a WARNING but doesn't raise an error. This is intentional — during debugging, you might want unfrozen shared components. In production, the warning alerts you.

**Q: Why one expert PER LAYER instead of one shared expert across layers?**
A: Each layer operates at a different "level of abstraction." Lower layers handle syntax, higher layers handle semantics. An expert needs different FFN weights at each level to specialize appropriately.
