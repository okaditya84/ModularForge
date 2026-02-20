# `calibration.py` — Post-Assembly LayerNorm Calibration

> **Source:** [calibration.py](file:///Users/aditya/Documents/Aditya/Python/LLM_In_Parts/modularforge/assembly/calibration.py) · **Lines:** 242 · **Prereqs:** `assembler.py`

---

## What This File Does

Fixes the **distribution shift** caused by combining independently trained experts. Each expert was trained with a single FFN per layer, but the assembled model routes through MULTIPLE experts with weighted outputs — producing different activation distributions than any single expert saw during training.

**The Fix:** Run a small calibration dataset through the assembled model, collect activation statistics, and adjust LayerNorm parameters. No weight updates — only normalization statistics. Fast (<1 min) and memory-light.

---

## How It Works

1. **Register hooks** on every `LayerNorm` to capture pre-normalization activations
2. **Forward pass** through 500-2000 calibration samples (no gradients)
3. **Collect statistics:** running sum and squared sum for mean/variance calculation
4. **Adjust LayerNorm:** Scale `weight` parameter based on observed variance
5. **Remove hooks** and return calibrated model

### The Hook Mechanism (Lines 126-239)

```python
hook = module.register_forward_hook(self._make_stats_hook(name, stats))
```

**Forward hooks** are PyTorch's callback mechanism. Every time a LayerNorm runs its `forward()`, the hook fires and collects the INPUT activations (before normalization).

```python
def hook(module, input, output):
    x = input[0].detach().float()
    batch_sum = x.sum(dim=list(range(x.dim() - 1)))
    batch_sq_sum = (x ** 2).sum(dim=list(range(x.dim() - 1)))
```

**`.detach().float()`:** Detach from computation graph (no gradients needed) and cast to float32 for numerical stability.

### Calibration Math (Lines 163-184)

```python
mean = stat["sum"] / stat["count"]
var = stat["sq_sum"] / stat["count"] - mean ** 2
var = torch.clamp(var, min=1e-6)

std = torch.sqrt(var + module.eps)
scale_factor = 1.0 / std.mean().item()
module.weight.data *= torch.clamp(torch.tensor(scale_factor), 0.5, 2.0)
```

**`torch.clamp(scale_factor, 0.5, 2.0)`:** Safety clamp. Prevents extreme weight changes that could destabilize the model. If calibration suggests a 10× scaling, that probably indicates a problem — better to cap at 2×.

---

## Q&A

**Q: Why calibrate LayerNorm and not other layers?**
A: LayerNorm has affine parameters (weight, bias) that assume a specific input distribution. When the distribution shifts (from single-expert to multi-expert routing), these parameters become suboptimal. Other layers (linear, embedding) don't have this distribution-dependent behavior.

**Q: How much does calibration help?**
A: Typically 1-5% perplexity improvement. Small but free — it takes <1 minute and uses negligible memory.
