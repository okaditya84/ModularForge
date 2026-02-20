# `metrics.py` — Evaluation Metrics

> **Source:** [metrics.py](file:///Users/aditya/Documents/Aditya/Python/LLM_In_Parts/modularforge/evaluation/metrics.py) · **Lines:** 226 · **Prereqs:** `assembled_model.py`

---

## What This File Does

Provides three core measurement tools: **perplexity computation** (language model quality), **memory tracking** (proving the memory-bounded guarantee), and **timing** (performance benchmarking).

---

## Perplexity (Lines 54-141)

```python
PPL = exp( (1/N) × Σ -log P(tokenᵢ | context) )
```

| PPL   | Meaning                                          |
| ----- | ------------------------------------------------ |
| 1     | Perfect prediction (impossible in practice)      |
| 30    | Good for a small model                           |
| 74    | What our model achieved — excellent for its size |
| 100   | Reasonable but room for improvement              |
| 1000+ | Essentially random guessing                      |

**Implementation:**

```python
criterion = nn.CrossEntropyLoss(ignore_index=pad_id, reduction="sum")
# "sum" not "mean" — we manually divide by token count for precision
...
avg_loss = total_loss / total_tokens
perplexity = math.exp(avg_loss)
```

**Why `reduction="sum"` instead of `"mean"`?** The default `"mean"` averages over all positions including padding (which is ignored by `ignore_index` but still affects the denominator in some edge cases). Using `"sum"` and dividing by our own `total_tokens` count (non-padding only) gives the correct average.

**Overflow protection (Lines 127-133):**

```python
if avg_loss > 100:
    return float("inf")
```

`exp(100) ≈ 2.7 × 10⁴³` — this would overflow float64. Any loss > 100 means the model is catastrophically bad, so returning infinity is correct and safe.

---

## MemoryTracker (Lines 144-201)

```python
with MemoryTracker("Assembly") as tracker:
    # do assembly
print(f"Peak: {tracker.peak_mb:.1f} MB")
```

Uses Python's `tracemalloc` module — the standard library's memory profiler. Tracks Python-level allocations (not GPU memory).

**Limitation:** Does NOT track PyTorch GPU memory. For CUDA tracking, use `torch.cuda.max_memory_allocated()`.

---

## Timer (Lines 204-226)

Simple context manager wrapping `time.time()`. Used for wall-clock timing of pipeline stages.

---

## Q&A

**Q: Why tracemalloc and not `psutil.Process().memory_info()`?**
A: `tracemalloc` tracks only Python allocations, giving you the "Python overhead." `psutil` tracks total process memory, which includes the Python interpreter, imported libraries, and OS overhead — harder to attribute to specific operations.
