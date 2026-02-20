# `evaluator.py` — Full Evaluation Pipeline

> **Source:** [evaluator.py](file:///Users/aditya/Documents/Aditya/Python/LLM_In_Parts/modularforge/evaluation/evaluator.py) · **Lines:** 350 · **Prereqs:** `metrics.py`, `generate.py`

---

## What This File Does

Orchestrates the complete evaluation: perplexity measurement, text generation, router statistics analysis, and results reporting.

---

## Evaluation Steps

```python
def evaluate(self, model, test_loader, output_dir, ...):
    # 1. Measure perplexity
    ppl = compute_perplexity(model, test_loader, self.device)

    # 2. Generate text samples from configured prompts
    generator = TextGenerator(model, self.tokenizer, self.device)
    samples = generator.generate_samples(prompts=self.config.evaluation.prompts, ...)

    # 3. Analyze router token distribution
    router_stats = self._compute_router_stats(model, test_loader)

    # 4. Save results + print report
    self._save_results(results, output_dir)
    self._print_report(results)
```

---

## Router Statistics (Lines 167-272)

Measures how evenly tokens are distributed across experts — the "health check" for the MoE routing.

```python
def _compute_router_stats(self, model, data_loader, max_batches=50):
    # Register hooks on all routers
    for layer_idx, moe_layer in enumerate(model.moe_layers):
        h = moe_layer.router.register_forward_hook(make_router_hook(layer_idx))

    # Process batches and count expert selections
    for input_ids, _ in data_loader:
        model(input_ids, ...)
        for layer_idx, indices in router_outputs.items():
            for expert_idx in range(n_experts):
                count = (indices[..., k] == expert_idx).sum().item()
                expert_counts[layer_idx, expert_idx] += count

    # Balance score: 1 - coefficient of variation
    cv = utilization.std(dim=-1) / utilization.mean(dim=-1)
    balance_scores = (1 - cv).tolist()
```

**Balance score interpretation:**

- **1.0** = perfect balance (each expert gets exactly 20% of tokens)
- **0.8** = good balance (slight skew, normal and acceptable)
- **0.5** = concerning (one expert dominates)
- **0.0** = complete collapse (all tokens go to one expert)

---

## JSON Serialization (Lines 336-349)

```python
@staticmethod
def _make_serializable(obj):
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    elif isinstance(obj, float):
        if math.isinf(obj) or math.isnan(obj):
            return str(obj)
```

**Why needed?** Python's `json.dump` can't handle PyTorch tensors, `inf`, or `NaN`. This recursive helper converts everything to JSON-compatible types.

---

## Q&A

**Q: Why only 50 batches for router stats?**
A: Processing the FULL test set is slow and unnecessary — 50 batches (~3000 sequences, ~1.5M tokens) gives a statistically reliable estimate of routing patterns. More batches would give only marginally better estimates.
