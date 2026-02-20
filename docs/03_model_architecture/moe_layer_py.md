# `moe_layer.py` — Complete MoE Transformer Layer

> **Source:** [moe_layer.py](file:///Users/aditya/Documents/Aditya/Python/LLM_In_Parts/modularforge/model/moe_layer.py) · **Lines:** 215 · **Prereqs:** `expert.py`, `router.py`

---

## What This File Does

Combines the **router** and **expert pool** into one complete MoE transformer layer. This is one "floor" of the assembled model building.

```
Input → FFN Norm → Router (selects 2 experts) → Selected Experts → Weighted Combine → + Input → Output
```

---

## Sparse Expert Forward (Lines 130-200)

This is the most performance-critical function in the codebase. It efficiently dispatches tokens to their selected experts.

### The Algorithm

```python
for expert_idx, expert in enumerate(self.experts):
    expert_mask = (flat_indices == expert_idx)    # Which tokens go to this expert?

    for k in range(self.config.top_k):            # For each k-slot
        slot_mask = expert_mask[:, k]              # Tokens where this expert is in slot k

        if not slot_mask.any():
            continue                                # Skip if no tokens assigned

        token_indices = slot_mask.nonzero(as_tuple=True)[0]
        expert_input = flat_x[token_indices]       # Gather assigned tokens

        expert_out = expert(expert_input) - expert_input  # Get expert's CONTRIBUTION only

        slot_weights = flat_weights[token_indices, k].unsqueeze(-1)
        flat_output[token_indices] += slot_weights * expert_out  # Weighted add
```

**Key insight: `expert(expert_input) - expert_input`**
The `ExpertFFN` includes a residual connection internally (`output = x + ffn(x)`). But the MoE layer adds its OWN residual connection (`output = x + moe_output`). If we used the expert's full output, we'd add the input TWICE. Subtracting the input gives us just the expert's learned contribution.

**Why iterate over experts (not tokens)?** Batching! If 100 tokens are assigned to Expert 0, we process them ALL in one forward pass through Expert 0. This is much faster than processing each token individually.

**Why `flat_x.reshape(-1, d_model)`?** Flattening batch and sequence dimensions lets us index individual tokens by their flat position, regardless of which batch element they came from.

---

## Memory Analysis

During inference, all experts are in memory (since they're all part of the assembled model). The compute savings come from only RUNNING 2 out of 5 experts per token.

```
Total params per MoE layer:
  Router:      512 × 5 × 2 = 5,120 (gate + noise weights)
  5 Experts:   5 × 2.1M = 10.5M
  FFN Norm:    512 + 512 = 1,024
  Total:       ~10.5M per layer

Active per token: Router + 2 Experts = 5K + 4.2M ≈ 4.2M
```

---

## Q&A

**Q: Why not use a `torch.scatter` approach instead of the for-loop?**
A: `torch.scatter` could be faster for large expert counts but is harder to read/debug. With 5 experts and 2 k-slots, the loop runs only 10 iterations total — negligible overhead. For 100+ experts (like GShard), you'd want a fully batched scatter approach.

**Q: What if an expert gets NO tokens?**
A: The `if not expert_mask.any(): continue` check handles this gracefully. The expert is simply skipped, contributing nothing to the output. This is normal — not every expert is relevant for every batch.
