# `assembler.py` — Streaming O(M) Assembly

> **Source:** [assembler.py](file:///Users/aditya/Documents/Aditya/Python/LLM_In_Parts/modularforge/assembly/assembler.py) · **Lines:** 356 · **Prereqs:** `assembled_model.py`, all training docs

---

## What This File Does

The **crown jewel** of ModularForge — assembles independently trained expert modules into a single working MoE model using **O(M) memory**, where M = size of one expert. Never loads more than one expert's weights into memory at a time.

---

## Assembly Algorithm

```
Step 1: Create empty AssembledMoEModel (structure only, random weights)
Step 2: Load shared.pt → assign to model.shared → delete from memory
Step 3: For each expert i = 0..4:
    a. Load expert_i.pt (just expert weights, ~33MB)
    b. Assign to model.moe_layers[*].experts[i]
    c. Delete loaded weights from memory
    d. gc.collect()
Step 4: Initialize router weights (uniform/kaiming/data_stats)
Step 5: Save complete model
```

**Peak memory:** Only ONE expert's weights exist in RAM at any time, plus the full model structure (which holds the already-assigned weights).

### Expert Weight Mapping (Lines 161-178)

```python
for layer_idx in range(self.config.model.n_layers):
    expert_in_layer = model.moe_layers[layer_idx].experts[expert_idx]

    layer_prefix = f"{layer_idx}."
    layer_state = {}
    for key, value in experts_state_dict.items():
        if key.startswith(layer_prefix):
            new_key = key[len(layer_prefix):]
            layer_state[new_key] = value

    expert_in_layer.load_state_dict(layer_state)
```

**What:** Expert checkpoints store weights as `"0.norm.weight"`, `"0.fc1.weight"`, etc. where the prefix `"0."` is the layer index. We strip this prefix to match the `ExpertFFN`'s expected state dict format.

### Router Initialization (Lines 246-308)

Three strategies:

| Strategy       | How                                                                                                                    | When                                                |
| -------------- | ---------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------- |
| `"uniform"`    | All gate weights = 0. Router starts with equal probability for all experts                                             | Simple baseline                                     |
| `"kaiming"`    | Standard Kaiming initialization                                                                                        | Default, good general purpose                       |
| `"data_stats"` | Use training data partition centroids as gate weights. Tokens similar to partition i's centroid get routed to expert i | Best when partitions have clear semantic boundaries |

### Save Formats (Lines 310-345)

Supports both PyTorch (`.pt`) and safetensors (`.safetensors`):

- **PyTorch:** Universal compatibility but uses `pickle` (security risk with untrusted files)
- **safetensors:** Safe, fast, memory-mappable — preferred for production

---

## Q&A

**Q: Why is this O(M) and not O(model_size)?**
A: The full model structure IS in memory (all expert `nn.Module` objects exist), but their weights are assigned one at a time. Once assigned via `load_state_dict`, the loaded checkpoint data is freed. The model structure itself is small — it's just Python objects tracking parameter shapes.

**Q: What if an expert checkpoint is missing?**
A: `_validate_inputs` checks ALL files exist before starting. Failing fast is better than assembling 4 experts then crashing on #5.
