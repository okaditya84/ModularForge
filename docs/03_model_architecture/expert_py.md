# `expert.py` — The Expert FFN Module

> **Source:** [expert.py](file:///Users/aditya/Documents/Aditya/Python/LLM_In_Parts/modularforge/model/expert.py) · **Lines:** 196 · **Prereqs:** `shared.py`

---

## What This File Does

Defines the **independently trainable specialist** — a Feed-Forward Network (FFN) that processes tokens after attention. Each expert is trained on a different data partition, learning different knowledge.

```
Input (512-dim) → LayerNorm → Linear(512→2048) → GELU → Dropout → Linear(2048→512) → Dropout → + Input
                  ↑ normalize    ↑ expand          ↑ activate      ↑ compress               ↑ residual
```

---

## Why a 2-Layer FFN?

The FFN is where transformers store **factual knowledge**. The attention layers decide WHAT to retrieve; the FFN layers decide the actual CONTENT.

Research (Geva et al., 2021 — "Transformer Feed-Forward Layers Are Key-Value Memories") shows that each row of `fc1.weight` is a "key" and each row of `fc2.weight` is a "value." When a token activates a key (via dot product), the corresponding value is added to the representation.

**Why 2 layers, not 3?** Standard in all transformers since "Attention Is All You Need." Adding a third layer barely helps but increases parameters by 50%.

---

## Line-by-Line Walkthrough

### Lines 87-125: `__init__`

```python
self.norm = nn.LayerNorm(d_model)
```

**Pre-norm for FFN:** Normalizes input before the FFN, matching the Pre-Norm architecture choice in `shared.py`.

```python
self.fc1 = nn.Linear(d_model, d_ff)     # 512 → 2048
self.fc2 = nn.Linear(d_ff, d_model)     # 2048 → 512
```

**Why 4× expansion (d_ff = 4 × d_model)?** This ratio was established in the original transformer paper and has become standard. The expansion creates a higher-dimensional "thinking space" where the model can represent more complex patterns, then compresses back.

**Parameter count per expert per layer:**

- `fc1`: 512 × 2048 + 2048 (bias) = 1,050,624
- `fc2`: 2048 × 512 + 512 (bias) = 1,049,088
- `norm`: 512 + 512 = 1,024
- **Total per layer:** ~2.1M params
- **Total per expert (4 layers):** ~8.4M params

```python
self.activation = nn.GELU()
```

**Why GELU instead of ReLU?**

| Activation  | Formula       | Pros                    | Cons                                        |
| ----------- | ------------- | ----------------------- | ------------------------------------------- |
| ReLU        | max(0, x)     | Simple, fast            | "Dead neurons" (outputs stuck at 0 forever) |
| **GELU** ✅ | x × Φ(x)      | Smooth, no dead neurons | Slightly slower                             |
| SwiGLU      | Used in LLaMA | Best quality            | Requires 3 linear layers instead of 2       |

GELU is the default in GPT-2, BERT, and most modern transformers. It's a smooth approximation of ReLU that allows small negative values through, preventing the "dead neuron" problem.

```python
self.dropout1 = nn.Dropout(dropout)
self.dropout2 = nn.Dropout(dropout)
```

**Two separate dropouts:** One after activation, one after the second linear. Each dropout instance has its own random mask each forward pass. Using the same dropout object for both would apply THE SAME mask, reducing the regularization effect.

---

### Lines 127-141: Weight Initialization

```python
nn.init.kaiming_normal_(self.fc1.weight, nonlinearity="relu")
nn.init.zeros_(self.fc1.bias)
```

**Why Kaiming (He) initialization?**

| Init Method       | When to Use           | Problem It Solves                                          |
| ----------------- | --------------------- | ---------------------------------------------------------- |
| Zeros             | Never for weights     | All neurons output the same thing                          |
| Xavier/Glorot     | sigmoid/tanh networks | Assumes linear activation                                  |
| **Kaiming/He** ✅ | ReLU-like activations | Accounts for the fact that ReLU zeros out half the outputs |

Kaiming scales the initial weights by `sqrt(2/fan_in)`, preventing signal explosion or collapse in deep networks. We use `nonlinearity="relu"` even though we use GELU — GELU is close enough to ReLU that the same scale factor works.

**Biases initialized to zero:** Convention. Non-zero biases can cause some neurons to be pre-activated, biasing the initial training.

---

### Lines 143-177: `forward()`

```python
residual = x
h = self.norm(x)          # Step 1: Normalize
h = self.fc1(h)            # Step 2: Expand 512 → 2048
h = self.activation(h)     # Step 3: Non-linearity
h = self.dropout1(h)       # Step 4: Regularize
h = self.fc2(h)            # Step 5: Compress 2048 → 512
h = self.dropout2(h)       # Step 6: Regularize
output = residual + h      # Step 7: Residual connection
```

**Why save `residual` before normalization?** The residual adds the ORIGINAL (pre-norm) input. This creates a "skip path" that allows gradients to flow directly from output to input without going through the FFN. Even if the FFN is poorly trained, information can still pass through unchanged.

---

### Lines 179-195: Utility Properties

```python
@property
def n_params(self) -> int:
    return sum(p.numel() for p in self.parameters())

@property
def memory_bytes(self) -> int:
    return self.n_params * 4  # 4 bytes per float32
```

**`numel()`:** Returns the total number of elements in a tensor. For a (512, 2048) weight matrix, `numel()` = 1,048,576.
**Why × 4?** Each float32 parameter uses 4 bytes. For float16, it would be × 2.

---

## Q&A

**Q: Why does each expert include its own LayerNorm?**
A: The expert's LayerNorm adapts to the expert's specific activation distribution. If all experts shared LayerNorm, the statistics would be an average across experts — suboptimal for any individual expert.

**Q: Why not use SwiGLU like LLaMA?**
A: SwiGLU uses 3 linear projections (gate + up + down) instead of 2, adding ~50% more parameters per expert. For our small model, the extra complexity isn't worth it. GELU is simpler and well-proven.

**Q: Could experts have different architectures?**
A: The current design requires all experts to have the same architecture (same `d_model`, `d_ff`) because they must be interchangeable from the router's perspective. Heterogeneous experts are an active research area.
