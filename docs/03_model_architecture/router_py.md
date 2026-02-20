# `router.py` — MoE Token Router

> **Source:** [router.py](file:///Users/aditya/Documents/Aditya/Python/LLM_In_Parts/modularforge/model/router.py) · **Lines:** 244 · **Prereqs:** `expert.py`

---

## What This File Does

The router is the **traffic controller** of the MoE model. For each token, it decides: (1) which experts should process this token, and (2) how much weight to give each expert's output.

```
Token "physics" → Router → [Expert 0: 0.7, Expert 3: 0.3]
                           (70% science expert, 30% general expert)
```

---

## Core Mechanism

```
token_vector (512-dim) → Linear → logits (5 values, one per expert) → Top-K(2) → Softmax → weights
```

1. **Linear layer** (`gate`): maps each 512-dim token to 5 scores (one per expert)
2. **Top-K selection**: picks the 2 highest-scoring experts
3. **Softmax over selected**: normalizes the 2 scores to sum to 1.0
4. **Only 2 experts run**: the other 3 are completely skipped (sparse computation)

---

## Key Design Decisions

### Why Top-K = 2?

| K        | Active Compute       | Quality             | Trade-off                                           |
| -------- | -------------------- | ------------------- | --------------------------------------------------- |
| 1        | 20% of total experts | Good for efficiency | May lose nuance — one expert must handle everything |
| **2** ✅ | 40% of total experts | Good balance        | Each token gets a "primary" and "secondary" opinion |
| 3        | 60% of total experts | Diminishing returns | Approaching dense computation cost                  |
| 5 (all)  | 100% — fully dense   | Maximum             | No efficiency gain, not MoE at all                  |

K=2 is the standard in Switch Transformer (Google) and GShard. It gives each token access to two "perspectives" while keeping compute sparse.

### Noise for Exploration (Lines 168-171)

```python
if self.training and self.noise_std > 0:
    noise_logits = self.noise_weight(x)
    noise = torch.randn_like(logits) * F.softplus(noise_logits) * self.noise_std
    logits = logits + noise
```

**Problem without noise:** The router might converge early to always selecting the same 1-2 "favorite" experts, leaving others unused. This is called **expert collapse**.

**Solution:** Add learned noise during training to encourage exploration. `F.softplus` ensures the noise scale is always positive. The noise is only added during training (`self.training`), not during inference.

**`noise_weight` (Lines 120):** A separate linear layer that learns how much noise to add per token. Some tokens may need more exploration (ambiguous topics) while others are clearly routed.

### Load Balancing Loss (Lines 186-237)

```python
def _compute_load_balance_loss(self, logits):
    # f_i = fraction of tokens where expert i is in top-k
    expert_mask = torch.zeros_like(flat_logits).scatter_(1, top_k_indices, 1.0)
    f = expert_mask.mean(dim=0)

    # P_i = mean routing probability for expert i
    probs = F.softmax(flat_logits, dim=-1)
    P = probs.mean(dim=0)

    # Loss = n_experts × Σ(f_i × P_i)
    loss = self.n_experts * (f * P).sum()
    return loss * self.load_balance_weight
```

**The Math:** `L = N × Σ(fᵢ × Pᵢ)` where:

- `fᵢ` = fraction of tokens actually routed to expert i
- `Pᵢ` = average routing probability for expert i
- `N` = number of experts

**Intuition:** If all experts get equal traffic: `fᵢ = Pᵢ = 1/N`, so `L = N × N × (1/N × 1/N) = 1`. If ONE expert gets all traffic: `L → N` (maximum penalty).

**`load_balance_weight=0.01`:** The balance loss is tiny compared to the language modeling loss. Too high (>0.1) forces uniform routing even when some experts are genuinely better for certain tokens.

---

## Q&A

**Q: Why not use a softmax over ALL experts (dense routing)?**
A: Dense routing means running all 5 experts for every token — 5× the compute. MoE's whole point is that only K experts run per token, giving capacity without proportional compute cost.

**Q: What if the router always picks the same 2 experts?**
A: The load balancing loss penalizes this. Additionally, training noise encourages exploration of all experts.

**Q: Is the router trained during expert training (Phase 2)?**
A: No. The router is only initialized during assembly (Phase 3) and is either calibrated post-assembly or fine-tuned. During individual expert training, there IS no router.
