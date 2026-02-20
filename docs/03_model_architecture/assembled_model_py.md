# `assembled_model.py` — The Fully Assembled MoE Model

> **Source:** [assembled_model.py](file:///Users/aditya/Documents/Aditya/Python/LLM_In_Parts/modularforge/model/assembled_model.py) · **Lines:** 375 · **Prereqs:** `shared.py`, `moe_layer.py`

---

## What This File Does

The **final product** of the ModularForge pipeline — a complete MoE language model that can do inference and text generation. It combines shared components with ALL experts and routers.

---

## Architecture

```python
self.shared = SharedComponents(config)
self.moe_layers = nn.ModuleList([
    MoETransformerLayer(config, layer_idx=i)
    for i in range(config.n_layers)
])
```

Each `MoETransformerLayer` contains all 5 experts and a router. At inference, only 2 experts run per token per layer.

### Forward Pass

```python
def forward(self, input_ids, attention_mask=None):
    x = self.shared.embed(input_ids)
    total_aux_loss = torch.tensor(0.0, device=x.device)

    for layer_idx in range(self.config.n_layers):
        x = self.shared.apply_attention(x, layer_idx, attention_mask)
        x, aux_loss = self.moe_layers[layer_idx](x, attention_mask)
        total_aux_loss = total_aux_loss + aux_loss

    logits = self.shared.predict(x)
    return logits, total_aux_loss
```

**Returns a tuple:** `(logits, aux_loss)`. The training loop adds `aux_loss` to the main loss for router load balancing. During inference, `aux_loss` is ignored.

---

## Text Generation (Lines 142-236)

### `generate()` Method

Implements autoregressive generation — one token at a time:

```python
for _ in range(max_new_tokens):
    logits, _ = self.forward(context)     # Run full model
    next_logits = logits[:, -1, :]        # Take LAST position's predictions

    # Temperature scaling
    next_logits = next_logits / temperature

    # Top-K filtering
    next_logits = self._top_k_filter(next_logits, top_k)

    # Top-P (nucleus) filtering
    next_logits = self._top_p_filter(next_logits, top_p)

    # Sample
    probs = F.softmax(next_logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)

    generated = torch.cat([generated, next_token], dim=-1)
```

**Why take ONLY the last position?** In autoregressive generation, we only need to predict what comes AFTER the entire current sequence. The logits at position `-1` represent the model's prediction for the NEXT token given everything before it.

### Sampling Strategies

**Temperature:**

```
temperature = 0.5:  [0.01, 0.04, 0.90, 0.05]  → very peaked → predictable
temperature = 1.0:  [0.05, 0.15, 0.60, 0.20]  → original distribution
temperature = 2.0:  [0.10, 0.20, 0.40, 0.30]  → very flat → random/creative
```

**Top-K (Lines 238-264):**

```python
values, _ = logits.topk(k, dim=-1)
min_value = values[..., -1:]
return logits.where(logits >= min_value, torch.full_like(logits, float("-inf")))
```

Keeps only the K highest logits, sets everything else to `-inf` (which becomes 0 after softmax).

**Top-P / Nucleus (Lines 266-297):**
Sort by probability, accumulate until reaching p% of total mass, zero out the rest. Adaptive — when the model is confident (one token has 90% probability), only 1-2 candidates remain. When uncertain (flat distribution), many candidates remain.

### Context Window Management

```python
if generated.shape[1] > self.config.max_seq_len:
    context = generated[:, -self.config.max_seq_len:]
```

**What:** If the generated sequence exceeds `max_seq_len`, use only the last `max_seq_len` tokens as context. This is a sliding window approach.
**Limitation:** The model "forgets" early tokens. For our `max_seq_len=512` this means the model can only consider the last ~400 words.

### `n_active_params_per_token` Property

```python
shared_params = self.shared.n_params
expert_params_per_layer = self.moe_layers[0].experts[0].n_params * self.config.top_k
```

Only `top_k` (2) out of 5 experts are active per token, so active compute is:
`18M shared + 4 layers × (2 × 2.1M expert) = 18M + 16.8M ≈ 35M` per token, despite having 50M+ total params.

---

## Q&A

**Q: Why no KV cache?**
A: KV caching stores computed key/value pairs from previous positions to avoid recomputing them at each step. It's essential for fast generation but adds implementation complexity. This project prioritizes clarity over speed.

**Q: Could this model be fine-tuned after assembly?**
A: Yes! You could unfreeze the router and/or experts and fine-tune on a downstream task. The assembled model is a standard `nn.Module`.
