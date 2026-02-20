# `shared.py` — Shared Components (The Foundation)

> **Source:** [shared.py](file:///Users/aditya/Documents/Aditya/Python/LLM_In_Parts/modularforge/model/shared.py) · **Lines:** 483 · **Prereqs:** `config.py`

---

## What This File Does

Defines everything in the transformer model that is **shared across all experts**: token embedding, positional encoding, multi-head self-attention, layer normalization, and the language model head. These are trained once on the full corpus and then **frozen** — they never change again.

```
Input: "The cat"  →  [token IDs: 435, 1024]
                            ↓
                    Token Embedding (shared)        → [512-dim vectors]
                    + Positional Encoding (shared)  → [position-aware vectors]
                    → Self-Attention × 4 layers (shared)  → [context-aware vectors]
                    → Output LayerNorm (shared)
                    → LM Head (shared, weight-tied) → [vocabulary logits]
                            ↓
Output: "sat" (predicted next token)
```

---

## Components Defined in This File

### 1. `SinusoidalPositionalEncoding` (Lines 55-120)

**Problem:** Token embeddings have NO sense of position. "The cat sat" and "sat cat The" produce the same embeddings (just in different order). The model needs to know WHERE each token is.

**Solution:** Add a unique position-dependent vector to each token's embedding.

```python
pe = torch.zeros(max_seq_len, d_model)
position = torch.arange(0, max_seq_len).unsqueeze(1).float()
div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
pe[:, 0::2] = torch.sin(position * div_term)  # Even dimensions: sin
pe[:, 1::2] = torch.cos(position * div_term)  # Odd dimensions: cos
```

**Why sinusoidal?**

| Method             | Pros                                      | Cons                                     |
| ------------------ | ----------------------------------------- | ---------------------------------------- |
| Learned embeddings | Flexible, can learn optimal patterns      | Can't extrapolate beyond training length |
| **Sinusoidal** ✅  | Extrapolates to any length, no parameters | Theoretically less expressive            |
| RoPE (Rotary)      | Best of both, used in LLaMA               | More complex implementation              |
| ALiBi (Alibi)      | Efficient, simple                         | Only works with causal attention         |

Sinusoidal was chosen for simplicity. Each position gets a unique fingerprint via sin/cos waves at different frequencies. Position 0 and position 100 have very different vectors, but position 100 and position 101 are similar — capturing the notion of "nearby" positions.

```python
self.register_buffer("pe", pe.unsqueeze(0))
```

**`register_buffer`:** Stores `pe` as part of the model but NOT as a learnable parameter. It moves to GPU with the model but doesn't receive gradients. Without this, the positional encoding would stay on CPU when you call `model.to("cuda")`.

---

### 2. `MultiHeadSelfAttention` (Lines 122-260)

The **core** of the transformer — allows each token to look at ALL other tokens and extract relevant information.

```python
self.W_q = nn.Linear(d_model, d_model, bias=False)  # Query projection
self.W_k = nn.Linear(d_model, d_model, bias=False)  # Key projection
self.W_v = nn.Linear(d_model, d_model, bias=False)  # Value projection
self.W_o = nn.Linear(d_model, d_model, bias=False)  # Output projection
```

**Analogy — Library Search:**

- **Query (Q):** "I'm looking for a book about cats" (what I want)
- **Key (K):** Each book's title/description (what each book offers)
- **Value (V):** Each book's actual content (what you get)
- **Attention = softmax(Q·K^T / √d) × V:** Find the most relevant books and read their content

**Why 4 separate linear layers?** Each projection learns a different "aspect" of the representation:

- `W_q` learns what to ask for
- `W_k` learns what to advertise
- `W_v` learns what to provide
- `W_o` recombines multi-head outputs

**Why `bias=False`?** Modern transformers (LLaMA, GPT-NeoX) drop attention biases. It reduces parameters with negligible quality loss.

#### Multi-Head Split (Lines 185-200)

```python
def _split_heads(self, x, batch_size):
    x = x.view(batch_size, -1, self.n_heads, self.head_dim)
    return x.transpose(1, 2)  # (batch, n_heads, seq_len, head_dim)
```

**What:** Splits the 512-dim vector into 8 heads of 64 dimensions each.
**Why:** Each head independently attends to different aspects of the input. Head 1 might focus on syntactic relationships, head 2 on semantic similarity, head 3 on positional proximity, etc.

#### Causal Mask (Lines 210-230)

```python
causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
scores = scores.masked_fill(causal_mask, float("-inf"))
```

**What:** Prevents each token from "seeing" future tokens. Token at position 3 can only attend to positions 0, 1, 2, 3 — not 4, 5, 6, ...
**Why:** This is what makes the model _autoregressive_ — it generates one token at a time, left to right, never peeking ahead.

```
Token positions:    0  1  2  3  4
Causal mask:
  Position 0 sees: [✓  ✗  ✗  ✗  ✗]
  Position 1 sees: [✓  ✓  ✗  ✗  ✗]
  Position 2 sees: [✓  ✓  ✓  ✗  ✗]
  Position 3 sees: [✓  ✓  ✓  ✓  ✗]
  Position 4 sees: [✓  ✓  ✓  ✓  ✓]
```

---

### 3. `TransformerBlock` (Lines 262-310)

Wraps attention + LayerNorm into a reusable block with residual connection.

```python
def forward(self, x, attention_mask=None):
    residual = x
    x = self.norm(x)        # Pre-norm
    x = self.attention(x, attention_mask)
    x = residual + x        # Residual connection
    return x
```

**Why Pre-Norm (normalize BEFORE attention)?**

| Pattern         | Stability                       | Quality                 | Used By                    |
| --------------- | ------------------------------- | ----------------------- | -------------------------- |
| Post-Norm       | Unstable at depth, needs warmup | Slightly higher ceiling | Original Transformer, BERT |
| **Pre-Norm** ✅ | Very stable, easy to train      | Nearly as good          | GPT-2, GPT-3, LLaMA        |

Pre-Norm skips the need for careful learning rate warmup and is more forgiving with hyperparameters.

**Residual Connection (`residual + x`):** The "highway" that allows gradients to flow directly from output to input without degradation. Without it, deep networks (>6 layers) fail to train — gradients vanish before reaching early layers.

---

### 4. `SharedComponents` Class (Lines 316-483)

The main class that bundles everything together.

```python
self.embedding = nn.Embedding(config.vocab_size, config.d_model)
self.pos_encoding = SinusoidalPositionalEncoding(config.max_seq_len, config.d_model)
self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
self.output_norm = nn.LayerNorm(config.d_model)
self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
```

#### Weight Tying (Lines 375-380)

```python
self.lm_head.weight = self.embedding.weight
```

**What:** The LM head and embedding table share the SAME weight matrix.
**Why:** Embedding maps `token_id → vector` (lookup). LM head maps `vector → logits` (linear). Both fundamentally relate tokens to their vector representations. Sharing halves the parameters (saves ~8M params) and often IMPROVES quality because it encourages consistent token representations.

#### `freeze()` / `unfreeze()` Methods

```python
def freeze(self):
    for param in self.parameters():
        param.requires_grad = False
```

**What:** Sets `requires_grad=False` on all parameters. Gradients won't be computed for these parameters, saving memory and compute during expert training.
**Why:** After Phase 1 training, the shared components are "done." During Phase 2, only expert FFNs learn — the shared foundation stays fixed.

#### `embed()`, `apply_attention()`, `predict()` (Lines 412-470)

Clean API for the training model to use shared components step by step:

1. `embed(input_ids)` → token + positional encoding
2. `apply_attention(x, layer_idx, mask)` → one attention block
3. `predict(x)` → output norm + LM head → logits

This separation lets `ModuleTrainerModel` interleave shared attention with trainable experts layer by layer.

---

## Q&A

**Q: Why not make positional encoding learnable?**
A: Learnable encoding needs more parameters and can't generalize to sequences longer than `max_seq_len`. Sinusoidal works at any length and costs zero parameters.

**Q: Why freeze shared components?**
A: If experts could modify the shared components during their training, expert 4 would overwrite what expert 0 learned. Freezing ensures all experts build on the SAME foundation.

**Q: What about KV caching for faster generation?**
A: Not implemented — it would add complexity. The current implementation recomputes attention for the full context at each generation step. For production, you'd add KV caching in `MultiHeadSelfAttention.forward()`.
