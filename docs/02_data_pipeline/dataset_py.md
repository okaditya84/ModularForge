# `dataset.py` — PyTorch Dataset for Language Modeling

> **Source:** [dataset.py](file:///Users/aditya/Documents/Aditya/Python/LLM_In_Parts/modularforge/data/dataset.py) · **Lines:** 249 · **Prereqs:** `tokenizer.py`

---

## What This File Does

Converts a list of text articles into **fixed-length token sequences** that PyTorch can batch-feed to the model. Every training step gets a pair: `(input_ids, target_ids)` where the target is the input shifted by 1 — the classic **next-token prediction** setup.

```
Article: "The cat sat on the mat"
Tokenized: [BOS, The, cat, sat, on, the, mat, EOS]
chunk of length 5:  input  = [BOS, The, cat, sat, on]
                    target = [The, cat, sat, on, the]   ← shifted by 1
```

---

## Key Design Decision: Concatenate-Then-Chunk

Instead of treating each article independently (wasting short articles), we concatenate ALL articles into one giant token stream, then slice it into fixed-length windows.

| Approach                      | Pros                                         | Cons                                                                                        |
| ----------------------------- | -------------------------------------------- | ------------------------------------------------------------------------------------------- |
| **Per-article padding**       | Preserves article boundaries                 | Short articles waste space (all padding)                                                    |
| **Concatenate-then-chunk** ✅ | No wasted tokens, maximizes data utilization | Article boundaries are blurred (model sees tokens from adjacent articles in the same chunk) |

The concatenation approach is standard in GPT-2, GPT-3, LLaMA, etc. It works because the BOS/EOS tokens between articles serve as natural "separators" in the token stream.

---

## Line-by-Line Walkthrough

### Lines 72-126: `__init__`

```python
def __init__(self, texts, tokenizer, max_seq_len=512, stride=None):
```

**`max_seq_len=512`:** Every sequence is exactly 512 tokens (plus 1 for the target). This is the "page size."

**`stride=None`:** Controls overlap between consecutive chunks.

- `stride=512` (default, = `max_seq_len`): No overlap. Chunk 1 = tokens 0-511, chunk 2 = tokens 512-1023.
- `stride=256` (= `max_seq_len // 2`): 50% overlap. Chunk 1 = tokens 0-511, chunk 2 = tokens 256-767. 2× more data but each token appears in 2 chunks.

**Why stride exists:** Overlapping windows give the model more training examples with different "contexts" for each token. Trade-off: more data but also more computation.

```python
if max_seq_len < 4:
    raise ValueError(...)
```

**Edge case:** With `max_seq_len < 4`, there isn't enough room for `BOS + at_least_2_tokens + EOS`. This would create degenerate sequences.

---

### Lines 128-180: `_build_sequences()`

```python
all_tokens: list[int] = []
for text in texts:
    if not text or not text.strip():
        continue
    tokens = tokenizer.encode(text, add_special_tokens=True)
    all_tokens.extend(tokens)
```

**What:** Tokenizes every article and concatenates them into ONE giant list. Each article's tokens include BOS and EOS, so the stream looks like:

```
[BOS, art1_tok1, art1_tok2, ..., EOS, BOS, art2_tok1, ..., EOS, BOS, ...]
```

```python
chunk_len = self.max_seq_len + 1
```

**Why +1?** We need `seq_len` tokens for the input AND `seq_len` tokens for the target. Since the target is input shifted by 1, we need `seq_len + 1` total tokens per chunk:

```
chunk:  [t0, t1, t2, t3, t4]   (length = seq_len + 1 = 5)
input:  [t0, t1, t2, t3]       (first seq_len = 4)
target: [t1, t2, t3, t4]       (last seq_len = 4, shifted by 1)
```

```python
for start in range(0, total_tokens - chunk_len + 1, self.stride):
    chunk = all_tokens[start: start + chunk_len]
    self.sequences.append(torch.tensor(chunk, dtype=torch.long))
```

**`torch.long`:** 64-bit integers. Token IDs can be up to 32768, which fits in 16-bit, but PyTorch's embedding layer requires `long` (int64). Using `int32` would cause a runtime error.

**Why pre-tensorize?** Converting to tensors during `__init__` (not during `__getitem__`) avoids re-tokenizing every epoch. This uses more RAM but dramatically speeds up training.

---

### Lines 174-180: Handling the Last Partial Chunk

```python
remainder_start = len(self.sequences) * self.stride if self.sequences else 0
if remainder_start < total_tokens:
    remaining = all_tokens[remainder_start:]
    if len(remaining) >= 2:
        padded = remaining + [self.pad_id] * (chunk_len - len(remaining))
        self.sequences.append(torch.tensor(padded, dtype=torch.long))
```

**What:** If the last chunk has fewer tokens than `chunk_len`, pad it with PAD tokens instead of discarding it.
**Why `>= 2`?** You need at least 2 tokens to form an input-target pair. A single token has no target.
**Edge case:** If the corpus is very small and `stride` > `total_tokens`, `remainder_start` is 0, and the entire corpus becomes one padded chunk.

---

### Lines 186-217: `__getitem__`

```python
def __getitem__(self, idx):
    chunk = self.sequences[idx]
    input_ids = chunk[:-1]   # All but last token
    target_ids = chunk[1:]   # All but first token (shifted by 1)
    return input_ids, target_ids
```

**What:** The core of next-token prediction. For every position, the model sees tokens up to that position and must predict the NEXT one.

```
chunk:   [BOS, The, cat, sat, EOS]
input:   [BOS, The, cat, sat]         ← model sees these
target:  [The, cat, sat, EOS]         ← model predicts these
                                        (shifted right by 1)
```

Position 0: sees `[BOS]`, must predict `The`
Position 1: sees `[BOS, The]`, must predict `cat`
Position 2: sees `[BOS, The, cat]`, must predict `sat`
Position 3: sees `[BOS, The, cat, sat]`, must predict `EOS`

---

### Lines 224-241: `get_attention_mask`

```python
def get_attention_mask(self, input_ids):
    return (input_ids != self.pad_id).long()
```

**What:** Creates a mask where real tokens = 1 and padding = 0.

```
input:  [BOS, The, cat, PAD, PAD]
mask:   [  1,   1,   1,   0,   0]
```

**Why:** Tells the attention mechanism to IGNORE padding. Without this, the model would waste capacity trying to process meaningless PAD tokens and might learn spurious patterns from padding.

---

## Q&A

**Q: Why int64 (`torch.long`) instead of int32?**
A: PyTorch's `nn.Embedding` requires `long` (int64) for index tensors. Using int32 causes `RuntimeError: Expected tensor for argument #1 'indices' to have one of the following scalar types: Long`.

**Q: Is pre-tokenizing wasteful on memory?**
A: For WikiText-103 (~103M tokens), storing as int64 tensors uses ~800MB. This is manageable on modern machines. For larger datasets (>1B tokens), you'd switch to memory-mapped files or on-the-fly tokenization.

**Q: Why not use HuggingFace's `transformers.DataCollatorForLanguageModeling`?**
A: That class handles masking (for BERT-style training), not next-token targets. Our approach is simpler and specific to autoregressive (GPT-style) training.
