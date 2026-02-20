# `tokenizer.py` — BPE Tokenizer

> **Source:** [tokenizer.py](file:///Users/aditya/Documents/Aditya/Python/LLM_In_Parts/modularforge/data/tokenizer.py) · **Lines:** 410 · **Prereqs:** `config.py`

---

## What This File Does

Converts raw text like `"The cat sat"` into a list of integers like `[2, 435, 1024, 847, 3]`. These integers are what the neural network actually processes. This tokenizer is **shared** across all expert modules — it defines the common "vocabulary" that every expert understands.

---

## Why BPE (Byte-Pair Encoding)?

| Method              | How It Works                   | Vocab Size | Unknown Words? | Example                        |
| ------------------- | ------------------------------ | ---------- | -------------- | ------------------------------ |
| **Character-level** | Each letter is a token         | 26+special | Never          | `"cat"` → `[c, a, t]`          |
| **Word-level**      | Each word is a token           | 100K+      | Yes (OOV)      | `"cat"` → `[cat]`              |
| **BPE** ✅          | Merge frequent character pairs | 16K        | Almost never   | `"cat"` → `[ca, t]` or `[cat]` |

BPE starts with individual bytes, finds the most frequently co-occurring pair, merges them into one token, and repeats until reaching the target vocabulary size. Common words become single tokens; rare words are broken into subword pieces.

---

## Line-by-Line Walkthrough

### Lines 31-49: Imports and Constants

```python
from tokenizers import Tokenizer, pre_tokenizers, models, trainers, decoders
from tokenizers.processors import TemplateProcessing
```

**What:** HuggingFace's `tokenizers` library — written in Rust, 10-100× faster than pure Python alternatives.
**Why not `tiktoken` (OpenAI)?** `tiktoken` is extremely fast but only loads pre-trained tokenizers (GPT-3, GPT-4). It doesn't support training a new tokenizer from scratch. We need to train our own on WikiText.
**Why not `sentencepiece` (Google)?** Viable alternative, but HuggingFace `tokenizers` has a nicer Python API and natively integrates with the HuggingFace ecosystem.

```python
PAD_TOKEN = "<pad>"    # ID 0: fills empty space in fixed-length batches
UNK_TOKEN = "<unk>"    # ID 1: replaces truly unknown tokens (very rare with BPE)
BOS_TOKEN = "<bos>"    # ID 2: "beginning of sequence" marker
EOS_TOKEN = "<eos>"    # ID 3: "end of sequence" marker
```

**Why these 4 special tokens?**

- **PAD**: Batches must have uniform length. Shorter sequences get padded. The model learns to IGNORE padding.
- **BOS/EOS**: Tell the model where sequences start and end. Critical for generation (model knows when to stop).
- **UNK**: Safety net for characters not in vocabulary. BPE makes this extremely rare since it falls back to individual bytes.

---

### Lines 52-104: `ModularForgeTokenizer` Class

```python
def __init__(self, vocab_size: int = 16384):
    if vocab_size < len(SPECIAL_TOKENS):
        raise ValueError(...)
    self.vocab_size = vocab_size
    self._tokenizer: Optional[Tokenizer] = None
```

**Why `_tokenizer` is `None` initially?** The tokenizer needs to be either trained on data or loaded from disk before it can be used. The underscore prefix indicates it's a private attribute — external code should use the `tokenizer` property which raises a helpful error if it hasn't been initialized.

**Why 16384 vocab size?**

| Size         | Trade-offs                                                                    |
| ------------ | ----------------------------------------------------------------------------- |
| 1000         | Too small. Common words like "the" get split: `th` + `e`. Inefficient         |
| 8192         | Okay for very small models. Most common words are single tokens               |
| **16384** ✅ | Good balance. Almost all English words are 1-2 tokens                         |
| 32768        | What we use in default.yaml. Better coverage, slightly larger embedding table |
| 50000+       | Diminishing returns. Lots of rare tokens that barely appear in training data  |

---

### Lines 108-203: `train()` Method

```python
self._tokenizer = Tokenizer(models.BPE())
```

**What:** Creates an empty BPE tokenizer model. No vocabulary yet — just the structure.

```python
self._tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
```

**What:** The pre-tokenizer runs BEFORE BPE, splitting text into initial tokens.
**`ByteLevel`:** Converts text to bytes first, then applies BPE on byte sequences. This means it can handle ANY text (code, math, emoji, Chinese) without unknown tokens.
**`add_prefix_space=False`:** GPT-2 adds a space before the first word (`" Hello"` instead of `"Hello"`). We don't want this because it creates an asymmetry between the first word and other words.

**Why ByteLevel instead of Whitespace?**

| Pre-tokenizer      | Behavior                 | Problem                                                                |
| ------------------ | ------------------------ | ---------------------------------------------------------------------- |
| `Whitespace`       | Splits on spaces         | Can't handle words within words (`"don't"` → `["don't"]` — no subword) |
| **`ByteLevel`** ✅ | Operates on bytes        | Universal — handles everything. No OOV tokens ever                     |
| `CharDelimiter`    | Splits on specific chars | Manual, brittle, misses edge cases                                     |

```python
self._tokenizer.decoder = decoders.ByteLevel()
```

**What:** The inverse of ByteLevel encoding. Converts byte-level tokens back to readable text.
**Edge Case:** Without a matching decoder, decoded text would show raw byte representations like `Ġ` instead of spaces.

```python
trainer = trainers.BpeTrainer(
    vocab_size=self.vocab_size,
    min_frequency=2,
    special_tokens=SPECIAL_TOKENS,
    show_progress=True,
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
)
```

**`min_frequency=2`:** A character pair must appear at least 2 times to be merged. This prevents very rare pairs from wasting vocabulary slots.
**`initial_alphabet=pre_tokenizers.ByteLevel.alphabet()`:** Starts with all 256 byte values as initial tokens. This guarantees that ANY byte can be represented, even if it's never seen during training.

```python
self._tokenizer.post_processor = TemplateProcessing(
    single=f"{BOS_TOKEN} $A {EOS_TOKEN}",
    special_tokens=[
        (BOS_TOKEN, bos_id),
        (EOS_TOKEN, eos_id),
    ],
)
```

**What:** Automatically wraps every encoded text with BOS and EOS tokens.
Example: `"Hello"` → `[BOS, Hello, EOS]` = `[2, 435, 3]`
**Why automatic?** If you manually add BOS/EOS, you'll inevitably forget somewhere and get mysterious bugs.

---

### Lines 207-272: `encode()` Method

```python
if not add_special_tokens:
    saved_processor = self.tokenizer.post_processor
    self.tokenizer.post_processor = None
```

**What:** Temporarily disables BOS/EOS injection when the caller doesn't want special tokens.
**Why this dance?** The HuggingFace tokenizer library doesn't have a built-in flag for this, so we save, disable, encode, then restore. This is thread-unsafe (a known limitation), but ModularForge doesn't use multi-threaded encoding.

```python
if max_length is not None and len(ids) > max_length:
    ids = ids[:max_length]
    if add_special_tokens:
        eos = self.tokenizer.token_to_id(EOS_TOKEN)
        ids[-1] = eos
```

**Edge Case:** When truncating, we REPLACE the last token with EOS. Why? Without EOS, the model doesn't know the sequence ended — it might try to "continue" it during generation, producing garbage.

---

### Lines 318-336: `decode()` Method

```python
def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
    if not ids:
        return ""
    return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)
```

**Why `skip_special_tokens=True` by default?** When displaying generated text to humans, you don't want to see `<bos>The cat sat<eos><pad><pad>`. The special tokens are only meaningful to the model.

---

### Lines 367-404: Save/Load

```python
def save(self, path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    self.tokenizer.save(str(path))
```

**Why `mkdir(parents=True, exist_ok=True)`?** Creates the directory AND all parent directories. `exist_ok=True` means no error if it already exists. Without this, saving to `data/tokenizer.json` fails if `data/` doesn't exist.

**Why save as JSON?** The HuggingFace tokenizers library uses JSON internally. The entire vocabulary, merge rules, and special tokens are stored in a single human-readable file. You can literally open `tokenizer.json` and read the vocabulary.

---

## Q&A

**Q: Why not use a pre-trained tokenizer (like GPT-2's)?**
A: Pre-trained tokenizers are optimized for their training data (internet text). WikiText-103 has a different distribution. Training our own gives better compression (fewer tokens per article) and ensures optimal coverage.

**Q: Could two experts have different tokenizers?**
A: No. The tokenizer defines the vocabulary, which determines the embedding table dimensions. All experts share the same embedding table, so they MUST use the same tokenizer.

**Q: What happens if the tokenizer file is corrupt?**
A: `Tokenizer.from_file()` will raise a `tokenizers.TokenizerError`. There's no recovery — you'd need to retrain it.
