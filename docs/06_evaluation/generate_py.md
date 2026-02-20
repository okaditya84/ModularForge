# `generate.py` — Text Generation Interface

> **Source:** [generate.py](file:///Users/aditya/Documents/Aditya/Python/LLM_In_Parts/modularforge/evaluation/generate.py) · **Lines:** 206 · **Prereqs:** `assembled_model.py`, `tokenizer.py`

---

## What This File Does

High-level interface for generating text from the assembled model. Handles the encode → generate → decode pipeline and provides both batch and interactive modes.

---

## `generate_samples()` (Lines 72-157)

```python
for prompt in prompts:
    prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
    prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)

    output_ids = self.model.generate(
        input_ids=prompt_tensor,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        eos_token_id=self.tokenizer.eos_id,
    )

    full_text = self.tokenizer.decode(output_ids[0].tolist(), skip_special_tokens=True)
```

**Prompt separation (Lines 142-146):**

```python
if full_text.startswith(prompt_text):
    continuation = full_text[len(prompt_text):]
else:
    continuation = full_text
```

**Why "best-effort"?** Tokenization is NOT perfectly reversible. Encoding `"Hello"` then decoding might give `" Hello"` (with leading space) due to BPE byte-level encoding. The `startswith` check handles the common case; the fallback returns the full text.

---

## `interactive_generate()` (Lines 159-205)

A REPL-style interface for terminal use. Type a prompt, see the model's continuation. Type `quit` to exit.

**`except (EOFError, KeyboardInterrupt)`:** Handles both Ctrl+D (EOF) and Ctrl+C (interrupt) gracefully instead of crashing with a traceback.

---

## Q&A

**Q: Why is generation so much slower than training?**
A: Training processes all tokens in parallel (one forward pass for the entire batch). Generation is sequential — you MUST generate token N before you can generate token N+1 (it depends on all previous tokens). This makes generation inherently serial.
