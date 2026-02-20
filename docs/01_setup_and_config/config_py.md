# `config.py` — The Master Configuration System

> **Source:** [config.py](file:///Users/aditya/Documents/Aditya/Python/LLM_In_Parts/modularforge/config.py) · **Lines:** 748 · **Prereqs:** `setup.py`, basic Python dataclasses

---

## What This File Does

This is the **single source of truth** for every setting in ModularForge. Every hyperparameter, path, and setting lives here. Change a value in the config YAML, and it automatically propagates to every component — no hunting through code.

Think of it as the **blueprint** for your entire experiment.

---

## Line-by-Line Walkthrough

### Lines 1-26: Module Docstring

The docstring shows three usage patterns:

1. **Load from YAML** — normal usage from config file
2. **Create programmatically** — for testing or custom experiments
3. **Save to YAML** — for experiment reproducibility

### Lines 28-36: Imports

```python
from __future__ import annotations
```

**What:** Enables PEP 604 style type hints (`str | Path` instead of `Union[str, Path]`).
**Why:** Without this, Python 3.10 would try to evaluate `str | Path` at import time, which can cause `NameError` for forward references. With this import, all annotations are treated as strings and only evaluated when needed.
**Alternative:** `from typing import Union` and use `Union[str, Path]` everywhere. **Rejected** because it's more verbose.

```python
import yaml
```

**Why YAML not JSON?** YAML supports comments (critical for configs), is more human-readable, and handles multi-line strings better. JSON would work functionally but is harder to annotate.
**Why not TOML?** TOML (`tomllib`) needs Python 3.11+ and can't represent nested structures as cleanly as YAML for deeply nested configs.

```python
from dataclasses import dataclass, field, asdict
```

**Why dataclasses?**

| Alternative              | Why Not                                                                                                      |
| ------------------------ | ------------------------------------------------------------------------------------------------------------ |
| Plain `dict`             | No type checking, no validation, no IDE autocomplete, easy typos                                             |
| `NamedTuple`             | Immutable (can't modify after creation), no default values before 3.6.1                                      |
| `pydantic.BaseModel`     | Adds a heavy dependency for what dataclasses handle fine. Pydantic is better for API schemas, not ML configs |
| `attrs`                  | Similar to dataclasses but requires `pip install attrs`. Dataclasses are built-in since Python 3.7           |
| `OmegaConf` (from Hydra) | Very powerful but complex. Overkill for a single-file config. Good for huge projects with config composition |

**`field`:** Used to define default values for mutable types (like lists). You can't write `prompts: list[str] = ["a", "b"]` directly because Python would share the same list object across all instances.

**`asdict`:** Recursively converts a dataclass to a nested dictionary. Used for saving to YAML.

```python
from typing import Optional, Literal
```

**`Optional[int]`:** Means "this value can be `int` or `None`".
**`Literal["random", "clustered", "curriculum"]`:** Restricts the value to one of these exact strings. Better than plain `str` because IDEs can autocomplete and type checkers can verify.

---

### Lines 41-188: `ModelConfig` — Architecture Hyperparameters

```python
@dataclass
class ModelConfig:
```

Each field has a default value matching the "small but functional" target:

| Parameter          | Default                       | What It Controls                                                                                                                  | Why This Value |
| ------------------ | ----------------------------- | --------------------------------------------------------------------------------------------------------------------------------- | -------------- |
| `d_model=512`      | Width of all vectors          | 512 is the "sweet spot" for small models. GPT-2 Small uses 768, but 512 trains faster and still learns meaningful representations |
| `n_heads=8`        | Parallel attention readers    | 8 heads × 64 dims/head = 512. More heads = more diverse attention patterns. 8 is standard for d_model=512                         |
| `n_layers=4`       | Depth of processing           | 4 is minimal but sufficient. GPT-2 uses 12. Each layer adds ~6M params                                                            |
| `d_ff=2048`        | Expert brain size (4×d_model) | 4× multiplier is the "Attention Is All You Need" standard. This is where most "knowledge" is stored                               |
| `n_experts=5`      | Number of specialists         | 5 is enough to demonstrate MoE while keeping training feasible on one GPU                                                         |
| `top_k=2`          | Active experts per token      | 2 gives good quality-efficiency tradeoff. K=1 loses too much info, K=3+ is diminishing returns                                    |
| `vocab_size=16384` | Dictionary size               | 16K is compact but sufficient for English. GPT-2 uses 50K, but larger vocab = larger embedding table                              |
| `max_seq_len=512`  | Maximum context window        | 512 tokens ≈ 1 paragraph. Longer sequences use quadratically more memory in attention                                             |
| `dropout=0.1`      | Random information hiding     | 10% dropout is standard for transformers. Prevents overfitting without hurting learning too much                                  |

#### The `validate()` Method (Lines 124-164)

Defensive programming — catches invalid configs EARLY before they cause cryptic errors hours into training.

```python
if self.d_model % self.n_heads != 0:
    raise ValueError(...)
```

**Why this check?** Multi-head attention divides `d_model` into `n_heads` equal pieces. If `512 / 7 = 73.14`, you can't have fractional dimensions. This would cause a shape mismatch error deep inside the attention code, which would be very confusing. Catching it here gives a clear error message.

#### Properties (Lines 166-187)

```python
@property
def head_dim(self) -> int:
    return self.d_model // self.n_heads
```

**Why a property instead of a field?** Because `head_dim` is computed FROM other fields. If it were a separate field, you could accidentally set `d_model=512, n_heads=8, head_dim=128` — inconsistent! A property ensures it's always correct.

```python
@property
def total_params_estimate(self) -> int:
    embedding_params = self.vocab_size * self.d_model  # 16384 × 512 = 8.4M
    pos_params = self.max_seq_len * self.d_model       # 512 × 512 = 0.3M
    attn_params = self.n_layers * 4 * self.d_model * self.d_model  # 4 × 4 × 512² = 4.2M
    expert_total = self.n_layers * self.n_experts * self.expert_params  # 4 × 5 × 2.1M = 42M
    ...
```

**What:** A quick estimate of total parameters, used for logging. Helps you check if your config makes sense before training.

---

### Lines 190-330: `TrainingConfig`

Key design decisions:

```python
learning_rate: float = 3e-4
```

**Why 3e-4?** This is the "Karpathy constant" — Andrej Karpathy observed this is a universally good starting point for Adam/AdamW with transformers. Too high (1e-3) → unstable. Too low (1e-5) → painfully slow.

```python
gradient_accumulation_steps: int = 1
```

**What:** Simulates larger batch sizes without more memory. With `batch_size=16` and `accumulation=4`, the effective batch size is 64, but you only hold 16 sequences in memory.
**When to increase:** If you get OOM errors, increase this instead of decreasing batch_size.

```python
def resolve_device(self) -> torch.device:
    if torch.cuda.is_available():        # NVIDIA GPU (Kaggle/Colab)
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")       # Apple Silicon
    else:
        return torch.device("cpu")
```

**Why this priority order?** CUDA > MPS > CPU. CUDA is fastest for large models. MPS (Apple Silicon) is good but has some PyTorch compatibility gaps. CPU always works but is slowest.
**Edge Case:** `hasattr(torch.backends, "mps")` — the `mps` attribute doesn't exist on old PyTorch versions (pre-1.12). The `hasattr` check prevents `AttributeError`.

---

### Lines 332-400: `DataConfig`

```python
partition_strategy: Literal["random", "clustered", "curriculum"] = "clustered"
```

**Why "clustered" is the default:**

| Strategy       | Pros                                              | Cons                                                     |
| -------------- | ------------------------------------------------- | -------------------------------------------------------- |
| `random`       | Simple, unbiased                                  | Experts don't specialize — they all learn similar things |
| `clustered` ✅ | Experts become domain specialists, best diversity | Needs `sentence-transformers`                            |
| `curriculum`   | Interesting for difficulty-based learning         | Harder experts may lack simple pattern coverage          |

```python
overlap_ratio: float = 0.1
```

**Why 10% overlap?** Each expert gets 10% extra data from other partitions. This "bridge" ensures experts share some common knowledge, improving coherence when assembled. Without overlap, experts might learn incompatible representations for common words. 0% = fully specialized, 50% = half the data is shared (too much).

---

### Lines 458-522: `EvalConfig`

```python
temperature: float = 0.8
```

**Why 0.8?** Full temperature (1.0) can be too random. 0.8 slightly sharpens the distribution, producing more coherent text while still being creative. For formal text, use 0.3-0.5. For brainstorming, use 1.0-1.5.

```python
prompts: list[str] = field(default_factory=lambda: [...])
```

**Why `field(default_factory=...)`?** Python dataclasses cannot use mutable defaults directly. If you wrote `prompts: list = ["a", "b"]`, ALL instances would share the SAME list. `default_factory` creates a new list for each instance.

---

### Lines 524-748: `ModularForgeConfig` — The Master Config

```python
@dataclass
class ModularForgeConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    ...
```

**What:** A composition of all sub-configs into one master config.
**Why composition?** Each component only needs to access its relevant section (`config.model.d_model`), keeping concerns separated.

#### Cross-Config Validation (Lines 569-576)

```python
if self.data.tokenizer_vocab_size != self.model.vocab_size:
    raise ValueError(...)
```

**Why:** If the tokenizer has 16384 words but the model's embedding table has 32000 slots, the model will crash at runtime. This catches the mismatch at config load time.

#### `from_yaml()` Class Method (Lines 584-629)

```python
@classmethod
def from_yaml(cls, path: str | Path) -> ModularForgeConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    config = cls(
        model=ModelConfig(**raw.get("model", {})),
        ...
    )
```

**`yaml.safe_load`:** Uses the safe loader that can't execute arbitrary Python code. `yaml.load` (without `safe_`) is a security vulnerability — it can execute arbitrary Python embedded in YAML.

**`**raw.get("model", {})`:** The `\*\*`unpacks the dictionary into keyword arguments.`raw.get("model", {})`returns an empty dict if`model` section is missing, so all defaults apply.

#### `for_smoke_test()` (Lines 662-728)

Creates a tiny config for quick pipeline validation: tiny model (d_model=64, n_layers=2), minimal data (200 articles), 1 epoch each. Completes in <5 minutes on any hardware.

---

## Q&A

**Q: Why not use environment variables for config?**
A: Environment variables are flat key-value pairs, terrible for nested configs with 30+ parameters. YAML files are version-controlled, self-documenting, and sharable.

**Q: Why not use Hydra (from Facebook)?**
A: Hydra is powerful for large-scale experiments with config composition, grid search, and multi-run. But it adds complexity (special directory structure, decorator syntax) that's overkill for a single-experiment research project.

**Q: Why are some things in config and some hardcoded?**
A: We put things in config that you might want to change between experiments (hyperparameters, paths, strategies). Things like "use CrossEntropyLoss" or "use AdamW" are hardcoded because they're fundamental design choices that shouldn't change casually.
