# `setup.py` — Package Installation Script

> **Source:** [setup.py](file:///Users/aditya/Documents/Aditya/Python/LLM_In_Parts/setup.py) · **Lines:** 50 · **Prereqs:** None (start here!)

---

## What This File Does

This file tells Python's package manager (`pip`) how to **install ModularForge as a package** on your machine. Without it, statements like `from modularforge.model.expert import ExpertFFN` would fail with `ModuleNotFoundError`.

When you run `pip install -e .`, Python reads this file and creates a "link" from your Python environment to this project folder.

---

## Line-by-Line Walkthrough

### Lines 1-11: Module Docstring

```python
"""
ModularForge — Setup Script
============================
Installs ModularForge as a local editable package...
"""
```

**What:** Documentation for the file.
**Why:** Every Python file should start with a docstring explaining its purpose. This appears in `help(setup)`.

---

### Line 13: Import

```python
from setuptools import setup, find_packages
```

**What:** Imports two functions from `setuptools` — Python's standard library for packaging.

| Function          | Purpose                                                                      |
| ----------------- | ---------------------------------------------------------------------------- |
| `setup()`         | The main function that tells pip everything about your package               |
| `find_packages()` | Automatically discovers all Python packages (directories with `__init__.py`) |

**Why `setuptools` and not alternatives?**

| Alternative           | Why Not                                                                               |
| --------------------- | ------------------------------------------------------------------------------------- |
| `distutils`           | Deprecated since Python 3.12. `setuptools` is its modern replacement.                 |
| `poetry`              | More opinionated, requires `pyproject.toml`. Good for new projects but overkill here. |
| `flit`                | Simpler than poetry but can't handle complex build steps.                             |
| `pyproject.toml` only | The modern standard, but `setup.py` is still widely used and well understood.         |

**Edge Case:** If `setuptools` is not installed (extremely rare in modern Python ≥3.4), you'd get `ImportError`. Fix: `pip install setuptools`.

---

### Lines 15-49: The `setup()` Call

```python
setup(
    name="modularforge",
```

**What:** The package name. This is what appears when you run `pip list`.
**Why "modularforge"?** Matches the directory name `modularforge/` that contains the code.
**Edge Case:** If another pip package named `modularforge` existed on PyPI, you'd get conflicts when installing from PyPI. Since we install locally with `-e .`, this doesn't matter.

---

```python
    version="0.1.0",
```

**What:** [Semantic Versioning](https://semver.org/) — `MAJOR.MINOR.PATCH`.

- `0` = pre-release (not production-ready)
- `1` = first minor version
- `0` = no patches yet

**Why 0.1.0?** Convention for initial development. Once stable, it would become `1.0.0`.

---

```python
    author="Aditya",
    description=(
        "ModularForge: Memory-Bounded Modular Training and Zero-Retraining "
        "Assembly of Large Language Models on Resource-Constrained Hardware"
    ),
```

**What:** Metadata about who wrote it and what it does.
**Why parentheses for description?** Python's implicit string concatenation — putting two strings next to each other in parentheses joins them. Keeps lines short.

---

```python
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
```

**What:** Uses the project's README as the long description (shown on PyPI).
**Why `encoding="utf-8"`?** Without it, Windows might try to read the file as `latin-1` and crash on special characters.
**Edge Case:** If `README.md` doesn't exist, this line raises `FileNotFoundError`. The README must exist in the same directory as `setup.py`.

---

```python
    url="https://github.com/aditya/ModularForge",
    packages=find_packages(),
```

**`find_packages()`:** Scans the current directory recursively for any folder containing `__init__.py`. It finds:

- `modularforge/`
- `modularforge/data/`
- `modularforge/model/`
- `modularforge/training/`
- `modularforge/assembly/`
- `modularforge/evaluation/`

**Why not list them manually?** Automatic discovery means you never forget to add a new subpackage. Manual listing (e.g., `packages=["modularforge", "modularforge.data", ...]`) is error-prone.

---

```python
    python_requires=">=3.10",
```

**What:** Minimum Python version.
**Why 3.10?** The codebase uses:

- `match/case` syntax (3.10+)
- `list[str]` type hints in function signatures (3.9+, but 3.10 is safer)
- `X | Y` union types (3.10+)

**Why not 3.12?** Unnecessarily restrictive. 3.10 is the oldest version that supports all features used.

---

```python
    install_requires=[
        "torch>=2.1.0",        # Deep learning framework
        "safetensors>=0.4.0",  # Fast, safe tensor serialization
        "datasets>=2.14.0",    # HuggingFace dataset loading
        "tokenizers>=0.15.0",  # Fast BPE tokenizer
        "numpy>=1.24.0",       # Numerical computing
        "tqdm>=4.65.0",        # Progress bars
        "pyyaml>=6.0",         # YAML config parsing
        "sentence-transformers>=2.2.0",  # For clustered partitioning
        "scikit-learn>=1.3.0",           # KMeans clustering
    ],
```

**What it does:** When you run `pip install -e .`, pip automatically installs all these dependencies.

**Why these specific minimum versions?**

| Dependency           | Min Version                                              | Reason |
| -------------------- | -------------------------------------------------------- | ------ |
| `torch>=2.1.0`       | `torch.compile` stability, MPS support for Apple Silicon |
| `safetensors>=0.4.0` | Memory-mapped loading support                            |
| `datasets>=2.14.0`   | `trust_remote_code` parameter support                    |
| `tokenizers>=0.15.0` | `ByteLevel` pre-tokenizer API stability                  |
| `numpy>=1.24.0`      | Compatible with PyTorch 2.1+                             |
| `pyyaml>=6.0`        | `yaml.safe_load` security fixes                          |

**Why `sentence-transformers` is a required dependency (not optional):** The default config uses `clustered` partitioning, which needs sentence embeddings. Making it optional would cause the default pipeline to fail.

**Alternative considered:** Making `sentence-transformers` optional with a `try/except` import. **Why rejected:** The fallback (random partitioning) gives worse results, and a user following the README would be confused by the different behavior.

---

```python
    extras_require={
        "dev": ["pytest>=7.0", "tensorboard>=2.14.0"],
    },
```

**What:** Optional dependencies installed via `pip install -e ".[dev]"`.
**Why separate?** Normal users don't need `pytest` (testing) or `tensorboard` (visualization). Separating them keeps the base install lighter.

---

```python
    classifiers=[
        "Development Status :: 3 - Alpha",
        ...
    ],
```

**What:** PyPI classifiers — metadata tags for discoverability on pypi.org.
**Why "3 - Alpha"?** The project is functional but not production-hardened.

---

## Q&A

**Q: Why `setup.py` instead of `pyproject.toml`?**
A: `pyproject.toml` is the modern standard, but `setup.py` still works everywhere and is more immediately understandable. For a research project, simplicity wins.

**Q: What does `pip install -e .` actually do?**
A: The `-e` flag means "editable". Instead of copying files into `site-packages/`, pip creates a `.egg-link` file that points to your project directory. Changes to your code take effect immediately without reinstalling.

**Q: Why is there a `modularforge.egg-info/` directory?**
A: Created automatically by `pip install -e .`. Contains metadata about the installed package. Safe to delete (it gets regenerated), and it's in `.gitignore`.
