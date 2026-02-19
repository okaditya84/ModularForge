"""
modularforge.evaluation — Metrics, Evaluation & Generation
============================================================
This subpackage measures how well the assembled ModularForge model
performs and generates sample outputs for qualitative assessment.

What We Measure:
    1. Perplexity — How "surprised" the model is by test data.
       Lower = better. A perplexity of 30 means the model is as
       uncertain as if it had to choose from 30 equally likely words.

    2. Peak Memory — Maximum RAM used at each pipeline stage.
       Must stay under the memory budget (e.g., 200MB for training).

    3. Assembly Time — How long it takes to combine all modules.
       Target: < 60 seconds for 5 modules.

    4. Text Quality — Generated text samples for human evaluation.
       Should produce coherent, grammatical English.

Components:
    - metrics.py   — Perplexity, memory profiling, timing utilities
    - evaluator.py — Full evaluation pipeline (runs all metrics)
    - generate.py  — Text generation (greedy, top-k, top-p, temperature)

Information Flow:
    Assembled model + Test data → Evaluator
        → Perplexity score
        → Memory usage report
        → Generated text samples
        → Comparison table (ModularForge vs. Monolithic baseline)
"""

from modularforge.evaluation.metrics import compute_perplexity, MemoryTracker
from modularforge.evaluation.evaluator import Evaluator
from modularforge.evaluation.generate import TextGenerator
