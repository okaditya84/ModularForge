"""
ModularForge
============
Memory-Bounded Modular Training and Zero-Retraining Assembly of
Large Language Models on Resource-Constrained Hardware.

This package provides a complete framework for:
    1. Decomposing a transformer LLM into small, independent expert modules
    2. Training each module sequentially on different data partitions
    3. Assembling all modules into a Mixture-of-Experts model via streaming
    4. Running inference with the assembled model

The entire pipeline (training, assembly, inference) is designed to run
within a constant memory budget — never exceeding the size of a single
module in RAM at any stage.

Quick Start:
    >>> from modularforge.config import ModularForgeConfig
    >>> config = ModularForgeConfig.from_yaml("configs/default.yaml")
    >>> print(config)

Subpackages:
    - modularforge.data       — Tokenization, dataset, and partitioning
    - modularforge.model      — Expert, shared, router, and MoE architecture
    - modularforge.training   — Training engine for shared and expert modules
    - modularforge.assembly   — Streaming assembly and post-assembly calibration
    - modularforge.evaluation — Metrics, evaluation, and text generation
"""

__version__ = "0.1.0"
__author__ = "Aditya"
