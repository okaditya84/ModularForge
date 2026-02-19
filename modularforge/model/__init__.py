"""
modularforge.model — Model Architecture
========================================
This subpackage defines all neural network components of ModularForge.

Architecture Overview (think of it as building a team of specialists):

    ┌──────────────────────────────────────────────────────────┐
    │                   Assembled MoE Model                    │
    │                                                          │
    │  ┌─ SharedComponents ────────────────────────────────┐   │
    │  │  Token Embedding (shared "dictionary" all use)    │   │
    │  │  Positional Encoding (tells model word positions) │   │
    │  │  Self-Attention (shared "reading comprehension")  │   │
    │  │  LM Head (predicts next word, tied to embedding)  │   │
    │  └───────────────────────────────────────────────────┘   │
    │                                                          │
    │  ┌─ MoE Layer ──────────────────────────────────────┐   │
    │  │  Router: decides which expert handles each token  │   │
    │  │  ┌─────────┐ ┌─────────┐       ┌─────────┐      │   │
    │  │  │Expert 0 │ │Expert 1 │  ...  │Expert N │      │   │
    │  │  │(Code)   │ │(Math)   │       │(General)│      │   │
    │  │  └─────────┘ └─────────┘       └─────────┘      │   │
    │  └───────────────────────────────────────────────────┘   │
    └──────────────────────────────────────────────────────────┘

Components:
    - shared.py             — Shared components (embedding, attention, LM head)
    - expert.py             — Expert FFN module (the independently trained unit)
    - router.py             — MoE gating network (routes tokens to experts)
    - moe_layer.py          — Complete MoE transformer layer
    - module_trainer_model.py — Training wrapper (shared frozen + 1 expert trainable)
    - assembled_model.py    — Full assembled model for inference

Information Flow During Training:
    Input tokens → SharedComponents (frozen) → Expert FFN (trainable) → LM Head → Loss

Information Flow During Inference:
    Input tokens → SharedComponents → Router → Top-K Experts → Weighted sum → LM Head → Output
"""

from modularforge.model.shared import SharedComponents
from modularforge.model.expert import ExpertFFN
from modularforge.model.router import MoERouter
from modularforge.model.moe_layer import MoETransformerLayer
from modularforge.model.module_trainer_model import ModuleTrainerModel
from modularforge.model.assembled_model import AssembledMoEModel
