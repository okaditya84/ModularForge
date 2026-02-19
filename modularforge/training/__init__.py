"""
modularforge.training — Training Engine
========================================
This subpackage manages the entire training process for ModularForge.

Training happens in two phases, like building a house:

    Phase 1: "Build the Foundation" (SharedTrainer)
        Train embedding + attention layers on the FULL corpus.
        These become the shared foundation that every expert stands on.
        → Output: frozen shared checkpoint

    Phase 2: "Train the Specialists" (ModuleTrainer)
        For each expert module (sequentially, one at a time):
            1. Load the frozen shared checkpoint
            2. Train ONLY the expert FFN on its data partition
            3. Save the expert checkpoint
            4. Free all memory
            5. Move to next expert
        → Output: N expert checkpoints

Memory Budget:
    At no point during training does RAM usage exceed:
        shared_params + expert_params + optimizer_states + gradients + batch
        ≈ 18M + 8M + 16M + 8M + batch ≈ ~160MB

Components:
    - trainer.py         — Core training loop (gradient accumulation, scheduling,
                           checkpointing, memory tracking, logging)
    - shared_trainer.py  — Phase 1: trains shared components on full corpus
    - module_trainer.py  — Phase 2: trains expert modules sequentially

Information Flow:
    Full corpus → SharedTrainer → frozen shared.pt
    Partition 0 → ModuleTrainer(shared.pt) → expert_0.pt
    Partition 1 → ModuleTrainer(shared.pt) → expert_1.pt
    ...
    Partition N → ModuleTrainer(shared.pt) → expert_N.pt
"""

from modularforge.training.trainer import Trainer
from modularforge.training.shared_trainer import SharedTrainer
from modularforge.training.module_trainer import ModuleTrainer
