"""
modularforge.assembly — Streaming Assembly & Calibration
=========================================================
This subpackage contains the most novel contribution of ModularForge:
the ability to assemble independently trained modules into a single
working model using O(M) memory — never loading more than one module
at a time.

Analogy:
    Imagine building a bookshelf from IKEA. You don't need all the
    pieces in your hands at once. You take one shelf, screw it in,
    put it down, take the next shelf, screw it in, and so on. At no
    point do you hold more than one shelf. That's streaming assembly.

Assembly Process:
    1. Read shared checkpoint → write shared tensors to output file
    2. For each expert (i = 0..N-1):
        a. Load expert_i.pt from disk (~40MB)
        b. Write expert_i tensors to the correct position in output file
        c. Delete expert_i from memory (gc.collect)
    3. Initialize router weights (heuristic)
    4. Save in safetensors format
    → Output: assembled_model.safetensors

Post-Assembly Calibration (Optional):
    After assembly, run a tiny calibration pass (~1000 samples) that
    recalibrates LayerNorm statistics without updating any weights.
    This corrects the distribution shift caused by combining
    independently trained modules.

Components:
    - assembler.py    — Streaming O(M) memory assembly algorithm
    - calibration.py  — Post-assembly LayerNorm calibration (MAGIC-inspired)

Information Flow:
    shared.pt + expert_0.pt + ... + expert_N.pt
        → Assembler (streaming, O(M) memory)
        → assembled_model.safetensors
        → Calibration (optional, tiny dataset)
        → calibrated_model.safetensors
"""

from modularforge.assembly.assembler import StreamingAssembler
from modularforge.assembly.calibration import LayerNormCalibrator
