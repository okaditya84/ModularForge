"""
modularforge.data — Data Pipeline
==================================
This subpackage handles everything related to preparing data for
ModularForge training:

    1. **Tokenizer** (`tokenizer.py`):
       Trains a Byte-Pair Encoding (BPE) tokenizer on the full corpus.
       Think of this as building the "vocabulary" that all modules share.

    2. **Dataset** (`dataset.py`):
       Converts raw text into fixed-length token sequences that PyTorch
       can batch and feed to the model during training.

    3. **Partitioner** (`partitioner.py`):
       Splits the training corpus into N disjoint partitions — one per
       expert module. Supports random, topic-clustered, and curriculum
       partitioning strategies.

Information Flow:
    Raw text (WikiText-103)
        → Tokenizer (builds vocabulary)
        → Partitioner (splits text into N groups)
        → Dataset (converts each group into token sequences)
        → DataLoader (feeds batches to trainer)
"""

from modularforge.data.tokenizer import ModularForgeTokenizer
from modularforge.data.dataset import TextDataset
from modularforge.data.partitioner import DataPartitioner
