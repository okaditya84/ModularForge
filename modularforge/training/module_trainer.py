"""
ModularForge Module Trainer
=============================
Trains expert modules SEQUENTIALLY — one at a time, each on its own
data partition. This is Phase 2 of the ModularForge pipeline and the
key to the memory-bounded guarantee.

The Sequential Training Loop:
    For module_idx in range(n_experts):
        1. Load frozen shared components
        2. Create a new ExpertFFN (randomly initialized)
        3. Wrap in ModuleTrainerModel
        4. Train on partition[module_idx]
        5. Save expert checkpoint
        6. Delete model + optimizer + data from memory
        7. gc.collect() — force garbage collection
        → Peak memory NEVER exceeds shared + 1 expert + optimizer

Analogy:
    Like training 5 different chefs one at a time in the same kitchen.
    Chef 1 learns Italian cuisine, saves their recipes, leaves.
    Chef 2 comes in, learns Japanese cuisine, saves recipes, leaves.
    And so on. The kitchen (shared components) stays the same, and
    you never have more than one chef in the kitchen at a time.

Usage:
    >>> module_trainer = ModuleTrainer(config, shared_path="outputs/shared.pt")
    >>> module_trainer.train_all(partitioned_data, output_dir="outputs")
    # Produces: expert_0.pt, expert_1.pt, ..., expert_N.pt
"""

from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader

from modularforge.config import ModularForgeConfig
from modularforge.data.dataset import TextDataset
from modularforge.data.tokenizer import ModularForgeTokenizer
from modularforge.model.shared import SharedComponents
from modularforge.model.module_trainer_model import ModuleTrainerModel
from modularforge.training.trainer import Trainer

logger = logging.getLogger(__name__)


class ModuleTrainer:
    """
    Phase 2 trainer: sequentially trains each expert module on its
    data partition.

    Parameters
    ----------
    config : ModularForgeConfig
        Full configuration.

    shared_path : str
        Path to the saved shared components checkpoint.
        This will be loaded fresh for EACH expert to guarantee
        consistent initialization.

    tokenizer : ModularForgeTokenizer
        Trained tokenizer for creating datasets from text partitions.
    """

    def __init__(
        self,
        config: ModularForgeConfig,
        shared_path: str,
        tokenizer: ModularForgeTokenizer,
    ):
        self.config = config
        self.shared_path = shared_path
        self.tokenizer = tokenizer

        if not Path(shared_path).exists():
            raise FileNotFoundError(
                f"Shared checkpoint not found: {shared_path}. "
                f"Run SharedTrainer first."
            )

    def train_all(
        self,
        partitions: list[list[str]],
        val_texts: Optional[list[str]] = None,
        output_dir: str = "outputs",
    ) -> list[dict]:
        """
        Train all expert modules sequentially.

        For each partition, this method:
        1. Creates a fresh model (shared frozen + new expert)
        2. Trains the expert
        3. Saves the expert checkpoint
        4. Frees all memory

        Parameters
        ----------
        partitions : list[list[str]]
            Data partitions — one list of text articles per expert.
            partitions[i] is used to train expert i.

        val_texts : list[str] or None
            Validation text (shared across all experts for fair comparison).

        output_dir : str
            Directory to save expert checkpoints.

        Returns
        -------
        list[dict]
            Training results for each expert (loss, memory, time, etc.)
        """
        if len(partitions) != self.config.model.n_experts:
            raise ValueError(
                f"Number of partitions ({len(partitions)}) must match "
                f"n_experts ({self.config.model.n_experts})"
            )

        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Create validation loader once (shared across experts)
        val_loader = None
        if val_texts:
            val_dataset = TextDataset(
                texts=val_texts,
                tokenizer=self.tokenizer,
                max_seq_len=self.config.model.max_seq_len,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.training.batch_size,
                shuffle=False,
                num_workers=min(self.config.training.num_workers, 2),
                pin_memory=False,
            )

        all_results = []

        for expert_idx in range(self.config.model.n_experts):
            logger.info(
                f"\n{'='*60}\n"
                f"Training Expert {expert_idx}/{self.config.model.n_experts - 1}\n"
                f"Partition size: {len(partitions[expert_idx]):,} articles\n"
                f"{'='*60}"
            )

            result = self._train_single_expert(
                expert_idx=expert_idx,
                texts=partitions[expert_idx],
                val_loader=val_loader,
                output_dir=output_dir,
            )

            all_results.append(result)

            # Force memory cleanup between experts
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        logger.info(
            f"\nAll {self.config.model.n_experts} experts trained. "
            f"Checkpoints in {output_dir}/"
        )

        return all_results

    def _train_single_expert(
        self,
        expert_idx: int,
        texts: list[str],
        val_loader: Optional[DataLoader],
        output_dir: str,
    ) -> dict:
        """
        Train a single expert module.

        This method is self-contained: it loads shared components,
        creates the model, trains it, saves the result, and cleans up.
        Memory after this call returns to baseline.

        Parameters
        ----------
        expert_idx : int
            Index of the expert to train.
        texts : list[str]
            Text articles for this expert's partition.
        val_loader : DataLoader or None
            Validation data loader.
        output_dir : str
            Directory to save the expert checkpoint.

        Returns
        -------
        dict
            Training results for this expert.
        """
        # 1. Load fresh shared components and freeze them
        shared = SharedComponents(self.config.model)
        shared_state = torch.load(
            self.shared_path,
            map_location="cpu",
            weights_only=False,
        )
        shared.load_state_dict(shared_state)
        shared.freeze()

        # Free the loaded state dict
        del shared_state
        gc.collect()

        # 2. Create training model with fresh expert
        model = ModuleTrainerModel(
            shared=shared,
            config=self.config.model,
            expert_idx=expert_idx,
        )

        # 3. Create training dataset from this partition's texts
        train_dataset = TextDataset(
            texts=texts,
            tokenizer=self.tokenizer,
            max_seq_len=self.config.model.max_seq_len,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=min(self.config.training.num_workers, 2),
            pin_memory=False,
            drop_last=True,  # Avoid tiny last batch
        )

        # 4. Train
        trainer = Trainer(
            model=model,
            config=self.config,
            train_loader=train_loader,
            val_loader=val_loader,
            name=f"expert_{expert_idx}",
        )

        results = trainer.train(
            epochs=self.config.training.epochs_expert,
            output_dir=output_dir,
        )

        # 5. Save expert checkpoint
        expert_path = str(Path(output_dir) / f"expert_{expert_idx}.pt")
        model.save_expert(expert_path)
        results["expert_path"] = expert_path

        # 6. Clean up everything
        del model, trainer, train_dataset, train_loader, shared
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info(f"Expert {expert_idx} complete — memory cleaned up")

        return results
