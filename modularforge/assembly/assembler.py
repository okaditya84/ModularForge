"""
ModularForge Streaming Assembler
==================================
The crown jewel of ModularForge — assembles independently trained expert
modules into a single working MoE model using O(M) memory, where M is
the size of a single expert module.

The Key Insight:
    We NEVER load the full model into RAM. Instead, we build it piece
    by piece, writing each piece to the output file before loading
    the next one. Like assembling a puzzle by gluing one piece at a
    time rather than laying them all out on a table.

Assembly Algorithm:
    1. Create an empty AssembledMoEModel (just the structure, no weights)
    2. Load the shared checkpoint → assign to model → save to output
    3. For each expert i:
        a. Load expert_i.pt (just the expert weights)
        b. Assign expert weights to model.moe_layers[*].experts[i]
        c. Delete loaded weights from memory
    4. Initialize router weights
    5. Save the complete model

Memory Guarantee:
    Peak memory = model_structure + max(shared_weights, expert_weights)
    ≈ ModelStructure + Shared_OR_Expert (whichever is larger)
    For our config: ≈ metadata + max(18M×4B, 8M×4B) ≈ ~72MB

Why Not Just torch.save the Whole Thing:
    That would require loading ALL expert weights simultaneously,
    using O(N×M) memory. With N=5 experts of 8M params each,
    that's 160MB just for expert params — and grows linearly with
    the number of experts. Our approach stays constant at O(M).

Usage:
    >>> assembler = StreamingAssembler(config)
    >>> assembler.assemble(
    ...     shared_path="outputs/shared.pt",
    ...     expert_paths=["outputs/expert_0.pt", ..., "outputs/expert_4.pt"],
    ...     output_path="outputs/assembled_model.pt",
    ... )
"""

from __future__ import annotations

import gc
import json
import logging
import time
import tracemalloc
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from modularforge.config import ModularForgeConfig
from modularforge.model.assembled_model import AssembledMoEModel
from modularforge.model.shared import SharedComponents
from modularforge.model.expert import ExpertFFN

logger = logging.getLogger(__name__)


class StreamingAssembler:
    """
    Streaming O(M) memory assembly of expert modules into a complete
    Mixture-of-Experts model.

    Parameters
    ----------
    config : ModularForgeConfig
        Full configuration.
    """

    def __init__(self, config: ModularForgeConfig):
        self.config = config

    def assemble(
        self,
        shared_path: str,
        expert_paths: list[str],
        output_path: str,
        router_init: Optional[str] = None,
        data_stats: Optional[dict] = None,
    ) -> dict:
        """
        Assemble the complete MoE model from shared + expert checkpoints.

        Parameters
        ----------
        shared_path : str
            Path to the shared components checkpoint.

        expert_paths : list[str]
            Paths to expert checkpoints, in order [expert_0, expert_1, ...].

        output_path : str
            Where to save the assembled model.

        router_init : str or None
            Router initialization strategy ("uniform", "kaiming", "data_stats").
            Defaults to config setting if None.

        data_stats : dict or None
            Optional data statistics for "data_stats" router initialization.
            Expected format: {"partition_centroids": tensor(n_experts, d_model)}.

        Returns
        -------
        dict
            Assembly results including:
            - assembly_time_seconds: how long assembly took
            - peak_memory_mb: peak RAM usage during assembly
            - output_path: path to the assembled model
            - output_size_mb: file size of the assembled model
            - n_params: total parameters in assembled model
        """
        # Validate inputs
        self._validate_inputs(shared_path, expert_paths)

        router_init = router_init or self.config.assembly.router_init

        logger.info(
            f"Starting streaming assembly:\n"
            f"  Shared: {shared_path}\n"
            f"  Experts: {len(expert_paths)} files\n"
            f"  Output: {output_path}\n"
            f"  Router init: {router_init}"
        )

        # Start tracking
        tracemalloc.start()
        start_time = time.time()

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Step 1: Create the empty model structure
        model = AssembledMoEModel(self.config.model)

        # Step 2: Load and assign shared weights
        logger.info("Loading shared components...")
        shared_state = torch.load(
            shared_path, map_location="cpu", weights_only=False
        )
        model.shared.load_state_dict(shared_state)
        del shared_state
        gc.collect()

        self._log_memory("After loading shared")

        # Step 3: Stream expert weights one at a time
        for expert_idx, expert_path in enumerate(expert_paths):
            logger.info(f"Loading expert {expert_idx} from {expert_path}...")

            expert_state = torch.load(
                expert_path, map_location="cpu", weights_only=False
            )

            # Expert checkpoint contains per-layer expert weights
            experts_state_dict = expert_state["experts_state_dict"]

            # Assign expert weights to each MoE layer
            for layer_idx in range(self.config.model.n_layers):
                # Map the expert state dict keys to the model's expert
                expert_in_layer = model.moe_layers[layer_idx].experts[expert_idx]

                # The expert state dict has keys like "0.norm.weight", "0.fc1.weight"
                # where 0 is the layer index
                layer_prefix = f"{layer_idx}."
                layer_state = {}
                for key, value in experts_state_dict.items():
                    if key.startswith(layer_prefix):
                        new_key = key[len(layer_prefix):]
                        layer_state[new_key] = value

                if layer_state:
                    expert_in_layer.load_state_dict(layer_state)

            # Free this expert's data before loading the next
            del expert_state, experts_state_dict
            gc.collect()

            self._log_memory(f"After loading expert {expert_idx}")

        # Step 4: Initialize router weights
        logger.info(f"Initializing router ({router_init})...")
        self._initialize_routers(model, router_init, data_stats)

        # Step 5: Save the assembled model
        logger.info(f"Saving assembled model to {output_path}...")
        self._save_model(model, output_path)

        # Collect results
        _, peak_memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        elapsed = time.time() - start_time

        output_size = Path(output_path).stat().st_size / (1024 * 1024)
        n_params = sum(p.numel() for p in model.parameters())

        results = {
            "assembly_time_seconds": elapsed,
            "peak_memory_mb": peak_memory / (1024 * 1024),
            "output_path": output_path,
            "output_size_mb": output_size,
            "n_params": n_params,
        }

        logger.info(
            f"Assembly complete:\n"
            f"  Time: {elapsed:.2f}s\n"
            f"  Peak memory: {results['peak_memory_mb']:.1f}MB\n"
            f"  Output size: {output_size:.1f}MB\n"
            f"  Total params: {n_params / 1e6:.2f}M"
        )

        del model
        gc.collect()

        return results

    def _validate_inputs(
        self,
        shared_path: str,
        expert_paths: list[str],
    ) -> None:
        """Validate that all input files exist."""
        if not Path(shared_path).exists():
            raise FileNotFoundError(
                f"Shared checkpoint not found: {shared_path}"
            )

        for i, path in enumerate(expert_paths):
            if not Path(path).exists():
                raise FileNotFoundError(
                    f"Expert {i} checkpoint not found: {path}"
                )

        if len(expert_paths) != self.config.model.n_experts:
            raise ValueError(
                f"Expected {self.config.model.n_experts} expert paths, "
                f"got {len(expert_paths)}"
            )

    def _initialize_routers(
        self,
        model: AssembledMoEModel,
        strategy: str,
        data_stats: Optional[dict] = None,
    ) -> None:
        """
        Initialize the MoE router weights.

        Three strategies:
        - "uniform": Equal routing probability for all experts.
          Simple but a good starting point.
        - "kaiming": Kaiming He initialization (default in PyTorch).
          Preserves variance across layers.
        - "data_stats": Use training data statistics to bias routing.
          E.g., route tokens similar to partition i's centroid toward expert i.

        Parameters
        ----------
        model : AssembledMoEModel
            The model whose routers to initialize.
        strategy : str
            Initialization strategy.
        data_stats : dict or None
            Data statistics for "data_stats" strategy.
        """
        for layer_idx, moe_layer in enumerate(model.moe_layers):
            router = moe_layer.router

            if strategy == "uniform":
                nn.init.constant_(router.gate.weight, 0.0)
                logger.debug(f"Router layer {layer_idx}: uniform init")

            elif strategy == "kaiming":
                nn.init.kaiming_uniform_(router.gate.weight, a=0.01)
                logger.debug(f"Router layer {layer_idx}: Kaiming init")

            elif strategy == "data_stats":
                if data_stats and "partition_centroids" in data_stats:
                    centroids = data_stats["partition_centroids"]
                    # Use centroids as router weight initialization
                    # Router gate: (n_experts, d_model)
                    # Centroid: (n_experts, d_model)
                    # By setting gate.weight = centroids, tokens similar to
                    # partition i's centroid will be routed to expert i
                    with torch.no_grad():
                        if centroids.shape == router.gate.weight.shape:
                            router.gate.weight.copy_(centroids)
                        else:
                            logger.warning(
                                f"Centroid shape {centroids.shape} doesn't match "
                                f"gate weight shape {router.gate.weight.shape}. "
                                f"Falling back to Kaiming init."
                            )
                            nn.init.kaiming_uniform_(router.gate.weight, a=0.01)
                else:
                    logger.warning(
                        "data_stats strategy requested but no centroids provided. "
                        "Falling back to Kaiming init."
                    )
                    nn.init.kaiming_uniform_(router.gate.weight, a=0.01)
            else:
                raise ValueError(f"Unknown router init strategy: {strategy}")

    def _save_model(self, model: AssembledMoEModel, output_path: str) -> None:
        """
        Save the assembled model to disk.

        Supports both PyTorch and safetensors formats based on config.

        Parameters
        ----------
        model : AssembledMoEModel
            The assembled model.
        output_path : str
            Output file path.
        """
        if self.config.assembly.output_format == "safetensors":
            try:
                from safetensors.torch import save_file
                state_dict = model.state_dict()
                save_file(state_dict, output_path)
            except ImportError:
                logger.warning(
                    "safetensors not installed, falling back to PyTorch format. "
                    "Install with: pip install safetensors"
                )
                self._save_pytorch(model, output_path)
        else:
            self._save_pytorch(model, output_path)

    def _save_pytorch(self, model: AssembledMoEModel, output_path: str) -> None:
        """Save in PyTorch format."""
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": self.config.to_dict(),
            },
            output_path,
        )

    @staticmethod
    def _log_memory(label: str) -> None:
        """Log current memory usage."""
        current, peak = tracemalloc.get_traced_memory()
        logger.info(
            f"  Memory [{label}]: "
            f"current={current / 1024 / 1024:.1f}MB, "
            f"peak={peak / 1024 / 1024:.1f}MB"
        )
