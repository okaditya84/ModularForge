"""
ModularForge Evaluator
========================
Complete evaluation pipeline that runs all metrics and generates
a comprehensive results report.

What Gets Measured:
    1. Perplexity on test set (lower = better)
    2. Peak memory at each pipeline stage
    3. Assembly time
    4. Text generation quality (qualitative samples)
    5. Router statistics (expert utilization balance)

Output:
    A structured results dictionary and a formatted report suitable
    for inclusion in a paper or README.

Usage:
    >>> evaluator = Evaluator(config, tokenizer, device)
    >>> results = evaluator.evaluate(
    ...     model=assembled_model,
    ...     test_loader=test_loader,
    ...     output_dir="outputs/eval",
    ... )
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from modularforge.config import ModularForgeConfig
from modularforge.data.tokenizer import ModularForgeTokenizer
from modularforge.evaluation.metrics import compute_perplexity, MemoryTracker
from modularforge.evaluation.generate import TextGenerator

logger = logging.getLogger(__name__)


class Evaluator:
    """
    Complete evaluation pipeline for ModularForge models.

    Parameters
    ----------
    config : ModularForgeConfig
        Full configuration.
    tokenizer : ModularForgeTokenizer
        Trained tokenizer.
    device : torch.device
        Evaluation device.
    """

    def __init__(
        self,
        config: ModularForgeConfig,
        tokenizer: ModularForgeTokenizer,
        device: torch.device,
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.device = device

    def evaluate(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        output_dir: str = "outputs/eval",
        assembly_results: Optional[dict] = None,
        training_results: Optional[dict] = None,
    ) -> dict:
        """
        Run the full evaluation pipeline.

        Parameters
        ----------
        model : nn.Module
            The assembled MoE model to evaluate.
        test_loader : DataLoader
            Test data loader.
        output_dir : str
            Directory to save evaluation results and generated samples.
        assembly_results : dict or None
            Results from the assembly step (for reporting).
        training_results : dict or None
            Results from training (for reporting).

        Returns
        -------
        dict
            Comprehensive evaluation results.
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        model = model.to(self.device)
        model.eval()

        results = {
            "model_info": self._get_model_info(model),
            "perplexity": None,
            "generated_samples": [],
            "router_stats": None,
            "assembly_info": assembly_results,
            "training_info": training_results,
        }

        # 1. Perplexity
        logger.info("Computing perplexity...")
        with MemoryTracker("Perplexity evaluation") as mem:
            ppl = compute_perplexity(model, test_loader, self.device)
        results["perplexity"] = {
            "value": ppl,
            "memory_mb": mem.peak_mb,
            "time_seconds": mem.duration_seconds,
        }

        # 2. Text Generation
        logger.info("Generating text samples...")
        generator = TextGenerator(model, self.tokenizer, self.device)
        samples = generator.generate_samples(
            prompts=self.config.evaluation.prompts,
            max_new_tokens=self.config.evaluation.generate_max_tokens,
            temperature=self.config.evaluation.temperature,
            top_k=self.config.evaluation.top_k,
            top_p=self.config.evaluation.top_p,
        )
        results["generated_samples"] = samples

        # 3. Router Statistics
        logger.info("Computing router statistics...")
        results["router_stats"] = self._compute_router_stats(model, test_loader)

        # 4. Save results
        self._save_results(results, output_dir)
        self._print_report(results)

        return results

    def _get_model_info(self, model: nn.Module) -> dict:
        """Collect model metadata."""
        n_total = sum(p.numel() for p in model.parameters())

        info = {
            "total_params": n_total,
            "total_params_M": n_total / 1e6,
            "d_model": self.config.model.d_model,
            "n_layers": self.config.model.n_layers,
            "n_experts": self.config.model.n_experts,
            "top_k": self.config.model.top_k,
            "vocab_size": self.config.model.vocab_size,
            "max_seq_len": self.config.model.max_seq_len,
        }

        # Active params per token (for sparse MoE)
        if hasattr(model, "n_active_params_per_token"):
            info["active_params_per_token"] = model.n_active_params_per_token
            info["active_params_per_token_M"] = model.n_active_params_per_token / 1e6

        return info

    @torch.no_grad()
    def _compute_router_stats(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        max_batches: int = 50,
    ) -> dict:
        """
        Analyze how the router distributes tokens across experts.

        Measures:
        - Expert utilization: fraction of tokens routed to each expert
        - Balance score: how evenly tokens are distributed (1.0 = perfect)
        - Expert selection frequency per layer

        Parameters
        ----------
        model : nn.Module
            The assembled model.
        data_loader : DataLoader
            Data to analyze routing on.
        max_batches : int
            Maximum batches to process for stats.

        Returns
        -------
        dict
            Router statistics.
        """
        model.eval()

        if not hasattr(model, "moe_layers"):
            return {"error": "Model does not have MoE layers"}

        n_experts = self.config.model.n_experts
        n_layers = self.config.model.n_layers

        # Expert selection counts: (n_layers, n_experts)
        expert_counts = torch.zeros(n_layers, n_experts)
        total_tokens = 0

        # Hook to capture router outputs
        router_outputs = {}

        def make_router_hook(layer_idx):
            def hook(module, input, output):
                weights, indices, _ = output
                router_outputs[layer_idx] = indices.cpu()
            return hook

        # Register hooks
        hooks = []
        for layer_idx, moe_layer in enumerate(model.moe_layers):
            h = moe_layer.router.register_forward_hook(make_router_hook(layer_idx))
            hooks.append(h)

        # Process batches
        n_processed = 0
        for input_ids, _ in data_loader:
            if n_processed >= max_batches:
                break

            input_ids = input_ids.to(self.device)
            attention_mask = (input_ids != 0).long()
            model(input_ids, attention_mask=attention_mask)

            # Accumulate counts
            for layer_idx, indices in router_outputs.items():
                for k in range(indices.shape[-1]):
                    for expert_idx in range(n_experts):
                        count = (indices[..., k] == expert_idx).sum().item()
                        expert_counts[layer_idx, expert_idx] += count

            total_tokens += (input_ids != 0).sum().item()
            n_processed += 1
            router_outputs.clear()

        # Remove hooks
        for h in hooks:
            h.remove()

        # Compute statistics
        if total_tokens == 0:
            return {"error": "No tokens processed"}

        # Normalize to get utilization fractions
        utilization = expert_counts / expert_counts.sum(dim=-1, keepdim=True).clamp(min=1)

        # Balance score: 1 - coefficient of variation
        # Perfect balance = all experts get 1/n_experts = 20% each
        cv = utilization.std(dim=-1) / utilization.mean(dim=-1).clamp(min=1e-8)
        balance_scores = (1 - cv).tolist()

        stats = {
            "expert_utilization": utilization.tolist(),
            "balance_scores_per_layer": balance_scores,
            "average_balance_score": sum(balance_scores) / max(len(balance_scores), 1),
            "total_tokens_analyzed": total_tokens,
        }

        logger.info(
            f"Router balance: avg={stats['average_balance_score']:.3f} "
            f"(1.0=perfect)"
        )

        return stats

    def _save_results(self, results: dict, output_dir: str) -> None:
        """Save evaluation results to JSON."""
        # Make results JSON-serializable
        clean_results = self._make_serializable(results)

        path = Path(output_dir) / "evaluation_results.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(clean_results, f, indent=2, ensure_ascii=False)

        logger.info(f"Results saved to {path}")

        # Save generated samples separately for easy reading
        samples_path = Path(output_dir) / "generated_samples.txt"
        with open(samples_path, "w", encoding="utf-8") as f:
            for i, sample in enumerate(results.get("generated_samples", [])):
                f.write(f"{'='*60}\n")
                f.write(f"Sample {i+1}\n")
                f.write(f"{'='*60}\n")
                f.write(f"Prompt: {sample['prompt']}\n")
                f.write(f"Generated: {sample['generated']}\n\n")

        logger.info(f"Generated samples saved to {samples_path}")

    def _print_report(self, results: dict) -> None:
        """Print a formatted evaluation report."""
        print("\n" + "=" * 60)
        print("ModularForge Evaluation Report")
        print("=" * 60)

        # Model Info
        info = results.get("model_info", {})
        print(f"\nModel: {info.get('total_params_M', 0):.2f}M params "
              f"({info.get('n_experts', '?')} experts)")

        # Perplexity
        ppl = results.get("perplexity", {})
        if ppl:
            ppl_val = ppl.get("value", float("inf"))
            print(f"Perplexity: {ppl_val:.2f}")

        # Router Balance
        router = results.get("router_stats", {})
        if router and "average_balance_score" in router:
            print(f"Router Balance: {router['average_balance_score']:.3f}")

        # Assembly Info
        assembly = results.get("assembly_info", {})
        if assembly:
            print(f"Assembly Time: {assembly.get('assembly_time_seconds', 0):.2f}s")
            print(f"Assembly Peak Memory: {assembly.get('peak_memory_mb', 0):.1f}MB")

        # Generated Samples
        samples = results.get("generated_samples", [])
        if samples:
            print(f"\n--- Generated Samples ({len(samples)}) ---")
            for i, s in enumerate(samples[:3]):  # Show first 3
                print(f"\n  [{i+1}] '{s['prompt']}'")
                continuation = s.get('continuation', '')[:200]
                print(f"      → {continuation}...")

        print("\n" + "=" * 60)

    @staticmethod
    def _make_serializable(obj):
        """Convert non-serializable objects for JSON."""
        if isinstance(obj, dict):
            return {k: Evaluator._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [Evaluator._make_serializable(v) for v in obj]
        elif isinstance(obj, (torch.Tensor,)):
            return obj.tolist()
        elif isinstance(obj, float):
            if math.isinf(obj) or math.isnan(obj):
                return str(obj)
            return obj
        return obj
