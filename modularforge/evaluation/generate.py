"""
ModularForge Text Generator
==============================
Text generation utilities for the assembled MoE model.

This module provides a high-level interface for generating text from
the assembled model with different sampling strategies.

Sampling Strategies:
    1. GREEDY (temperature=0): Always pick the most likely next token.
       Deterministic but can be repetitive and boring.

    2. TOP-K (top_k=50): Only consider the top 50 most likely tokens,
       then sample from them. Good balance of quality and diversity.

    3. TOP-P / NUCLEUS (top_p=0.9): Only consider tokens whose cumulative
       probability adds up to 90%. Adaptive — when the model is confident,
       it considers fewer tokens; when uncertain, more tokens.

    4. TEMPERATURE: Scale the logits before softmax.
       - Low (0.3): Sharp distribution → conservative, repetitive text
       - Medium (0.8): Balanced → good default
       - High (1.5): Flat distribution → creative but potentially incoherent

Usage:
    >>> generator = TextGenerator(model, tokenizer, device)
    >>> samples = generator.generate_samples(
    ...     prompts=["The history of"],
    ...     max_new_tokens=100,
    ... )
    >>> for prompt, generated in samples:
    ...     print(f"Prompt: {prompt}")
    ...     print(f"Generated: {generated}")
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

from modularforge.data.tokenizer import ModularForgeTokenizer

logger = logging.getLogger(__name__)


class TextGenerator:
    """
    High-level text generation interface.

    Parameters
    ----------
    model : nn.Module
        The assembled MoE model (must have a .generate() method).
    tokenizer : ModularForgeTokenizer
        The trained tokenizer.
    device : torch.device
        Device to run generation on.
    """

    def __init__(
        self,
        model,
        tokenizer: ModularForgeTokenizer,
        device: torch.device,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def generate_samples(
        self,
        prompts: list[str],
        max_new_tokens: int = 200,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
    ) -> list[dict]:
        """
        Generate text continuations for a list of prompts.

        Parameters
        ----------
        prompts : list[str]
            Seed texts to continue from.
        max_new_tokens : int
            Maximum number of new tokens to generate per prompt.
        temperature : float
            Sampling temperature.
        top_k : int
            Top-k filtering parameter.
        top_p : float
            Nucleus sampling parameter.

        Returns
        -------
        list[dict]
            List of dicts with keys:
            - "prompt": the original prompt
            - "generated": the full generated text (prompt + continuation)
            - "continuation": just the continuation (without prompt)
            - "n_tokens": number of tokens generated
        """
        self.model.eval()
        results = []

        for prompt in prompts:
            logger.info(f"Generating from prompt: '{prompt[:50]}...'")

            # Encode prompt
            prompt_ids = self.tokenizer.encode(
                prompt,
                add_special_tokens=True,
            )
            prompt_tensor = torch.tensor(
                [prompt_ids], dtype=torch.long, device=self.device
            )

            # Generate
            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids=prompt_tensor,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    eos_token_id=self.tokenizer.eos_id,
                )

            # Decode
            generated_ids = output_ids[0].tolist()
            full_text = self.tokenizer.decode(
                generated_ids, skip_special_tokens=True
            )

            # Separate prompt from continuation
            prompt_text = self.tokenizer.decode(
                prompt_ids, skip_special_tokens=True
            )

            # Best-effort prompt removal from generated text
            if full_text.startswith(prompt_text):
                continuation = full_text[len(prompt_text):]
            else:
                continuation = full_text

            results.append({
                "prompt": prompt,
                "generated": full_text,
                "continuation": continuation.strip(),
                "n_tokens": len(generated_ids) - len(prompt_ids),
            })

            logger.info(f"  Generated {results[-1]['n_tokens']} tokens")

        return results

    def interactive_generate(
        self,
        max_new_tokens: int = 200,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
    ) -> None:
        """
        Interactive text generation loop (for terminal use).

        Type a prompt and see the model's continuation.
        Type 'quit' or 'exit' to stop.

        Parameters
        ----------
        max_new_tokens : int
            Maximum tokens to generate per prompt.
        temperature, top_k, top_p : float
            Sampling parameters.
        """
        print("\n" + "=" * 60)
        print("ModularForge Interactive Generator")
        print("Type a prompt and press Enter. Type 'quit' to exit.")
        print("=" * 60 + "\n")

        while True:
            try:
                prompt = input(">>> ")
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            if prompt.lower() in ("quit", "exit", "q"):
                break

            if not prompt.strip():
                continue

            results = self.generate_samples(
                prompts=[prompt],
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )

            print(f"\n{results[0]['generated']}\n")
