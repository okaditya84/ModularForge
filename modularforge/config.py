"""
ModularForge Configuration System
===================================
Centralized configuration for all ModularForge components using Python
dataclasses. Every hyperparameter, path, and setting lives here.

Think of this as the "blueprint" for your entire experiment. Change a
value here and it propagates everywhere — no hunting through code.

Usage:
    # Load from YAML file:
    >>> config = ModularForgeConfig.from_yaml("configs/default.yaml")

    # Create programmatically:
    >>> config = ModularForgeConfig(
    ...     model=ModelConfig(d_model=512, n_experts=5),
    ...     training=TrainingConfig(learning_rate=3e-4),
    ... )

    # Save to YAML:
    >>> config.to_yaml("configs/my_experiment.yaml")

    # Access nested values:
    >>> config.model.d_model        # 512
    >>> config.training.batch_size  # 16
"""

from __future__ import annotations

import os
import yaml
import torch
import logging
from dataclasses import dataclass, field, asdict
from typing import Optional, Literal
from pathlib import Path

logger = logging.getLogger(__name__)


# =============================================================================
# Model Configuration
# =============================================================================

@dataclass
class ModelConfig:
    """
    Architecture hyperparameters for the ModularForge model.

    Analogy: If the model is a building, these parameters define the
    building's blueprint — how tall it is (n_layers), how wide each
    floor is (d_model), how many specialist offices there are (n_experts),
    and how many words are in its dictionary (vocab_size).

    Parameters
    ----------
    d_model : int
        Dimensionality of the model's internal representations.
        Every token is represented as a vector of this size.
        Analogy: The "width" of the information highway — bigger means
        the model can carry more nuanced information per token.
        Typical values: 256 (tiny), 512 (small), 768 (medium), 1024 (large).

    n_heads : int
        Number of attention heads in multi-head self-attention.
        Each head independently "looks at" different aspects of the input.
        Analogy: Like having multiple readers each focusing on different
        aspects of a document (grammar, meaning, context, etc.).
        Must evenly divide d_model. Typical: d_model // 64.

    n_layers : int
        Number of transformer blocks stacked on top of each other.
        More layers = deeper understanding but slower training.
        Analogy: Layers of analysis — first layer understands words,
        second understands phrases, third understands sentences, etc.

    d_ff : int
        Dimensionality of the feed-forward network inside each expert.
        Usually 4× d_model. This is where most of the model's "knowledge"
        is stored.
        Analogy: The size of each specialist's "brain" — bigger brains
        can memorize more patterns.

    n_experts : int
        Number of expert modules (the independently trainable units).
        Each expert is trained on a different data partition.
        Analogy: Number of specialists on your team — one for code,
        one for math, one for science, etc.

    top_k : int
        How many experts are activated per token during inference.
        Higher = more compute but potentially better quality.
        Typical: 1 or 2.

    vocab_size : int
        Size of the tokenizer's vocabulary.
        Analogy: How many unique "words" the model knows.
        Must match the trained tokenizer.

    max_seq_len : int
        Maximum sequence length the model can process.
        Analogy: The maximum "paragraph length" the model can read at once.

    dropout : float
        Dropout probability for regularization (prevents overfitting).
        Analogy: Randomly "forgetting" some information during training
        forces the model to be more robust.

    expert_dropout : float
        Separate dropout rate inside expert FFN layers.
        Can be tuned independently from the main dropout.
    """
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 4
    d_ff: int = 2048
    n_experts: int = 5
    top_k: int = 2
    vocab_size: int = 16384
    max_seq_len: int = 512
    dropout: float = 0.1
    expert_dropout: float = 0.1

    def validate(self) -> None:
        """
        Check that all model parameters are valid and consistent.

        Raises
        ------
        ValueError
            If any parameter is invalid or inconsistent with others.
        """
        if self.d_model <= 0:
            raise ValueError(f"d_model must be positive, got {self.d_model}")
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads "
                f"({self.n_heads}). Each head gets d_model/n_heads = "
                f"{self.d_model / self.n_heads:.1f} dimensions, which "
                f"must be an integer."
            )
        if self.n_experts < 1:
            raise ValueError(f"n_experts must be >= 1, got {self.n_experts}")
        if self.top_k < 1 or self.top_k > self.n_experts:
            raise ValueError(
                f"top_k ({self.top_k}) must be between 1 and n_experts "
                f"({self.n_experts})"
            )
        if self.vocab_size < 100:
            raise ValueError(
                f"vocab_size ({self.vocab_size}) is suspiciously small. "
                f"Minimum recommended: 1000"
            )
        if self.max_seq_len < 16:
            raise ValueError(
                f"max_seq_len ({self.max_seq_len}) is too small. "
                f"Minimum: 16"
            )
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        if not 0.0 <= self.expert_dropout < 1.0:
            raise ValueError(
                f"expert_dropout must be in [0, 1), got {self.expert_dropout}"
            )

    @property
    def head_dim(self) -> int:
        """Dimension per attention head (d_model / n_heads)."""
        return self.d_model // self.n_heads

    @property
    def expert_params(self) -> int:
        """Approximate parameter count per expert FFN."""
        # Two linear layers: d_model→d_ff and d_ff→d_model, plus biases
        return 2 * self.d_model * self.d_ff + self.d_model + self.d_ff

    @property
    def total_params_estimate(self) -> int:
        """Rough estimate of total assembled model parameters."""
        embedding_params = self.vocab_size * self.d_model  # Token + LM head
        pos_params = self.max_seq_len * self.d_model
        attn_params = self.n_layers * 4 * self.d_model * self.d_model  # Q,K,V,O
        expert_total = self.n_layers * self.n_experts * self.expert_params
        norm_params = self.n_layers * 2 * self.d_model  # 2 norms per layer
        router_params = self.n_layers * self.d_model * self.n_experts
        return (embedding_params + pos_params + attn_params +
                expert_total + norm_params + router_params)


# =============================================================================
# Training Configuration
# =============================================================================

@dataclass
class TrainingConfig:
    """
    Hyperparameters for the training process.

    Analogy: If ModelConfig is the blueprint, TrainingConfig is the
    "construction plan" — how fast workers build (learning_rate), how
    many bricks they carry at once (batch_size), and how many passes
    they make over the site (epochs).

    Parameters
    ----------
    learning_rate : float
        Step size for the optimizer. Too high = unstable learning,
        too low = painfully slow convergence.
        Analogy: How big of a step you take when walking downhill —
        big steps are fast but you might overshoot the valley.

    batch_size : int
        Number of sequences processed in parallel per step.
        Limited by available RAM. Larger = more stable gradients
        but more memory.

    epochs_shared : int
        Number of full passes through the corpus for training shared
        components (embedding + attention). These need more training
        because they're the foundation for everything.

    epochs_expert : int
        Number of full passes through its partition for each expert.
        Experts are smaller and see less data, so they need fewer epochs.

    warmup_steps : int
        Number of steps where learning rate linearly increases from 0.
        Analogy: "Warming up" before exercise — start gentle to avoid
        destroying randomly initialized weights.

    weight_decay : float
        L2 regularization coefficient. Penalizes large weights to
        prevent overfitting.

    max_grad_norm : float
        Maximum gradient norm for clipping. Prevents "exploding gradients"
        that can destroy training.

    gradient_accumulation_steps : int
        Accumulate gradients over this many steps before updating weights.
        Effectively multiplies batch size without multiplying memory.
        Analogy: Writing down notes from multiple meetings before
        making a decision, rather than reacting to each meeting.

    seed : int
        Random seed for reproducibility. Same seed = same results.

    num_workers : int
        Number of parallel data loading workers.
        For M4 Pro: 4-6 is optimal. Set to 0 for Kaggle.

    use_amp : bool
        Whether to use Automatic Mixed Precision (float16/bfloat16).
        Speeds up training on GPU but not useful on CPU.

    device : str
        Device to train on. Auto-detected if set to "auto".
        Options: "auto", "cpu", "mps" (Apple Silicon), "cuda" (NVIDIA GPU).

    checkpoint_every : int
        Save a checkpoint every N steps. Set to 0 to only save at end.

    log_every : int
        Print training stats every N steps.

    eval_every : int
        Run validation every N steps. Set to 0 for only end-of-epoch eval.
    """
    learning_rate: float = 3e-4
    batch_size: int = 16
    epochs_shared: int = 3
    epochs_expert: int = 5
    warmup_steps: int = 200
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 1
    seed: int = 42
    num_workers: int = 4
    use_amp: bool = False
    device: str = "auto"
    checkpoint_every: int = 1000
    log_every: int = 50
    eval_every: int = 500

    def validate(self) -> None:
        """Validate training parameters."""
        if self.learning_rate <= 0:
            raise ValueError(
                f"learning_rate must be positive, got {self.learning_rate}"
            )
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.epochs_shared < 1:
            raise ValueError(
                f"epochs_shared must be >= 1, got {self.epochs_shared}"
            )
        if self.epochs_expert < 1:
            raise ValueError(
                f"epochs_expert must be >= 1, got {self.epochs_expert}"
            )
        if self.gradient_accumulation_steps < 1:
            raise ValueError(
                f"gradient_accumulation_steps must be >= 1, got "
                f"{self.gradient_accumulation_steps}"
            )

    def resolve_device(self) -> torch.device:
        """
        Auto-detect the best available device.

        Priority: CUDA (Kaggle GPU) > MPS (Apple Silicon) > CPU

        Returns
        -------
        torch.device
            The resolved device.
        """
        if self.device != "auto":
            return torch.device(self.device)

        if torch.cuda.is_available():
            logger.info("Using CUDA device (GPU detected)")
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            logger.info("Using MPS device (Apple Silicon detected)")
            return torch.device("mps")
        else:
            logger.info("Using CPU device")
            return torch.device("cpu")


# =============================================================================
# Data Configuration
# =============================================================================

@dataclass
class DataConfig:
    """
    Configuration for data loading, tokenization, and partitioning.

    Parameters
    ----------
    dataset_name : str
        HuggingFace dataset identifier.
        Default: "wikitext" with config "wikitext-103-raw-v1".

    dataset_config : str
        Sub-configuration for the HuggingFace dataset.

    tokenizer_vocab_size : int
        Vocabulary size for the BPE tokenizer.
        Must match model.vocab_size.

    partition_strategy : str
        How to split data among expert modules.
        - "random": shuffle and split evenly
        - "clustered": semantic clustering via sentence embeddings
        - "curriculum": sort by complexity then split

    overlap_ratio : float
        Fraction of data shared between adjacent partitions.
        Higher overlap = more shared knowledge = better coherence.
        Range: [0.0, 0.5]. Recommended: 0.1 (10%).

    min_article_length : int
        Minimum character length for an article to be included.
        Filters out stubs and empty articles.

    data_dir : str
        Directory to store downloaded/processed data.

    max_articles : int or None
        Maximum number of articles to use. None = use all.
        Set to a small number for smoke testing.
    """
    dataset_name: str = "wikitext"
    dataset_config: str = "wikitext-103-raw-v1"
    tokenizer_vocab_size: int = 16384
    partition_strategy: Literal["random", "clustered", "curriculum"] = "clustered"
    overlap_ratio: float = 0.1
    min_article_length: int = 100
    data_dir: str = "data"
    max_articles: Optional[int] = None

    def validate(self) -> None:
        """Validate data parameters."""
        if self.tokenizer_vocab_size < 100:
            raise ValueError(
                f"tokenizer_vocab_size ({self.tokenizer_vocab_size}) too small"
            )
        if self.partition_strategy not in ("random", "clustered", "curriculum"):
            raise ValueError(
                f"Unknown partition_strategy: '{self.partition_strategy}'. "
                f"Choose from: random, clustered, curriculum"
            )
        if not 0.0 <= self.overlap_ratio <= 0.5:
            raise ValueError(
                f"overlap_ratio must be in [0, 0.5], got {self.overlap_ratio}"
            )


# =============================================================================
# Assembly Configuration
# =============================================================================

@dataclass
class AssemblyConfig:
    """
    Configuration for the streaming assembly process.

    Parameters
    ----------
    output_format : str
        Format for the assembled model file.
        Currently supports: "safetensors", "pytorch".

    router_init : str
        How to initialize the MoE router weights.
        - "uniform": equal probability for all experts
        - "kaiming": Kaiming He initialization
        - "data_stats": based on training data statistics

    calibrate : bool
        Whether to run post-assembly LayerNorm calibration.

    calibration_samples : int
        Number of samples for LayerNorm calibration.
        More samples = better calibration but slower.

    output_dir : str
        Directory to save assembled model files.
    """
    output_format: Literal["safetensors", "pytorch"] = "safetensors"
    router_init: Literal["uniform", "kaiming", "data_stats"] = "kaiming"
    calibrate: bool = True
    calibration_samples: int = 1000
    output_dir: str = "outputs"

    def validate(self) -> None:
        """Validate assembly parameters."""
        if self.output_format not in ("safetensors", "pytorch"):
            raise ValueError(
                f"Unknown output_format: '{self.output_format}'. "
                f"Choose from: safetensors, pytorch"
            )
        if self.router_init not in ("uniform", "kaiming", "data_stats"):
            raise ValueError(
                f"Unknown router_init: '{self.router_init}'. "
                f"Choose from: uniform, kaiming, data_stats"
            )
        if self.calibration_samples < 0:
            raise ValueError(
                f"calibration_samples must be >= 0, "
                f"got {self.calibration_samples}"
            )


# =============================================================================
# Evaluation Configuration
# =============================================================================

@dataclass
class EvalConfig:
    """
    Configuration for model evaluation and text generation.

    Parameters
    ----------
    eval_batch_size : int
        Batch size for evaluation (can be larger than training batch
        since no gradients are stored).

    generate_samples : int
        Number of text samples to generate for qualitative evaluation.

    generate_max_tokens : int
        Maximum number of tokens to generate per sample.

    temperature : float
        Sampling temperature. Higher = more random, lower = more focused.
        1.0 = standard, 0.7 = slightly focused, 0.0 = greedy (argmax).

    top_k : int
        Top-k sampling: only consider the k most likely next tokens.
        0 = disabled (consider all tokens).

    top_p : float
        Nucleus sampling: only consider tokens whose cumulative probability
        exceeds p. 1.0 = disabled.

    prompts : list[str]
        Seed prompts for text generation evaluation.
    """
    eval_batch_size: int = 32
    generate_samples: int = 5
    generate_max_tokens: int = 200
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.9
    prompts: list[str] = field(default_factory=lambda: [
        "The scientific method is",
        "In mathematics, a prime number",
        "The history of computing began",
        "Machine learning algorithms can",
        "The structure of DNA was",
    ])

    def validate(self) -> None:
        """Validate evaluation parameters."""
        if self.eval_batch_size < 1:
            raise ValueError(
                f"eval_batch_size must be >= 1, got {self.eval_batch_size}"
            )
        if self.temperature < 0:
            raise ValueError(
                f"temperature must be >= 0, got {self.temperature}"
            )
        if self.top_k < 0:
            raise ValueError(f"top_k must be >= 0, got {self.top_k}")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}")


# =============================================================================
# Master Configuration
# =============================================================================

@dataclass
class ModularForgeConfig:
    """
    Master configuration combining all sub-configurations.

    This is the single source of truth for an experiment. Pass this
    object to any ModularForge component and it will extract the
    settings it needs.

    Usage:
        # From YAML file:
        >>> config = ModularForgeConfig.from_yaml("configs/default.yaml")

        # Programmatic:
        >>> config = ModularForgeConfig()
        >>> config.validate()

        # Save:
        >>> config.to_yaml("configs/my_experiment.yaml")
    """
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    assembly: AssemblyConfig = field(default_factory=AssemblyConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)

    def validate(self) -> None:
        """
        Validate all sub-configurations and cross-config consistency.

        Raises
        ------
        ValueError
            If any parameter is invalid or configs are inconsistent.
        """
        self.model.validate()
        self.training.validate()
        self.data.validate()
        self.assembly.validate()
        self.evaluation.validate()

        # Cross-config consistency checks
        if self.data.tokenizer_vocab_size != self.model.vocab_size:
            raise ValueError(
                f"Tokenizer vocab_size ({self.data.tokenizer_vocab_size}) "
                f"must match model vocab_size ({self.model.vocab_size}). "
                f"These must be the same so the model can understand the "
                f"tokenizer's output."
            )

        logger.info(
            f"Config validated: {self.model.total_params_estimate / 1e6:.1f}M "
            f"params, {self.model.n_experts} experts, "
            f"device={self.training.device}"
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> ModularForgeConfig:
        """
        Load configuration from a YAML file.

        Parameters
        ----------
        path : str or Path
            Path to the YAML configuration file.

        Returns
        -------
        ModularForgeConfig
            Loaded and validated configuration.

        Raises
        ------
        FileNotFoundError
            If the YAML file does not exist.
        yaml.YAMLError
            If the YAML file is malformed.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"Config file not found: {path}. "
                f"Create one from configs/default.yaml as a template."
            )

        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        if raw is None:
            raise ValueError(f"Config file is empty: {path}")

        # Build sub-configs from nested dicts
        config = cls(
            model=ModelConfig(**raw.get("model", {})),
            training=TrainingConfig(**raw.get("training", {})),
            data=DataConfig(**raw.get("data", {})),
            assembly=AssemblyConfig(**raw.get("assembly", {})),
            evaluation=EvalConfig(**raw.get("evaluation", {})),
        )

        config.validate()
        return config

    def to_yaml(self, path: str | Path) -> None:
        """
        Save configuration to a YAML file.

        Creates parent directories if they don't exist.

        Parameters
        ----------
        path : str or Path
            Output YAML file path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = asdict(self)

        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(
                data,
                f,
                default_flow_style=False,
                sort_keys=False,
                allow_unicode=True,
            )

        logger.info(f"Config saved to {path}")

    def to_dict(self) -> dict:
        """Convert to nested dictionary."""
        return asdict(self)

    @classmethod
    def for_smoke_test(cls) -> ModularForgeConfig:
        """
        Create a minimal configuration for quick smoke testing.

        Uses tiny dimensions and few epochs so the entire pipeline
        completes in under 5 minutes on any hardware.

        Returns
        -------
        ModularForgeConfig
            Smoke-test configuration.
        """
        return cls(
            model=ModelConfig(
                d_model=64,
                n_heads=4,
                n_layers=2,
                d_ff=128,
                n_experts=3,
                top_k=2,
                vocab_size=1000,
                max_seq_len=64,
                dropout=0.0,
                expert_dropout=0.0,
            ),
            training=TrainingConfig(
                learning_rate=1e-3,
                batch_size=4,
                epochs_shared=1,
                epochs_expert=1,
                warmup_steps=10,
                weight_decay=0.0,
                gradient_accumulation_steps=1,
                seed=42,
                num_workers=0,
                use_amp=False,
                device="cpu",
                checkpoint_every=0,
                log_every=10,
                eval_every=0,
            ),
            data=DataConfig(
                tokenizer_vocab_size=1000,
                partition_strategy="random",
                overlap_ratio=0.0,
                min_article_length=20,
                data_dir="data_smoke",
                max_articles=200,
            ),
            assembly=AssemblyConfig(
                output_format="pytorch",
                router_init="kaiming",
                calibrate=False,
                calibration_samples=50,
                output_dir="outputs_smoke",
            ),
            evaluation=EvalConfig(
                eval_batch_size=4,
                generate_samples=2,
                generate_max_tokens=50,
                temperature=0.8,
                top_k=10,
                top_p=0.9,
                prompts=["The", "In the"],
            ),
        )

    def __repr__(self) -> str:
        """Pretty-print the configuration."""
        lines = [
            "ModularForgeConfig(",
            f"  Model:    {self.model.total_params_estimate / 1e6:.1f}M params "
            f"({self.model.n_experts} experts × ~{self.model.expert_params / 1e6:.1f}M each)",
            f"  Dims:     d_model={self.model.d_model}, d_ff={self.model.d_ff}, "
            f"n_heads={self.model.n_heads}, n_layers={self.model.n_layers}",
            f"  Training: lr={self.training.learning_rate}, "
            f"batch_size={self.training.batch_size}, "
            f"epochs={self.training.epochs_shared}/{self.training.epochs_expert} "
            f"(shared/expert)",
            f"  Data:     {self.data.partition_strategy} partitioning, "
            f"{self.data.overlap_ratio:.0%} overlap",
            f"  Device:   {self.training.device}",
            ")",
        ]
        return "\n".join(lines)
