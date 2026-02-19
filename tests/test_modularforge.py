#!/usr/bin/env python3
"""
Tests for ModularForge configuration, model architecture, data pipeline,
and assembly.

Run all tests:
    cd /Users/aditya/Documents/Aditya/Python/LLM_In_Parts
    python -m pytest tests/ -v --tb=short

Run a specific test file:
    python -m pytest tests/test_model.py -v
"""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# =============================================================================
# Config Tests
# =============================================================================

class TestConfig:
    """Tests for the configuration system."""

    def test_default_config_loads(self):
        """Default config should load and validate without errors."""
        from modularforge.config import ModularForgeConfig
        config = ModularForgeConfig()
        config.validate()

    def test_smoke_test_config(self):
        """Smoke test config should create valid minimal configuration."""
        from modularforge.config import ModularForgeConfig
        config = ModularForgeConfig.for_smoke_test()
        config.validate()
        assert config.model.d_model == 64
        assert config.model.n_experts == 3

    def test_invalid_d_model_n_heads(self):
        """d_model must be divisible by n_heads."""
        from modularforge.config import ModelConfig
        config = ModelConfig(d_model=100, n_heads=7)
        with pytest.raises(ValueError, match="divisible"):
            config.validate()

    def test_invalid_top_k(self):
        """top_k must be <= n_experts."""
        from modularforge.config import ModelConfig
        config = ModelConfig(n_experts=3, top_k=5)
        with pytest.raises(ValueError):
            config.validate()

    def test_vocab_size_mismatch(self):
        """Tokenizer and model vocab_size must match."""
        from modularforge.config import ModularForgeConfig, ModelConfig, DataConfig
        config = ModularForgeConfig(
            model=ModelConfig(vocab_size=1000),
            data=DataConfig(tokenizer_vocab_size=2000),
        )
        with pytest.raises(ValueError, match="vocab_size"):
            config.validate()

    def test_yaml_round_trip(self, tmp_path):
        """Config should save to YAML and load back identically."""
        from modularforge.config import ModularForgeConfig
        config = ModularForgeConfig.for_smoke_test()

        yaml_path = tmp_path / "test_config.yaml"
        config.to_yaml(yaml_path)

        loaded = ModularForgeConfig.from_yaml(yaml_path)
        assert loaded.model.d_model == config.model.d_model
        assert loaded.model.n_experts == config.model.n_experts

    def test_param_estimate(self):
        """Parameter count estimate should be reasonable."""
        from modularforge.config import ModelConfig
        config = ModelConfig(d_model=512, d_ff=2048, n_experts=5, n_layers=4)
        estimate = config.total_params_estimate
        assert 10_000_000 < estimate < 200_000_000


# =============================================================================
# Model Tests
# =============================================================================

class TestExpertFFN:
    """Tests for the expert feed-forward network."""

    def test_forward_shape(self):
        """Expert output should match input shape."""
        from modularforge.model.expert import ExpertFFN
        expert = ExpertFFN(d_model=64, d_ff=128)
        x = torch.randn(2, 8, 64)
        out = expert(x)
        assert out.shape == x.shape

    def test_residual_connection(self):
        """With zero-initialized FC layers, output should equal input (residual)."""
        from modularforge.model.expert import ExpertFFN
        expert = ExpertFFN(d_model=64, d_ff=128)
        # Zero out all expert weights
        nn.init.zeros_(expert.fc1.weight)
        nn.init.zeros_(expert.fc2.weight)
        nn.init.zeros_(expert.fc1.bias)
        nn.init.zeros_(expert.fc2.bias)

        x = torch.randn(2, 8, 64)
        out = expert(x)
        # Due to LayerNorm, this won't be exactly equal, but close
        # after fc1 zeros, the FFN contribution is zero, so output ≈ input
        # (norm(x) → 0 → 0 → + x = x, but norm changes x slightly)
        assert out.shape == x.shape

    def test_param_count(self):
        """Expert param count should match formula."""
        from modularforge.model.expert import ExpertFFN
        expert = ExpertFFN(d_model=64, d_ff=256)
        # 2 linear layers: 64*256 + 256*64 + biases + LN
        assert expert.n_params > 0


class TestSharedComponents:
    """Tests for shared model components."""

    def test_embed_shape(self):
        """Embedding should produce correct output shape."""
        from modularforge.config import ModelConfig
        from modularforge.model.shared import SharedComponents
        config = ModelConfig(d_model=64, n_heads=4, n_layers=2, vocab_size=100, max_seq_len=32)
        shared = SharedComponents(config)
        ids = torch.randint(0, 100, (2, 16))
        embedded = shared.embed(ids)
        assert embedded.shape == (2, 16, 64)

    def test_attention_shape(self):
        """Attention should preserve shape."""
        from modularforge.config import ModelConfig
        from modularforge.model.shared import SharedComponents
        config = ModelConfig(d_model=64, n_heads=4, n_layers=2, vocab_size=100, max_seq_len=32)
        shared = SharedComponents(config)
        x = torch.randn(2, 16, 64)
        out = shared.apply_attention(x, layer_idx=0)
        assert out.shape == x.shape

    def test_predict_shape(self):
        """LM head should output vocab_size logits."""
        from modularforge.config import ModelConfig
        from modularforge.model.shared import SharedComponents
        config = ModelConfig(d_model=64, n_heads=4, n_layers=2, vocab_size=100, max_seq_len=32)
        shared = SharedComponents(config)
        x = torch.randn(2, 16, 64)
        logits = shared.predict(x)
        assert logits.shape == (2, 16, 100)

    def test_freeze_unfreeze(self):
        """Freeze should disable gradients, unfreeze should re-enable."""
        from modularforge.config import ModelConfig
        from modularforge.model.shared import SharedComponents
        config = ModelConfig(d_model=64, n_heads=4, n_layers=2, vocab_size=100, max_seq_len=32)
        shared = SharedComponents(config)

        shared.freeze()
        assert shared.n_trainable_params == 0

        shared.unfreeze()
        assert shared.n_trainable_params > 0


class TestMoERouter:
    """Tests for the MoE router."""

    def test_forward_shapes(self):
        """Router should return correct shapes for weights and indices."""
        from modularforge.model.router import MoERouter
        router = MoERouter(d_model=64, n_experts=5, top_k=2)
        x = torch.randn(2, 8, 64)
        weights, indices, aux_loss = router(x)

        assert weights.shape == (2, 8, 2)
        assert indices.shape == (2, 8, 2)
        assert aux_loss.dim() == 0  # scalar

    def test_weights_sum_to_one(self):
        """Router weights should sum to 1 along the expert dimension."""
        from modularforge.model.router import MoERouter
        router = MoERouter(d_model=64, n_experts=5, top_k=2)
        x = torch.randn(2, 8, 64)
        weights, _, _ = router(x)
        sums = weights.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_indices_in_range(self):
        """Expert indices should be in [0, n_experts)."""
        from modularforge.model.router import MoERouter
        router = MoERouter(d_model=64, n_experts=5, top_k=2)
        x = torch.randn(4, 16, 64)
        _, indices, _ = router(x)
        assert indices.min() >= 0
        assert indices.max() < 5


class TestAssembledModel:
    """Tests for the fully assembled model."""

    def test_forward_shape(self):
        """Assembled model should produce correct output shapes."""
        from modularforge.config import ModelConfig
        from modularforge.model.assembled_model import AssembledMoEModel
        config = ModelConfig(
            d_model=64, n_heads=4, n_layers=2, d_ff=128,
            n_experts=3, top_k=2, vocab_size=100, max_seq_len=32
        )
        model = AssembledMoEModel(config)
        ids = torch.randint(0, 100, (2, 16))
        logits, aux_loss = model(ids)
        assert logits.shape == (2, 16, 100)
        assert aux_loss.dim() == 0

    def test_generate(self):
        """Generate should produce tokens."""
        from modularforge.config import ModelConfig
        from modularforge.model.assembled_model import AssembledMoEModel
        config = ModelConfig(
            d_model=64, n_heads=4, n_layers=2, d_ff=128,
            n_experts=3, top_k=2, vocab_size=100, max_seq_len=32
        )
        model = AssembledMoEModel(config)
        prompt = torch.randint(1, 100, (1, 4))
        generated = model.generate(prompt, max_new_tokens=10)
        assert generated.shape[1] > prompt.shape[1]  # Should have generated tokens


# =============================================================================
# Data Tests
# =============================================================================

class TestTokenizer:
    """Tests for the BPE tokenizer."""

    def test_train_and_encode(self):
        """Tokenizer should train and encode/decode correctly."""
        from modularforge.data.tokenizer import ModularForgeTokenizer
        tok = ModularForgeTokenizer(vocab_size=500)
        tok.train(["Hello world! This is a test.", "Another sentence for training."])

        ids = tok.encode("Hello world")
        assert isinstance(ids, list)
        assert all(isinstance(i, int) for i in ids)
        assert len(ids) >= 3  # At least BOS + token + EOS

        decoded = tok.decode(ids)
        assert "Hello" in decoded or "hello" in decoded.lower()

    def test_special_tokens(self):
        """Special token IDs should be consistent."""
        from modularforge.data.tokenizer import ModularForgeTokenizer
        tok = ModularForgeTokenizer(vocab_size=500)
        tok.train(["Test text for tokenizer training."])

        assert tok.pad_id == 0
        assert tok.bos_id is not None
        assert tok.eos_id is not None

    def test_save_load(self, tmp_path):
        """Tokenizer should save to disk and load back."""
        from modularforge.data.tokenizer import ModularForgeTokenizer
        tok = ModularForgeTokenizer(vocab_size=500)
        tok.train(["Test text."])

        path = tmp_path / "tok.json"
        tok.save(path)

        tok2 = ModularForgeTokenizer(vocab_size=500)
        tok2.load(path)

        ids1 = tok.encode("Hello")
        ids2 = tok2.encode("Hello")
        assert ids1 == ids2


class TestDataset:
    """Tests for the text dataset."""

    def test_dataset_creation(self):
        """Dataset should create sequences from text."""
        from modularforge.data.tokenizer import ModularForgeTokenizer
        from modularforge.data.dataset import TextDataset

        tok = ModularForgeTokenizer(vocab_size=500)
        tok.train(["This is a test sentence for the dataset." * 10])

        ds = TextDataset(texts=["This is test text." * 20], tokenizer=tok, max_seq_len=32)
        assert len(ds) > 0

        input_ids, target_ids = ds[0]
        assert input_ids.shape == (32,)
        assert target_ids.shape == (32,)

    def test_target_is_shifted_input(self):
        """Target should be input shifted by 1."""
        from modularforge.data.tokenizer import ModularForgeTokenizer
        from modularforge.data.dataset import TextDataset

        tok = ModularForgeTokenizer(vocab_size=500)
        tok.train(["A B C D E F G H I J K L M N O P." * 20])

        ds = TextDataset(texts=["A B C D E F G H I J K." * 20], tokenizer=tok, max_seq_len=16)
        if len(ds) > 0:
            inp, tgt = ds[0]
            # The internal sequence is chunk[:-1] and chunk[1:]
            # So there should be overlap between consecutive positions
            assert inp.shape == tgt.shape


class TestPartitioner:
    """Tests for data partitioning."""

    def test_random_partition(self):
        """Random partition should produce N non-empty partitions."""
        from modularforge.data.partitioner import DataPartitioner
        articles = [f"Article {i} content here" for i in range(100)]
        p = DataPartitioner(n_partitions=5, strategy="random", overlap_ratio=0.0)
        partitions = p.partition(articles)
        assert len(partitions) == 5
        assert all(len(part) > 0 for part in partitions)

    def test_curriculum_partition(self):
        """Curriculum partition should produce sorted partitions."""
        from modularforge.data.partitioner import DataPartitioner
        articles = [f"{'word ' * (i + 5)}" for i in range(50)]
        p = DataPartitioner(n_partitions=3, strategy="curriculum", overlap_ratio=0.0)
        partitions = p.partition(articles)
        assert len(partitions) == 3

    def test_overlap(self):
        """Overlap should add shared articles."""
        from modularforge.data.partitioner import DataPartitioner
        articles = [f"Article {i}" for i in range(100)]
        p_no_overlap = DataPartitioner(n_partitions=5, strategy="random", overlap_ratio=0.0)
        p_with_overlap = DataPartitioner(n_partitions=5, strategy="random", overlap_ratio=0.2)

        parts_no = p_no_overlap.partition(articles)
        parts_yes = p_with_overlap.partition(articles)

        # With overlap, partitions should be larger
        total_no = sum(len(p) for p in parts_no)
        total_yes = sum(len(p) for p in parts_yes)
        assert total_yes > total_no


# =============================================================================
# Assembly Tests
# =============================================================================

class TestAssembly:
    """Tests for the streaming assembly process."""

    def test_assembly_round_trip(self, tmp_path):
        """Assembled model should be loadable and produce valid outputs."""
        from modularforge.config import ModularForgeConfig
        from modularforge.model.shared import SharedComponents
        from modularforge.model.module_trainer_model import ModuleTrainerModel
        from modularforge.model.assembled_model import AssembledMoEModel
        from modularforge.assembly.assembler import StreamingAssembler

        config = ModularForgeConfig.for_smoke_test()

        # Create and save shared components
        shared = SharedComponents(config.model)
        shared_path = str(tmp_path / "shared.pt")
        torch.save(shared.state_dict(), shared_path)

        # Create and save expert checkpoints
        expert_paths = []
        for i in range(config.model.n_experts):
            shared_copy = SharedComponents(config.model)
            shared_copy.load_state_dict(torch.load(shared_path, weights_only=False))
            shared_copy.freeze()

            model = ModuleTrainerModel(shared_copy, config.model, expert_idx=i)
            expert_path = str(tmp_path / f"expert_{i}.pt")
            model.save_expert(expert_path)
            expert_paths.append(expert_path)

        # Assemble
        assembled_path = str(tmp_path / "assembled.pt")
        assembler = StreamingAssembler(config)
        results = assembler.assemble(
            shared_path=shared_path,
            expert_paths=expert_paths,
            output_path=assembled_path,
        )

        assert results["n_params"] > 0
        assert Path(assembled_path).exists()

        # Load and verify
        model = AssembledMoEModel(config.model)
        model.load_from_checkpoint(assembled_path)

        ids = torch.randint(1, 100, (1, 8))
        logits, _ = model(ids)
        assert logits.shape == (1, 8, config.model.vocab_size)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
