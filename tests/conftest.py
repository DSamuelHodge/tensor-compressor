"""
Pytest configuration and fixtures.
"""

import pytest
import torch
import torch.nn as nn
from transformers import LlamaConfig, LlamaModel
from tensor_compressor import CompressionConfig
from tensor_compressor.core.tensor import MPOTensor

@pytest.fixture
def device():
    """Get available device"""
    return "cuda" if torch.cuda.is_available() else "cpu"

@pytest.fixture
def sample_tensor(device):
    """Create a sample tensor for testing"""
    return torch.randn(64, 64, device=device)

@pytest.fixture
def compression_config():
    """Create a test compression configuration"""
    return CompressionConfig(
        bond_dim=32,
        target_compression=0.85,
        accuracy_threshold=0.95,
        device="cpu"  # Use CPU for testing
    )

@pytest.fixture
def tiny_llama_config():
    """Create a tiny LLaMA config for testing"""
    return LlamaConfig(
        vocab_size=100,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        max_position_embeddings=128
    )

@pytest.fixture
def tiny_llama_model(tiny_llama_config):
    """Create a tiny LLaMA model for testing"""
    return LlamaModel(tiny_llama_config)

@pytest.fixture
def mock_linear_layer():
    """Create a mock linear layer for testing"""
    layer = nn.Linear(64, 64)
    # Initialize with a known pattern for testing
    nn.init.eye_(layer.weight)
    nn.init.zeros_(layer.bias)
    return layer