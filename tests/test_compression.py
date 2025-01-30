"""
Tests for core compression functionality.
"""

import pytest
import torch
import numpy as np
from tensor_compressor.core.tensor import MPOTensor
from tensor_compressor.models.llama import LlamaAdapter

def test_mpo_tensor_creation(sample_tensor):
    """Test MPO tensor creation and reconstruction"""
    bond_dim = 16
    mpo = MPOTensor.from_tensor(sample_tensor, bond_dim)
    
    # Check tensor shapes
    assert len(mpo.left.shape) == 3
    assert len(mpo.core.shape) == 2
    assert len(mpo.right.shape) == 3
    
    # Check bond dimension
    assert mpo.core.shape[0] <= bond_dim
    assert mpo.core.shape[1] <= bond_dim
    
    # Check reconstruction
    reconstructed = mpo.materialize()
    assert reconstructed.shape == sample_tensor.shape
    
    # Check compression ratio
    assert 0 < mpo.compression_ratio < 1

def test_compression_accuracy(sample_tensor):
    """Test compression accuracy with different bond dimensions"""
    errors = []
    bond_dims = [8, 16, 32]
    
    for bond_dim in bond_dims:
        mpo = MPOTensor.from_tensor(sample_tensor, bond_dim)
        reconstructed = mpo.materialize()
        error = torch.norm(sample_tensor - reconstructed).item()
        errors.append(error)
    
    # Error should decrease with larger bond dimensions
    assert all(errors[i] > errors[i+1] for i in range(len(errors)-1))

def test_llama_adapter(tiny_llama_model, compression_config):
    """Test LLaMA adapter functionality"""
    adapter = LlamaAdapter(tiny_llama_model, compression_config)
    
    # Check layer detection
    attn_layers = adapter.get_attention_layers()
    mlp_layers = adapter.get_mlp_layers()
    
    assert len(attn_layers) > 0
    assert len(mlp_layers) > 0
    
    # Test compression
    compressed_model = adapter.compress_model()
    
    # Model should still be a valid LLaMA model
    assert hasattr(compressed_model, "layers")
    
    # Check compression stats
    stats = adapter.get_compression_summary()
    assert stats["overall_compression_ratio"] > 0

def test_device_handling(compression_config):
    """Test device management in compression"""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
        
    # Create tensor on CPU
    tensor = torch.randn(64, 64)
    
    # Move to GPU
    compression_config.device = "cuda"
    mpo = MPOTensor.from_tensor(tensor, compression_config.bond_dim)
    
    assert mpo.left.device.type == "cuda"
    assert mpo.core.device.type == "cuda"
    assert mpo.right.device.type == "cuda"
    
    # Move back to CPU
    mpo = mpo.to("cpu")
    assert mpo.left.device.type == "cpu"
    assert mpo.core.device.type == "cpu"
    assert mpo.right.device.type == "cpu"

def test_compression_stability(mock_linear_layer, compression_config):
    """Test compression stability with different random seeds"""
    weight = mock_linear_layer.weight.data
    results = []
    
    for seed in range(5):
        torch.manual_seed(seed)
        mpo = MPOTensor.from_tensor(weight, compression_config.bond_dim)
        reconstructed = mpo.materialize()
        results.append(torch.norm(weight - reconstructed).item())
    
    # Results should be similar across seeds
    assert max(results) - min(results) < 1e-5

@pytest.mark.parametrize("shape", [
    (32, 32),
    (64, 128),
    (128, 64),
    (256, 256)
])
def test_different_shapes(shape, compression_config):
    """Test compression with different matrix shapes"""
    tensor = torch.randn(*shape)
    mpo = MPOTensor.from_tensor(tensor, compression_config.bond_dim)
    reconstructed = mpo.materialize()
    
    assert reconstructed.shape == shape
    # Error should be reasonable
    error = torch.norm(tensor - reconstructed) / torch.norm(tensor)
    assert error < 0.1  # 10% relative error threshold