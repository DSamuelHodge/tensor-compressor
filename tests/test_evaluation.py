"""
Tests for evaluation module.
"""

import pytest
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from tensor_compressor.evaluation.metrics import (
    CompressionMetrics,
    PerformanceMetrics,
    AccuracyMetrics,
    calculate_compression_ratio,
    calculate_reconstruction_error,
    calculate_memory_savings,
    measure_inference_latency
)
from tensor_compressor.evaluation.tasks.mmlu import MMLUEvaluator

def test_compression_metrics():
    """Test compression metrics calculations"""
    original_size = 1000
    compressed_size = 200
    
    ratio = calculate_compression_ratio(original_size, compressed_size)
    assert 0 <= ratio <= 1
    assert np.isclose(ratio, 0.8)
    
    savings = calculate_memory_savings(original_size, compressed_size)
    assert savings == (original_size - compressed_size) * 4  # float32
    
    metrics = CompressionMetrics(
        compression_ratio=ratio,
        memory_saved=savings,
        parameter_reduction=ratio,
        reconstruction_error=0.1
    )
    assert metrics.compression_ratio == ratio
    assert metrics.memory_saved == savings

def test_reconstruction_error():
    """Test reconstruction error calculation"""
    original = torch.randn(10, 10)
    # Add small perturbation
    noise = torch.randn(10, 10) * 0.1
    reconstructed = original + noise
    
    error = calculate_reconstruction_error(original, reconstructed)
    assert 0 <= error <= 1
    
    # Test perfect reconstruction
    error = calculate_reconstruction_error(original, original)
    assert np.isclose(error, 0.0)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_performance_metrics():
    """Test performance metrics on GPU"""
    model = torch.nn.Linear(100, 100).cuda()
    input_tensor = torch.randn(1, 32, 100).cuda()
    
    metrics = measure_inference_latency(
        model,
        input_tensor,
        num_runs=5,
        warmup_runs=2
    )
    
    assert isinstance(metrics, PerformanceMetrics)
    assert metrics.latency > 0
    assert metrics.throughput > 0
    assert metrics.memory_usage > 0

def test_accuracy_metrics():
    """Test accuracy metrics"""
    metrics = AccuracyMetrics(
        perplexity=10.5,
        accuracy=0.85,
        f1_score=0.83,
        exact_match=0.80,
        task_specific={"subject1": 0.9, "subject2": 0.8}
    )
    
    assert metrics.perplexity == 10.5
    assert metrics.accuracy == 0.85
    assert metrics.f1_score == 0.83
    assert metrics.exact_match == 0.80
    assert len(metrics.task_specific) == 2

@pytest.fixture
def tiny_model():
    """Create a tiny model for testing"""
    config = {
        "vocab_size": 1000,
        "hidden_size": 128,
        "num_attention_heads": 2,
        "num_hidden_layers": 2
    }
    return AutoModelForCausalLM.from_pretrained("gpt2", config=config)

@pytest.fixture
def tokenizer():
    """Get tokenizer for testing"""
    return AutoTokenizer.from_pretrained("gpt2")

def test_mmlu_evaluator(tiny_model, tokenizer):
    """Test MMLU evaluator"""
    evaluator = MMLUEvaluator(
        model=tiny_model,
        tokenizer=tokenizer,
        device="cpu",
        subjects=["high_school_mathematics"]  # Test with one subject for speed
    )
    
    # Test dataset loading
    datasets = evaluator.load_dataset()
    assert len(datasets) == 1
    assert "high_school_mathematics" in datasets
    assert len(datasets["high_school_mathematics"]) > 0
    
    # Test prompt formatting
    example = datasets["high_school_mathematics"][0]
    prompt = evaluator.format_prompt(example)
    assert isinstance(prompt, str)
    assert "A." in prompt
    assert "B." in prompt
    assert "C." in prompt
    assert "D." in prompt
    assert prompt.endswith("Answer:")
    
    # Test performance measurement
    metrics = evaluator.measure_performance(num_samples=2)
    assert isinstance(metrics, PerformanceMetrics)
    assert metrics.latency > 0
    assert metrics.throughput > 0

@pytest.mark.slow
def test_full_evaluation(tiny_model, tokenizer):
    """Test full evaluation pipeline"""
    evaluator = MMLUEvaluator(
        model=tiny_model,
        tokenizer=tokenizer,
        device="cpu",
        subjects=["high_school_mathematics"]
    )
    
    metrics = evaluator.evaluate()
    assert isinstance(metrics, AccuracyMetrics)
    assert isinstance(metrics.accuracy, float)
    assert 0 <= metrics.accuracy <= 1
    assert "high_school_mathematics" in metrics.task_specific