"""
Core evaluation metrics for model compression.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, List, Union
import torch
import numpy as np

@dataclass
class CompressionMetrics:
    """Metrics for compression quality"""
    compression_ratio: float
    memory_saved: int  # in bytes
    parameter_reduction: float
    reconstruction_error: float

@dataclass
class PerformanceMetrics:
    """Metrics for model performance"""
    latency: float  # in seconds
    throughput: float  # tokens/second
    memory_usage: int  # peak memory in bytes
    device_utilization: Optional[float] = None  # GPU utilization if applicable

@dataclass
class AccuracyMetrics:
    """Task-specific accuracy metrics"""
    perplexity: Optional[float] = None
    accuracy: Optional[float] = None
    f1_score: Optional[float] = None
    exact_match: Optional[float] = None
    task_specific: Dict[str, float] = field(default_factory=dict)

@dataclass
class EvaluationResult:
    """Complete evaluation results"""
    compression: CompressionMetrics
    performance: PerformanceMetrics
    accuracy: AccuracyMetrics
    metadata: Dict[str, Union[str, float]] = field(default_factory=dict)

def calculate_compression_ratio(original_size: int, compressed_size: int) -> float:
    """Calculate compression ratio"""
    return 1.0 - (compressed_size / original_size)

def calculate_reconstruction_error(original: torch.Tensor, reconstructed: torch.Tensor) -> float:
    """Calculate relative reconstruction error"""
    with torch.no_grad():
        error = torch.norm(original - reconstructed) / torch.norm(original)
        return error.item()

def calculate_memory_savings(
    original_params: int,
    compressed_params: int,
    dtype: torch.dtype = torch.float32
) -> int:
    """Calculate memory savings in bytes"""
    bytes_per_param = {
        torch.float32: 4,
        torch.float16: 2,
        torch.bfloat16: 2,
        torch.int8: 1
    }.get(dtype, 4)
    
    return (original_params - compressed_params) * bytes_per_param

def calculate_perplexity(model_outputs, labels: torch.Tensor) -> float:
    """Calculate perplexity from model outputs"""
    loss = model_outputs.loss
    return torch.exp(loss).item()

def calculate_accuracy(predictions: torch.Tensor, labels: torch.Tensor) -> float:
    """Calculate accuracy for classification tasks"""
    correct = (predictions == labels).sum().item()
    total = labels.numel()
    return correct / total

def calculate_f1_score(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    average: str = 'macro'
) -> float:
    """Calculate F1 score"""
    from sklearn.metrics import f1_score
    return f1_score(
        labels.cpu().numpy(),
        predictions.cpu().numpy(),
        average=average
    )

def measure_inference_latency(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    num_runs: int = 10,
    warmup_runs: int = 2
) -> PerformanceMetrics:
    """Measure model inference latency and throughput"""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
    
    # Warmup runs
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(input_tensor)
    
    # Measurement runs
    latencies = []
    with torch.no_grad():
        for _ in range(num_runs):
            if torch.cuda.is_available():
                start_event.record()
                _ = model(input_tensor)
                end_event.record()
                torch.cuda.synchronize()
                latencies.append(start_event.elapsed_time(end_event) / 1000)  # convert to seconds
            else:
                import time
                start_time = time.time()
                _ = model(input_tensor)
                latencies.append(time.time() - start_time)
    
    avg_latency = np.mean(latencies)
    throughput = input_tensor.shape[1] / avg_latency  # tokens/second
    
    memory_usage = (
        torch.cuda.max_memory_allocated()
        if torch.cuda.is_available()
        else 0
    )
    
    return PerformanceMetrics(
        latency=avg_latency,
        throughput=throughput,
        memory_usage=memory_usage,
        device_utilization=None  # Could be implemented with pynvml
    )