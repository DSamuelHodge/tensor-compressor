"""
Utilities for tensor manipulation and analysis.
"""

import numpy as np
import torch
from typing import Tuple, List, Dict, Optional

def find_optimal_factors(n: int) -> Tuple[int, int]:
    """
    Find factors closest to square root for optimal tensor decomposition.
    
    Args:
        n: Number to factorize
        
    Returns:
        Tuple of two factors that multiply to n
    """
    sqrt_n = int(np.sqrt(n))
    for i in range(sqrt_n, 0, -1):
        if n % i == 0:
            return i, n // i
    return 1, n

def calculate_compression_stats(original_size: int, compressed_size: int) -> Dict[str, float]:
    """
    Calculate compression statistics.
    
    Args:
        original_size: Size of original tensor in bytes
        compressed_size: Size of compressed tensor in bytes
        
    Returns:
        Dictionary containing compression metrics
    """
    ratio = 1 - (compressed_size / original_size)
    savings = original_size - compressed_size
    return {
        "compression_ratio": ratio,
        "space_saved": savings,
        "size_reduction_percent": ratio * 100
    }

def estimate_memory_usage(shape: Tuple[int, ...], dtype: torch.dtype = torch.float32) -> int:
    """
    Estimate memory usage in bytes for a tensor.
    
    Args:
        shape: Tensor shape
        dtype: Tensor data type
        
    Returns:
        Estimated memory usage in bytes
    """
    elem_size = {
        torch.float32: 4,
        torch.float16: 2,
        torch.bfloat16: 2,
        torch.int32: 4,
        torch.int64: 8
    }.get(dtype, 4)
    
    return int(np.prod(shape)) * elem_size

def is_power_of_two(n: int) -> bool:
    """Check if a number is a power of 2."""
    return (n & (n - 1) == 0) and n != 0

def find_nearest_power_of_two(n: int) -> int:
    """Find the nearest power of 2."""
    return 2 ** round(np.log2(n))

def optimize_tensor_shape(shape: Tuple[int, ...]) -> Tuple[int, ...]:
    """
    Optimize tensor shape for efficient computation.
    
    Args:
        shape: Original shape
        
    Returns:
        Optimized shape with dimensions rounded to powers of 2
    """
    return tuple(find_nearest_power_of_two(dim) for dim in shape)

def analyze_tensor_sparsity(tensor: torch.Tensor, threshold: float = 1e-6) -> Dict[str, float]:
    """
    Analyze sparsity patterns in a tensor.
    
    Args:
        tensor: Input tensor
        threshold: Value threshold for considering an element zero
        
    Returns:
        Dictionary containing sparsity metrics
    """
    total_elements = tensor.numel()
    zeros = torch.sum(torch.abs(tensor) < threshold).item()
    sparsity = zeros / total_elements
    
    return {
        "sparsity_ratio": sparsity,
        "non_zero_elements": total_elements - zeros,
        "zero_elements": zeros
    }