"""
Device management utilities for tensor operations.
"""

import torch
from typing import Union, Optional
from ..core.tensor import MPOTensor

def get_device(device: Optional[str] = None) -> torch.device:
    """
    Get the appropriate device for tensor operations.
    
    Args:
        device: Optional device specification. If None, uses CUDA if available.
        
    Returns:
        torch.device: Device to use for computations
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if isinstance(device, torch.device):
        return device
    
    if device.startswith("cuda") and not torch.cuda.is_available():
        print(f"Warning: CUDA requested but not available. Falling back to CPU.")
        return torch.device("cpu")
    
    return torch.device(device)

def move_to_device(tensor: Union[torch.Tensor, MPOTensor], device: Optional[str] = None) -> Union[torch.Tensor, MPOTensor]:
    """
    Move a tensor or MPOTensor to the specified device.
    
    Args:
        tensor: Tensor to move
        device: Target device
        
    Returns:
        Tensor on the target device
    """
    device = get_device(device)
    
    if isinstance(tensor, torch.Tensor):
        return tensor.to(device)
    elif isinstance(tensor, MPOTensor):
        return tensor.to(device)
    else:
        raise TypeError(f"Unsupported tensor type: {type(tensor)}")