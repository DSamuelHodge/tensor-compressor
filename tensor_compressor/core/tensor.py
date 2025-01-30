"""
Implementation of Matrix Product Operator (MPO) tensor decomposition.
"""

from typing import Tuple, Optional, List
import torch
import numpy as np
from dataclasses import dataclass
from ..utils.device import get_device
from ..utils.tensor import find_optimal_factors

@dataclass
class MPOTensor:
    """Matrix Product Operator representation of a tensor"""
    
    # Core tensors
    left: torch.Tensor
    core: torch.Tensor
    right: torch.Tensor
    
    # Original tensor metadata
    orig_shape: Tuple[int, ...]
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    def __post_init__(self):
        """Move tensors to specified device"""
        self.device = get_device(self.device)
        self.left = self.left.to(self.device)
        self.core = self.core.to(self.device)
        self.right = self.right.to(self.device)
    
    @classmethod
    def from_tensor(cls, tensor: torch.Tensor, bond_dim: int) -> 'MPOTensor':
        """Create MPO from a tensor using SVD decomposition"""
        device = tensor.device
        orig_shape = tensor.shape
        rows, cols = orig_shape
        
        # Find optimal reshape factors
        row_factors = find_optimal_factors(rows)
        col_factors = find_optimal_factors(cols)
        
        # Reshape and perform SVD
        reshaped = tensor.reshape(row_factors + col_factors)
        flattened = reshaped.reshape(-1, reshaped.shape[-1])
        
        U, S, Vh = torch.linalg.svd(flattened, full_matrices=False)
        
        # Truncate based on bond dimension and energy preservation
        total_energy = torch.sum(S)
        cum_energy = torch.cumsum(S, dim=0) / total_energy
        dynamic_bond_dim = torch.where(cum_energy > 0.99)[0][0].item()
        bond_dim = min(bond_dim, dynamic_bond_dim)
        
        U = U[:, :bond_dim]
        S = S[:bond_dim]
        Vh = Vh[:bond_dim, :]
        
        # Create MPO structure
        core_tensor = torch.diag(S)
        left_tensor = U.reshape(*row_factors, -1)
        right_tensor = Vh.reshape(-1, *col_factors)
        
        return cls(left_tensor, core_tensor, right_tensor, orig_shape, str(device))
    
    def materialize(self) -> torch.Tensor:
        """Reconstruct the original tensor from MPO format"""
        with torch.no_grad():
            temp = torch.matmul(
                self.left.reshape(-1, self.left.size(-1)),
                self.core
            )
            result = torch.matmul(
                temp,
                self.right.reshape(self.right.size(0), -1)
            )
            return result.reshape(self.orig_shape)
    
    @property
    def compression_ratio(self) -> float:
        """Calculate the compression ratio"""
        original_params = np.prod(self.orig_shape)
        compressed_params = sum(
            np.prod(t.shape) for t in [self.left, self.core, self.right]
        )
        return 1 - (compressed_params / original_params)
    
    def to(self, device: str) -> 'MPOTensor':
        """Move MPO to specified device"""
        return MPOTensor(
            left=self.left.to(device),
            core=self.core.to(device),
            right=self.right.to(device),
            orig_shape=self.orig_shape,
            device=device
        )
        
    def __repr__(self) -> str:
        return (f"MPOTensor(shape={self.orig_shape}, "
                f"compression_ratio={self.compression_ratio:.2%}, "
                f"device={self.device})")
