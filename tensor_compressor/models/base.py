"""
Base classes for model-specific compression adapters.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any
import torch
import torch.nn as nn
from dataclasses import dataclass
from ..config import CompressionConfig
from ..core.tensor import MPOTensor

@dataclass
class LayerCompression:
    """Metadata about layer compression"""
    original_size: int
    compressed_size: int
    compression_ratio: float
    accuracy_impact: Optional[float] = None

class ModelAdapter(ABC):
    """Base class for model-specific compression adapters"""
    
    def __init__(self, model: nn.Module, config: CompressionConfig):
        self.model = model
        self.config = config
        self.compressed_layers: Dict[str, Union[MPOTensor, List[MPOTensor]]] = {}
        self.compression_stats: Dict[str, LayerCompression] = {}
        self.device = config.device_type
        
    @abstractmethod
    def get_attention_layers(self) -> Dict[str, nn.Module]:
        """Return attention layers that can be compressed"""
        pass
    
    @abstractmethod
    def get_mlp_layers(self) -> Dict[str, nn.Module]:
        """Return MLP layers that can be compressed"""
        pass
    
    @abstractmethod
    def replace_layer(self, name: str, compressed_layer: nn.Module) -> None:
        """Replace original layer with compressed version"""
        pass
        
    def compress_layer(self, layer: nn.Module, name: str) -> None:
        """Compress a single layer"""
        if not isinstance(layer, nn.Linear):
            return
            
        weight = layer.weight.data
        original_size = weight.numel() * weight.element_size()
        
        # Handle attention heads separately
        if "attn" in name:
            head_size = getattr(layer, "head_dim", weight.size(0))
            reshaped = weight.view(-1, head_size, weight.size(-1))
            compressed_heads = []
            
            for head_idx in range(reshaped.size(0)):
                head = reshaped[head_idx]
                mpo = MPOTensor.from_tensor(head, self.config.bond_dim)
                compressed_heads.append(mpo)
                
            self.compressed_layers[name] = compressed_heads
            compressed_size = sum(head.numel() * head.element_size() 
                                for head in compressed_heads)
        else:
            mpo = MPOTensor.from_tensor(weight, self.config.bond_dim)
            self.compressed_layers[name] = mpo
            compressed_size = mpo.numel() * mpo.element_size()
            
        compression_ratio = 1 - (compressed_size / original_size)
        self.compression_stats[name] = LayerCompression(
            original_size=original_size,
            compressed_size=compressed_size,
            compression_ratio=compression_ratio
        )
    
    def compress_model(self) -> nn.Module:
        """Compress the entire model"""
        # Compress attention layers
        for name, layer in self.get_attention_layers().items():
            self.compress_layer(layer, name)
            
        # Compress MLP layers
        for name, layer in self.get_mlp_layers().items():
            self.compress_layer(layer, name)
            
        # Replace original layers with compressed versions
        for name, compressed in self.compressed_layers.items():
            self.replace_layer(name, compressed)
            
        return self.model
    
    def get_compression_summary(self) -> Dict[str, Any]:
        """Get summary of compression results"""
        total_original = sum(stat.original_size for stat in self.compression_stats.values())
        total_compressed = sum(stat.compressed_size for stat in self.compression_stats.values())
        overall_ratio = 1 - (total_compressed / total_original)
        
        return {
            "total_original_size": total_original,
            "total_compressed_size": total_compressed,
            "overall_compression_ratio": overall_ratio,
            "layer_stats": self.compression_stats
        }