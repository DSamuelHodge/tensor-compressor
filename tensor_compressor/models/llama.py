"""
LLaMA-specific model compression adapter.
"""

from typing import Dict, List, Optional, Union, cast
import torch
import torch.nn as nn
from transformers import LlamaModel, LlamaForCausalLM
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaMLP
from .base import ModelAdapter, LayerCompression
from ..config import CompressionConfig
from ..core.tensor import MPOTensor

class LlamaAdapter(ModelAdapter):
    """Adapter for compressing LLaMA models"""
    
    def __init__(self, model: Union[LlamaModel, LlamaForCausalLM], config: CompressionConfig):
        super().__init__(model, config)
        self._validate_model()
        self.base_model = self.model.model if isinstance(self.model, LlamaForCausalLM) else self.model
        
    def _validate_model(self) -> None:
        """Validate model type and architecture"""
        if not isinstance(self.model, (LlamaModel, LlamaForCausalLM)):
            raise ValueError(f"Expected LlamaModel or LlamaForCausalLM, got {type(self.model)}")
            
        # Validate rotary embeddings configuration
        for layer in self.base_model.layers:
            if not hasattr(layer.self_attn, "rotary_emb"):
                raise ValueError("LLaMA attention layers must use rotary embeddings")
    
    def get_attention_layers(self) -> Dict[str, nn.Module]:
        """Get all attention layers in the model"""
        layers = {}
        for idx, block in enumerate(self.base_model.layers):
            attn = cast(LlamaAttention, block.self_attn)
            # Get query, key, value projections
            for name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                layer = getattr(attn, name)
                layers[f"layer_{idx}.self_attn.{name}"] = layer
        return layers
    
    def get_mlp_layers(self) -> Dict[str, nn.Module]:
        """Get all MLP layers in the model"""
        layers = {}
        for idx, block in enumerate(self.base_model.layers):
            mlp = cast(LlamaMLP, block.mlp)
            for name in ["gate_proj", "up_proj", "down_proj"]:
                layer = getattr(mlp, name)
                layers[f"layer_{idx}.mlp.{name}"] = layer
        return layers
    
    def _compress_attention(self, layer: LlamaAttention, name: str) -> None:
        """Compress attention layer with special handling for rotary embeddings"""
        head_dim = layer.head_dim
        num_heads = layer.num_heads
        
        # Handle Q, K separately due to rotary embeddings
        for proj in ["q_proj", "k_proj"]:
            weight = getattr(layer, proj).weight
            # Reshape considering head dimension for proper rotary embedding application
            reshaped = weight.view(num_heads, head_dim, -1)
            compressed_heads = []
            
            for head_idx in range(num_heads):
                head = reshaped[head_idx]
                mpo = MPOTensor.from_tensor(head, self.config.bond_dim)
                compressed_heads.append(mpo)
                
            self.compressed_layers[f"{name}.{proj}"] = compressed_heads
            
        # Handle V and O projections normally
        for proj in ["v_proj", "o_proj"]:
            weight = getattr(layer, proj).weight
            mpo = MPOTensor.from_tensor(weight, self.config.bond_dim)
            self.compressed_layers[f"{name}.{proj}"] = mpo
    
    def replace_layer(self, name: str, compressed_layer: Union[MPOTensor, List[MPOTensor]]) -> None:
        """Replace a layer with its compressed version"""
        parts = name.split('.')
        module = self.base_model
        
        # Navigate to parent module
        for part in parts[:-1]:
            if part.startswith("layer_"):
                idx = int(part.split("_")[1])
                module = module.layers[idx]
            else:
                module = getattr(module, part)
        
        # Get the target layer name
        layer_name = parts[-1]
        original_layer = getattr(module, layer_name)
        
        if isinstance(compressed_layer, list):  # Multi-head compression
            # Reconstruct weight matrix from compressed heads
            reconstructed = []
            for head in compressed_layer:
                reconstructed.append(head.materialize())
            weight = torch.stack(reconstructed).reshape(original_layer.weight.shape)
        else:
            weight = compressed_layer.materialize()
        
        # Create new layer with compressed weights
        compressed_linear = nn.Linear(
            in_features=original_layer.in_features,
            out_features=original_layer.out_features,
            bias=original_layer.bias is not None,
            device=self.device
        )
        compressed_linear.weight.data = weight
        if original_layer.bias is not None:
            compressed_linear.bias.data = original_layer.bias.data
            
        # Replace the layer
        setattr(module, layer_name, compressed_linear)
    
    def compress_model(self) -> nn.Module:
        """Compress the LLaMA model with architecture-specific handling"""
        # First compress embedding layer if configured
        if self.config.compress_embeddings:
            embed_weight = self.base_model.embed_tokens.weight
            mpo = MPOTensor.from_tensor(embed_weight, self.config.bond_dim)
            self.compressed_layers["embed_tokens"] = mpo
        
        # Compress attention and MLP layers
        for name, layer in self.get_attention_layers().items():
            if isinstance(layer, LlamaAttention):
                self._compress_attention(layer, name)
            else:
                self.compress_layer(layer, name)
                
        for name, layer in self.get_mlp_layers().items():
            self.compress_layer(layer, name)
            
        # Replace layers with compressed versions
        for name, compressed in self.compressed_layers.items():
            self.replace_layer(name, compressed)
            
        return self.model
