from dataclasses import dataclass, field
from typing import Optional, List, Union
import torch

@dataclass
class CompressionConfig:
    """Configuration for model compression"""
    
    # Compression parameters
    bond_dim: int
    target_compression: float
    accuracy_threshold: float
    
    # Device and precision settings
    device: str = field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu"
    )
    dtype: torch.dtype = torch.float32
    seed: Optional[int] = None
    
    # Layer-specific settings
    attention_layers: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    mlp_layers: List[str] = field(
        default_factory=lambda: ["gate_proj", "up_proj", "down_proj"]
    )
    
    # Evaluation settings
    eval_tasks: List[str] = field(
        default_factory=lambda: ["mmlu", "hellaswag", "boolq", "triviaqa", "gsm8k"]
    )
    batch_size: int = 1
    
    def __post_init__(self):
        """Validate configuration and set random seeds"""
        if not 0 < self.target_compression < 1:
            raise ValueError("Target compression must be between 0 and 1")
        
        if not 0 < self.accuracy_threshold <= 1:
            raise ValueError("Accuracy threshold must be between 0 and 1")
            
        if self.bond_dim < 1:
            raise ValueError("Bond dimension must be positive")
            
        if self.seed is not None:
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)
                
    @property
    def device_type(self) -> torch.device:
        """Get torch device object"""
        return torch.device(self.device)