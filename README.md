# Tensor Compressor

A flexible tensor compression framework for Large Language Models using Matrix Product Operators (MPO).

## Features

- SVD-based tensor compression using Matrix Product Operators
- Support for various model architectures (LLaMA, etc.)
- GPU acceleration with CPU fallback
- Configurable compression parameters
- Accuracy-aware compression

## Installation

```bash
pip install poetry
poetry install
```

## Quick Start

```python
from tensor_compressor import CompressionConfig
from tensor_compressor.models import LlamaCompressor

# Load model
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

# Configure compression
config = CompressionConfig(
    bond_dim=32,
    target_compression=0.88,
    accuracy_threshold=0.95
)

# Create compressor
compressor = LlamaCompressor(model, config)

# Compress model
compressed_model = compressor.compress()
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.