"""
Example script for compressing a LLaMA model and evaluating its performance.
"""

import argparse
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
from datasets import load_dataset
from tensor_compressor import CompressionConfig
from tensor_compressor.models import LlamaAdapter
from tensor_compressor.evaluation import ComprehensiveEvaluator

def parse_args():
    parser = argparse.ArgumentParser(description="Compress and evaluate LLaMA model")
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-2-7b-chat-hf",
        help="Name or path of the model to compress"
    )
    parser.add_argument(
        "--bond_dim",
        type=int,
        default=32,
        help="Bond dimension for compression"
    )
    parser.add_argument(
        "--target_compression",
        type=float,
        default=0.88,
        help="Target compression ratio"
    )
    parser.add_argument(
        "--accuracy_threshold",
        type=float,
        default=0.95,
        help="Minimum accuracy threshold"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./compressed_model",
        help="Directory to save compressed model"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for compression"
    )
    return parser.parse_args()

def load_evaluation_datasets():
    """Load datasets for evaluation"""
    datasets = {}
    
    # Load MMLU dataset
    mmlu = load_dataset("cais/mmlu", "all", split="validation")
    datasets["mmlu"] = [
        {
            "question": example["question"],
            "answers": [example["answer"]]
        }
        for example in mmlu
    ]
    
    # Load GSM8K dataset
    gsm8k = load_dataset("gsm8k", "main", split="test")
    datasets["gsm8k"] = [
        {
            "question": example["question"],
            "answers": [example["answer"].split("####")[1].strip()]
        }
        for example in gsm8k
    ]
    
    return datasets

def main():
    args = parse_args()
    
    print(f"Loading model: {args.model_name}")
    model = LlamaForCausalLM.from_pretrained(args.model_name)
    tokenizer = LlamaTokenizer.from_pretrained(args.model_name)
    
    # Configure compression
    config = CompressionConfig(
        bond_dim=args.bond_dim,
        target_compression=args.target_compression,
        accuracy_threshold=args.accuracy_threshold,
        device=args.device
    )
    
    print("Compressing model...")
    adapter = LlamaAdapter(model, config)
    compressed_model = adapter.compress_model()
    
    # Get compression statistics
    stats = adapter.get_compression_summary()
    print("\nCompression Results:")
    print(f"Overall compression ratio: {stats['overall_compression_ratio']:.2%}")
    print(f"Original size: {stats['total_original_size'] / 1e9:.2f} GB")
    print(f"Compressed size: {stats['total_compressed_size'] / 1e9:.2f} GB")
    
    # Load evaluation datasets
    print("\nLoading evaluation datasets...")
    datasets = load_evaluation_datasets()
    
    # Run evaluation
    print("\nEvaluating models...")
    evaluator = ComprehensiveEvaluator(
        original_model=model,
        compressed_model=compressed_model,
        tokenizer=tokenizer,
        device=args.device
    )
    
    results = evaluator.run_evaluation(datasets)
    
    # Print results
    print("\nEvaluation Results:")
    for model_type, model_results in results.items():
        print(f"\n{model_type.title()} Model:")
        for dataset_name, metrics in model_results.items():
            print(f"\n{dataset_name} Dataset:")
            print(f"Accuracy: {metrics.accuracy:.2%}")
            if metrics.perplexity:
                print(f"Perplexity: {metrics.perplexity:.2f}")
            if metrics.latency:
                print(f"Latency: {metrics.latency*1000:.2f} ms")
            if metrics.memory_usage:
                print(f"Peak Memory: {metrics.memory_usage/1e9:.2f} GB")
    
    # Save compressed model
    print(f"\nSaving compressed model to {args.output_dir}")
    compressed_model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()