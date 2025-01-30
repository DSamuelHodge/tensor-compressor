"""
Evaluation harness for compressed models.
"""

from typing import Dict, List, Optional, Union, Iterator
import torch
import torch.nn as nn
from dataclasses import dataclass
from tqdm import tqdm
import numpy as np
from transformers import PreTrainedTokenizer
import time

@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics"""
    accuracy: float
    perplexity: Optional[float] = None
    latency: Optional[float] = None
    memory_usage: Optional[int] = None
    additional_metrics: Optional[Dict[str, float]] = None

class ModelEvaluator:
    """Evaluator for compressed language models"""
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer: PreTrainedTokenizer,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(device)
        self.model.eval()
    
    def evaluate_perplexity(
        self,
        dataset: Iterator[str],
        max_length: int = 512,
        batch_size: int = 1
    ) -> float:
        """
        Evaluate model perplexity on a dataset.
        
        Args:
            dataset: Iterator of text samples
            max_length: Maximum sequence length
            batch_size: Batch size for evaluation
            
        Returns:
            Perplexity score
        """
        total_loss = 0.0
        total_tokens = 0
        
        with torch.no_grad():
            for text_batch in self._batch_iterator(dataset, batch_size):
                inputs = self.tokenizer(
                    text_batch,
                    max_length=max_length,
                    truncation=True,
                    padding=True,
                    return_tensors="pt"
                ).to(self.device)
                
                labels = inputs["input_ids"].clone()
                outputs = self.model(**inputs, labels=labels)
                
                loss = outputs.loss
                total_loss += loss.item() * labels.numel()
                total_tokens += labels.numel()
        
        return torch.exp(torch.tensor(total_loss / total_tokens)).item()
    
    def evaluate_accuracy(
        self,
        dataset: List[Dict[str, Union[str, List[str]]]],
        metric: str = "exact_match"
    ) -> float:
        """
        Evaluate model accuracy on a dataset.
        
        Args:
            dataset: List of examples with questions and answers
            metric: Metric to use for evaluation
            
        Returns:
            Accuracy score
        """
        correct = 0
        total = 0
        
        with torch.no_grad():
            for example in tqdm(dataset, desc="Evaluating"):
                question = example["question"]
                references = example["answers"]
                
                prediction = self.generate_answer(question)
                if self._is_correct(prediction, references, metric):
                    correct += 1
                total += 1
        
        return correct / total if total > 0 else 0.0
    
    def measure_latency(
        self,
        input_text: str,
        num_runs: int = 10,
        warmup: int = 2
    ) -> Dict[str, float]:
        """
        Measure model inference latency.
        
        Args:
            input_text: Input text for measurement
            num_runs: Number of runs for averaging
            warmup: Number of warmup runs
            
        Returns:
            Dictionary with latency statistics
        """
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt"
        ).to(self.device)
        
        # Warmup runs
        for _ in range(warmup):
            _ = self.model.generate(**inputs, max_length=20)
        
        latencies = []
        with torch.no_grad():
            for _ in range(num_runs):
                start_time = time.time()
                _ = self.model.generate(**inputs, max_length=20)
                latencies.append(time.time() - start_time)
        
        return {
            "mean_latency": np.mean(latencies),
            "std_latency": np.std(latencies),
            "min_latency": np.min(latencies),
            "max_latency": np.max(latencies)
        }
    
    def measure_memory(self) -> Dict[str, int]:
        """
        Measure model memory usage.
        
        Returns:
            Dictionary with memory statistics in bytes
        """
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
            
            # Trigger memory allocation
            sample_input = self.tokenizer(
                "Sample text for memory measurement",
                return_tensors="pt"
            ).to(self.device)
            with torch.no_grad():
                _ = self.model(**sample_input)
            
            peak_memory = torch.cuda.max_memory_allocated()
            current_memory = torch.cuda.memory_allocated()
        else:
            peak_memory = 0
            current_memory = 0
        
        return {
            "peak_memory": peak_memory,
            "current_memory": current_memory
        }
    
    def _batch_iterator(
        self,
        dataset: Iterator[str],
        batch_size: int
    ) -> Iterator[List[str]]:
        """Create batches from dataset"""
        batch = []
        for item in dataset:
            batch.append(item)
            if len(batch) == batch_size:
                yield batch
                batch = []
        if batch:
            yield batch
    
    def _is_correct(
        self,
        prediction: str,
        references: List[str],
        metric: str
    ) -> bool:
        """Check if prediction matches references"""
        if metric == "exact_match":
            return any(pred.strip() == ref.strip() 
                      for pred in [prediction] 
                      for ref in references)
        elif metric == "contains":
            return any(ref.strip() in prediction.strip() 
                      for ref in references)
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    def generate_answer(
        self,
        question: str,
        max_length: int = 50,
        num_beams: int = 4
    ) -> str:
        """Generate answer for a question"""
        inputs = self.tokenizer(
            question,
            return_tensors="pt"
        ).to(self.device)
        
        outputs = self.model.generate(
            **inputs,
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=True
        )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

class ComprehensiveEvaluator:
    """Run comprehensive evaluation on compressed models"""
    
    def __init__(
        self,
        original_model: nn.Module,
        compressed_model: nn.Module,
        tokenizer: PreTrainedTokenizer,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.original_evaluator = ModelEvaluator(original_model, tokenizer, device)
        self.compressed_evaluator = ModelEvaluator(compressed_model, tokenizer, device)
    
    def run_evaluation(
        self,
        datasets: Dict[str, Union[Iterator[str], List[Dict[str, Union[str, List[str]]]]]],
        metrics: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, EvaluationMetrics]]:
        """
        Run comprehensive evaluation comparing original and compressed models.
        
        Args:
            datasets: Dictionary mapping dataset names to evaluation datasets
            metrics: List of metrics to evaluate
            
        Returns:
            Dictionary with evaluation results
        """
        metrics = metrics or ["perplexity", "accuracy", "latency", "memory"]
        results = {"original": {}, "compressed": {}}
        
        for name, dataset in datasets.items():
            # Evaluate original model
            original_metrics = self._evaluate_model(
                self.original_evaluator, dataset, metrics
            )
            results["original"][name] = original_metrics
            
            # Evaluate compressed model
            compressed_metrics = self._evaluate_model(
                self.compressed_evaluator, dataset, metrics
            )
            results["compressed"][name] = compressed_metrics
        
        return results
    
    def _evaluate_model(
        self,
        evaluator: ModelEvaluator,
        dataset: Union[Iterator[str], List[Dict[str, Union[str, List[str]]]]],
        metrics: List[str]
    ) -> EvaluationMetrics:
        """Run evaluation for a single model"""
        results = {}
        
        if "perplexity" in metrics and isinstance(dataset, Iterator):
            results["perplexity"] = evaluator.evaluate_perplexity(dataset)
            
        if "accuracy" in metrics and isinstance(dataset, list):
            results["accuracy"] = evaluator.evaluate_accuracy(dataset)
            
        if "latency" in metrics:
            latency_stats = evaluator.measure_latency(
                "Sample text for latency measurement"
            )
            results["latency"] = latency_stats["mean_latency"]
            
        if "memory" in metrics:
            memory_stats = evaluator.measure_memory()
            results["memory_usage"] = memory_stats["peak_memory"]
        
        return EvaluationMetrics(**results)