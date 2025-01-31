"""
Base classes for model evaluation.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
from .metrics import AccuracyMetrics, PerformanceMetrics

class TaskEvaluator(ABC):
    """Base class for task-specific evaluators"""
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(device)
        self.model.eval()
    
    @abstractmethod
    def load_dataset(self) -> Any:
        """Load task-specific dataset"""
        pass
    
    @abstractmethod
    def evaluate(self) -> AccuracyMetrics:
        """Evaluate model on task"""
        pass
    
    @abstractmethod
    def measure_performance(self, num_samples: int = 100) -> PerformanceMetrics:
        """Measure performance metrics on task"""
        pass
    
    def format_prompt(self, example: Dict[str, Any]) -> str:
        """Format example into prompt"""
        raise NotImplementedError
    
    def preprocess_example(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess example before evaluation"""
        return example
    
    def postprocess_output(
        self,
        output: torch.Tensor,
        example: Dict[str, Any]
    ) -> Any:
        """Postprocess model output"""
        return output

class ComparisonEvaluator:
    """Compare original and compressed models"""
    
    def __init__(
        self,
        original_model: PreTrainedModel,
        compressed_model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        tasks: List[TaskEvaluator],
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.original_evaluators = [
            task_cls(original_model, tokenizer, device)
            for task_cls in tasks
        ]
        self.compressed_evaluators = [
            task_cls(compressed_model, tokenizer, device)
            for task_cls in tasks
        ]
    
    def evaluate(self) -> Dict[str, Dict[str, Union[AccuracyMetrics, PerformanceMetrics]]]:
        """Run evaluation on all tasks"""
        results = {
            "original": {},
            "compressed": {}
        }
        
        # Evaluate original model
        for evaluator in self.original_evaluators:
            task_name = evaluator.__class__.__name__
            results["original"][task_name] = {
                "accuracy": evaluator.evaluate(),
                "performance": evaluator.measure_performance()
            }
        
        # Evaluate compressed model
        for evaluator in self.compressed_evaluators:
            task_name = evaluator.__class__.__name__
            results["compressed"][task_name] = {
                "accuracy": evaluator.evaluate(),
                "performance": evaluator.measure_performance()
            }
        
        return results
    
    def compare_results(
        self,
        results: Dict[str, Dict[str, Dict[str, Union[AccuracyMetrics, PerformanceMetrics]]]]
    ) -> Dict[str, Dict[str, float]]:
        """Compare results between original and compressed models"""
        comparison = {}
        
        for task_name in results["original"]:
            orig_metrics = results["original"][task_name]
            comp_metrics = results["compressed"][task_name]
            
            comparison[task_name] = {
                "accuracy_change": (
                    comp_metrics["accuracy"].accuracy -
                    orig_metrics["accuracy"].accuracy
                ),
                "latency_change": (
                    comp_metrics["performance"].latency /
                    orig_metrics["performance"].latency - 1
                ),
                "memory_change": (
                    comp_metrics["performance"].memory_usage /
                    orig_metrics["performance"].memory_usage - 1
                )
            }
        
        return comparison