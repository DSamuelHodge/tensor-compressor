"""
MMLU (Massive Multitask Language Understanding) task evaluator.
"""

from typing import Dict, List, Optional, Union
import torch
from datasets import load_dataset
from transformers import PreTrainedModel, PreTrainedTokenizer
from ..metrics import AccuracyMetrics, PerformanceMetrics
from ..evaluator import TaskEvaluator

class MMLUEvaluator(TaskEvaluator):
    """Evaluator for MMLU benchmark"""
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        subjects: Optional[List[str]] = None
    ):
        super().__init__(model, tokenizer, device)
        self.subjects = subjects or [
            "abstract_algebra", "anatomy", "astronomy", "business_ethics",
            "clinical_knowledge", "college_biology", "college_chemistry",
            "college_computer_science", "college_mathematics", "college_medicine",
            "college_physics", "computer_security", "conceptual_physics",
            "econometrics", "electrical_engineering", "elementary_mathematics",
            "formal_logic", "global_facts", "high_school_biology",
            "high_school_chemistry", "high_school_computer_science",
            "high_school_european_history", "high_school_geography",
            "high_school_government_and_politics", "high_school_macroeconomics",
            "high_school_mathematics", "high_school_microeconomics",
            "high_school_physics", "high_school_psychology",
            "high_school_statistics", "high_school_us_history",
            "high_school_world_history", "human_aging", "human_sexuality",
            "international_law", "jurisprudence", "logical_fallacies",
            "machine_learning", "management", "marketing", "medical_genetics",
            "miscellaneous", "moral_debates", "moral_scenarios", "nutrition",
            "philosophy", "prehistory", "professional_accounting",
            "professional_law", "professional_medicine", "professional_psychology",
            "public_relations", "security_studies", "sociology",
            "us_foreign_policy", "virology", "world_religions"
        ]
    
    def load_dataset(self) -> Dict[str, List[Dict[str, Union[str, List[str]]]]]:
        """Load MMLU dataset by subject"""
        datasets = {}
        for subject in self.subjects:
            try:
                ds = load_dataset("cais/mmlu", subject, split="validation")
                datasets[subject] = [
                    {
                        "question": row["question"],
                        "choices": [row["choices"][i] for i in range(4)],
                        "answer": row["answer"]
                    }
                    for row in ds
                ]
            except Exception as e:
                print(f"Failed to load {subject}: {e}")
                continue
        return datasets
    
    def format_prompt(self, example: Dict[str, Union[str, List[str]]]) -> str:
        """Format MMLU example into prompt"""
        choices = example["choices"]
        prompt = f"{example['question']}\n\n"
        for i, choice in enumerate(choices):
            prompt += f"{chr(65 + i)}. {choice}\n"
        prompt += "\nAnswer:"
        return prompt
    
    def evaluate(self) -> AccuracyMetrics:
        """Evaluate model on MMLU benchmark"""
        datasets = self.load_dataset()
        results = {}
        total_correct = 0
        total_questions = 0
        
        for subject, examples in datasets.items():
            correct = 0
            
            for example in examples:
                prompt = self.format_prompt(example)
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=1,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
                
                predicted = self.tokenizer.decode(outputs[0][-1]).strip().upper()
                if predicted == chr(65 + example["answer"]):
                    correct += 1
            
            accuracy = correct / len(examples)
            results[subject] = accuracy
            total_correct += correct
            total_questions += len(examples)
        
        overall_accuracy = total_correct / total_questions
        
        return AccuracyMetrics(
            accuracy=overall_accuracy,
            task_specific=results
        )
    
    def measure_performance(self, num_samples: int = 100) -> PerformanceMetrics:
        """Measure performance metrics on MMLU task"""
        datasets = self.load_dataset()
        flat_examples = [
            example
            for examples in datasets.values()
            for example in examples[:num_samples // len(datasets)]
        ]
        
        if not flat_examples:
            raise ValueError("No examples available for performance measurement")
        
        # Prepare batch for measurement
        prompts = [self.format_prompt(example) for example in flat_examples]
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        # Measure performance
        from ..metrics import measure_inference_latency
        return measure_inference_latency(self.model, inputs["input_ids"])