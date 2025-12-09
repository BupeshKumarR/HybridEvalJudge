#!/usr/bin/env python3
"""
Script to run benchmark validation on FEVER and TruthfulQA datasets.

Usage:
    python scripts/run_benchmarks.py --dataset fever --preset balanced
    python scripts/run_benchmarks.py --dataset truthfulqa --preset fast
    python scripts/run_benchmarks.py --all --preset balanced
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@dataclass
class BenchmarkResult:
    """Results from benchmark evaluation."""
    dataset: str
    total_samples: int
    correct: int
    incorrect: int
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    avg_confidence: float
    avg_latency_ms: float
    errors: int


class BenchmarkEvaluator:
    """Evaluates the toolkit on benchmark datasets."""
    
    def __init__(self, preset: str = "balanced", benchmarks_dir: str = "benchmarks"):
        self.preset = preset
        self.benchmarks_dir = Path(benchmarks_dir)
        self.toolkit = None
        
    def _initialize_toolkit(self):
        """Initialize the evaluation toolkit (lazy loading)."""
        if self.toolkit is not None:
            return
        
        print(f"Initializing toolkit with preset: {self.preset}")
        print("Note: This is a mock implementation for demonstration.")
        print("In production, this would load actual models.")
        
        # Mock toolkit for demonstration
        # In production, this would be:
        # from llm_judge_auditor import EvaluationToolkit
        # self.toolkit = EvaluationToolkit.from_preset(self.preset)
        
        self.toolkit = MockToolkit()
        print("✓ Toolkit initialized")
    
    def evaluate_fever(self, max_samples: int = None) -> BenchmarkResult:
        """Evaluate on FEVER dataset."""
        print("\n" + "=" * 60)
        print("Evaluating on FEVER dataset")
        print("=" * 60)
        
        self._initialize_toolkit()
        
        fever_file = self.benchmarks_dir / "fever" / "dev.jsonl"
        if not fever_file.exists():
            raise FileNotFoundError(
                f"FEVER dataset not found at {fever_file}. "
                "Run 'python scripts/download_benchmarks.py' first."
            )
        
        # Load dataset
        samples = []
        with open(fever_file, "r") as f:
            for line in f:
                samples.append(json.loads(line))
                if max_samples and len(samples) >= max_samples:
                    break
        
        print(f"Loaded {len(samples)} samples")
        
        # Evaluate each sample
        correct = 0
        incorrect = 0
        errors = 0
        total_confidence = 0.0
        total_latency = 0.0
        
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        
        for i, sample in enumerate(samples, 1):
            try:
                start_time = time.time()
                
                # Evaluate claim against evidence
                result = self._evaluate_claim(
                    claim=sample["claim"],
                    evidence=sample["evidence"],
                    true_label=sample["label"]
                )
                
                latency = (time.time() - start_time) * 1000  # ms
                total_latency += latency
                
                if result["correct"]:
                    correct += 1
                    if sample["label"] in ["SUPPORTS", "REFUTES"]:
                        true_positives += 1
                else:
                    incorrect += 1
                    if sample["label"] in ["SUPPORTS", "REFUTES"]:
                        false_negatives += 1
                    else:
                        false_positives += 1
                
                total_confidence += result["confidence"]
                
                if i % 10 == 0:
                    print(f"Progress: {i}/{len(samples)} samples processed")
                
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                errors += 1
        
        # Calculate metrics
        accuracy = correct / len(samples) if samples else 0.0
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        avg_confidence = total_confidence / len(samples) if samples else 0.0
        avg_latency = total_latency / len(samples) if samples else 0.0
        
        result = BenchmarkResult(
            dataset="FEVER",
            total_samples=len(samples),
            correct=correct,
            incorrect=incorrect,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            avg_confidence=avg_confidence,
            avg_latency_ms=avg_latency,
            errors=errors
        )
        
        self._print_results(result)
        return result
    
    def evaluate_truthfulqa(self, max_samples: int = None) -> BenchmarkResult:
        """Evaluate on TruthfulQA dataset."""
        print("\n" + "=" * 60)
        print("Evaluating on TruthfulQA dataset")
        print("=" * 60)
        
        self._initialize_toolkit()
        
        truthfulqa_file = self.benchmarks_dir / "truthfulqa" / "truthfulqa.json"
        if not truthfulqa_file.exists():
            raise FileNotFoundError(
                f"TruthfulQA dataset not found at {truthfulqa_file}. "
                "Run 'python scripts/download_benchmarks.py' first."
            )
        
        # Load dataset
        with open(truthfulqa_file, "r") as f:
            samples = json.load(f)
        
        if max_samples:
            samples = samples[:max_samples]
        
        print(f"Loaded {len(samples)} samples")
        
        # Evaluate each sample
        correct = 0
        incorrect = 0
        errors = 0
        total_confidence = 0.0
        total_latency = 0.0
        
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        
        for i, sample in enumerate(samples, 1):
            try:
                start_time = time.time()
                
                # Evaluate answer against question
                result = self._evaluate_truthfulness(
                    question=sample["question"],
                    answer=sample["best_answer"],
                    correct_answers=sample["correct_answers"],
                    incorrect_answers=sample["incorrect_answers"]
                )
                
                latency = (time.time() - start_time) * 1000  # ms
                total_latency += latency
                
                if result["correct"]:
                    correct += 1
                    true_positives += 1
                else:
                    incorrect += 1
                    false_negatives += 1
                
                total_confidence += result["confidence"]
                
                if i % 10 == 0:
                    print(f"Progress: {i}/{len(samples)} samples processed")
                
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                errors += 1
        
        # Calculate metrics
        accuracy = correct / len(samples) if samples else 0.0
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        avg_confidence = total_confidence / len(samples) if samples else 0.0
        avg_latency = total_latency / len(samples) if samples else 0.0
        
        result = BenchmarkResult(
            dataset="TruthfulQA",
            total_samples=len(samples),
            correct=correct,
            incorrect=incorrect,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            avg_confidence=avg_confidence,
            avg_latency_ms=avg_latency,
            errors=errors
        )
        
        self._print_results(result)
        return result
    
    def _evaluate_claim(self, claim: str, evidence: str, true_label: str) -> Dict:
        """Evaluate a single claim (mock implementation)."""
        # Mock evaluation - in production, use actual toolkit
        # result = self.toolkit.evaluate(source_text=evidence, candidate_output=claim)
        
        # Simple heuristic for demonstration
        import random
        confidence = random.uniform(0.7, 0.95)
        
        # Mock prediction based on simple heuristics
        if "not" in claim.lower() or "never" in claim.lower():
            predicted_label = "REFUTES"
        else:
            predicted_label = "SUPPORTS"
        
        correct = predicted_label == true_label
        
        return {
            "correct": correct,
            "confidence": confidence,
            "predicted_label": predicted_label
        }
    
    def _evaluate_truthfulness(self, question: str, answer: str, 
                               correct_answers: List[str], 
                               incorrect_answers: List[str]) -> Dict:
        """Evaluate truthfulness of an answer (mock implementation)."""
        # Mock evaluation - in production, use actual toolkit
        import random
        confidence = random.uniform(0.7, 0.95)
        
        # Simple check if answer is in correct answers
        correct = any(answer.lower() in ca.lower() for ca in correct_answers)
        
        return {
            "correct": correct,
            "confidence": confidence
        }
    
    def _print_results(self, result: BenchmarkResult):
        """Print benchmark results."""
        print("\n" + "-" * 60)
        print(f"Results for {result.dataset}")
        print("-" * 60)
        print(f"Total Samples:     {result.total_samples}")
        print(f"Correct:           {result.correct}")
        print(f"Incorrect:         {result.incorrect}")
        print(f"Errors:            {result.errors}")
        print(f"Accuracy:          {result.accuracy:.2%}")
        print(f"Precision:         {result.precision:.2%}")
        print(f"Recall:            {result.recall:.2%}")
        print(f"F1 Score:          {result.f1_score:.2%}")
        print(f"Avg Confidence:    {result.avg_confidence:.2%}")
        print(f"Avg Latency:       {result.avg_latency_ms:.2f} ms")
        print("-" * 60)
    
    def save_results(self, results: List[BenchmarkResult], output_file: str):
        """Save results to JSON file."""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        results_dict = [asdict(r) for r in results]
        
        with open(output_path, "w") as f:
            json.dump(results_dict, f, indent=2)
        
        print(f"\n✓ Results saved to {output_path}")
    
    def compare_to_baseline(self, results: List[BenchmarkResult]):
        """Compare results to baseline performance."""
        print("\n" + "=" * 60)
        print("Comparison to Baselines")
        print("=" * 60)
        
        # Baseline values from literature
        baselines = {
            "FEVER": {
                "accuracy": 0.85,
                "f1_score": 0.82,
                "source": "MiniCheck baseline"
            },
            "TruthfulQA": {
                "accuracy": 0.75,
                "f1_score": 0.70,
                "source": "GPT-3.5 baseline"
            }
        }
        
        for result in results:
            if result.dataset in baselines:
                baseline = baselines[result.dataset]
                print(f"\n{result.dataset}:")
                print(f"  Baseline ({baseline['source']}):")
                print(f"    Accuracy: {baseline['accuracy']:.2%}")
                print(f"    F1 Score: {baseline['f1_score']:.2%}")
                print(f"  Our Results:")
                print(f"    Accuracy: {result.accuracy:.2%} ({self._format_diff(result.accuracy, baseline['accuracy'])})")
                print(f"    F1 Score: {result.f1_score:.2%} ({self._format_diff(result.f1_score, baseline['f1_score'])})")
    
    def _format_diff(self, value: float, baseline: float) -> str:
        """Format difference from baseline."""
        diff = value - baseline
        sign = "+" if diff >= 0 else ""
        return f"{sign}{diff:.2%}"


class MockToolkit:
    """Mock toolkit for demonstration purposes."""
    pass


def main():
    parser = argparse.ArgumentParser(
        description="Run benchmark validation for LLM Judge Auditor"
    )
    parser.add_argument(
        "--dataset",
        choices=["fever", "truthfulqa", "all"],
        default="all",
        help="Which dataset to evaluate (default: all)"
    )
    parser.add_argument(
        "--preset",
        default="balanced",
        help="Toolkit preset to use (default: balanced)"
    )
    parser.add_argument(
        "--benchmarks-dir",
        default="benchmarks",
        help="Directory containing benchmark datasets (default: benchmarks)"
    )
    parser.add_argument(
        "--output",
        default="benchmarks/results/benchmark_results.json",
        help="Output file for results (default: benchmarks/results/benchmark_results.json)"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate per dataset (default: all)"
    )
    
    args = parser.parse_args()
    
    evaluator = BenchmarkEvaluator(
        preset=args.preset,
        benchmarks_dir=args.benchmarks_dir
    )
    
    results = []
    
    try:
        if args.dataset in ["fever", "all"]:
            result = evaluator.evaluate_fever(max_samples=args.max_samples)
            results.append(result)
        
        if args.dataset in ["truthfulqa", "all"]:
            result = evaluator.evaluate_truthfulqa(max_samples=args.max_samples)
            results.append(result)
        
        # Save results
        evaluator.save_results(results, args.output)
        
        # Compare to baselines
        evaluator.compare_to_baseline(results)
        
        print("\n" + "=" * 60)
        print("Benchmark validation complete!")
        print("=" * 60)
        
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nPlease run 'python scripts/download_benchmarks.py' first.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError during benchmark evaluation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
