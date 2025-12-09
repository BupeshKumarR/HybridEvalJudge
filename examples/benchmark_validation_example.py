#!/usr/bin/env python3
"""
Example: Benchmark Validation

This example demonstrates how to run benchmark validation on FEVER and TruthfulQA datasets.

Usage:
    python examples/benchmark_validation_example.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))


def example_download_benchmarks():
    """Example: Download benchmark datasets."""
    print("=" * 70)
    print("Example 1: Downloading Benchmark Datasets")
    print("=" * 70)
    
    from download_benchmarks import BenchmarkDownloader
    
    # Initialize downloader
    downloader = BenchmarkDownloader(output_dir="benchmarks")
    
    # Download all benchmarks
    print("\nDownloading all benchmark datasets...")
    downloader.download_all()
    
    print("\n✓ Benchmarks downloaded successfully!")


def example_run_fever_benchmark():
    """Example: Run FEVER benchmark validation."""
    print("\n" + "=" * 70)
    print("Example 2: Running FEVER Benchmark")
    print("=" * 70)
    
    from run_benchmarks import BenchmarkEvaluator
    
    # Initialize evaluator with balanced preset
    evaluator = BenchmarkEvaluator(preset="balanced", benchmarks_dir="benchmarks")
    
    # Run FEVER evaluation (limited samples for demo)
    print("\nRunning FEVER evaluation...")
    result = evaluator.evaluate_fever(max_samples=5)
    
    print(f"\n✓ FEVER evaluation complete!")
    print(f"  Accuracy: {result.accuracy:.2%}")
    print(f"  F1 Score: {result.f1_score:.2%}")
    
    return result


def example_run_truthfulqa_benchmark():
    """Example: Run TruthfulQA benchmark validation."""
    print("\n" + "=" * 70)
    print("Example 3: Running TruthfulQA Benchmark")
    print("=" * 70)
    
    from run_benchmarks import BenchmarkEvaluator
    
    # Initialize evaluator with fast preset
    evaluator = BenchmarkEvaluator(preset="fast", benchmarks_dir="benchmarks")
    
    # Run TruthfulQA evaluation
    print("\nRunning TruthfulQA evaluation...")
    result = evaluator.evaluate_truthfulqa(max_samples=3)
    
    print(f"\n✓ TruthfulQA evaluation complete!")
    print(f"  Accuracy: {result.accuracy:.2%}")
    print(f"  F1 Score: {result.f1_score:.2%}")
    
    return result


def example_compare_presets():
    """Example: Compare different presets on benchmarks."""
    print("\n" + "=" * 70)
    print("Example 4: Comparing Presets")
    print("=" * 70)
    
    from run_benchmarks import BenchmarkEvaluator
    
    presets = ["fast", "balanced"]
    results = {}
    
    for preset in presets:
        print(f"\nEvaluating with {preset} preset...")
        evaluator = BenchmarkEvaluator(preset=preset, benchmarks_dir="benchmarks")
        
        # Run on small sample
        fever_result = evaluator.evaluate_fever(max_samples=5)
        results[preset] = fever_result
    
    # Compare results
    print("\n" + "-" * 70)
    print("Preset Comparison (FEVER)")
    print("-" * 70)
    print(f"{'Preset':<15} {'Accuracy':<12} {'F1 Score':<12} {'Latency (ms)':<15}")
    print("-" * 70)
    
    for preset, result in results.items():
        print(f"{preset:<15} {result.accuracy:<12.2%} {result.f1_score:<12.2%} {result.avg_latency_ms:<15.2f}")
    
    print("-" * 70)


def example_save_and_load_results():
    """Example: Save and load benchmark results."""
    print("\n" + "=" * 70)
    print("Example 5: Saving and Loading Results")
    print("=" * 70)
    
    from run_benchmarks import BenchmarkEvaluator
    import json
    
    # Run evaluation
    evaluator = BenchmarkEvaluator(preset="balanced", benchmarks_dir="benchmarks")
    fever_result = evaluator.evaluate_fever(max_samples=5)
    truthfulqa_result = evaluator.evaluate_truthfulqa(max_samples=3)
    
    # Save results
    results = [fever_result, truthfulqa_result]
    output_file = "benchmarks/results/example_results.json"
    evaluator.save_results(results, output_file)
    
    print(f"\n✓ Results saved to {output_file}")
    
    # Load and display results
    print("\nLoading results...")
    with open(output_file, "r") as f:
        loaded_results = json.load(f)
    
    print("\nLoaded Results:")
    for result in loaded_results:
        print(f"\n{result['dataset']}:")
        print(f"  Accuracy: {result['accuracy']:.2%}")
        print(f"  F1 Score: {result['f1_score']:.2%}")
        print(f"  Avg Latency: {result['avg_latency_ms']:.2f} ms")


def example_baseline_comparison():
    """Example: Compare results to published baselines."""
    print("\n" + "=" * 70)
    print("Example 6: Baseline Comparison")
    print("=" * 70)
    
    from run_benchmarks import BenchmarkEvaluator
    
    # Run evaluations
    evaluator = BenchmarkEvaluator(preset="balanced", benchmarks_dir="benchmarks")
    
    fever_result = evaluator.evaluate_fever(max_samples=5)
    truthfulqa_result = evaluator.evaluate_truthfulqa(max_samples=3)
    
    # Compare to baselines
    results = [fever_result, truthfulqa_result]
    evaluator.compare_to_baseline(results)


def example_custom_evaluation():
    """Example: Custom evaluation logic."""
    print("\n" + "=" * 70)
    print("Example 7: Custom Evaluation Logic")
    print("=" * 70)
    
    from run_benchmarks import BenchmarkEvaluator
    
    class CustomEvaluator(BenchmarkEvaluator):
        """Custom evaluator with additional metrics."""
        
        def _evaluate_claim(self, claim, evidence, true_label):
            """Custom evaluation with additional checks."""
            # Call parent implementation
            result = super()._evaluate_claim(claim, evidence, true_label)
            
            # Add custom logic
            if len(claim.split()) < 5:
                result["note"] = "Short claim"
            
            return result
    
    # Use custom evaluator
    evaluator = CustomEvaluator(preset="balanced", benchmarks_dir="benchmarks")
    result = evaluator.evaluate_fever(max_samples=5)
    
    print(f"\n✓ Custom evaluation complete!")
    print(f"  Accuracy: {result.accuracy:.2%}")


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("Benchmark Validation Examples")
    print("=" * 70)
    
    try:
        # Example 1: Download benchmarks
        example_download_benchmarks()
        
        # Example 2: Run FEVER benchmark
        example_run_fever_benchmark()
        
        # Example 3: Run TruthfulQA benchmark
        example_run_truthfulqa_benchmark()
        
        # Example 4: Compare presets
        example_compare_presets()
        
        # Example 5: Save and load results
        example_save_and_load_results()
        
        # Example 6: Baseline comparison
        example_baseline_comparison()
        
        # Example 7: Custom evaluation
        example_custom_evaluation()
        
        print("\n" + "=" * 70)
        print("All examples completed successfully!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
