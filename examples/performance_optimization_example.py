"""
Example demonstrating performance optimization features.

This example shows how to:
1. Enable profiling to identify bottlenecks
2. Use parallel judge evaluation for faster processing
3. Analyze performance metrics
4. Optimize batch processing

Requirements: 1.1, 1.2, 5.1
"""

import logging
import time
from pathlib import Path

from llm_judge_auditor.config import ToolkitConfig
from llm_judge_auditor.evaluation_toolkit import EvaluationToolkit
from llm_judge_auditor.models import EvaluationRequest

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def example_profiling():
    """
    Example 1: Using profiling to identify bottlenecks.
    """
    print("\n" + "=" * 80)
    print("Example 1: Performance Profiling")
    print("=" * 80)

    # Create toolkit with profiling enabled
    config = ToolkitConfig(
        verifier_model="mock-verifier",  # Use mock for demo
        judge_models=["mock-judge-1", "mock-judge-2", "mock-judge-3"],
        device="cpu",
        quantize=False,
        enable_retrieval=False,
    )

    toolkit = EvaluationToolkit(config, enable_profiling=True)

    # Run some evaluations
    source = "The Eiffel Tower was completed in 1889 in Paris, France."
    candidates = [
        "The Eiffel Tower was built in 1889.",
        "The Eiffel Tower is located in Paris.",
        "The Eiffel Tower was completed in the 19th century.",
    ]

    print("\nRunning evaluations with profiling enabled...")
    for i, candidate in enumerate(candidates, 1):
        print(f"\nEvaluating candidate {i}...")
        try:
            result = toolkit.evaluate(
                source_text=source, candidate_output=candidate, parallel_judges=False
            )
            print(f"  Score: {result.consensus_score:.2f}")
        except Exception as e:
            print(f"  Error: {e}")

    # Get profiling summary
    print("\n" + "-" * 80)
    print("Profiling Summary:")
    print("-" * 80)
    print(toolkit.get_profiling_summary())

    # Get bottlenecks
    print("\n" + "-" * 80)
    print("Top Bottlenecks:")
    print("-" * 80)
    bottlenecks = toolkit.get_profiling_bottlenecks(5)
    for name, total_time in bottlenecks:
        print(f"  {name}: {total_time:.3f}s")


def example_parallel_evaluation():
    """
    Example 2: Comparing sequential vs parallel judge evaluation.
    """
    print("\n" + "=" * 80)
    print("Example 2: Sequential vs Parallel Judge Evaluation")
    print("=" * 80)

    # Create toolkit with multiple judges
    config = ToolkitConfig(
        verifier_model="mock-verifier",
        judge_models=["mock-judge-1", "mock-judge-2", "mock-judge-3"],
        device="cpu",
        quantize=False,
        enable_retrieval=False,
    )

    toolkit = EvaluationToolkit(config)

    source = "The Eiffel Tower was completed in 1889 in Paris, France."
    candidate = "The Eiffel Tower was built in 1889 in Paris."

    # Sequential evaluation
    print("\nSequential evaluation (judges run one after another):")
    start_time = time.time()
    try:
        result_seq = toolkit.evaluate(
            source_text=source, candidate_output=candidate, parallel_judges=False
        )
        seq_time = time.time() - start_time
        print(f"  Time: {seq_time:.3f}s")
        print(f"  Score: {result_seq.consensus_score:.2f}")
    except Exception as e:
        print(f"  Error: {e}")
        seq_time = 0

    # Parallel evaluation
    print("\nParallel evaluation (judges run concurrently):")
    start_time = time.time()
    try:
        result_par = toolkit.evaluate(
            source_text=source, candidate_output=candidate, parallel_judges=True
        )
        par_time = time.time() - start_time
        print(f"  Time: {par_time:.3f}s")
        print(f"  Score: {result_par.consensus_score:.2f}")
    except Exception as e:
        print(f"  Error: {e}")
        par_time = 0

    # Compare
    if seq_time > 0 and par_time > 0:
        speedup = seq_time / par_time
        print(f"\nSpeedup: {speedup:.2f}x faster with parallel evaluation")


def example_batch_optimization():
    """
    Example 3: Optimized batch processing with parallel judges.
    """
    print("\n" + "=" * 80)
    print("Example 3: Optimized Batch Processing")
    print("=" * 80)

    # Create toolkit
    config = ToolkitConfig(
        verifier_model="mock-verifier",
        judge_models=["mock-judge-1", "mock-judge-2"],
        device="cpu",
        quantize=False,
        enable_retrieval=False,
    )

    toolkit = EvaluationToolkit(config, enable_profiling=True)

    # Create batch requests
    requests = [
        EvaluationRequest(
            source_text="Paris is the capital of France.",
            candidate_output="Paris is in France.",
            task="factual_accuracy",
            criteria=["correctness"],
            use_retrieval=False,
        ),
        EvaluationRequest(
            source_text="The Earth orbits the Sun.",
            candidate_output="The Sun orbits the Earth.",
            task="factual_accuracy",
            criteria=["correctness"],
            use_retrieval=False,
        ),
        EvaluationRequest(
            source_text="Water boils at 100°C at sea level.",
            candidate_output="Water boils at 100°C.",
            task="factual_accuracy",
            criteria=["correctness"],
            use_retrieval=False,
        ),
    ]

    # Batch evaluation without parallel judges
    print(f"\nBatch evaluation of {len(requests)} requests (sequential judges):")
    start_time = time.time()
    try:
        batch_result_seq = toolkit.batch_evaluate(
            requests, continue_on_error=True, parallel_judges=False
        )
        seq_time = time.time() - start_time
        print(f"  Time: {seq_time:.3f}s")
        print(f"  Successful: {len(batch_result_seq.results)}/{len(requests)}")
        print(f"  Mean score: {batch_result_seq.statistics.get('mean', 0):.2f}")
    except Exception as e:
        print(f"  Error: {e}")
        seq_time = 0

    # Reset profiling
    toolkit.reset_profiling()

    # Batch evaluation with parallel judges
    print(f"\nBatch evaluation of {len(requests)} requests (parallel judges):")
    start_time = time.time()
    try:
        batch_result_par = toolkit.batch_evaluate(
            requests, continue_on_error=True, parallel_judges=True
        )
        par_time = time.time() - start_time
        print(f"  Time: {par_time:.3f}s")
        print(f"  Successful: {len(batch_result_par.results)}/{len(requests)}")
        print(f"  Mean score: {batch_result_par.statistics.get('mean', 0):.2f}")
    except Exception as e:
        print(f"  Error: {e}")
        par_time = 0

    # Compare
    if seq_time > 0 and par_time > 0:
        speedup = seq_time / par_time
        print(f"\nSpeedup: {speedup:.2f}x faster with parallel judges")

    # Show profiling results
    print("\n" + "-" * 80)
    print("Profiling Results:")
    print("-" * 80)
    print(toolkit.get_profiling_summary())


def example_model_caching():
    """
    Example 4: Demonstrating model loading optimization with caching.
    """
    print("\n" + "=" * 80)
    print("Example 4: Model Loading Optimization")
    print("=" * 80)

    config = ToolkitConfig(
        verifier_model="mock-verifier",
        judge_models=["mock-judge-1", "mock-judge-2"],
        device="cpu",
        quantize=False,
        enable_retrieval=False,
    )

    print("\nFirst toolkit initialization (models loaded from disk):")
    start_time = time.time()
    try:
        toolkit1 = EvaluationToolkit(config)
        init_time1 = time.time() - start_time
        print(f"  Initialization time: {init_time1:.3f}s")
    except Exception as e:
        print(f"  Error: {e}")

    print("\nSecond toolkit initialization (models cached):")
    start_time = time.time()
    try:
        toolkit2 = EvaluationToolkit(config)
        init_time2 = time.time() - start_time
        print(f"  Initialization time: {init_time2:.3f}s")
        
        if init_time1 > 0 and init_time2 > 0:
            speedup = init_time1 / init_time2
            print(f"  Speedup: {speedup:.2f}x faster with caching")
    except Exception as e:
        print(f"  Error: {e}")


def example_performance_tracking():
    """
    Example 5: Using performance tracking to monitor component performance.
    """
    print("\n" + "=" * 80)
    print("Example 5: Performance Tracking")
    print("=" * 80)

    config = ToolkitConfig(
        verifier_model="mock-verifier",
        judge_models=["mock-judge-1", "mock-judge-2"],
        device="cpu",
        quantize=False,
        enable_retrieval=False,
    )

    toolkit = EvaluationToolkit(config)

    # Run some evaluations
    print("\nRunning evaluations...")
    source = "The Eiffel Tower was completed in 1889."
    candidates = [
        "The Eiffel Tower was built in 1889.",
        "The Eiffel Tower is in Paris.",
        "The Eiffel Tower was completed in the 19th century.",
    ]

    for i, candidate in enumerate(candidates, 1):
        print(f"  Evaluating candidate {i}...")
        try:
            result = toolkit.evaluate(
                source_text=source, candidate_output=candidate, parallel_judges=True
            )
            print(f"    Score: {result.consensus_score:.2f}")
        except Exception as e:
            print(f"    Error: {e}")

    # Get performance report
    print("\n" + "-" * 80)
    print("Performance Report:")
    print("-" * 80)
    try:
        report = toolkit.get_performance_report()
        
        print("\nVerifier Metrics:")
        verifier_metrics = report.get("verifier_metrics", {})
        print(f"  Total evaluations: {verifier_metrics.get('total_evaluations', 0)}")
        print(f"  Average latency: {verifier_metrics.get('average_latency', 0):.3f}s")
        print(f"  Average confidence: {verifier_metrics.get('average_confidence', 0):.3f}")
        
        print("\nJudge Metrics:")
        judge_metrics = report.get("judge_metrics", {})
        print(f"  Total evaluations: {judge_metrics.get('total_evaluations', 0)}")
        print(f"  Average latency: {judge_metrics.get('average_latency', 0):.3f}s")
        print(f"  Average confidence: {judge_metrics.get('average_confidence', 0):.3f}")
        
        print("\nDisagreements:")
        disagreements = report.get("disagreements", {})
        print(f"  Total: {disagreements.get('total_count', 0)}")
    except Exception as e:
        print(f"  Error: {e}")


def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print("Performance Optimization Examples")
    print("=" * 80)
    print("\nThese examples demonstrate various performance optimization features:")
    print("1. Profiling to identify bottlenecks")
    print("2. Parallel judge evaluation for faster processing")
    print("3. Optimized batch processing")
    print("4. Model loading optimization with caching")
    print("5. Performance tracking and monitoring")

    try:
        # Note: These examples use mock models for demonstration
        # In real usage, replace with actual model names
        
        example_profiling()
        example_parallel_evaluation()
        example_batch_optimization()
        example_model_caching()
        example_performance_tracking()

    except Exception as e:
        logger.error(f"Example failed: {e}", exc_info=True)
        print(f"\nNote: These examples require actual models to be available.")
        print(f"Replace 'mock-*' model names with real model identifiers.")

    print("\n" + "=" * 80)
    print("Examples complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
