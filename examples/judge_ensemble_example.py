"""
Example usage of the JudgeEnsemble component.

This script demonstrates how to use the JudgeEnsemble for:
1. Single judge evaluation
2. Ensemble evaluation with multiple judges
3. Pairwise comparison of candidate outputs
"""

import logging

from llm_judge_auditor.components import JudgeEnsemble, ModelManager, PromptManager
from llm_judge_auditor.config import DeviceType, ToolkitConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def example_single_evaluation():
    """Example: Evaluate with a single judge model."""
    print("\n" + "=" * 80)
    print("Example 1: Single Judge Evaluation")
    print("=" * 80)

    # Create configuration
    config = ToolkitConfig(
        verifier_model=None,  # Not needed for this example
        judge_models=["microsoft/Phi-3-mini-4k-instruct"],
        quantize=True,
        device=DeviceType.AUTO,
    )

    # Initialize components
    model_manager = ModelManager(config)
    prompt_manager = PromptManager()

    # Load judge models
    print("\nLoading judge models...")
    model_manager.load_judge_ensemble()

    # Create ensemble
    ensemble = JudgeEnsemble(
        model_manager=model_manager,
        prompt_manager=prompt_manager,
        max_length=2048,
        temperature=0.1,
    )

    # Example evaluation
    source_text = """
    The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France.
    It is named after the engineer Gustave Eiffel, whose company designed and built the tower.
    Constructed from 1887 to 1889, it was initially criticized by some of France's leading
    artists and intellectuals for its design, but it has become a global cultural icon of
    France and one of the most recognizable structures in the world.
    """

    candidate_output = """
    The Eiffel Tower was built in Paris between 1887 and 1889. It was designed by
    Gustave Eiffel's company and is made of wrought iron. The tower is located on
    the Champ de Mars and has become an iconic symbol of France.
    """

    print("\nEvaluating candidate output...")
    result = ensemble.evaluate_single(
        judge_name="microsoft/Phi-3-mini-4k-instruct",
        source_text=source_text,
        candidate_output=candidate_output,
    )

    print(f"\nJudge: {result.model_name}")
    print(f"Score: {result.score:.1f}/100")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"\nReasoning:\n{result.reasoning}")
    print(f"\nFlagged Issues: {len(result.flagged_issues)}")
    for issue in result.flagged_issues:
        print(f"  - [{issue.severity.value}] {issue.description}")


def example_ensemble_evaluation():
    """Example: Evaluate with multiple judges (ensemble)."""
    print("\n" + "=" * 80)
    print("Example 2: Ensemble Evaluation")
    print("=" * 80)

    # Create configuration with multiple judges
    config = ToolkitConfig(
        verifier_model=None,
        judge_models=[
            "microsoft/Phi-3-mini-4k-instruct",
            # Add more models if you have sufficient resources:
            # "meta-llama/Meta-Llama-3-8B-Instruct",
            # "mistralai/Mistral-7B-Instruct-v0.2",
        ],
        quantize=True,
        device=DeviceType.AUTO,
    )

    # Initialize components
    model_manager = ModelManager(config)
    prompt_manager = PromptManager()

    # Load judge models
    print("\nLoading judge ensemble...")
    model_manager.load_judge_ensemble()

    # Create ensemble
    ensemble = JudgeEnsemble(
        model_manager=model_manager,
        prompt_manager=prompt_manager,
    )

    # Example with hallucination
    source_text = "The Great Wall of China was built over many centuries by various dynasties."

    candidate_output = """
    The Great Wall of China was built in 1850 by Emperor Napoleon Bonaparte.
    It stretches from Beijing to Moscow and is made entirely of marble.
    """

    print("\nEvaluating candidate with hallucinations...")
    results = ensemble.evaluate_all(
        source_text=source_text,
        candidate_output=candidate_output,
    )

    print(f"\nEnsemble Results ({len(results)} judges):")
    for result in results:
        print(f"\n  Judge: {result.model_name}")
        print(f"  Score: {result.score:.1f}/100")
        print(f"  Confidence: {result.confidence:.2f}")
        print(f"  Issues: {len(result.flagged_issues)}")

    # Calculate average score
    avg_score = sum(r.score for r in results) / len(results)
    print(f"\nAverage Score: {avg_score:.1f}/100")


def example_pairwise_comparison():
    """Example: Compare two candidate outputs."""
    print("\n" + "=" * 80)
    print("Example 3: Pairwise Comparison")
    print("=" * 80)

    # Create configuration
    config = ToolkitConfig(
        verifier_model=None,
        judge_models=["microsoft/Phi-3-mini-4k-instruct"],
        quantize=True,
        device=DeviceType.AUTO,
    )

    # Initialize components
    model_manager = ModelManager(config)
    prompt_manager = PromptManager()

    # Load judge models
    print("\nLoading judge models...")
    model_manager.load_judge_ensemble()

    # Create ensemble
    ensemble = JudgeEnsemble(
        model_manager=model_manager,
        prompt_manager=prompt_manager,
    )

    # Example comparison
    source_text = """
    Python was created by Guido van Rossum and first released in 1991.
    It emphasizes code readability with significant indentation.
    """

    candidate_a = """
    Python is a programming language created by Guido van Rossum in 1991.
    It is known for its emphasis on code readability and uses indentation
    to define code blocks.
    """

    candidate_b = """
    Python was invented by James Gosling in 1995. It is a compiled language
    that runs on the Java Virtual Machine and uses curly braces for code blocks.
    """

    print("\nComparing two candidates...")
    result = ensemble.pairwise_compare(
        source_text=source_text,
        candidate_a=candidate_a,
        candidate_b=candidate_b,
    )

    print(f"\nWinner: Candidate {result.winner}")
    print(f"\nReasoning:\n{result.reasoning}")


def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print("JudgeEnsemble Component Examples")
    print("=" * 80)
    print("\nThese examples demonstrate the JudgeEnsemble component.")
    print("Note: Examples require downloading models, which may take time.")
    print("Ensure you have sufficient disk space and memory.")

    try:
        # Run examples
        example_single_evaluation()
        example_ensemble_evaluation()
        example_pairwise_comparison()

        print("\n" + "=" * 80)
        print("All examples completed successfully!")
        print("=" * 80)

    except Exception as e:
        logger.error(f"Example failed: {e}", exc_info=True)
        print(f"\nError: {e}")
        print("\nNote: These examples require models to be downloaded.")
        print("If you encounter errors, ensure you have:")
        print("  1. Sufficient disk space (10+ GB)")
        print("  2. Sufficient memory (8+ GB RAM)")
        print("  3. Internet connection for model downloads")


if __name__ == "__main__":
    main()
