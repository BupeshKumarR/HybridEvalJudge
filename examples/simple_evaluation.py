"""
Simple evaluation example for LLM Judge Auditor toolkit.

This example demonstrates the most basic usage:
1. Initialize the toolkit with a preset
2. Evaluate a candidate output against source text
3. View the results

This is the recommended starting point for new users.
"""

from llm_judge_auditor import EvaluationToolkit


def main():
    """Run a simple evaluation example."""
    print("=" * 80)
    print("Simple Evaluation Example")
    print("=" * 80)

    # Step 1: Initialize toolkit with a preset
    print("\n1. Initializing toolkit with 'fast' preset...")
    print("   (This uses minimal resources for quick evaluation)")
    toolkit = EvaluationToolkit.from_preset("fast")
    print("   ✓ Toolkit initialized")

    # Step 2: Define source text and candidate output
    print("\n2. Setting up evaluation...")
    source_text = """
    The Eiffel Tower is a wrought-iron lattice tower located in Paris, France.
    It was designed by engineer Gustave Eiffel and completed in 1889.
    The tower stands 330 meters tall and was the world's tallest structure until 1930.
    """

    candidate_output = """
    The Eiffel Tower is in Paris and was built by Gustave Eiffel in 1889.
    It is 330 meters tall.
    """

    print("   Source text: Facts about the Eiffel Tower")
    print("   Candidate output: Summary of those facts")

    # Step 3: Evaluate
    print("\n3. Running evaluation...")
    result = toolkit.evaluate(
        source_text=source_text,
        candidate_output=candidate_output,
    )
    print("   ✓ Evaluation complete")

    # Step 4: Display results
    print("\n4. Results:")
    print("-" * 80)
    print(f"   Consensus Score:    {result.consensus_score:.2f}/100")
    print(f"   Confidence:         {result.report.confidence:.2f}")
    print(f"   Disagreement:       {result.report.disagreement_level:.2f}")
    print(f"   Issues Found:       {len(result.flagged_issues)}")

    # Show individual judge scores
    print("\n   Individual Judge Scores:")
    for model_name, score in result.report.individual_scores.items():
        print(f"     • {model_name}: {score:.2f}")

    # Show verifier verdicts
    print(f"\n   Verifier Verdicts: {len(result.verifier_verdicts)} claims checked")
    for i, verdict in enumerate(result.verifier_verdicts[:3], 1):  # Show first 3
        print(f"     {i}. {verdict.label.value} (confidence: {verdict.confidence:.2f})")

    # Show any issues
    if result.flagged_issues:
        print("\n   Flagged Issues:")
        for issue in result.flagged_issues:
            print(f"     • [{issue.severity.value}] {issue.description}")
    else:
        print("\n   ✓ No issues detected - output appears factually accurate!")

    # Step 5: Export results (optional)
    print("\n5. Exporting results...")
    output_file = "simple_evaluation_result.json"
    result.save_to_file(output_file)
    print(f"   ✓ Results saved to: {output_file}")

    print("\n" + "=" * 80)
    print("Evaluation complete!")
    print("=" * 80)
    print("\nNext steps:")
    print("  • Try examples/batch_processing_example.py for multiple evaluations")
    print("  • See examples/evaluation_toolkit_example.py for advanced features")
    print("  • Read docs/CLI_USAGE.md for command-line usage")


if __name__ == "__main__":
    main()
