"""
Example demonstrating batch evaluation with the LLM Judge Auditor toolkit.

This example shows how to:
1. Create multiple evaluation requests
2. Process them in batch with error resilience
3. Access batch statistics
4. Save results to JSON
"""

from llm_judge_auditor import EvaluationToolkit, EvaluationRequest


def main():
    """Run batch evaluation example."""
    print("=" * 80)
    print("Batch Processing Example")
    print("=" * 80)

    # Create toolkit with fast preset for quick demonstration
    print("\n1. Initializing toolkit with 'fast' preset...")
    toolkit = EvaluationToolkit.from_preset("fast")

    # Create multiple evaluation requests
    print("\n2. Creating batch of evaluation requests...")
    requests = [
        EvaluationRequest(
            source_text="Paris is the capital of France. It is located on the Seine River.",
            candidate_output="Paris is the capital of France and sits on the Seine River.",
            task="factual_accuracy",
        ),
        EvaluationRequest(
            source_text="The Earth orbits the Sun once every 365.25 days.",
            candidate_output="The Sun orbits the Earth every year.",
            task="factual_accuracy",
        ),
        EvaluationRequest(
            source_text="Water boils at 100 degrees Celsius at sea level.",
            candidate_output="Water boils at 100°C at standard atmospheric pressure.",
            task="factual_accuracy",
        ),
        EvaluationRequest(
            source_text="The Great Wall of China was built over many centuries.",
            candidate_output="The Great Wall of China was completed in one year.",
            task="factual_accuracy",
        ),
        EvaluationRequest(
            source_text="Python is a high-level programming language.",
            candidate_output="Python is a programming language known for its simplicity.",
            task="factual_accuracy",
        ),
    ]

    print(f"   Created {len(requests)} evaluation requests")

    # Process batch with error resilience
    print("\n3. Processing batch (with error resilience enabled)...")
    batch_result = toolkit.batch_evaluate(
        requests=requests,
        continue_on_error=True,  # Continue even if some evaluations fail
    )

    # Display results
    print("\n4. Batch Processing Results:")
    print("-" * 80)
    print(f"Total requests:        {batch_result.metadata['total_requests']}")
    print(f"Successful:            {batch_result.metadata['successful_evaluations']}")
    print(f"Failed:                {batch_result.metadata['failed_evaluations']}")
    print(f"Success rate:          {batch_result.metadata['success_rate']:.1%}")

    # Display statistics
    print("\n5. Batch Statistics:")
    print("-" * 80)
    stats = batch_result.statistics
    print(f"Mean score:            {stats['mean']:.2f}")
    print(f"Median score:          {stats['median']:.2f}")
    print(f"Std deviation:         {stats['std']:.2f}")
    print(f"Min score:             {stats['min']:.2f}")
    print(f"Max score:             {stats['max']:.2f}")
    print(f"25th percentile:       {stats['p25']:.2f}")
    print(f"75th percentile:       {stats['p75']:.2f}")

    # Display individual results
    print("\n6. Individual Results:")
    print("-" * 80)
    for idx, result in enumerate(batch_result.results):
        print(f"\nRequest {idx + 1}:")
        print(f"  Source:     {result.request.source_text[:60]}...")
        print(f"  Candidate:  {result.request.candidate_output[:60]}...")
        print(f"  Score:      {result.consensus_score:.2f}")
        print(f"  Confidence: {result.report.confidence:.2f}")
        print(f"  Issues:     {len(result.flagged_issues)}")

    # Display errors if any
    if batch_result.errors:
        print("\n7. Errors:")
        print("-" * 80)
        for error in batch_result.errors:
            print(f"\nRequest {error['request_index'] + 1}:")
            print(f"  Error type: {error['error_type']}")
            print(f"  Message:    {error['error_message']}")

    # Save results to JSON
    print("\n8. Saving results to JSON...")
    output_file = "batch_results.json"
    batch_result.save_to_file(output_file)
    print(f"   Results saved to: {output_file}")

    # Demonstrate accessing specific result details
    print("\n9. Detailed Analysis of First Result:")
    print("-" * 80)
    if batch_result.results:
        first_result = batch_result.results[0]
        print(f"Consensus score:       {first_result.consensus_score:.2f}")
        print(f"Verifier verdicts:     {len(first_result.verifier_verdicts)}")
        print(f"Judge evaluations:     {len(first_result.judge_results)}")

        print("\nIndividual judge scores:")
        for judge_name, score in first_result.report.individual_scores.items():
            print(f"  {judge_name}: {score:.2f}")

        print("\nVerifier verdicts:")
        for verdict in first_result.verifier_verdicts[:3]:  # Show first 3
            print(f"  {verdict.label.value}: confidence={verdict.confidence:.2f}")

    print("\n" + "=" * 80)
    print("Batch processing example complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
