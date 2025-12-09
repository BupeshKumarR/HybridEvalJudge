"""
Example demonstrating the StreamingEvaluator for large documents.

This example shows how to use the StreamingEvaluator to process large documents
that cannot fit into memory at once. The evaluator chunks documents into
manageable segments and processes them incrementally.
"""

import io
import logging

from llm_judge_auditor import EvaluationToolkit, StreamingEvaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def example_streaming_evaluation():
    """
    Demonstrate streaming evaluation with large documents.
    """
    print("=" * 80)
    print("Streaming Evaluator Example")
    print("=" * 80)

    # Initialize toolkit with a preset
    print("\n1. Initializing EvaluationToolkit with 'fast' preset...")
    toolkit = EvaluationToolkit.from_preset("fast")

    # Create streaming evaluator
    print("\n2. Creating StreamingEvaluator with chunk_size=512, overlap=50...")
    streaming = StreamingEvaluator(toolkit, chunk_size=512, overlap=50)

    # Create sample large documents
    print("\n3. Creating sample large documents...")

    # Source text: A long article about the Eiffel Tower
    source_text = """
    The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France.
    It is named after the engineer Gustave Eiffel, whose company designed and built the tower.
    
    Construction of the tower began in 1887 and was completed in 1889. The tower was built
    as the entrance arch to the 1889 World's Fair, marking the centennial celebration of the
    French Revolution. The tower is 330 meters (1,083 ft) tall, about the same height as an
    81-story building.
    
    During its construction, the Eiffel Tower surpassed the Washington Monument to become
    the tallest man-made structure in the world, a title it held for 41 years until the
    Chrysler Building in New York City was finished in 1930.
    
    The tower has three levels for visitors, with restaurants on the first and second levels.
    The top level's upper platform is 276 m (906 ft) above the ground – the highest
    observation deck accessible to the public in the European Union.
    
    The Eiffel Tower is the most-visited paid monument in the world; 6.91 million people
    ascended it in 2015. The tower received its 250 millionth visitor in 2010.
    
    The tower has been used for various purposes throughout its history, including as a
    radio transmission tower. It has also been featured in numerous films, television shows,
    and works of art, making it one of the most recognizable structures in the world.
    """ * 3  # Repeat to make it longer

    # Candidate text: A response with some accurate and some inaccurate information
    candidate_text = """
    The Eiffel Tower is located in Paris, France, and was completed in 1889. It was designed
    by Gustave Eiffel's company for the 1889 World's Fair. The tower stands at 330 meters tall.
    
    The Eiffel Tower was the tallest structure in the world when it was completed, surpassing
    the Washington Monument. It held this record until 1930 when the Chrysler Building was
    completed in New York.
    
    The tower has three levels for visitors and includes restaurants. The top observation
    deck is 276 meters above the ground, making it the highest public observation deck in
    the European Union.
    
    Millions of people visit the Eiffel Tower each year, making it one of the most popular
    paid monuments in the world. The tower has been used for radio transmission and has
    appeared in many films and artworks.
    
    The tower's construction began in 1887 and took two years to complete. It was initially
    criticized by some of Paris's leading artists and intellectuals, but has since become
    a global cultural icon of France.
    """ * 3  # Repeat to make it longer

    print(f"   Source text length: {len(source_text)} characters")
    print(f"   Candidate text length: {len(candidate_text)} characters")

    # Create streams from the text
    source_stream = io.StringIO(source_text)
    candidate_stream = io.StringIO(candidate_text)

    # Evaluate using streaming
    print("\n4. Evaluating streams (this may take a moment)...")
    result = streaming.evaluate_stream(
        source_stream=source_stream,
        candidate_stream=candidate_stream,
        task="factual_accuracy",
        criteria=["correctness"],
        use_retrieval=False,
    )

    # Display results
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)

    print(f"\nConsensus Score: {result.consensus_score:.2f}/100")
    print(f"Confidence: {result.report.confidence:.2f}")
    print(f"Disagreement Level: {result.report.disagreement_level:.2f}")

    print(f"\nNumber of chunks processed: {result.report.metadata.get('num_chunks', 'N/A')}")
    print(
        f"Total characters processed: {result.report.metadata.get('total_characters', 'N/A')}"
    )

    print("\nIndividual Judge Scores:")
    for model_name, score in result.report.individual_scores.items():
        print(f"  - {model_name}: {score:.2f}")

    print(f"\nVerifier Verdicts: {len(result.verifier_verdicts)}")
    for i, verdict in enumerate(result.verifier_verdicts[:3], 1):  # Show first 3
        print(f"  {i}. {verdict.label.value} (confidence: {verdict.confidence:.2f})")
        if verdict.reasoning:
            print(f"     Reasoning: {verdict.reasoning[:100]}...")

    print(f"\nFlagged Issues: {len(result.flagged_issues)}")
    for i, issue in enumerate(result.flagged_issues[:3], 1):  # Show first 3
        print(f"  {i}. {issue.type.value} ({issue.severity.value})")
        print(f"     {issue.description[:100]}...")

    print("\nHallucination Categories:")
    for category, count in result.report.hallucination_categories.items():
        if count > 0:
            print(f"  - {category}: {count}")

    print("\n" + "=" * 80)


def example_streaming_with_files():
    """
    Demonstrate streaming evaluation with actual files.
    """
    print("\n" + "=" * 80)
    print("Streaming Evaluation with Files")
    print("=" * 80)

    print("\nNote: This example would work with actual files:")
    print("""
    toolkit = EvaluationToolkit.from_preset("balanced")
    streaming = StreamingEvaluator(toolkit, chunk_size=1024, overlap=100)
    
    with open("large_source.txt") as source_file, \\
         open("large_candidate.txt") as candidate_file:
        result = streaming.evaluate_stream(
            source_stream=source_file,
            candidate_stream=candidate_file
        )
    
    print(f"Final Score: {result.consensus_score:.2f}")
    
    # Save results
    with open("streaming_results.json", "w") as f:
        f.write(result.to_json())
    """)


def example_custom_chunk_size():
    """
    Demonstrate using different chunk sizes and overlap settings.
    """
    print("\n" + "=" * 80)
    print("Custom Chunk Size Configuration")
    print("=" * 80)

    print("\nYou can configure chunk size and overlap based on your needs:")
    print("""
    # Small chunks for fine-grained evaluation
    streaming_small = StreamingEvaluator(toolkit, chunk_size=256, overlap=25)
    
    # Large chunks for faster processing
    streaming_large = StreamingEvaluator(toolkit, chunk_size=2048, overlap=200)
    
    # No overlap for maximum speed (may lose context at boundaries)
    streaming_no_overlap = StreamingEvaluator(toolkit, chunk_size=1024, overlap=0)
    
    # High overlap for maximum context preservation
    streaming_high_overlap = StreamingEvaluator(toolkit, chunk_size=512, overlap=128)
    """)

    print("\nRecommendations:")
    print("  - chunk_size=512, overlap=50: Good balance (default)")
    print("  - chunk_size=256, overlap=25: Fine-grained, slower")
    print("  - chunk_size=1024, overlap=100: Faster, less granular")
    print("  - overlap should be ~10% of chunk_size for good context")


if __name__ == "__main__":
    try:
        # Run the main example
        example_streaming_evaluation()

        # Show file-based example
        example_streaming_with_files()

        # Show configuration options
        example_custom_chunk_size()

        print("\n" + "=" * 80)
        print("Example completed successfully!")
        print("=" * 80)

    except Exception as e:
        logger.error(f"Example failed: {str(e)}", exc_info=True)
        print(f"\nError: {str(e)}")
        print("Note: This example requires the evaluation toolkit to be properly configured.")
