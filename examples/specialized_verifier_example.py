"""
Example demonstrating the SpecializedVerifier component.

This example shows how to use the SpecializedVerifier for statement-level
fact-checking with three-way classification (SUPPORTED, REFUTED, NOT_ENOUGH_INFO).
"""

from llm_judge_auditor.components import ModelManager, SpecializedVerifier
from llm_judge_auditor.config import ToolkitConfig
from llm_judge_auditor.models import Passage


def main():
    """Demonstrate SpecializedVerifier usage."""
    print("=" * 70)
    print("Specialized Verifier Example")
    print("=" * 70)
    print()

    # Create a minimal config for the verifier
    # Note: In a real scenario, you would specify an actual verifier model
    config = ToolkitConfig(
        verifier_model="google/flan-t5-base",  # Example model
        judge_models=[],  # Not needed for this example
        quantize=True,
        device="auto",
    )

    print("1. Initializing Model Manager...")
    model_manager = ModelManager(config)

    print("2. Loading verifier model...")
    try:
        model, tokenizer = model_manager.load_verifier()
        print(f"   ✓ Verifier loaded successfully")
    except Exception as e:
        print(f"   ✗ Could not load model: {e}")
        print("   Note: This example requires a valid verifier model to be available.")
        return

    print()
    print("3. Creating SpecializedVerifier...")
    verifier = SpecializedVerifier(
        model=model,
        tokenizer=tokenizer,
        device=config.device.value,
        max_length=512,
        batch_size=4,
    )
    print("   ✓ SpecializedVerifier created")

    print()
    print("4. Example 1: Single Statement Verification")
    print("-" * 70)
    
    statement = "The Eiffel Tower was completed in 1889."
    context = "The Eiffel Tower was built for the 1889 World's Fair in Paris, France."
    
    print(f"Statement: {statement}")
    print(f"Context: {context}")
    print()
    
    verdict = verifier.verify_statement(statement, context)
    
    print(f"Result:")
    print(f"  Label: {verdict.label.value}")
    print(f"  Confidence: {verdict.confidence:.2f}")
    print(f"  Reasoning: {verdict.reasoning}")

    print()
    print("5. Example 2: Verification with Retrieved Passages")
    print("-" * 70)
    
    statement = "Paris is the capital of France."
    context = "France is a country in Western Europe."
    passages = [
        Passage(
            text="Paris is the capital and most populous city of France.",
            source="Wikipedia:Paris",
            relevance_score=0.95,
        ),
        Passage(
            text="The city of Paris is located in northern France.",
            source="Wikipedia:Geography_of_France",
            relevance_score=0.82,
        ),
    ]
    
    print(f"Statement: {statement}")
    print(f"Context: {context}")
    print(f"Retrieved Passages: {len(passages)}")
    print()
    
    verdict = verifier.verify_statement(statement, context, passages)
    
    print(f"Result:")
    print(f"  Label: {verdict.label.value}")
    print(f"  Confidence: {verdict.confidence:.2f}")
    print(f"  Evidence: {len(verdict.evidence)} passages")
    print(f"  Reasoning: {verdict.reasoning}")

    print()
    print("6. Example 3: Batch Verification")
    print("-" * 70)
    
    statements = [
        "The Eiffel Tower is in Paris.",
        "The Eiffel Tower was built in 1850.",
        "The Eiffel Tower is made of iron.",
    ]
    contexts = [
        "The Eiffel Tower is located in Paris, France.",
        "The Eiffel Tower was completed in 1889.",
        "The Eiffel Tower is a wrought iron lattice tower.",
    ]
    
    print(f"Verifying {len(statements)} statements...")
    print()
    
    verdicts = verifier.batch_verify(statements, contexts)
    
    for i, (stmt, verdict) in enumerate(zip(statements, verdicts), 1):
        print(f"{i}. {stmt}")
        print(f"   → {verdict.label.value} (confidence: {verdict.confidence:.2f})")

    print()
    print("7. Example 4: Full Text Verification")
    print("-" * 70)
    
    candidate_text = (
        "The Eiffel Tower is located in Paris, France. "
        "It was completed in 1889 for the World's Fair. "
        "The tower is approximately 300 meters tall."
    )
    source_context = (
        "The Eiffel Tower is a wrought iron lattice tower in Paris, France. "
        "It was completed in 1889 and stands at 330 meters tall including antennas."
    )
    
    print(f"Candidate Text: {candidate_text}")
    print(f"Source Context: {source_context}")
    print()
    
    verdicts = verifier.verify_text(candidate_text, source_context)
    
    print(f"Extracted and verified {len(verdicts)} statements:")
    print()
    
    for i, verdict in enumerate(verdicts, 1):
        print(f"{i}. Label: {verdict.label.value}")
        print(f"   Confidence: {verdict.confidence:.2f}")
        print(f"   Reasoning: {verdict.reasoning[:100]}...")
        print()

    print()
    print("=" * 70)
    print("Example completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
