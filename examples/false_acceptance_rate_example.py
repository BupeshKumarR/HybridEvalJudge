"""
Example demonstrating False Acceptance Rate (FAR) computation.

This script shows how to use the FalseAcceptanceCalculator to measure
model abstention behavior on queries about non-existent entities:
- Evaluating individual query-response pairs
- Computing FAR across multiple queries
- Customizing abstention detection patterns

Validates: Requirements 20.1, 20.2, 20.3, 20.4
"""

from llm_judge_auditor.components.hallucination_metrics import (
    FalseAcceptanceCalculator,
)


def main():
    """Demonstrate False Acceptance Rate computation."""
    print("=" * 70)
    print("False Acceptance Rate (FAR) Example")
    print("=" * 70)

    # Initialize calculator with default abstention patterns
    calculator = FalseAcceptanceCalculator()

    # =========================================================================
    # 1. Basic Abstention Detection
    # =========================================================================
    print("\n1. Basic Abstention Detection")
    print("-" * 70)
    print("Detecting when a model correctly refuses to answer")
    print()

    # Examples of abstention responses
    abstention_responses = [
        "I don't know the answer to that question.",
        "I'm not sure about that. I don't have information on this topic.",
        "I cannot find any information about this.",
        "There is no record of this in my knowledge base.",
        "I'm unable to provide information on this topic.",
    ]

    print("Abstention responses (should be detected):")
    for response in abstention_responses:
        result = calculator.evaluate_abstention(
            query="Tell me about XYZ",
            response=response,
            is_nonexistent=True
        )
        status = "✓ Abstained" if result.did_abstain else "✗ Did not abstain"
        print(f"  [{status}] \"{response[:50]}...\"")

    # Examples of non-abstention responses
    non_abstention_responses = [
        "XYZ is a famous scientist who discovered...",
        "The XYZ company was founded in 1985 and...",
        "According to my knowledge, XYZ refers to...",
    ]

    print("\nNon-abstention responses (should NOT be detected as abstention):")
    for response in non_abstention_responses:
        result = calculator.evaluate_abstention(
            query="Tell me about XYZ",
            response=response,
            is_nonexistent=True
        )
        status = "✓ Abstained" if result.did_abstain else "✗ Did not abstain"
        print(f"  [{status}] \"{response[:50]}...\"")

    # =========================================================================
    # 2. False Acceptance Detection
    # =========================================================================
    print("\n2. False Acceptance Detection")
    print("-" * 70)
    print("False acceptance = model generates content for non-existent entity")
    print()

    # Query about a non-existent entity
    result = calculator.evaluate_abstention(
        query="Who is Dr. Zephyr Moonstone?",
        response="Dr. Zephyr Moonstone is a renowned physicist who made significant contributions to quantum mechanics.",
        is_nonexistent=True
    )
    print(f"Query: \"{result.query}\"")
    print(f"Response: \"{result.response[:60]}...\"")
    print(f"Is non-existent entity: {result.is_nonexistent_entity}")
    print(f"Did abstain: {result.did_abstain}")
    print(f"Is false acceptance: {result.is_false_acceptance}")

    # Correct abstention for non-existent entity
    result2 = calculator.evaluate_abstention(
        query="Who is Dr. Zephyr Moonstone?",
        response="I don't have any information about Dr. Zephyr Moonstone. This person doesn't appear to exist in my knowledge base.",
        is_nonexistent=True
    )
    print(f"\nQuery: \"{result2.query}\"")
    print(f"Response: \"{result2.response[:60]}...\"")
    print(f"Did abstain: {result2.did_abstain}")
    print(f"Is false acceptance: {result2.is_false_acceptance}")

    # =========================================================================
    # 3. Computing FAR Across Multiple Queries
    # =========================================================================
    print("\n3. Computing FAR Across Multiple Queries")
    print("-" * 70)
    print("FAR = failed_abstentions / total_nonexistent_queries")
    print()

    # Simulate a batch of queries
    queries = [
        "Who is Dr. Zephyr Moonstone?",
        "What is the Chromatic Resonance Theory?",
        "Tell me about the Battle of Silverdale",
        "Who invented the Quantum Flux Capacitor?",
        "What is the capital of France?",  # Real entity
    ]
    
    responses = [
        "Dr. Zephyr Moonstone is a physicist...",  # False acceptance
        "I don't know about the Chromatic Resonance Theory. I cannot find any information.",  # Correct abstention
        "The Battle of Silverdale occurred in 1847...",  # False acceptance
        "I'm not familiar with the Quantum Flux Capacitor. There is no record of this.",  # Correct abstention
        "The capital of France is Paris.",  # Real entity, correct response
    ]
    
    is_nonexistent = [True, True, True, True, False]

    far_result = calculator.evaluate_and_compute_far(
        queries=queries,
        responses=responses,
        is_nonexistent_flags=is_nonexistent
    )

    print(f"Total queries: {len(queries)}")
    print(f"Non-existent entity queries: {far_result.total_nonexistent_queries}")
    print(f"Correct refusals: {far_result.correct_refusals}")
    print(f"Failed abstentions: {far_result.failed_abstentions}")
    print(f"False Acceptance Rate: {far_result.value:.2%}")

    print("\nDetailed results:")
    for result in far_result.abstention_results:
        if result.is_nonexistent_entity:
            status = "✓ Correct refusal" if result.did_abstain else "✗ False acceptance"
            print(f"  [{status}] \"{result.query}\"")

    # =========================================================================
    # 4. Perfect Abstention Behavior
    # =========================================================================
    print("\n4. Perfect Abstention Behavior")
    print("-" * 70)

    perfect_queries = [
        "Who is the fictional character Zyx?",
        "What is the made-up Blorp Protocol?",
    ]
    perfect_responses = [
        "I don't know anything about Zyx. This doesn't appear to be a real character.",
        "I'm not familiar with the Blorp Protocol. I cannot find any information about it.",
    ]
    perfect_flags = [True, True]

    perfect_far = calculator.evaluate_and_compute_far(
        queries=perfect_queries,
        responses=perfect_responses,
        is_nonexistent_flags=perfect_flags
    )

    print(f"FAR: {perfect_far.value:.2%}")
    print(f"Correct refusals: {perfect_far.correct_refusals}/{perfect_far.total_nonexistent_queries}")
    print("(Lower FAR is better - 0% means perfect abstention)")

    # =========================================================================
    # 5. Poor Abstention Behavior
    # =========================================================================
    print("\n5. Poor Abstention Behavior")
    print("-" * 70)

    poor_queries = [
        "Who is Dr. Imaginary Person?",
        "What is the Fake Theory of Everything?",
        "Tell me about the Non-existent Event",
    ]
    poor_responses = [
        "Dr. Imaginary Person was a scientist who...",
        "The Fake Theory of Everything states that...",
        "The Non-existent Event happened in 1900...",
    ]
    poor_flags = [True, True, True]

    poor_far = calculator.evaluate_and_compute_far(
        queries=poor_queries,
        responses=poor_responses,
        is_nonexistent_flags=poor_flags
    )

    print(f"FAR: {poor_far.value:.2%}")
    print(f"Failed abstentions: {poor_far.failed_abstentions}/{poor_far.total_nonexistent_queries}")
    print("(Higher FAR indicates model is prone to hallucination)")

    # =========================================================================
    # 6. Custom Abstention Patterns
    # =========================================================================
    print("\n6. Custom Abstention Patterns")
    print("-" * 70)

    # Create calculator with custom patterns
    custom_patterns = [
        "no data available",
        "cannot verify",
        "unverified claim",
        "no reliable source",
    ]
    custom_calculator = FalseAcceptanceCalculator(abstention_patterns=custom_patterns)

    # Test with custom pattern
    custom_result = custom_calculator.evaluate_abstention(
        query="What is the XYZ phenomenon?",
        response="There is no data available about the XYZ phenomenon.",
        is_nonexistent=True
    )
    print(f"Custom pattern detected abstention: {custom_result.did_abstain}")

    # Add a new pattern dynamically
    custom_calculator.add_abstention_pattern("insufficient evidence")
    result_after_add = custom_calculator.evaluate_abstention(
        query="What is ABC?",
        response="There is insufficient evidence to answer this question.",
        is_nonexistent=True
    )
    print(f"After adding pattern, detected abstention: {result_after_add.did_abstain}")

    # =========================================================================
    # 7. Case Sensitivity
    # =========================================================================
    print("\n7. Case Sensitivity")
    print("-" * 70)

    # Default: case-insensitive
    case_insensitive = FalseAcceptanceCalculator(case_sensitive=False)
    result_lower = case_insensitive.evaluate_abstention(
        query="Test",
        response="I DON'T KNOW THE ANSWER",
        is_nonexistent=True
    )
    print(f"Case-insensitive (default): detected = {result_lower.did_abstain}")

    # Case-sensitive
    case_sensitive = FalseAcceptanceCalculator(case_sensitive=True)
    result_upper = case_sensitive.evaluate_abstention(
        query="Test",
        response="I DON'T KNOW THE ANSWER",
        is_nonexistent=True
    )
    print(f"Case-sensitive: detected = {result_upper.did_abstain}")

    # =========================================================================
    # 8. Mixed Query Types
    # =========================================================================
    print("\n8. Mixed Query Types (Real and Non-existent)")
    print("-" * 70)

    mixed_queries = [
        ("Who is Albert Einstein?", "Albert Einstein was a physicist...", False),
        ("Who is Dr. Fake Person?", "Dr. Fake Person invented...", True),
        ("What is gravity?", "Gravity is a fundamental force...", False),
        ("What is the Imaginary Force?", "I don't know about that.", True),
    ]

    results = []
    for query, response, is_fake in mixed_queries:
        result = calculator.evaluate_abstention(query, response, is_fake)
        results.append(result)

    far = calculator.compute_far(results)
    
    print(f"Total queries: {len(mixed_queries)}")
    print(f"Real entity queries: {sum(1 for _, _, f in mixed_queries if not f)}")
    print(f"Non-existent entity queries: {far.total_nonexistent_queries}")
    print(f"FAR (for non-existent only): {far.value:.2%}")

    print("\n" + "=" * 70)
    print("Example complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
