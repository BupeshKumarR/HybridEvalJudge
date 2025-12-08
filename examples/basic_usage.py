"""
Basic usage example for LLM Judge Auditor toolkit.

This example demonstrates how to:
1. Load a preset configuration
2. Create custom configurations
3. Use the data models
"""

from llm_judge_auditor import (
    ToolkitConfig,
    AggregationStrategy,
    EvaluationRequest,
    Claim,
    ClaimType,
)


def main():
    print("=" * 60)
    print("LLM Judge Auditor - Basic Usage Example")
    print("=" * 60)

    # Example 1: Load a preset configuration
    print("\n1. Loading preset configurations:")
    print("-" * 40)

    fast_config = ToolkitConfig.from_preset("fast")
    print(f"Fast preset: {len(fast_config.judge_models)} judge(s), retrieval={fast_config.enable_retrieval}")

    balanced_config = ToolkitConfig.from_preset("balanced")
    print(f"Balanced preset: {len(balanced_config.judge_models)} judge(s), retrieval={balanced_config.enable_retrieval}")

    # Example 2: Create a custom configuration
    print("\n2. Creating custom configuration:")
    print("-" * 40)

    custom_config = ToolkitConfig(
        verifier_model="custom-verifier",
        judge_models=["judge-1", "judge-2"],
        quantize=True,
        enable_retrieval=True,
        aggregation_strategy=AggregationStrategy.WEIGHTED_AVERAGE,
        judge_weights={"judge-1": 0.6, "judge-2": 0.4},
    )
    print(f"Custom config created with {custom_config.aggregation_strategy} aggregation")

    # Example 3: Create an evaluation request
    print("\n3. Creating evaluation request:")
    print("-" * 40)

    request = EvaluationRequest(
        source_text="The Eiffel Tower is located in Paris, France.",
        candidate_output="The Eiffel Tower is in Paris.",
        task="factual_accuracy",
        criteria=["correctness", "completeness"],
    )
    print(f"Request created for task: {request.task}")
    print(f"Criteria: {', '.join(request.criteria)}")

    # Example 4: Create claims
    print("\n4. Creating claims:")
    print("-" * 40)

    claim1 = Claim(
        text="Paris is the capital of France.",
        source_span=(0, 31),
        claim_type=ClaimType.FACTUAL,
    )
    print(f"Claim: '{claim1.text}' (type: {claim1.claim_type})")

    claim2 = Claim(
        text="The tower was completed in 1889.",
        source_span=(32, 64),
        claim_type=ClaimType.TEMPORAL,
    )
    print(f"Claim: '{claim2.text}' (type: {claim2.claim_type})")

    print("\n" + "=" * 60)
    print("Setup complete! Ready to implement evaluation components.")
    print("=" * 60)


if __name__ == "__main__":
    main()
