"""
Example demonstrating the usage of core data models and JSON serialization.

This script shows how to create evaluation results and serialize them to JSON.
"""

import json
from datetime import datetime

from llm_judge_auditor.models import (
    AggregationMetadata,
    Claim,
    ClaimType,
    EvaluationRequest,
    EvaluationResult,
    Issue,
    IssueType,
    IssueSeverity,
    JudgeResult,
    Passage,
    Report,
    Verdict,
    VerdictLabel,
)


def create_sample_evaluation():
    """Create a sample evaluation result demonstrating all data models."""

    # Create an evaluation request
    request = EvaluationRequest(
        source_text="The Earth orbits the Sun and takes approximately 365.25 days to complete one orbit.",
        candidate_output="The Earth revolves around the Sun in about 365 days, which is why we have leap years.",
        task="factual_accuracy",
        criteria=["correctness", "completeness"],
        use_retrieval=True,
    )

    # Create claims extracted from the candidate output
    claim1 = Claim(
        text="The Earth revolves around the Sun",
        source_span=(0, 37),
        claim_type=ClaimType.FACTUAL,
    )

    claim2 = Claim(
        text="Earth's orbit takes about 365 days",
        source_span=(41, 76),
        claim_type=ClaimType.NUMERICAL,
    )

    # Create retrieved passages
    passage1 = Passage(
        text="Earth orbits the Sun at an average distance of about 150 million kilometers.",
        source="Wikipedia:Earth",
        relevance_score=0.92,
    )

    passage2 = Passage(
        text="A year on Earth is approximately 365.25 days, which is why we add a leap day every four years.",
        source="Wikipedia:Year",
        relevance_score=0.88,
    )

    # Create verifier verdicts
    verdict1 = Verdict(
        label=VerdictLabel.SUPPORTED,
        confidence=0.98,
        evidence=["Earth orbits the Sun - confirmed by astronomical observations"],
        reasoning="The claim about Earth orbiting the Sun is scientifically accurate and well-established.",
    )

    verdict2 = Verdict(
        label=VerdictLabel.SUPPORTED,
        confidence=0.95,
        evidence=["365.25 days orbital period - confirmed"],
        reasoning="The orbital period is approximately correct, though slightly simplified.",
    )

    # Create issues detected
    issue1 = Issue(
        type=IssueType.NUMERICAL_ERROR,
        severity=IssueSeverity.LOW,
        description="Minor simplification: 365 vs 365.25 days",
        evidence=["Source mentions 365.25 days, candidate says 365 days"],
    )

    # Create judge results
    judge1_result = JudgeResult(
        model_name="llama-3-8b",
        score=92.0,
        reasoning="The candidate output is factually accurate. It correctly states that Earth orbits the Sun and provides a reasonable approximation of the orbital period. The mention of leap years shows understanding of the 365.25 day cycle. Minor point deduction for slight simplification of the exact period.",
        flagged_issues=[issue1],
        confidence=0.90,
    )

    judge2_result = JudgeResult(
        model_name="mistral-7b",
        score=94.0,
        reasoning="Excellent response that captures the key facts accurately. The explanation of leap years demonstrates good understanding of why the orbital period matters. Very minor simplification of 365.25 to 365 days is acceptable in context.",
        flagged_issues=[],
        confidence=0.92,
    )

    # Create aggregation metadata
    aggregation_metadata = AggregationMetadata(
        strategy="mean",
        individual_scores={"llama-3-8b": 92.0, "mistral-7b": 94.0},
        variance=2.0,
        is_low_confidence=False,
        weights=None,
    )

    # Create comprehensive report
    report = Report(
        metadata={
            "timestamp": datetime.now().isoformat(),
            "models": ["llama-3-8b", "mistral-7b"],
            "verifier": "MiniCheck/flan-t5-large",
            "version": "1.0.0",
            "retrieval_enabled": True,
        },
        consensus_score=93.0,
        individual_scores={"llama-3-8b": 92.0, "mistral-7b": 94.0},
        verifier_verdicts=[verdict1, verdict2],
        retrieval_provenance=[passage1, passage2],
        reasoning={
            "llama-3-8b": judge1_result.reasoning,
            "mistral-7b": judge2_result.reasoning,
        },
        confidence=0.91,
        disagreement_level=2.0,
        flagged_issues=[issue1],
        hallucination_categories={"numerical_error": 1},
    )

    # Create final evaluation result
    result = EvaluationResult(
        request=request,
        consensus_score=93.0,
        verifier_verdicts=[verdict1, verdict2],
        judge_results=[judge1_result, judge2_result],
        aggregation_metadata=aggregation_metadata,
        report=report,
        flagged_issues=[issue1],
    )

    return result


def main():
    """Main function demonstrating data model usage."""

    print("=" * 80)
    print("LLM Judge Auditor - Data Models Example")
    print("=" * 80)
    print()

    # Create a sample evaluation
    print("Creating sample evaluation result...")
    result = create_sample_evaluation()
    print("✓ Evaluation result created")
    print()

    # Display key information
    print("Evaluation Summary:")
    print(f"  Consensus Score: {result.consensus_score}/100")
    print(f"  Number of Judges: {len(result.judge_results)}")
    print(f"  Number of Verdicts: {len(result.verifier_verdicts)}")
    print(f"  Flagged Issues: {len(result.flagged_issues)}")
    print()

    # Serialize to JSON
    print("Serializing to JSON...")
    json_str = result.to_json(indent=2)
    print("✓ JSON serialization successful")
    print()

    # Show a snippet of the JSON
    print("JSON Output (first 500 characters):")
    print("-" * 80)
    print(json_str[:500] + "...")
    print("-" * 80)
    print()

    # Deserialize from JSON
    print("Deserializing from JSON...")
    restored_result = EvaluationResult.from_json(json_str)
    print("✓ JSON deserialization successful")
    print()

    # Verify round-trip
    print("Verifying round-trip integrity...")
    assert restored_result.consensus_score == result.consensus_score
    assert len(restored_result.judge_results) == len(result.judge_results)
    assert restored_result.judge_results[0].model_name == result.judge_results[0].model_name
    assert restored_result.verifier_verdicts[0].label == result.verifier_verdicts[0].label
    print("✓ Round-trip verification passed")
    print()

    # Save to file
    output_file = "evaluation_result.json"
    print(f"Saving to {output_file}...")
    with open(output_file, "w") as f:
        f.write(json_str)
    print(f"✓ Saved to {output_file}")
    print()

    # Load from file
    print(f"Loading from {output_file}...")
    with open(output_file, "r") as f:
        loaded_json = f.read()
    loaded_result = EvaluationResult.from_json(loaded_json)
    print("✓ Loaded from file successfully")
    print()

    # Display detailed information
    print("Detailed Evaluation Information:")
    print("-" * 80)
    print(f"Source Text: {result.request.source_text}")
    print()
    print(f"Candidate Output: {result.request.candidate_output}")
    print()
    print("Judge Scores:")
    for judge_result in result.judge_results:
        print(f"  - {judge_result.model_name}: {judge_result.score}/100")
    print()
    print("Verifier Verdicts:")
    for i, verdict in enumerate(result.verifier_verdicts, 1):
        print(f"  {i}. {verdict.label.value} (confidence: {verdict.confidence:.2f})")
    print()
    print("Flagged Issues:")
    for issue in result.flagged_issues:
        print(f"  - {issue.type.value} ({issue.severity.value}): {issue.description}")
    print("-" * 80)
    print()

    print("Example completed successfully!")


if __name__ == "__main__":
    main()
