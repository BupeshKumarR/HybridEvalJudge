"""
Example demonstrating the ReportGenerator component.

This example shows how to:
1. Generate evaluation reports
2. Export reports to JSON format
3. Export reports to CSV format
4. Export reports to Markdown format
5. Export reports to plain text format
6. Generate retrieval provenance summaries
7. Generate hallucination summaries
"""

from pathlib import Path

from llm_judge_auditor.components.report_generator import ReportGenerator
from llm_judge_auditor.models import (
    Issue,
    IssueSeverity,
    IssueType,
    Passage,
    Report,
    Verdict,
    VerdictLabel,
)


def create_sample_report() -> Report:
    """Create a sample report for demonstration."""
    metadata = {
        "timestamp": "2024-01-01T12:00:00",
        "task": "factual_accuracy",
        "criteria": ["correctness"],
        "retrieval_enabled": True,
        "verifier_model": "minicheck-flan-t5-large",
        "judge_models": ["llama-3-8b", "mistral-7b"],
        "aggregation_strategy": "mean",
        "num_retrieved_passages": 2,
        "num_verifier_verdicts": 2,
        "num_judge_results": 2,
    }

    individual_scores = {
        "llama-3-8b": 75.0,
        "mistral-7b": 80.0,
    }

    verifier_verdicts = [
        Verdict(
            label=VerdictLabel.SUPPORTED,
            confidence=0.9,
            evidence=["The source text confirms this claim."],
            reasoning="The claim is directly supported by the source text.",
        ),
        Verdict(
            label=VerdictLabel.REFUTED,
            confidence=0.85,
            evidence=["The source text contradicts this claim."],
            reasoning="The claim contradicts information in the source text.",
        ),
    ]

    retrieval_provenance = [
        Passage(
            text="Paris is the capital and most populous city of France.",
            source="Wikipedia:Paris",
            relevance_score=0.95,
        ),
        Passage(
            text="France is a country primarily located in Western Europe.",
            source="Wikipedia:France",
            relevance_score=0.88,
        ),
    ]

    reasoning = {
        "llama-3-8b": "The candidate output contains mostly accurate information about Paris and France. However, there is one factual error regarding the population figure.",
        "mistral-7b": "The output is generally accurate and well-structured. Minor issues with specificity in some claims, but overall factually sound.",
    }

    flagged_issues = [
        Issue(
            type=IssueType.HALLUCINATION,
            severity=IssueSeverity.HIGH,
            description="Refuted claim: The claim contradicts information in the source text.",
            evidence=["The source text contradicts this claim."],
        ),
        Issue(
            type=IssueType.NUMERICAL_ERROR,
            severity=IssueSeverity.MEDIUM,
            description="Population figure is incorrect",
            evidence=["Source states 2.1 million, candidate states 3 million"],
        ),
    ]

    hallucination_categories = {
        "factual_error": 1,
        "unsupported_claim": 0,
        "temporal_inconsistency": 0,
        "numerical_error": 1,
        "bias": 0,
        "inconsistency": 0,
        "other": 0,
    }

    report = Report(
        metadata=metadata,
        consensus_score=77.5,
        individual_scores=individual_scores,
        verifier_verdicts=verifier_verdicts,
        retrieval_provenance=retrieval_provenance,
        reasoning=reasoning,
        confidence=0.85,
        disagreement_level=2.5,
        flagged_issues=flagged_issues,
        hallucination_categories=hallucination_categories,
    )

    return report


def main():
    """Run the report generator example."""
    print("=" * 60)
    print("Report Generator Example")
    print("=" * 60)
    print()

    # Create a sample report
    print("Creating sample evaluation report...")
    report = create_sample_report()
    print(f"✓ Report created with consensus score: {report.consensus_score:.2f}")
    print()

    # Initialize the report generator
    print("Initializing ReportGenerator...")
    generator = ReportGenerator()
    print("✓ ReportGenerator initialized")
    print()

    # Create output directory
    output_dir = Path("output/reports")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"✓ Output directory created: {output_dir}")
    print()

    # Export to JSON
    print("Exporting report to JSON...")
    json_path = output_dir / "evaluation_report.json"
    generator.export_json(report, str(json_path), indent=2)
    print(f"✓ JSON report saved to: {json_path}")
    print()

    # Export to Markdown
    print("Exporting report to Markdown...")
    md_path = output_dir / "evaluation_report.md"
    generator.export_markdown(report, str(md_path))
    print(f"✓ Markdown report saved to: {md_path}")
    print()

    # Export to plain text
    print("Exporting report to plain text...")
    txt_path = output_dir / "evaluation_report.txt"
    generator.export_text(report, str(txt_path))
    print(f"✓ Text report saved to: {txt_path}")
    print()

    # Export to CSV
    print("Exporting report to CSV...")
    csv_path = output_dir / "evaluation_report.csv"
    generator.export_csv(report, str(csv_path))
    print(f"✓ CSV report saved to: {csv_path}")
    print()

    # Get retrieval provenance summary
    print("Generating retrieval provenance summary...")
    provenance_summary = generator.get_retrieval_provenance_summary(report)
    print(f"✓ Retrieved {provenance_summary['total_passages']} passages from {len(provenance_summary['sources'])} sources")
    print(f"  - Average relevance score: {provenance_summary['avg_relevance_score']:.4f}")
    print(f"  - Min relevance score: {provenance_summary['min_relevance_score']:.4f}")
    print(f"  - Max relevance score: {provenance_summary['max_relevance_score']:.4f}")
    print()

    # Get hallucination summary
    print("Generating hallucination summary...")
    hallucination_summary = generator.get_hallucination_summary(report)
    print(f"✓ Found {hallucination_summary['total_hallucinations']} total hallucinations")
    print(f"  - Severity distribution:")
    for severity, count in hallucination_summary['severity_distribution'].items():
        if count > 0:
            print(f"    - {severity.title()}: {count}")
    print()

    # Display report summary
    print("=" * 60)
    print("Report Summary")
    print("=" * 60)
    print(f"Consensus Score: {report.consensus_score:.2f}/100")
    print(f"Confidence: {report.confidence:.2f}")
    print(f"Disagreement Level: {report.disagreement_level:.2f}")
    print()
    print("Individual Judge Scores:")
    for model_name, score in report.individual_scores.items():
        print(f"  - {model_name}: {score:.2f}/100")
    print()
    print(f"Verifier Verdicts: {len(report.verifier_verdicts)}")
    print(f"Retrieved Passages: {len(report.retrieval_provenance)}")
    print(f"Flagged Issues: {len(report.flagged_issues)}")
    print()
    print("Hallucination Categories:")
    for category, count in report.hallucination_categories.items():
        if count > 0:
            print(f"  - {category.replace('_', ' ').title()}: {count}")
    print()

    # Show sample of Markdown output
    print("=" * 60)
    print("Sample Markdown Output (first 500 characters)")
    print("=" * 60)
    with open(md_path, "r") as f:
        content = f.read()
        print(content[:500])
        print("...")
    print()

    print("=" * 60)
    print("Example complete!")
    print("=" * 60)
    print()
    print("Check the output directory for the generated reports:")
    print(f"  - JSON: {json_path}")
    print(f"  - CSV: {csv_path}")
    print(f"  - Markdown: {md_path}")
    print(f"  - Text: {txt_path}")


if __name__ == "__main__":
    main()
