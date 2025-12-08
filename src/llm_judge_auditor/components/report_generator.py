"""
Report Generator for the LLM Judge Auditor toolkit.

This module provides the ReportGenerator class for creating comprehensive,
transparent evaluation reports with full provenance. Supports JSON and
Markdown export formats.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from llm_judge_auditor.models import (
    EvaluationResult,
    Issue,
    IssueSeverity,
    IssueType,
    Passage,
    Report,
    Verdict,
    VerdictLabel,
)


class ReportGenerator:
    """
    Generate comprehensive evaluation reports with multiple export formats.

    The ReportGenerator creates transparent reports that include:
    - Metadata (timestamp, model versions, evaluation parameters)
    - Consensus scores and individual judge scores
    - Chain-of-thought reasoning from each judge
    - Confidence levels and disagreement metrics
    - Individual verdicts alongside consensus
    - Retrieval provenance
    - Categorized hallucinations

    Supports export to JSON and Markdown formats.

    Example:
        >>> generator = ReportGenerator()
        >>> report = generator.generate_report(evaluation_result)
        >>> generator.export_json(report, "report.json")
        >>> generator.export_markdown(report, "report.md")
    """

    def __init__(self):
        """Initialize the ReportGenerator."""
        pass

    def generate_report(self, evaluation: EvaluationResult) -> Report:
        """
        Generate a comprehensive evaluation report from an EvaluationResult.

        This method extracts all relevant information from the evaluation result
        and compiles it into a structured Report object.

        Args:
            evaluation: EvaluationResult containing all evaluation data

        Returns:
            Report object with comprehensive evaluation details

        Example:
            >>> toolkit = EvaluationToolkit.from_preset("balanced")
            >>> result = toolkit.evaluate(source, candidate)
            >>> generator = ReportGenerator()
            >>> report = generator.generate_report(result)
        """
        # The report is already generated in the evaluation result
        # This method can be used to regenerate or customize the report
        return evaluation.report

    def export_json(self, report: Report, path: str, indent: Optional[int] = 2) -> None:
        """
        Export report to JSON format.

        Args:
            report: Report object to export
            path: File path for the JSON output
            indent: Number of spaces for indentation (None for compact)

        Example:
            >>> generator = ReportGenerator()
            >>> generator.export_json(report, "evaluation_report.json")
        """
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report.to_json(indent=indent))

    def export_markdown(self, report: Report, path: str) -> None:
        """
        Export report to Markdown format.

        Creates a human-readable Markdown report with all evaluation details
        organized into sections.

        Args:
            report: Report object to export
            path: File path for the Markdown output

        Example:
            >>> generator = ReportGenerator()
            >>> generator.export_markdown(report, "evaluation_report.md")
        """
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        markdown_content = self._generate_markdown(report)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(markdown_content)

    def export_text(self, report: Report, path: str) -> None:
        """
        Export report to plain text format.

        Creates a simple text report with key evaluation details.

        Args:
            report: Report object to export
            path: File path for the text output

        Example:
            >>> generator = ReportGenerator()
            >>> generator.export_text(report, "evaluation_report.txt")
        """
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        text_content = self._generate_text(report)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text_content)

    def _generate_markdown(self, report: Report) -> str:
        """
        Generate Markdown content from a report.

        Args:
            report: Report object to convert

        Returns:
            Markdown-formatted string
        """
        lines = []

        # Title
        lines.append("# LLM Evaluation Report")
        lines.append("")

        # Metadata section
        lines.append("## Metadata")
        lines.append("")
        lines.append(f"- **Timestamp**: {report.metadata.get('timestamp', 'N/A')}")
        lines.append(f"- **Task**: {report.metadata.get('task', 'N/A')}")
        lines.append(f"- **Criteria**: {', '.join(report.metadata.get('criteria', []))}")
        lines.append(f"- **Retrieval Enabled**: {report.metadata.get('retrieval_enabled', False)}")
        lines.append(f"- **Verifier Model**: {report.metadata.get('verifier_model', 'N/A')}")
        lines.append(f"- **Judge Models**: {', '.join(report.metadata.get('judge_models', []))}")
        lines.append(f"- **Aggregation Strategy**: {report.metadata.get('aggregation_strategy', 'N/A')}")
        lines.append("")

        # Scores section
        lines.append("## Evaluation Scores")
        lines.append("")
        lines.append(f"### Consensus Score: **{report.consensus_score:.2f}**/100")
        lines.append("")
        lines.append(f"- **Confidence**: {report.confidence:.2f}")
        lines.append(f"- **Disagreement Level**: {report.disagreement_level:.2f}")
        lines.append("")

        # Individual judge scores
        lines.append("### Individual Judge Scores")
        lines.append("")
        for model_name, score in report.individual_scores.items():
            lines.append(f"- **{model_name}**: {score:.2f}/100")
        lines.append("")

        # Chain-of-thought reasoning
        lines.append("## Chain-of-Thought Reasoning")
        lines.append("")
        for model_name, reasoning in report.reasoning.items():
            lines.append(f"### {model_name}")
            lines.append("")
            lines.append(reasoning)
            lines.append("")

        # Verifier verdicts
        lines.append("## Specialized Verifier Verdicts")
        lines.append("")
        if report.verifier_verdicts:
            for idx, verdict in enumerate(report.verifier_verdicts, 1):
                lines.append(f"### Verdict {idx}")
                lines.append("")
                lines.append(f"- **Label**: {verdict.label.value}")
                lines.append(f"- **Confidence**: {verdict.confidence:.2f}")
                if verdict.reasoning:
                    lines.append(f"- **Reasoning**: {verdict.reasoning}")
                if verdict.evidence:
                    lines.append(f"- **Evidence**:")
                    for evidence in verdict.evidence:
                        lines.append(f"  - {evidence}")
                lines.append("")
        else:
            lines.append("*No verifier verdicts available*")
            lines.append("")

        # Retrieval provenance
        lines.append("## Retrieval Provenance")
        lines.append("")
        if report.retrieval_provenance:
            lines.append(f"Retrieved {len(report.retrieval_provenance)} passages:")
            lines.append("")
            for idx, passage in enumerate(report.retrieval_provenance, 1):
                lines.append(f"### Passage {idx}")
                lines.append("")
                lines.append(f"- **Source**: {passage.source}")
                lines.append(f"- **Relevance Score**: {passage.relevance_score:.4f}")
                lines.append(f"- **Text**: {passage.text}")
                lines.append("")
        else:
            lines.append("*No retrieval performed or no passages retrieved*")
            lines.append("")

        # Flagged issues
        lines.append("## Flagged Issues")
        lines.append("")
        if report.flagged_issues:
            for idx, issue in enumerate(report.flagged_issues, 1):
                lines.append(f"### Issue {idx}")
                lines.append("")
                lines.append(f"- **Type**: {issue.type.value}")
                lines.append(f"- **Severity**: {issue.severity.value}")
                lines.append(f"- **Description**: {issue.description}")
                if issue.evidence:
                    lines.append(f"- **Evidence**:")
                    for evidence in issue.evidence:
                        lines.append(f"  - {evidence}")
                lines.append("")
        else:
            lines.append("*No issues flagged*")
            lines.append("")

        # Hallucination categories
        lines.append("## Hallucination Categories")
        lines.append("")
        if any(count > 0 for count in report.hallucination_categories.values()):
            for category, count in report.hallucination_categories.items():
                if count > 0:
                    lines.append(f"- **{category.replace('_', ' ').title()}**: {count}")
            lines.append("")
        else:
            lines.append("*No hallucinations detected*")
            lines.append("")

        return "\n".join(lines)

    def _generate_text(self, report: Report) -> str:
        """
        Generate plain text content from a report.

        Args:
            report: Report object to convert

        Returns:
            Plain text formatted string
        """
        lines = []

        # Title
        lines.append("=" * 60)
        lines.append("LLM EVALUATION REPORT")
        lines.append("=" * 60)
        lines.append("")

        # Metadata
        lines.append("METADATA")
        lines.append("-" * 60)
        lines.append(f"Timestamp: {report.metadata.get('timestamp', 'N/A')}")
        lines.append(f"Task: {report.metadata.get('task', 'N/A')}")
        lines.append(f"Criteria: {', '.join(report.metadata.get('criteria', []))}")
        lines.append(f"Retrieval Enabled: {report.metadata.get('retrieval_enabled', False)}")
        lines.append(f"Verifier Model: {report.metadata.get('verifier_model', 'N/A')}")
        lines.append(f"Judge Models: {', '.join(report.metadata.get('judge_models', []))}")
        lines.append(f"Aggregation Strategy: {report.metadata.get('aggregation_strategy', 'N/A')}")
        lines.append("")

        # Scores
        lines.append("EVALUATION SCORES")
        lines.append("-" * 60)
        lines.append(f"Consensus Score: {report.consensus_score:.2f}/100")
        lines.append(f"Confidence: {report.confidence:.2f}")
        lines.append(f"Disagreement Level: {report.disagreement_level:.2f}")
        lines.append("")

        # Individual scores
        lines.append("Individual Judge Scores:")
        for model_name, score in report.individual_scores.items():
            lines.append(f"  - {model_name}: {score:.2f}/100")
        lines.append("")

        # Reasoning
        lines.append("CHAIN-OF-THOUGHT REASONING")
        lines.append("-" * 60)
        for model_name, reasoning in report.reasoning.items():
            lines.append(f"{model_name}:")
            lines.append(reasoning)
            lines.append("")

        # Verifier verdicts
        lines.append("SPECIALIZED VERIFIER VERDICTS")
        lines.append("-" * 60)
        if report.verifier_verdicts:
            for idx, verdict in enumerate(report.verifier_verdicts, 1):
                lines.append(f"Verdict {idx}:")
                lines.append(f"  Label: {verdict.label.value}")
                lines.append(f"  Confidence: {verdict.confidence:.2f}")
                if verdict.reasoning:
                    lines.append(f"  Reasoning: {verdict.reasoning}")
                lines.append("")
        else:
            lines.append("No verifier verdicts available")
            lines.append("")

        # Flagged issues
        lines.append("FLAGGED ISSUES")
        lines.append("-" * 60)
        if report.flagged_issues:
            for idx, issue in enumerate(report.flagged_issues, 1):
                lines.append(f"Issue {idx}:")
                lines.append(f"  Type: {issue.type.value}")
                lines.append(f"  Severity: {issue.severity.value}")
                lines.append(f"  Description: {issue.description}")
                lines.append("")
        else:
            lines.append("No issues flagged")
            lines.append("")

        # Hallucination categories
        lines.append("HALLUCINATION CATEGORIES")
        lines.append("-" * 60)
        if any(count > 0 for count in report.hallucination_categories.values()):
            for category, count in report.hallucination_categories.items():
                if count > 0:
                    lines.append(f"  {category.replace('_', ' ').title()}: {count}")
            lines.append("")
        else:
            lines.append("No hallucinations detected")
            lines.append("")

        lines.append("=" * 60)
        lines.append("END OF REPORT")
        lines.append("=" * 60)

        return "\n".join(lines)
