"""
Report Generator for the LLM Judge Auditor toolkit.

This module provides the ReportGenerator class for creating comprehensive,
transparent evaluation reports with full provenance. Supports JSON, CSV,
and Markdown export formats.
"""

import csv
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
    - Retrieval provenance with source tracking
    - Categorized hallucinations by type

    Supports export to JSON, CSV, and Markdown formats.

    Example:
        >>> generator = ReportGenerator()
        >>> report = generator.generate_report(evaluation_result)
        >>> generator.export_json(report, "report.json")
        >>> generator.export_csv(report, "report.csv")
        >>> generator.export_markdown(report, "report.md")
    """

    def __init__(self):
        """Initialize the ReportGenerator."""
        pass

    def get_retrieval_provenance_summary(self, report: Report) -> Dict[str, Any]:
        """
        Generate a detailed summary of retrieval provenance.

        This method provides enhanced tracking of where information came from,
        including source distribution, relevance statistics, and passage details.

        Args:
            report: Report object containing retrieval provenance

        Returns:
            Dictionary with detailed provenance information including:
            - total_passages: Total number of retrieved passages
            - sources: List of unique sources
            - source_distribution: Count of passages per source
            - avg_relevance_score: Average relevance score across all passages
            - min_relevance_score: Minimum relevance score
            - max_relevance_score: Maximum relevance score
            - passages_by_source: Passages grouped by source

        Example:
            >>> generator = ReportGenerator()
            >>> provenance = generator.get_retrieval_provenance_summary(report)
            >>> print(f"Retrieved from {len(provenance['sources'])} sources")
        """
        if not report.retrieval_provenance:
            return {
                "total_passages": 0,
                "sources": [],
                "source_distribution": {},
                "avg_relevance_score": 0.0,
                "min_relevance_score": 0.0,
                "max_relevance_score": 0.0,
                "passages_by_source": {},
            }

        # Calculate statistics
        total_passages = len(report.retrieval_provenance)
        sources = list(set(p.source for p in report.retrieval_provenance))
        relevance_scores = [p.relevance_score for p in report.retrieval_provenance]

        # Count passages per source
        source_distribution = {}
        passages_by_source = {}
        for passage in report.retrieval_provenance:
            source_distribution[passage.source] = source_distribution.get(passage.source, 0) + 1
            if passage.source not in passages_by_source:
                passages_by_source[passage.source] = []
            passages_by_source[passage.source].append({
                "text": passage.text,
                "relevance_score": passage.relevance_score,
            })

        return {
            "total_passages": total_passages,
            "sources": sources,
            "source_distribution": source_distribution,
            "avg_relevance_score": sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.0,
            "min_relevance_score": min(relevance_scores) if relevance_scores else 0.0,
            "max_relevance_score": max(relevance_scores) if relevance_scores else 0.0,
            "passages_by_source": passages_by_source,
        }

    def get_hallucination_summary(self, report: Report) -> Dict[str, Any]:
        """
        Generate a detailed summary of hallucination categorization.

        This method provides enhanced analysis of detected hallucinations,
        including type distribution, severity breakdown, and detailed issue lists.

        Args:
            report: Report object containing hallucination categories and flagged issues

        Returns:
            Dictionary with detailed hallucination information including:
            - total_hallucinations: Total count of all hallucinations
            - categories: Hallucination counts by type
            - severity_distribution: Count of issues by severity level
            - issues_by_type: Detailed issues grouped by type
            - issues_by_severity: Detailed issues grouped by severity

        Example:
            >>> generator = ReportGenerator()
            >>> summary = generator.get_hallucination_summary(report)
            >>> print(f"Found {summary['total_hallucinations']} hallucinations")
        """
        total_hallucinations = sum(report.hallucination_categories.values())

        # Group issues by severity
        severity_distribution = {
            "low": 0,
            "medium": 0,
            "high": 0,
        }
        issues_by_severity = {
            "low": [],
            "medium": [],
            "high": [],
        }

        for issue in report.flagged_issues:
            severity_key = issue.severity.value
            severity_distribution[severity_key] += 1
            issues_by_severity[severity_key].append({
                "type": issue.type.value,
                "description": issue.description,
                "evidence": issue.evidence,
            })

        # Group issues by type
        issues_by_type = {}
        for issue in report.flagged_issues:
            issue_type = issue.type.value
            if issue_type not in issues_by_type:
                issues_by_type[issue_type] = []
            issues_by_type[issue_type].append({
                "severity": issue.severity.value,
                "description": issue.description,
                "evidence": issue.evidence,
            })

        return {
            "total_hallucinations": total_hallucinations,
            "categories": report.hallucination_categories,
            "severity_distribution": severity_distribution,
            "issues_by_type": issues_by_type,
            "issues_by_severity": issues_by_severity,
        }

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

    def export_csv(self, report: Report, path: str) -> None:
        """
        Export report to CSV format.

        Creates a CSV file with evaluation summary and detailed breakdowns.
        The CSV includes multiple sections:
        - Summary: consensus score, confidence, disagreement
        - Individual Scores: per-judge scores
        - Verifier Verdicts: statement-level verdicts
        - Retrieval Provenance: retrieved passages with sources
        - Flagged Issues: detected issues with severity
        - Hallucination Categories: counts by type

        Args:
            report: Report object to export
            path: File path for the CSV output

        Example:
            >>> generator = ReportGenerator()
            >>> generator.export_csv(report, "evaluation_report.csv")
        """
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)

            # Write summary section
            writer.writerow(["EVALUATION SUMMARY"])
            writer.writerow(["Metric", "Value"])
            writer.writerow(["Timestamp", report.metadata.get("timestamp", "N/A")])
            writer.writerow(["Task", report.metadata.get("task", "N/A")])
            writer.writerow(["Criteria", ", ".join(report.metadata.get("criteria", []))])
            writer.writerow(["Consensus Score", f"{report.consensus_score:.2f}"])
            writer.writerow(["Confidence", f"{report.confidence:.2f}"])
            writer.writerow(["Disagreement Level", f"{report.disagreement_level:.2f}"])
            writer.writerow(["Retrieval Enabled", report.metadata.get("retrieval_enabled", False)])
            writer.writerow(["Verifier Model", report.metadata.get("verifier_model", "N/A")])
            writer.writerow(["Judge Models", ", ".join(report.metadata.get("judge_models", []))])
            writer.writerow(["Aggregation Strategy", report.metadata.get("aggregation_strategy", "N/A")])
            writer.writerow([])

            # Write individual scores section
            writer.writerow(["INDIVIDUAL JUDGE SCORES"])
            writer.writerow(["Judge Model", "Score"])
            for model_name, score in report.individual_scores.items():
                writer.writerow([model_name, f"{score:.2f}"])
            writer.writerow([])

            # Write verifier verdicts section
            writer.writerow(["VERIFIER VERDICTS"])
            writer.writerow(["Verdict #", "Label", "Confidence", "Reasoning", "Evidence"])
            for idx, verdict in enumerate(report.verifier_verdicts, 1):
                evidence_str = "; ".join(verdict.evidence) if verdict.evidence else "None"
                writer.writerow([
                    idx,
                    verdict.label.value,
                    f"{verdict.confidence:.2f}",
                    verdict.reasoning,
                    evidence_str
                ])
            if not report.verifier_verdicts:
                writer.writerow(["No verifier verdicts available"])
            writer.writerow([])

            # Write retrieval provenance section
            writer.writerow(["RETRIEVAL PROVENANCE"])
            writer.writerow(["Passage #", "Source", "Relevance Score", "Text"])
            for idx, passage in enumerate(report.retrieval_provenance, 1):
                writer.writerow([
                    idx,
                    passage.source,
                    f"{passage.relevance_score:.4f}",
                    passage.text
                ])
            if not report.retrieval_provenance:
                writer.writerow(["No retrieval performed or no passages retrieved"])
            writer.writerow([])

            # Write flagged issues section
            writer.writerow(["FLAGGED ISSUES"])
            writer.writerow(["Issue #", "Type", "Severity", "Description", "Evidence"])
            for idx, issue in enumerate(report.flagged_issues, 1):
                evidence_str = "; ".join(issue.evidence) if issue.evidence else "None"
                writer.writerow([
                    idx,
                    issue.type.value,
                    issue.severity.value,
                    issue.description,
                    evidence_str
                ])
            if not report.flagged_issues:
                writer.writerow(["No issues flagged"])
            writer.writerow([])

            # Write hallucination categories section
            writer.writerow(["HALLUCINATION CATEGORIES"])
            writer.writerow(["Category", "Count"])
            for category, count in report.hallucination_categories.items():
                writer.writerow([category.replace("_", " ").title(), count])
            writer.writerow([])

            # Write reasoning section
            writer.writerow(["CHAIN-OF-THOUGHT REASONING"])
            writer.writerow(["Judge Model", "Reasoning"])
            for model_name, reasoning in report.reasoning.items():
                writer.writerow([model_name, reasoning])
            writer.writerow([])

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
