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
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    pass  # For future type hints if needed

from llm_judge_auditor.components.hallucination_metrics import HallucinationProfile
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

    def get_hallucination_profile_summary(
        self, hallucination_profile: Optional[HallucinationProfile]
    ) -> Dict[str, Any]:
        """
        Generate a detailed summary of hallucination profile metrics.

        This method provides a comprehensive view of all hallucination quantification
        metrics including MiHR, MaHR, FactScore, Consensus F1, Fleiss' Kappa,
        uncertainty, and reliability classification.

        Args:
            hallucination_profile: HallucinationProfile object with computed metrics

        Returns:
            Dictionary with detailed hallucination metrics including:
            - mihr: Micro Hallucination Rate details
            - mahr: Macro Hallucination Rate details (if available)
            - factscore: FactScore value
            - consensus_f1: Consensus F1 metrics (if available)
            - fleiss_kappa: Fleiss' Kappa metrics (if available)
            - uncertainty: Uncertainty quantification (if available)
            - reliability: Reliability classification
            - is_high_risk: High risk flag
            - disputed_claims_count: Number of disputed claims
            - consensus_claims_count: Number of consensus claims

        Example:
            >>> generator = ReportGenerator()
            >>> summary = generator.get_hallucination_profile_summary(profile)
            >>> print(f"MiHR: {summary['mihr']['value']}")

        Requirements: 8.1, 19.4
        """
        if not hallucination_profile:
            return {
                "available": False,
                "mihr": None,
                "mahr": None,
                "factscore": None,
                "consensus_f1": None,
                "fleiss_kappa": None,
                "uncertainty": None,
                "reliability": None,
                "is_high_risk": None,
                "disputed_claims_count": 0,
                "consensus_claims_count": 0,
            }

        summary = {
            "available": True,
            "reliability": hallucination_profile.reliability.value,
            "is_high_risk": hallucination_profile.is_high_risk,
            "disputed_claims_count": len(hallucination_profile.disputed_claims),
            "consensus_claims_count": len(hallucination_profile.consensus_claims),
        }

        # MiHR details
        if hallucination_profile.mihr:
            summary["mihr"] = {
                "value": hallucination_profile.mihr.value,
                "unsupported_claims": hallucination_profile.mihr.unsupported_claims,
                "total_claims": hallucination_profile.mihr.total_claims,
                "has_claims": hallucination_profile.mihr.has_claims,
            }
        else:
            summary["mihr"] = None

        # MaHR details
        if hallucination_profile.mahr:
            summary["mahr"] = {
                "value": hallucination_profile.mahr.value,
                "responses_with_hallucinations": hallucination_profile.mahr.responses_with_hallucinations,
                "total_responses": hallucination_profile.mahr.total_responses,
            }
        else:
            summary["mahr"] = None

        # FactScore
        summary["factscore"] = hallucination_profile.factscore

        # Consensus F1 details
        if hallucination_profile.consensus_f1:
            summary["consensus_f1"] = {
                "precision": hallucination_profile.consensus_f1.precision,
                "recall": hallucination_profile.consensus_f1.recall,
                "f1": hallucination_profile.consensus_f1.f1,
            }
        else:
            summary["consensus_f1"] = None

        # Fleiss' Kappa details
        if hallucination_profile.fleiss_kappa:
            summary["fleiss_kappa"] = {
                "kappa": hallucination_profile.fleiss_kappa.kappa,
                "interpretation": hallucination_profile.fleiss_kappa.interpretation,
                "observed_agreement": hallucination_profile.fleiss_kappa.observed_agreement,
                "expected_agreement": hallucination_profile.fleiss_kappa.expected_agreement,
                "is_undefined": hallucination_profile.fleiss_kappa.is_undefined,
            }
        else:
            summary["fleiss_kappa"] = None

        # Uncertainty details
        if hallucination_profile.uncertainty:
            summary["uncertainty"] = {
                "shannon_entropy": hallucination_profile.uncertainty.shannon_entropy,
                "epistemic": hallucination_profile.uncertainty.epistemic,
                "aleatoric": hallucination_profile.uncertainty.aleatoric,
                "total": hallucination_profile.uncertainty.total,
                "is_high_uncertainty": hallucination_profile.uncertainty.is_high_uncertainty,
            }
        else:
            summary["uncertainty"] = None

        return summary

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

    def export_json_with_profile(
        self,
        report: Report,
        hallucination_profile: Optional[HallucinationProfile],
        path: str,
        indent: Optional[int] = 2,
    ) -> None:
        """
        Export report to JSON format with hallucination profile metrics.

        This method exports the full report including comprehensive hallucination
        quantification metrics (MiHR, MaHR, FactScore, Consensus F1, Fleiss' Kappa,
        uncertainty quantification).

        Args:
            report: Report object to export
            hallucination_profile: HallucinationProfile with computed metrics
            path: File path for the JSON output
            indent: Number of spaces for indentation (None for compact)

        Example:
            >>> generator = ReportGenerator()
            >>> generator.export_json_with_profile(report, profile, "report.json")

        Requirements: 8.1, 19.4
        """
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Build combined output
        output_data = report.to_dict()
        
        # Add hallucination profile metrics
        output_data["hallucination_profile"] = (
            hallucination_profile.to_dict() if hallucination_profile else None
        )
        output_data["hallucination_metrics_summary"] = self.get_hallucination_profile_summary(
            hallucination_profile
        )

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=indent)

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

    def export_csv_with_profile(
        self,
        report: Report,
        hallucination_profile: Optional[HallucinationProfile],
        path: str,
    ) -> None:
        """
        Export report to CSV format with hallucination profile metrics.

        Creates a CSV file with evaluation summary, detailed breakdowns, and
        comprehensive hallucination quantification metrics.

        Args:
            report: Report object to export
            hallucination_profile: HallucinationProfile with computed metrics
            path: File path for the CSV output

        Example:
            >>> generator = ReportGenerator()
            >>> generator.export_csv_with_profile(report, profile, "report.csv")

        Requirements: 8.1, 19.4
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

            # Write hallucination profile metrics section
            writer.writerow(["HALLUCINATION PROFILE METRICS"])
            writer.writerow(["Metric", "Value", "Details"])
            
            if hallucination_profile:
                # Reliability and risk
                writer.writerow(["Reliability", hallucination_profile.reliability.value, ""])
                writer.writerow(["High Risk", str(hallucination_profile.is_high_risk), ""])
                
                # MiHR
                if hallucination_profile.mihr:
                    mihr_val = f"{hallucination_profile.mihr.value:.4f}" if hallucination_profile.mihr.value is not None else "N/A"
                    writer.writerow([
                        "MiHR (Micro Hallucination Rate)",
                        mihr_val,
                        f"{hallucination_profile.mihr.unsupported_claims}/{hallucination_profile.mihr.total_claims} unsupported claims"
                    ])
                else:
                    writer.writerow(["MiHR (Micro Hallucination Rate)", "N/A", ""])
                
                # MaHR
                if hallucination_profile.mahr:
                    writer.writerow([
                        "MaHR (Macro Hallucination Rate)",
                        f"{hallucination_profile.mahr.value:.4f}",
                        f"{hallucination_profile.mahr.responses_with_hallucinations}/{hallucination_profile.mahr.total_responses} responses with hallucinations"
                    ])
                else:
                    writer.writerow(["MaHR (Macro Hallucination Rate)", "N/A", "Single response evaluation"])
                
                # FactScore
                factscore_val = f"{hallucination_profile.factscore:.4f}" if hallucination_profile.factscore is not None else "N/A"
                writer.writerow(["FactScore", factscore_val, ""])
                
                # Consensus F1
                if hallucination_profile.consensus_f1:
                    writer.writerow([
                        "Consensus F1",
                        f"{hallucination_profile.consensus_f1.f1:.4f}",
                        f"Precision: {hallucination_profile.consensus_f1.precision:.4f}, Recall: {hallucination_profile.consensus_f1.recall:.4f}"
                    ])
                else:
                    writer.writerow(["Consensus F1", "N/A", "Requires multiple model responses"])
                
                # Fleiss' Kappa
                if hallucination_profile.fleiss_kappa and not hallucination_profile.fleiss_kappa.is_undefined:
                    writer.writerow([
                        "Fleiss' Kappa",
                        f"{hallucination_profile.fleiss_kappa.kappa:.4f}",
                        f"Interpretation: {hallucination_profile.fleiss_kappa.interpretation}"
                    ])
                else:
                    writer.writerow(["Fleiss' Kappa", "N/A", "Requires 2+ judges"])
                
                # Uncertainty
                if hallucination_profile.uncertainty:
                    writer.writerow([
                        "Total Uncertainty",
                        f"{hallucination_profile.uncertainty.total:.4f}",
                        f"Epistemic: {hallucination_profile.uncertainty.epistemic:.4f}, Aleatoric: {hallucination_profile.uncertainty.aleatoric:.4f}"
                    ])
                    writer.writerow([
                        "Shannon Entropy",
                        f"{hallucination_profile.uncertainty.shannon_entropy:.4f}",
                        f"High Uncertainty: {hallucination_profile.uncertainty.is_high_uncertainty}"
                    ])
                else:
                    writer.writerow(["Uncertainty", "N/A", "Requires probability outputs"])
                
                # Claim analysis
                writer.writerow(["Disputed Claims Count", str(len(hallucination_profile.disputed_claims)), ""])
                writer.writerow(["Consensus Claims Count", str(len(hallucination_profile.consensus_claims)), ""])
            else:
                writer.writerow(["Hallucination Profile", "Not Available", ""])
            
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

    def _generate_text_with_profile(
        self,
        report: Report,
        hallucination_profile: Optional[HallucinationProfile],
    ) -> str:
        """
        Generate plain text content from a report with hallucination profile.

        Args:
            report: Report object to convert
            hallucination_profile: HallucinationProfile with computed metrics

        Returns:
            Plain text formatted string with hallucination metrics

        Requirements: 8.1, 19.4
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

        # Hallucination Profile Metrics
        lines.append("HALLUCINATION PROFILE METRICS")
        lines.append("-" * 60)
        if hallucination_profile:
            lines.append(f"Reliability: {hallucination_profile.reliability.value}")
            lines.append(f"High Risk: {hallucination_profile.is_high_risk}")
            lines.append("")
            
            # MiHR
            if hallucination_profile.mihr:
                mihr_val = f"{hallucination_profile.mihr.value:.4f}" if hallucination_profile.mihr.value is not None else "N/A"
                lines.append(f"MiHR (Micro Hallucination Rate): {mihr_val}")
                lines.append(f"  Unsupported Claims: {hallucination_profile.mihr.unsupported_claims}/{hallucination_profile.mihr.total_claims}")
            else:
                lines.append("MiHR: N/A")
            
            # MaHR
            if hallucination_profile.mahr:
                lines.append(f"MaHR (Macro Hallucination Rate): {hallucination_profile.mahr.value:.4f}")
                lines.append(f"  Responses with Hallucinations: {hallucination_profile.mahr.responses_with_hallucinations}/{hallucination_profile.mahr.total_responses}")
            else:
                lines.append("MaHR: N/A (single response evaluation)")
            
            # FactScore
            factscore_val = f"{hallucination_profile.factscore:.4f}" if hallucination_profile.factscore is not None else "N/A"
            lines.append(f"FactScore: {factscore_val}")
            
            # Consensus F1
            if hallucination_profile.consensus_f1:
                lines.append(f"Consensus F1: {hallucination_profile.consensus_f1.f1:.4f}")
                lines.append(f"  Precision: {hallucination_profile.consensus_f1.precision:.4f}")
                lines.append(f"  Recall: {hallucination_profile.consensus_f1.recall:.4f}")
            else:
                lines.append("Consensus F1: N/A (requires multiple model responses)")
            
            # Fleiss' Kappa
            if hallucination_profile.fleiss_kappa and not hallucination_profile.fleiss_kappa.is_undefined:
                lines.append(f"Fleiss' Kappa: {hallucination_profile.fleiss_kappa.kappa:.4f}")
                lines.append(f"  Interpretation: {hallucination_profile.fleiss_kappa.interpretation}")
                lines.append(f"  Observed Agreement: {hallucination_profile.fleiss_kappa.observed_agreement:.4f}")
                lines.append(f"  Expected Agreement: {hallucination_profile.fleiss_kappa.expected_agreement:.4f}")
            else:
                lines.append("Fleiss' Kappa: N/A (requires 2+ judges)")
            
            # Uncertainty
            if hallucination_profile.uncertainty:
                lines.append(f"Total Uncertainty: {hallucination_profile.uncertainty.total:.4f}")
                lines.append(f"  Shannon Entropy: {hallucination_profile.uncertainty.shannon_entropy:.4f}")
                lines.append(f"  Epistemic: {hallucination_profile.uncertainty.epistemic:.4f}")
                lines.append(f"  Aleatoric: {hallucination_profile.uncertainty.aleatoric:.4f}")
                lines.append(f"  High Uncertainty: {hallucination_profile.uncertainty.is_high_uncertainty}")
            else:
                lines.append("Uncertainty: N/A (requires probability outputs)")
            
            lines.append("")
            lines.append(f"Disputed Claims: {len(hallucination_profile.disputed_claims)}")
            lines.append(f"Consensus Claims: {len(hallucination_profile.consensus_claims)}")
        else:
            lines.append("Hallucination profile not available")
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

    def export_text_with_profile(
        self,
        report: Report,
        hallucination_profile: Optional[HallucinationProfile],
        path: str,
    ) -> None:
        """
        Export report to plain text format with hallucination profile metrics.

        Args:
            report: Report object to export
            hallucination_profile: HallucinationProfile with computed metrics
            path: File path for the text output

        Example:
            >>> generator = ReportGenerator()
            >>> generator.export_text_with_profile(report, profile, "report.txt")

        Requirements: 8.1, 19.4
        """
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        text_content = self._generate_text_with_profile(report, hallucination_profile)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text_content)
