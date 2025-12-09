"""
Component Performance Tracker for the LLM Judge Auditor toolkit.

This module provides the PerformanceTracker class for tracking separate metrics
for the specialized verifier and judge ensemble components, including accuracy,
latency, confidence, and disagreement logging.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from llm_judge_auditor.models import ClaimType, JudgeResult, Verdict, VerdictLabel

logger = logging.getLogger(__name__)


@dataclass
class ComponentMetrics:
    """
    Metrics for a single component (verifier or judge).

    Attributes:
        component_name: Name of the component
        total_evaluations: Total number of evaluations performed
        total_latency: Cumulative latency in seconds
        confidence_scores: List of confidence scores
        claim_type_performance: Performance breakdown by claim type
    """

    component_name: str
    total_evaluations: int = 0
    total_latency: float = 0.0
    confidence_scores: List[float] = field(default_factory=list)
    claim_type_performance: Dict[str, Dict[str, int]] = field(default_factory=dict)

    def add_evaluation(
        self,
        latency: float,
        confidence: float,
        claim_type: Optional[ClaimType] = None,
        correct: Optional[bool] = None,
    ):
        """
        Record a single evaluation.

        Args:
            latency: Time taken for evaluation in seconds
            confidence: Confidence score (0.0 to 1.0)
            claim_type: Optional type of claim evaluated
            correct: Optional correctness indicator (if ground truth available)
        """
        self.total_evaluations += 1
        self.total_latency += latency
        self.confidence_scores.append(confidence)

        # Track by claim type if provided
        if claim_type:
            claim_type_str = claim_type.value if isinstance(claim_type, ClaimType) else str(claim_type)
            if claim_type_str not in self.claim_type_performance:
                self.claim_type_performance[claim_type_str] = {
                    "total": 0,
                    "correct": 0,
                    "incorrect": 0,
                }

            self.claim_type_performance[claim_type_str]["total"] += 1
            if correct is not None:
                if correct:
                    self.claim_type_performance[claim_type_str]["correct"] += 1
                else:
                    self.claim_type_performance[claim_type_str]["incorrect"] += 1

    @property
    def average_latency(self) -> float:
        """Calculate average latency per evaluation."""
        if self.total_evaluations == 0:
            return 0.0
        return self.total_latency / self.total_evaluations

    @property
    def average_confidence(self) -> float:
        """Calculate average confidence score."""
        if not self.confidence_scores:
            return 0.0
        return sum(self.confidence_scores) / len(self.confidence_scores)

    @property
    def accuracy(self) -> Optional[float]:
        """
        Calculate overall accuracy if correctness data is available.

        Returns:
            Accuracy as a float between 0 and 1, or None if no correctness data
        """
        total_correct = 0
        total_with_ground_truth = 0

        for claim_type_data in self.claim_type_performance.values():
            correct = claim_type_data.get("correct", 0)
            incorrect = claim_type_data.get("incorrect", 0)
            total_correct += correct
            total_with_ground_truth += correct + incorrect

        if total_with_ground_truth == 0:
            return None

        return total_correct / total_with_ground_truth

    def get_claim_type_accuracy(self, claim_type: str) -> Optional[float]:
        """
        Get accuracy for a specific claim type.

        Args:
            claim_type: The claim type to get accuracy for

        Returns:
            Accuracy for the claim type, or None if no data available
        """
        if claim_type not in self.claim_type_performance:
            return None

        data = self.claim_type_performance[claim_type]
        correct = data.get("correct", 0)
        incorrect = data.get("incorrect", 0)
        total = correct + incorrect

        if total == 0:
            return None

        return correct / total

    def to_dict(self) -> Dict:
        """Convert metrics to dictionary for reporting."""
        return {
            "component_name": self.component_name,
            "total_evaluations": self.total_evaluations,
            "average_latency": self.average_latency,
            "average_confidence": self.average_confidence,
            "accuracy": self.accuracy,
            "claim_type_performance": self.claim_type_performance,
        }


@dataclass
class Disagreement:
    """
    Record of a disagreement between verifier and judge ensemble.

    Attributes:
        statement: The statement being evaluated
        verifier_verdict: Verdict from the specialized verifier
        judge_consensus: Consensus from judge ensemble
        judge_results: Individual judge results
        verifier_confidence: Verifier's confidence score
        judge_confidence: Average judge confidence
        claim_type: Type of claim (if available)
    """

    statement: str
    verifier_verdict: VerdictLabel
    judge_consensus: float
    judge_results: List[JudgeResult]
    verifier_confidence: float
    judge_confidence: float
    claim_type: Optional[ClaimType] = None

    def to_dict(self) -> Dict:
        """Convert disagreement to dictionary for logging."""
        return {
            "statement": self.statement,
            "verifier_verdict": self.verifier_verdict.value,
            "verifier_confidence": self.verifier_confidence,
            "judge_consensus": self.judge_consensus,
            "judge_confidence": self.judge_confidence,
            "judge_results": [
                {
                    "model_name": jr.model_name,
                    "score": jr.score,
                    "reasoning": jr.reasoning[:200] + "..." if len(jr.reasoning) > 200 else jr.reasoning,
                }
                for jr in self.judge_results
            ],
            "claim_type": self.claim_type.value if self.claim_type else None,
        }


class PerformanceTracker:
    """
    Tracks performance metrics for verifier and judge ensemble components.

    This class provides functionality to:
    - Track separate metrics for verifier and judge ensemble
    - Record latency, confidence, and accuracy for each component
    - Log disagreements between components
    - Generate performance reports and comparisons

    Example:
        >>> tracker = PerformanceTracker()
        >>> with tracker.track_verifier():
        ...     verdict = verifier.verify_statement(...)
        >>> tracker.record_verifier_result(verdict, claim_type="factual")
        >>> report = tracker.generate_report()
    """

    def __init__(self):
        """Initialize the PerformanceTracker."""
        self.verifier_metrics = ComponentMetrics(component_name="specialized_verifier")
        self.judge_metrics = ComponentMetrics(component_name="judge_ensemble")
        self.disagreements: List[Disagreement] = []
        self._current_verifier_start: Optional[float] = None
        self._current_judge_start: Optional[float] = None

        logger.info("PerformanceTracker initialized")

    def start_verifier_timing(self):
        """Start timing a verifier evaluation."""
        self._current_verifier_start = time.time()

    def end_verifier_timing(self) -> float:
        """
        End timing a verifier evaluation and return the elapsed time.

        Returns:
            Elapsed time in seconds

        Raises:
            RuntimeError: If timing was not started
        """
        if self._current_verifier_start is None:
            raise RuntimeError("Verifier timing was not started")

        elapsed = time.time() - self._current_verifier_start
        self._current_verifier_start = None
        return elapsed

    def start_judge_timing(self):
        """Start timing a judge ensemble evaluation."""
        self._current_judge_start = time.time()

    def end_judge_timing(self) -> float:
        """
        End timing a judge ensemble evaluation and return the elapsed time.

        Returns:
            Elapsed time in seconds

        Raises:
            RuntimeError: If timing was not started
        """
        if self._current_judge_start is None:
            raise RuntimeError("Judge timing was not started")

        elapsed = time.time() - self._current_judge_start
        self._current_judge_start = None
        return elapsed

    def record_verifier_result(
        self,
        verdict: Verdict,
        latency: float,
        claim_type: Optional[ClaimType] = None,
        correct: Optional[bool] = None,
    ):
        """
        Record a verifier evaluation result.

        Args:
            verdict: The verdict from the verifier
            latency: Time taken for evaluation in seconds
            claim_type: Optional type of claim evaluated
            correct: Optional correctness indicator (if ground truth available)
        """
        self.verifier_metrics.add_evaluation(
            latency=latency,
            confidence=verdict.confidence,
            claim_type=claim_type,
            correct=correct,
        )

        logger.debug(
            f"Recorded verifier result: latency={latency:.3f}s, "
            f"confidence={verdict.confidence:.2f}, claim_type={claim_type}"
        )

    def record_judge_results(
        self,
        judge_results: List[JudgeResult],
        latency: float,
        claim_type: Optional[ClaimType] = None,
        correct: Optional[bool] = None,
    ):
        """
        Record judge ensemble evaluation results.

        Args:
            judge_results: List of results from all judges
            latency: Time taken for ensemble evaluation in seconds
            claim_type: Optional type of claim evaluated
            correct: Optional correctness indicator (if ground truth available)
        """
        # Calculate average confidence from all judges
        avg_confidence = (
            sum(jr.confidence for jr in judge_results) / len(judge_results)
            if judge_results
            else 0.0
        )

        self.judge_metrics.add_evaluation(
            latency=latency,
            confidence=avg_confidence,
            claim_type=claim_type,
            correct=correct,
        )

        logger.debug(
            f"Recorded judge results: {len(judge_results)} judges, "
            f"latency={latency:.3f}s, avg_confidence={avg_confidence:.2f}, "
            f"claim_type={claim_type}"
        )

    def log_disagreement(
        self,
        statement: str,
        verifier_verdict: Verdict,
        judge_results: List[JudgeResult],
        judge_consensus_score: float,
        claim_type: Optional[ClaimType] = None,
    ):
        """
        Log a disagreement between verifier and judge ensemble.

        A disagreement is defined as:
        - Verifier says REFUTED but judges give high score (>70)
        - Verifier says SUPPORTED but judges give low score (<30)
        - Verifier says NOT_ENOUGH_INFO but judges are very confident (score <20 or >80)

        Args:
            statement: The statement being evaluated
            verifier_verdict: Verdict from the specialized verifier
            judge_results: Results from all judges
            judge_consensus_score: Consensus score from aggregation
            claim_type: Optional type of claim
        """
        # Determine if there's a significant disagreement
        is_disagreement = False

        if verifier_verdict.label == VerdictLabel.REFUTED and judge_consensus_score > 70:
            is_disagreement = True
        elif verifier_verdict.label == VerdictLabel.SUPPORTED and judge_consensus_score < 30:
            is_disagreement = True
        elif verifier_verdict.label == VerdictLabel.NOT_ENOUGH_INFO and (
            judge_consensus_score < 20 or judge_consensus_score > 80
        ):
            is_disagreement = True

        if is_disagreement:
            avg_judge_confidence = (
                sum(jr.confidence for jr in judge_results) / len(judge_results)
                if judge_results
                else 0.0
            )

            disagreement = Disagreement(
                statement=statement,
                verifier_verdict=verifier_verdict.label,
                judge_consensus=judge_consensus_score,
                judge_results=judge_results,
                verifier_confidence=verifier_verdict.confidence,
                judge_confidence=avg_judge_confidence,
                claim_type=claim_type,
            )

            self.disagreements.append(disagreement)

            logger.warning(
                f"Disagreement detected: verifier={verifier_verdict.label.value} "
                f"(conf={verifier_verdict.confidence:.2f}), "
                f"judges={judge_consensus_score:.1f} "
                f"(conf={avg_judge_confidence:.2f})"
            )

    def generate_report(self) -> Dict:
        """
        Generate a comprehensive performance report.

        Returns:
            Dictionary containing performance metrics for both components,
            disagreement statistics, and comparative analysis

        Requirements: 13.1, 13.2, 13.3, 13.4
        """
        report = {
            "verifier_metrics": self.verifier_metrics.to_dict(),
            "judge_metrics": self.judge_metrics.to_dict(),
            "disagreements": {
                "total_count": len(self.disagreements),
                "disagreement_rate": self._calculate_disagreement_rate(),
                "disagreements_by_claim_type": self._analyze_disagreements_by_claim_type(),
                "recent_disagreements": [d.to_dict() for d in self.disagreements[-10:]],
            },
            "comparative_analysis": self._generate_comparative_analysis(),
        }

        logger.info(
            f"Generated performance report: {self.verifier_metrics.total_evaluations} "
            f"verifier evaluations, {self.judge_metrics.total_evaluations} judge evaluations, "
            f"{len(self.disagreements)} disagreements"
        )

        return report

    def _calculate_disagreement_rate(self) -> float:
        """Calculate the rate of disagreements relative to total evaluations."""
        total_evaluations = min(
            self.verifier_metrics.total_evaluations,
            self.judge_metrics.total_evaluations,
        )

        if total_evaluations == 0:
            return 0.0

        return len(self.disagreements) / total_evaluations

    def _analyze_disagreements_by_claim_type(self) -> Dict[str, int]:
        """Analyze disagreements broken down by claim type."""
        disagreements_by_type: Dict[str, int] = {}

        for disagreement in self.disagreements:
            if disagreement.claim_type:
                claim_type_str = disagreement.claim_type.value
                disagreements_by_type[claim_type_str] = (
                    disagreements_by_type.get(claim_type_str, 0) + 1
                )

        return disagreements_by_type

    def _generate_comparative_analysis(self) -> Dict:
        """
        Generate comparative analysis between verifier and judge ensemble.

        This identifies which component performs better on different claim types.

        Requirements: 13.4
        """
        analysis = {
            "latency_comparison": {
                "verifier_avg": self.verifier_metrics.average_latency,
                "judge_avg": self.judge_metrics.average_latency,
                "faster_component": (
                    "verifier"
                    if self.verifier_metrics.average_latency < self.judge_metrics.average_latency
                    else "judge_ensemble"
                ),
            },
            "confidence_comparison": {
                "verifier_avg": self.verifier_metrics.average_confidence,
                "judge_avg": self.judge_metrics.average_confidence,
                "more_confident_component": (
                    "verifier"
                    if self.verifier_metrics.average_confidence > self.judge_metrics.average_confidence
                    else "judge_ensemble"
                ),
            },
            "claim_type_performance": self._compare_claim_type_performance(),
        }

        # Add accuracy comparison if available
        verifier_accuracy = self.verifier_metrics.accuracy
        judge_accuracy = self.judge_metrics.accuracy

        if verifier_accuracy is not None and judge_accuracy is not None:
            analysis["accuracy_comparison"] = {
                "verifier_accuracy": verifier_accuracy,
                "judge_accuracy": judge_accuracy,
                "more_accurate_component": (
                    "verifier" if verifier_accuracy > judge_accuracy else "judge_ensemble"
                ),
            }

        return analysis

    def _compare_claim_type_performance(self) -> Dict:
        """
        Compare performance of verifier vs judges on different claim types.

        Requirements: 13.4
        """
        # Get all claim types that have been evaluated
        all_claim_types = set(self.verifier_metrics.claim_type_performance.keys()) | set(
            self.judge_metrics.claim_type_performance.keys()
        )

        comparison = {}

        for claim_type in all_claim_types:
            verifier_accuracy = self.verifier_metrics.get_claim_type_accuracy(claim_type)
            judge_accuracy = self.judge_metrics.get_claim_type_accuracy(claim_type)

            verifier_data = self.verifier_metrics.claim_type_performance.get(claim_type, {})
            judge_data = self.judge_metrics.claim_type_performance.get(claim_type, {})

            comparison[claim_type] = {
                "verifier": {
                    "accuracy": verifier_accuracy,
                    "total_evaluations": verifier_data.get("total", 0),
                },
                "judge": {
                    "accuracy": judge_accuracy,
                    "total_evaluations": judge_data.get("total", 0),
                },
            }

            # Determine which component performs better
            if verifier_accuracy is not None and judge_accuracy is not None:
                if verifier_accuracy > judge_accuracy:
                    comparison[claim_type]["better_component"] = "verifier"
                elif judge_accuracy > verifier_accuracy:
                    comparison[claim_type]["better_component"] = "judge_ensemble"
                else:
                    comparison[claim_type]["better_component"] = "tie"
            else:
                comparison[claim_type]["better_component"] = "insufficient_data"

        return comparison

    def reset(self):
        """Reset all metrics and disagreements."""
        self.verifier_metrics = ComponentMetrics(component_name="specialized_verifier")
        self.judge_metrics = ComponentMetrics(component_name="judge_ensemble")
        self.disagreements = []
        self._current_verifier_start = None
        self._current_judge_start = None

        logger.info("PerformanceTracker reset")

    def get_summary(self) -> str:
        """
        Get a human-readable summary of performance metrics.

        Returns:
            Formatted string with key performance indicators
        """
        lines = [
            "=== Performance Tracker Summary ===",
            "",
            "Verifier Metrics:",
            f"  Total Evaluations: {self.verifier_metrics.total_evaluations}",
            f"  Average Latency: {self.verifier_metrics.average_latency:.3f}s",
            f"  Average Confidence: {self.verifier_metrics.average_confidence:.2f}",
        ]

        if self.verifier_metrics.accuracy is not None:
            lines.append(f"  Accuracy: {self.verifier_metrics.accuracy:.2%}")

        lines.extend(
            [
                "",
                "Judge Ensemble Metrics:",
                f"  Total Evaluations: {self.judge_metrics.total_evaluations}",
                f"  Average Latency: {self.judge_metrics.average_latency:.3f}s",
                f"  Average Confidence: {self.judge_metrics.average_confidence:.2f}",
            ]
        )

        if self.judge_metrics.accuracy is not None:
            lines.append(f"  Accuracy: {self.judge_metrics.accuracy:.2%}")

        lines.extend(
            [
                "",
                "Disagreements:",
                f"  Total: {len(self.disagreements)}",
                f"  Rate: {self._calculate_disagreement_rate():.2%}",
                "",
            ]
        )

        return "\n".join(lines)
