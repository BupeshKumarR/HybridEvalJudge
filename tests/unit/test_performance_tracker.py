"""
Unit tests for the PerformanceTracker component.

Tests cover:
- Metric recording for verifier and judge ensemble
- Latency tracking
- Confidence tracking
- Disagreement logging
- Performance report generation
- Comparative analysis
"""

import pytest

from llm_judge_auditor.components.performance_tracker import (
    ComponentMetrics,
    Disagreement,
    PerformanceTracker,
)
from llm_judge_auditor.models import (
    ClaimType,
    Issue,
    IssueType,
    IssueSeverity,
    JudgeResult,
    Verdict,
    VerdictLabel,
)


class TestComponentMetrics:
    """Test ComponentMetrics class."""

    def test_initialization(self):
        """Test ComponentMetrics initialization."""
        metrics = ComponentMetrics(component_name="test_component")

        assert metrics.component_name == "test_component"
        assert metrics.total_evaluations == 0
        assert metrics.total_latency == 0.0
        assert metrics.confidence_scores == []
        assert metrics.claim_type_performance == {}

    def test_add_evaluation(self):
        """Test adding evaluation metrics."""
        metrics = ComponentMetrics(component_name="test")

        metrics.add_evaluation(
            latency=0.5,
            confidence=0.8,
            claim_type=ClaimType.FACTUAL,
            correct=True,
        )

        assert metrics.total_evaluations == 1
        assert metrics.total_latency == 0.5
        assert metrics.confidence_scores == [0.8]
        assert "factual" in metrics.claim_type_performance
        assert metrics.claim_type_performance["factual"]["total"] == 1
        assert metrics.claim_type_performance["factual"]["correct"] == 1

    def test_average_latency(self):
        """Test average latency calculation."""
        metrics = ComponentMetrics(component_name="test")

        metrics.add_evaluation(latency=0.5, confidence=0.8)
        metrics.add_evaluation(latency=1.0, confidence=0.9)
        metrics.add_evaluation(latency=0.7, confidence=0.85)

        assert metrics.average_latency == pytest.approx(0.733, rel=0.01)

    def test_average_confidence(self):
        """Test average confidence calculation."""
        metrics = ComponentMetrics(component_name="test")

        metrics.add_evaluation(latency=0.5, confidence=0.8)
        metrics.add_evaluation(latency=1.0, confidence=0.9)
        metrics.add_evaluation(latency=0.7, confidence=0.7)

        assert metrics.average_confidence == pytest.approx(0.8, rel=0.01)

    def test_accuracy_calculation(self):
        """Test accuracy calculation."""
        metrics = ComponentMetrics(component_name="test")

        # Add evaluations with correctness data
        metrics.add_evaluation(
            latency=0.5, confidence=0.8, claim_type=ClaimType.FACTUAL, correct=True
        )
        metrics.add_evaluation(
            latency=0.6, confidence=0.9, claim_type=ClaimType.FACTUAL, correct=True
        )
        metrics.add_evaluation(
            latency=0.7, confidence=0.7, claim_type=ClaimType.FACTUAL, correct=False
        )

        assert metrics.accuracy == pytest.approx(2 / 3, rel=0.01)

    def test_accuracy_none_without_ground_truth(self):
        """Test that accuracy is None when no ground truth is available."""
        metrics = ComponentMetrics(component_name="test")

        metrics.add_evaluation(latency=0.5, confidence=0.8)
        metrics.add_evaluation(latency=0.6, confidence=0.9)

        assert metrics.accuracy is None

    def test_claim_type_accuracy(self):
        """Test claim type specific accuracy."""
        metrics = ComponentMetrics(component_name="test")

        # Add factual claims
        metrics.add_evaluation(
            latency=0.5, confidence=0.8, claim_type=ClaimType.FACTUAL, correct=True
        )
        metrics.add_evaluation(
            latency=0.6, confidence=0.9, claim_type=ClaimType.FACTUAL, correct=False
        )

        # Add temporal claims
        metrics.add_evaluation(
            latency=0.7, confidence=0.85, claim_type=ClaimType.TEMPORAL, correct=True
        )
        metrics.add_evaluation(
            latency=0.8, confidence=0.75, claim_type=ClaimType.TEMPORAL, correct=True
        )

        factual_accuracy = metrics.get_claim_type_accuracy("factual")
        temporal_accuracy = metrics.get_claim_type_accuracy("temporal")

        assert factual_accuracy == pytest.approx(0.5, rel=0.01)
        assert temporal_accuracy == pytest.approx(1.0, rel=0.01)

    def test_to_dict(self):
        """Test conversion to dictionary."""
        metrics = ComponentMetrics(component_name="test")
        metrics.add_evaluation(latency=0.5, confidence=0.8)

        result = metrics.to_dict()

        assert result["component_name"] == "test"
        assert result["total_evaluations"] == 1
        assert result["average_latency"] == 0.5
        assert result["average_confidence"] == 0.8


class TestDisagreement:
    """Test Disagreement class."""

    def test_initialization(self):
        """Test Disagreement initialization."""
        verdict = Verdict(
            label=VerdictLabel.REFUTED,
            confidence=0.9,
            evidence=["Evidence 1"],
            reasoning="Test reasoning",
        )

        judge_result = JudgeResult(
            model_name="test_judge",
            score=85.0,
            reasoning="Judge reasoning",
            confidence=0.8,
        )

        disagreement = Disagreement(
            statement="Test statement",
            verifier_verdict=VerdictLabel.REFUTED,
            judge_consensus=85.0,
            judge_results=[judge_result],
            verifier_confidence=0.9,
            judge_confidence=0.8,
            claim_type=ClaimType.FACTUAL,
        )

        assert disagreement.statement == "Test statement"
        assert disagreement.verifier_verdict == VerdictLabel.REFUTED
        assert disagreement.judge_consensus == 85.0
        assert len(disagreement.judge_results) == 1
        assert disagreement.verifier_confidence == 0.9
        assert disagreement.judge_confidence == 0.8
        assert disagreement.claim_type == ClaimType.FACTUAL

    def test_to_dict(self):
        """Test conversion to dictionary."""
        judge_result = JudgeResult(
            model_name="test_judge",
            score=85.0,
            reasoning="Judge reasoning",
            confidence=0.8,
        )

        disagreement = Disagreement(
            statement="Test statement",
            verifier_verdict=VerdictLabel.REFUTED,
            judge_consensus=85.0,
            judge_results=[judge_result],
            verifier_confidence=0.9,
            judge_confidence=0.8,
        )

        result = disagreement.to_dict()

        assert result["statement"] == "Test statement"
        assert result["verifier_verdict"] == "REFUTED"
        assert result["judge_consensus"] == 85.0
        assert len(result["judge_results"]) == 1


class TestPerformanceTracker:
    """Test PerformanceTracker class."""

    def test_initialization(self):
        """Test PerformanceTracker initialization."""
        tracker = PerformanceTracker()

        assert tracker.verifier_metrics.component_name == "specialized_verifier"
        assert tracker.judge_metrics.component_name == "judge_ensemble"
        assert tracker.disagreements == []

    def test_verifier_timing(self):
        """Test verifier timing functionality."""
        tracker = PerformanceTracker()

        tracker.start_verifier_timing()
        # Simulate some work
        import time
        time.sleep(0.01)
        latency = tracker.end_verifier_timing()

        assert latency > 0
        assert latency < 1.0  # Should be very quick

    def test_judge_timing(self):
        """Test judge timing functionality."""
        tracker = PerformanceTracker()

        tracker.start_judge_timing()
        # Simulate some work
        import time
        time.sleep(0.01)
        latency = tracker.end_judge_timing()

        assert latency > 0
        assert latency < 1.0  # Should be very quick

    def test_timing_not_started_error(self):
        """Test that ending timing without starting raises error."""
        tracker = PerformanceTracker()

        with pytest.raises(RuntimeError, match="timing was not started"):
            tracker.end_verifier_timing()

        with pytest.raises(RuntimeError, match="timing was not started"):
            tracker.end_judge_timing()

    def test_record_verifier_result(self):
        """Test recording verifier results."""
        tracker = PerformanceTracker()

        verdict = Verdict(
            label=VerdictLabel.SUPPORTED,
            confidence=0.85,
            evidence=["Evidence"],
            reasoning="Reasoning",
        )

        tracker.record_verifier_result(
            verdict=verdict,
            latency=0.5,
            claim_type=ClaimType.FACTUAL,
            correct=True,
        )

        assert tracker.verifier_metrics.total_evaluations == 1
        assert tracker.verifier_metrics.total_latency == 0.5
        assert tracker.verifier_metrics.confidence_scores == [0.85]

    def test_record_judge_results(self):
        """Test recording judge results."""
        tracker = PerformanceTracker()

        judge_results = [
            JudgeResult(
                model_name="judge1",
                score=80.0,
                reasoning="Reasoning 1",
                confidence=0.8,
            ),
            JudgeResult(
                model_name="judge2",
                score=85.0,
                reasoning="Reasoning 2",
                confidence=0.9,
            ),
        ]

        tracker.record_judge_results(
            judge_results=judge_results,
            latency=1.0,
            claim_type=ClaimType.TEMPORAL,
            correct=True,
        )

        assert tracker.judge_metrics.total_evaluations == 1
        assert tracker.judge_metrics.total_latency == 1.0
        # Average confidence should be (0.8 + 0.9) / 2 = 0.85
        assert len(tracker.judge_metrics.confidence_scores) == 1
        assert tracker.judge_metrics.confidence_scores[0] == pytest.approx(0.85, rel=0.01)

    def test_log_disagreement_refuted_vs_high_score(self):
        """Test logging disagreement when verifier says REFUTED but judges give high score."""
        tracker = PerformanceTracker()

        verdict = Verdict(
            label=VerdictLabel.REFUTED,
            confidence=0.9,
            evidence=["Evidence"],
            reasoning="This is refuted",
        )

        judge_results = [
            JudgeResult(
                model_name="judge1",
                score=80.0,
                reasoning="Looks good",
                confidence=0.8,
            ),
        ]

        tracker.log_disagreement(
            statement="Test statement",
            verifier_verdict=verdict,
            judge_results=judge_results,
            judge_consensus_score=80.0,
        )

        assert len(tracker.disagreements) == 1
        assert tracker.disagreements[0].verifier_verdict == VerdictLabel.REFUTED
        assert tracker.disagreements[0].judge_consensus == 80.0

    def test_log_disagreement_supported_vs_low_score(self):
        """Test logging disagreement when verifier says SUPPORTED but judges give low score."""
        tracker = PerformanceTracker()

        verdict = Verdict(
            label=VerdictLabel.SUPPORTED,
            confidence=0.9,
            evidence=["Evidence"],
            reasoning="This is supported",
        )

        judge_results = [
            JudgeResult(
                model_name="judge1",
                score=20.0,
                reasoning="Not good",
                confidence=0.8,
            ),
        ]

        tracker.log_disagreement(
            statement="Test statement",
            verifier_verdict=verdict,
            judge_results=judge_results,
            judge_consensus_score=20.0,
        )

        assert len(tracker.disagreements) == 1

    def test_no_disagreement_when_aligned(self):
        """Test that no disagreement is logged when verifier and judges agree."""
        tracker = PerformanceTracker()

        verdict = Verdict(
            label=VerdictLabel.SUPPORTED,
            confidence=0.9,
            evidence=["Evidence"],
            reasoning="This is supported",
        )

        judge_results = [
            JudgeResult(
                model_name="judge1",
                score=85.0,
                reasoning="Looks good",
                confidence=0.8,
            ),
        ]

        tracker.log_disagreement(
            statement="Test statement",
            verifier_verdict=verdict,
            judge_results=judge_results,
            judge_consensus_score=85.0,
        )

        assert len(tracker.disagreements) == 0

    def test_generate_report(self):
        """Test generating performance report."""
        tracker = PerformanceTracker()

        # Add some verifier results
        verdict = Verdict(
            label=VerdictLabel.SUPPORTED,
            confidence=0.85,
            evidence=["Evidence"],
            reasoning="Reasoning",
        )
        tracker.record_verifier_result(verdict, latency=0.5)

        # Add some judge results
        judge_results = [
            JudgeResult(
                model_name="judge1",
                score=80.0,
                reasoning="Reasoning",
                confidence=0.8,
            ),
        ]
        tracker.record_judge_results(judge_results, latency=1.0)

        report = tracker.generate_report()

        assert "verifier_metrics" in report
        assert "judge_metrics" in report
        assert "disagreements" in report
        assert "comparative_analysis" in report

        assert report["verifier_metrics"]["total_evaluations"] == 1
        assert report["judge_metrics"]["total_evaluations"] == 1

    def test_comparative_analysis_latency(self):
        """Test comparative analysis for latency."""
        tracker = PerformanceTracker()

        # Verifier is faster
        verdict = Verdict(label=VerdictLabel.SUPPORTED, confidence=0.85)
        tracker.record_verifier_result(verdict, latency=0.3)

        judge_results = [
            JudgeResult(model_name="judge1", score=80.0, reasoning="R", confidence=0.8)
        ]
        tracker.record_judge_results(judge_results, latency=1.0)

        report = tracker.generate_report()
        analysis = report["comparative_analysis"]

        assert analysis["latency_comparison"]["verifier_avg"] == 0.3
        assert analysis["latency_comparison"]["judge_avg"] == 1.0
        assert analysis["latency_comparison"]["faster_component"] == "verifier"

    def test_comparative_analysis_confidence(self):
        """Test comparative analysis for confidence."""
        tracker = PerformanceTracker()

        # Judge is more confident
        verdict = Verdict(label=VerdictLabel.SUPPORTED, confidence=0.7)
        tracker.record_verifier_result(verdict, latency=0.5)

        judge_results = [
            JudgeResult(model_name="judge1", score=80.0, reasoning="R", confidence=0.9)
        ]
        tracker.record_judge_results(judge_results, latency=1.0)

        report = tracker.generate_report()
        analysis = report["comparative_analysis"]

        assert analysis["confidence_comparison"]["verifier_avg"] == 0.7
        assert analysis["confidence_comparison"]["judge_avg"] == 0.9
        assert analysis["confidence_comparison"]["more_confident_component"] == "judge_ensemble"

    def test_claim_type_performance_comparison(self):
        """Test claim type performance comparison."""
        tracker = PerformanceTracker()

        # Verifier better on factual claims
        verdict = Verdict(label=VerdictLabel.SUPPORTED, confidence=0.85)
        tracker.record_verifier_result(
            verdict, latency=0.5, claim_type=ClaimType.FACTUAL, correct=True
        )
        tracker.record_verifier_result(
            verdict, latency=0.5, claim_type=ClaimType.FACTUAL, correct=True
        )

        # Judge better on temporal claims
        judge_results = [
            JudgeResult(model_name="judge1", score=80.0, reasoning="R", confidence=0.8)
        ]
        tracker.record_judge_results(
            judge_results, latency=1.0, claim_type=ClaimType.FACTUAL, correct=False
        )
        tracker.record_judge_results(
            judge_results, latency=1.0, claim_type=ClaimType.TEMPORAL, correct=True
        )

        report = tracker.generate_report()
        analysis = report["comparative_analysis"]["claim_type_performance"]

        assert "factual" in analysis
        assert analysis["factual"]["verifier"]["accuracy"] == 1.0
        assert analysis["factual"]["judge"]["accuracy"] == 0.0
        assert analysis["factual"]["better_component"] == "verifier"

    def test_reset(self):
        """Test resetting performance tracker."""
        tracker = PerformanceTracker()

        # Add some data
        verdict = Verdict(label=VerdictLabel.SUPPORTED, confidence=0.85)
        tracker.record_verifier_result(verdict, latency=0.5)

        judge_results = [
            JudgeResult(model_name="judge1", score=80.0, reasoning="R", confidence=0.8)
        ]
        tracker.record_judge_results(judge_results, latency=1.0)

        # Reset
        tracker.reset()

        assert tracker.verifier_metrics.total_evaluations == 0
        assert tracker.judge_metrics.total_evaluations == 0
        assert tracker.disagreements == []

    def test_get_summary(self):
        """Test getting human-readable summary."""
        tracker = PerformanceTracker()

        # Add some data
        verdict = Verdict(label=VerdictLabel.SUPPORTED, confidence=0.85)
        tracker.record_verifier_result(verdict, latency=0.5)

        judge_results = [
            JudgeResult(model_name="judge1", score=80.0, reasoning="R", confidence=0.8)
        ]
        tracker.record_judge_results(judge_results, latency=1.0)

        summary = tracker.get_summary()

        assert "Performance Tracker Summary" in summary
        assert "Verifier Metrics" in summary
        assert "Judge Ensemble Metrics" in summary
        assert "Total Evaluations: 1" in summary

    def test_disagreement_rate_calculation(self):
        """Test disagreement rate calculation."""
        tracker = PerformanceTracker()

        # Add 10 evaluations
        for i in range(10):
            verdict = Verdict(label=VerdictLabel.SUPPORTED, confidence=0.85)
            tracker.record_verifier_result(verdict, latency=0.5)

            judge_results = [
                JudgeResult(
                    model_name="judge1", score=80.0, reasoning="R", confidence=0.8
                )
            ]
            tracker.record_judge_results(judge_results, latency=1.0)

        # Add 2 disagreements
        for i in range(2):
            verdict = Verdict(label=VerdictLabel.REFUTED, confidence=0.9)
            judge_results = [
                JudgeResult(
                    model_name="judge1", score=85.0, reasoning="R", confidence=0.8
                )
            ]
            tracker.log_disagreement(
                statement="Test",
                verifier_verdict=verdict,
                judge_results=judge_results,
                judge_consensus_score=85.0,
            )

        report = tracker.generate_report()
        disagreement_rate = report["disagreements"]["disagreement_rate"]

        assert disagreement_rate == pytest.approx(0.2, rel=0.01)  # 2/10 = 0.2
