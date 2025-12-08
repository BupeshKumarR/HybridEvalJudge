"""
Unit tests for the AggregationEngine component.
"""

import pytest

from llm_judge_auditor.components.aggregation_engine import (
    AggregationEngine,
    AggregationStrategy,
)
from llm_judge_auditor.models import (
    Issue,
    IssueType,
    IssueSeverity,
    JudgeResult,
    Verdict,
    VerdictLabel,
)


class TestAggregationEngine:
    """Test suite for AggregationEngine."""

    def test_initialization(self):
        """Test AggregationEngine initialization."""
        engine = AggregationEngine()
        assert engine.strategy == AggregationStrategy.MEAN
        assert engine.disagreement_threshold == 20.0
        assert engine.weights == {}

    def test_initialization_with_params(self):
        """Test AggregationEngine initialization with custom parameters."""
        weights = {"judge1": 0.6, "judge2": 0.4}
        engine = AggregationEngine(
            strategy=AggregationStrategy.WEIGHTED_AVERAGE,
            disagreement_threshold=15.0,
            weights=weights,
        )
        assert engine.strategy == AggregationStrategy.WEIGHTED_AVERAGE
        assert engine.disagreement_threshold == 15.0
        assert engine.weights == weights

    def test_set_strategy(self):
        """Test changing aggregation strategy."""
        engine = AggregationEngine()
        engine.set_strategy(AggregationStrategy.MEDIAN)
        assert engine.strategy == AggregationStrategy.MEDIAN

    def test_set_weights(self):
        """Test setting weights."""
        engine = AggregationEngine()
        weights = {"judge1": 0.5, "judge2": 0.5}
        engine.set_weights(weights)
        assert engine.weights == weights

    def test_aggregate_mean(self):
        """Test mean aggregation strategy."""
        engine = AggregationEngine(strategy=AggregationStrategy.MEAN)

        judge_results = [
            JudgeResult("judge1", 80.0, "Good", [], 0.9),
            JudgeResult("judge2", 90.0, "Excellent", [], 0.95),
            JudgeResult("judge3", 70.0, "Fair", [], 0.85),
        ]

        score, metadata = engine.aggregate_scores(judge_results)

        assert score == 80.0  # (80 + 90 + 70) / 3
        assert metadata.strategy == "mean"
        assert metadata.individual_scores == {
            "judge1": 80.0,
            "judge2": 90.0,
            "judge3": 70.0,
        }
        # Variance is 100, which exceeds threshold of 20
        assert metadata.is_low_confidence

    def test_aggregate_median(self):
        """Test median aggregation strategy."""
        engine = AggregationEngine(strategy=AggregationStrategy.MEDIAN)

        judge_results = [
            JudgeResult("judge1", 80.0, "Good", [], 0.9),
            JudgeResult("judge2", 90.0, "Excellent", [], 0.95),
            JudgeResult("judge3", 70.0, "Fair", [], 0.85),
        ]

        score, metadata = engine.aggregate_scores(judge_results)

        assert score == 80.0  # Median of [70, 80, 90]
        assert metadata.strategy == "median"

    def test_aggregate_weighted_average(self):
        """Test weighted average aggregation strategy."""
        weights = {"judge1": 0.5, "judge2": 0.3, "judge3": 0.2}
        engine = AggregationEngine(
            strategy=AggregationStrategy.WEIGHTED_AVERAGE, weights=weights
        )

        judge_results = [
            JudgeResult("judge1", 80.0, "Good", [], 0.9),
            JudgeResult("judge2", 90.0, "Excellent", [], 0.95),
            JudgeResult("judge3", 70.0, "Fair", [], 0.85),
        ]

        score, metadata = engine.aggregate_scores(judge_results)

        expected = (80.0 * 0.5 + 90.0 * 0.3 + 70.0 * 0.2)  # 81.0
        assert score == pytest.approx(expected)
        assert metadata.strategy == "weighted_average"
        assert metadata.weights == weights

    def test_aggregate_weighted_average_no_weights(self):
        """Test weighted average falls back to mean when no weights provided."""
        engine = AggregationEngine(strategy=AggregationStrategy.WEIGHTED_AVERAGE)

        judge_results = [
            JudgeResult("judge1", 80.0, "Good", [], 0.9),
            JudgeResult("judge2", 90.0, "Excellent", [], 0.95),
        ]

        score, metadata = engine.aggregate_scores(judge_results)

        assert score == 85.0  # Falls back to mean
        assert metadata.strategy == "weighted_average"

    def test_aggregate_majority_vote(self):
        """Test majority vote aggregation strategy."""
        engine = AggregationEngine(strategy=AggregationStrategy.MAJORITY_VOTE)

        # Most scores in high range [67-100]
        judge_results = [
            JudgeResult("judge1", 80.0, "Good", [], 0.9),
            JudgeResult("judge2", 90.0, "Excellent", [], 0.95),
            JudgeResult("judge3", 85.0, "Very good", [], 0.92),
            JudgeResult("judge4", 30.0, "Poor", [], 0.7),  # Outlier
        ]

        score, metadata = engine.aggregate_scores(judge_results)

        # Majority is in high bin, should return median of [80, 85, 90]
        assert score == 85.0
        assert metadata.strategy == "majority_vote"

    def test_disagreement_detection_low_variance(self):
        """Test disagreement detection with low variance."""
        engine = AggregationEngine(disagreement_threshold=20.0)

        judge_results = [
            JudgeResult("judge1", 80.0, "Good", [], 0.9),
            JudgeResult("judge2", 82.0, "Good", [], 0.9),
            JudgeResult("judge3", 78.0, "Good", [], 0.9),
        ]

        score, metadata = engine.aggregate_scores(judge_results)

        assert not metadata.is_low_confidence
        assert metadata.variance < 20.0

    def test_disagreement_detection_high_variance(self):
        """Test disagreement detection with high variance."""
        engine = AggregationEngine(disagreement_threshold=20.0)

        judge_results = [
            JudgeResult("judge1", 90.0, "Excellent", [], 0.9),
            JudgeResult("judge2", 50.0, "Average", [], 0.8),
            JudgeResult("judge3", 30.0, "Poor", [], 0.7),
        ]

        score, metadata = engine.aggregate_scores(judge_results)

        assert metadata.is_low_confidence
        assert metadata.variance > 20.0

    def test_detect_disagreement_method(self):
        """Test the detect_disagreement method."""
        engine = AggregationEngine(disagreement_threshold=20.0)

        judge_results = [
            JudgeResult("judge1", 90.0, "Excellent", [], 0.9),
            JudgeResult("judge2", 50.0, "Average", [], 0.8),
            JudgeResult("judge3", 30.0, "Poor", [], 0.7),
        ]

        disagreement = engine.detect_disagreement(judge_results)

        assert disagreement["has_disagreement"] is True
        assert disagreement["variance"] > 20.0
        assert disagreement["score_range"] == (30.0, 90.0)
        assert isinstance(disagreement["outliers"], list)

    def test_detect_disagreement_no_disagreement(self):
        """Test detect_disagreement with consistent scores."""
        engine = AggregationEngine(disagreement_threshold=20.0)

        judge_results = [
            JudgeResult("judge1", 80.0, "Good", [], 0.9),
            JudgeResult("judge2", 82.0, "Good", [], 0.9),
            JudgeResult("judge3", 78.0, "Good", [], 0.9),
        ]

        disagreement = engine.detect_disagreement(judge_results)

        assert disagreement["has_disagreement"] is False
        assert disagreement["variance"] < 20.0

    def test_detect_disagreement_empty_results(self):
        """Test detect_disagreement with empty results."""
        engine = AggregationEngine()

        disagreement = engine.detect_disagreement([])

        assert disagreement["has_disagreement"] is False
        assert disagreement["variance"] == 0.0
        assert disagreement["score_range"] == (0.0, 0.0)
        assert disagreement["outliers"] == []

    def test_incorporate_verifier_verdicts_refuted(self):
        """Test score adjustment when verifier finds refuted claims."""
        engine = AggregationEngine()

        judge_results = [
            JudgeResult("judge1", 80.0, "Good", [], 0.9),
            JudgeResult("judge2", 85.0, "Very good", [], 0.95),
        ]

        # Many refuted claims should lower the score
        verifier_verdicts = [
            Verdict(VerdictLabel.REFUTED, 0.9, ["evidence1"], "Refuted"),
            Verdict(VerdictLabel.REFUTED, 0.85, ["evidence2"], "Refuted"),
            Verdict(VerdictLabel.SUPPORTED, 0.8, ["evidence3"], "Supported"),
        ]

        score, metadata = engine.aggregate_scores(judge_results, verifier_verdicts)

        # Score should be lower than mean (82.5) due to refuted claims
        assert score < 82.5

    def test_incorporate_verifier_verdicts_supported(self):
        """Test score adjustment when verifier finds supported claims."""
        engine = AggregationEngine()

        judge_results = [
            JudgeResult("judge1", 80.0, "Good", [], 0.9),
            JudgeResult("judge2", 85.0, "Very good", [], 0.95),
        ]

        # Most claims supported should boost the score slightly
        verifier_verdicts = [
            Verdict(VerdictLabel.SUPPORTED, 0.9, ["evidence1"], "Supported"),
            Verdict(VerdictLabel.SUPPORTED, 0.95, ["evidence2"], "Supported"),
            Verdict(VerdictLabel.SUPPORTED, 0.92, ["evidence3"], "Supported"),
            Verdict(VerdictLabel.SUPPORTED, 0.88, ["evidence4"], "Supported"),
            Verdict(VerdictLabel.SUPPORTED, 0.91, ["evidence5"], "Supported"),
        ]

        score, metadata = engine.aggregate_scores(judge_results, verifier_verdicts)

        # Score should be slightly higher than mean (82.5) due to supported claims
        assert score >= 82.5

    def test_incorporate_verifier_verdicts_mixed(self):
        """Test score adjustment with mixed verifier verdicts."""
        engine = AggregationEngine()

        judge_results = [
            JudgeResult("judge1", 80.0, "Good", [], 0.9),
        ]

        # Mixed verdicts should have minimal adjustment
        verifier_verdicts = [
            Verdict(VerdictLabel.SUPPORTED, 0.9, ["evidence1"], "Supported"),
            Verdict(VerdictLabel.REFUTED, 0.85, ["evidence2"], "Refuted"),
            Verdict(
                VerdictLabel.NOT_ENOUGH_INFO, 0.7, ["evidence3"], "Not enough info"
            ),
        ]

        score, metadata = engine.aggregate_scores(judge_results, verifier_verdicts)

        # Score should be close to original (80.0), with 1/3 refuted (33% > 30% threshold)
        # so a penalty is applied
        assert 70.0 <= score <= 85.0

    def test_aggregate_scores_empty_results_raises_error(self):
        """Test that aggregating empty results raises ValueError."""
        engine = AggregationEngine()

        with pytest.raises(ValueError, match="Cannot aggregate empty judge results"):
            engine.aggregate_scores([])

    def test_aggregate_scores_single_judge(self):
        """Test aggregation with a single judge."""
        engine = AggregationEngine()

        judge_results = [
            JudgeResult("judge1", 75.0, "Good", [], 0.9),
        ]

        score, metadata = engine.aggregate_scores(judge_results)

        assert score == 75.0
        assert metadata.variance == 0.0
        assert not metadata.is_low_confidence

    def test_aggregate_scores_with_flagged_issues(self):
        """Test aggregation with judges that flagged issues."""
        engine = AggregationEngine()

        judge_results = [
            JudgeResult(
                "judge1",
                70.0,
                "Some issues",
                [
                    Issue(
                        IssueType.HALLUCINATION,
                        IssueSeverity.MEDIUM,
                        "Unsupported claim",
                        [],
                    )
                ],
                0.85,
            ),
            JudgeResult("judge2", 80.0, "Minor issues", [], 0.9),
        ]

        score, metadata = engine.aggregate_scores(judge_results)

        assert score == 75.0  # Mean of 70 and 80
        assert metadata.individual_scores == {"judge1": 70.0, "judge2": 80.0}

    def test_score_clamping_with_verifier(self):
        """Test that scores are clamped to [0, 100] range after verifier adjustment."""
        engine = AggregationEngine()

        judge_results = [
            JudgeResult("judge1", 10.0, "Poor", [], 0.7),
        ]

        # All refuted should apply large penalty
        verifier_verdicts = [
            Verdict(VerdictLabel.REFUTED, 0.9, ["evidence1"], "Refuted"),
            Verdict(VerdictLabel.REFUTED, 0.9, ["evidence2"], "Refuted"),
            Verdict(VerdictLabel.REFUTED, 0.9, ["evidence3"], "Refuted"),
        ]

        score, metadata = engine.aggregate_scores(judge_results, verifier_verdicts)

        # Score should be clamped to 0
        assert score >= 0.0
        assert score <= 100.0

    def test_weighted_average_with_missing_weights(self):
        """Test weighted average with some judges missing weights."""
        weights = {"judge1": 0.7}  # judge2 missing, should default to 1.0
        engine = AggregationEngine(
            strategy=AggregationStrategy.WEIGHTED_AVERAGE, weights=weights
        )

        judge_results = [
            JudgeResult("judge1", 80.0, "Good", [], 0.9),
            JudgeResult("judge2", 90.0, "Excellent", [], 0.95),
        ]

        score, metadata = engine.aggregate_scores(judge_results)

        # (80 * 0.7 + 90 * 1.0) / (0.7 + 1.0) = 146 / 1.7 ≈ 85.88
        expected = (80.0 * 0.7 + 90.0 * 1.0) / (0.7 + 1.0)
        assert score == pytest.approx(expected)

    def test_majority_vote_with_tie(self):
        """Test majority vote when bins are tied."""
        engine = AggregationEngine(strategy=AggregationStrategy.MAJORITY_VOTE)

        # Equal distribution across bins
        judge_results = [
            JudgeResult("judge1", 20.0, "Poor", [], 0.7),  # Low bin
            JudgeResult("judge2", 50.0, "Average", [], 0.8),  # Medium bin
            JudgeResult("judge3", 80.0, "Good", [], 0.9),  # High bin
        ]

        score, metadata = engine.aggregate_scores(judge_results)

        # Should return median of one of the bins
        assert 0.0 <= score <= 100.0

    def test_outlier_detection(self):
        """Test outlier detection in disagreement analysis."""
        engine = AggregationEngine()

        # Create results where outlier detection should work
        # IQR method works best with symmetric distributions
        judge_results = [
            JudgeResult("judge1", 75.0, "Good", [], 0.9),
            JudgeResult("judge2", 78.0, "Good", [], 0.9),
            JudgeResult("judge3", 80.0, "Good", [], 0.9),
            JudgeResult("judge4", 82.0, "Good", [], 0.9),
            JudgeResult("judge5", 85.0, "Good", [], 0.9),
            JudgeResult("judge6", 150.0, "Outlier", [], 0.9),  # Clear outlier above upper bound
        ]

        disagreement = engine.detect_disagreement(judge_results)

        # With this distribution, judge6 should be detected as outlier
        assert "judge6" in disagreement["outliers"]

    def test_metadata_structure(self):
        """Test that aggregation metadata has correct structure."""
        engine = AggregationEngine(strategy=AggregationStrategy.MEAN)

        judge_results = [
            JudgeResult("judge1", 80.0, "Good", [], 0.9),
            JudgeResult("judge2", 85.0, "Very good", [], 0.95),
        ]

        score, metadata = engine.aggregate_scores(judge_results)

        assert hasattr(metadata, "strategy")
        assert hasattr(metadata, "individual_scores")
        assert hasattr(metadata, "variance")
        assert hasattr(metadata, "is_low_confidence")
        assert hasattr(metadata, "weights")

        assert isinstance(metadata.strategy, str)
        assert isinstance(metadata.individual_scores, dict)
        assert isinstance(metadata.variance, float)
        assert isinstance(metadata.is_low_confidence, bool)
