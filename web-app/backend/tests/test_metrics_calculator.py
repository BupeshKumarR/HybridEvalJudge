"""
Tests for metrics calculator module.
"""
import pytest
import numpy as np
from uuid import uuid4

from app.services.metrics_calculator import (
    MetricsCalculator,
    HallucinationMetrics,
    ConfidenceMetrics,
    InterJudgeAgreement
)
from app.models import JudgeResult, VerifierVerdict, FlaggedIssue
from app.schemas import IssueType, IssueSeverity, VerifierLabel


class TestHallucinationScoreCalculation:
    """Tests for hallucination score calculation."""
    
    def test_perfect_score_low_hallucination(self):
        """Test that perfect consensus score results in low hallucination score."""
        # Create judge results with high scores
        judge_results = [
            self._create_judge_result(score=95.0, confidence=0.9),
            self._create_judge_result(score=98.0, confidence=0.95)
        ]
        
        verifier_verdicts = []
        consensus_score = 96.5
        
        metrics = MetricsCalculator.calculate_hallucination_score(
            judge_results,
            verifier_verdicts,
            consensus_score
        )
        
        assert isinstance(metrics, HallucinationMetrics)
        assert metrics.overall_score < 20  # Low hallucination
        assert 0 <= metrics.overall_score <= 100
    
    def test_low_score_high_hallucination(self):
        """Test that low consensus score results in high hallucination score."""
        # Create judge results with low scores and issues
        judge_results = [
            self._create_judge_result(score=30.0, confidence=0.7, num_issues=3),
            self._create_judge_result(score=25.0, confidence=0.65, num_issues=2)
        ]
        
        verifier_verdicts = []
        consensus_score = 27.5
        
        metrics = MetricsCalculator.calculate_hallucination_score(
            judge_results,
            verifier_verdicts,
            consensus_score
        )
        
        # Low score should result in higher hallucination than high score
        assert metrics.overall_score > 30  # Significant hallucination
        assert 0 <= metrics.overall_score <= 100
    
    def test_refuted_claims_increase_hallucination(self):
        """Test that refuted verifier verdicts increase hallucination score."""
        judge_results = [
            self._create_judge_result(score=70.0, confidence=0.8)
        ]
        
        # All claims refuted
        verifier_verdicts = [
            self._create_verifier_verdict(label=VerifierLabel.REFUTED),
            self._create_verifier_verdict(label=VerifierLabel.REFUTED),
            self._create_verifier_verdict(label=VerifierLabel.REFUTED)
        ]
        
        consensus_score = 70.0
        
        metrics = MetricsCalculator.calculate_hallucination_score(
            judge_results,
            verifier_verdicts,
            consensus_score
        )
        
        # Should have higher hallucination due to refuted claims
        assert metrics.overall_score > 30
    
    def test_supported_claims_low_hallucination(self):
        """Test that supported verifier verdicts result in lower hallucination score."""
        judge_results = [
            self._create_judge_result(score=85.0, confidence=0.9)
        ]
        
        # All claims supported
        verifier_verdicts = [
            self._create_verifier_verdict(label=VerifierLabel.SUPPORTED),
            self._create_verifier_verdict(label=VerifierLabel.SUPPORTED)
        ]
        
        consensus_score = 85.0
        
        metrics = MetricsCalculator.calculate_hallucination_score(
            judge_results,
            verifier_verdicts,
            consensus_score
        )
        
        assert metrics.overall_score < 30
    
    def test_breakdown_by_issue_type(self):
        """Test that breakdown by issue type is calculated correctly."""
        judge_results = [
            self._create_judge_result(
                score=60.0,
                confidence=0.8,
                issue_types=[IssueType.FACTUAL_ERROR, IssueType.HALLUCINATION],
                severities=[IssueSeverity.MEDIUM, IssueSeverity.HIGH]
            )
        ]
        
        verifier_verdicts = []
        consensus_score = 60.0
        
        metrics = MetricsCalculator.calculate_hallucination_score(
            judge_results,
            verifier_verdicts,
            consensus_score
        )
        
        assert isinstance(metrics.breakdown_by_type, dict)
        assert IssueType.FACTUAL_ERROR.value in metrics.breakdown_by_type
        assert IssueType.HALLUCINATION.value in metrics.breakdown_by_type
        # Check that the types with issues have non-zero scores
        assert metrics.breakdown_by_type[IssueType.FACTUAL_ERROR.value] > 0
        assert metrics.breakdown_by_type[IssueType.HALLUCINATION.value] > 0
    
    def test_severity_distribution(self):
        """Test that severity distribution is calculated correctly."""
        judge_results = [
            self._create_judge_result(
                score=50.0,
                confidence=0.7,
                issue_types=[IssueType.FACTUAL_ERROR, IssueType.HALLUCINATION, IssueType.UNSUPPORTED_CLAIM],
                severities=[IssueSeverity.HIGH, IssueSeverity.MEDIUM, IssueSeverity.LOW]
            )
        ]
        
        verifier_verdicts = []
        consensus_score = 50.0
        
        metrics = MetricsCalculator.calculate_hallucination_score(
            judge_results,
            verifier_verdicts,
            consensus_score
        )
        
        assert isinstance(metrics.severity_distribution, dict)
        assert metrics.severity_distribution[IssueSeverity.HIGH.value] == 1
        assert metrics.severity_distribution[IssueSeverity.MEDIUM.value] == 1
        assert metrics.severity_distribution[IssueSeverity.LOW.value] == 1
    
    def test_affected_text_spans(self):
        """Test that affected text spans are extracted correctly."""
        judge_results = [
            self._create_judge_result(
                score=60.0,
                confidence=0.8,
                issue_types=[IssueType.FACTUAL_ERROR, IssueType.HALLUCINATION],
                severities=[IssueSeverity.MEDIUM, IssueSeverity.HIGH],
                text_spans=[(10, 20), (30, 40)]
            )
        ]
        
        verifier_verdicts = []
        consensus_score = 60.0
        
        metrics = MetricsCalculator.calculate_hallucination_score(
            judge_results,
            verifier_verdicts,
            consensus_score
        )
        
        assert len(metrics.affected_text_spans) == 2
        assert metrics.affected_text_spans[0][0] == 10
        assert metrics.affected_text_spans[0][1] == 20
    
    # Helper methods
    def _create_judge_result(
        self,
        score: float,
        confidence: float,
        num_issues: int = 0,
        issue_types: list = None,
        severities: list = None,
        text_spans: list = None
    ) -> JudgeResult:
        """Create a mock judge result."""
        session_id = uuid4()
        judge_result = JudgeResult(
            id=uuid4(),
            session_id=session_id,
            judge_name="test_judge",
            score=score,
            confidence=confidence,
            reasoning="Test reasoning"
        )
        
        # Add flagged issues
        if issue_types and severities:
            for i, (issue_type, severity) in enumerate(zip(issue_types, severities)):
                text_span = text_spans[i] if text_spans and i < len(text_spans) else None
                issue = FlaggedIssue(
                    id=uuid4(),
                    judge_result_id=judge_result.id,
                    issue_type=issue_type,
                    severity=severity,
                    description=f"Test issue {i}",
                    text_span_start=text_span[0] if text_span else None,
                    text_span_end=text_span[1] if text_span else None
                )
                judge_result.flagged_issues.append(issue)
        elif num_issues > 0:
            for i in range(num_issues):
                issue = FlaggedIssue(
                    id=uuid4(),
                    judge_result_id=judge_result.id,
                    issue_type=IssueType.FACTUAL_ERROR,
                    severity=IssueSeverity.MEDIUM,
                    description=f"Test issue {i}"
                )
                judge_result.flagged_issues.append(issue)
        
        return judge_result
    
    def _create_verifier_verdict(
        self,
        label: VerifierLabel = VerifierLabel.SUPPORTED
    ) -> VerifierVerdict:
        """Create a mock verifier verdict."""
        return VerifierVerdict(
            id=uuid4(),
            session_id=uuid4(),
            claim_text="Test claim",
            label=label,
            confidence=0.9,
            reasoning="Test reasoning"
        )


class TestConfidenceMetrics:
    """Tests for confidence metrics calculation."""
    
    def test_high_confidence_narrow_interval(self):
        """Test that high confidence results in narrow confidence interval."""
        judge_results = [
            self._create_judge_result(score=85.0, confidence=0.95),
            self._create_judge_result(score=87.0, confidence=0.93),
            self._create_judge_result(score=86.0, confidence=0.94)
        ]
        
        metrics = MetricsCalculator.calculate_confidence_metrics(judge_results)
        
        assert isinstance(metrics, ConfidenceMetrics)
        assert metrics.mean_confidence > 0.9
        assert not metrics.is_low_confidence
        
        ci_width = metrics.confidence_interval[1] - metrics.confidence_interval[0]
        assert ci_width < 20  # Narrow interval
    
    def test_low_confidence_wide_interval(self):
        """Test that low confidence results in wide confidence interval."""
        judge_results = [
            self._create_judge_result(score=50.0, confidence=0.6),
            self._create_judge_result(score=80.0, confidence=0.65),
            self._create_judge_result(score=30.0, confidence=0.55)
        ]
        
        metrics = MetricsCalculator.calculate_confidence_metrics(judge_results)
        
        assert metrics.mean_confidence < 0.7
        # Wide variance in scores should result in low confidence flag
    
    def test_bootstrap_confidence_interval(self):
        """Test that bootstrap confidence interval is calculated correctly."""
        judge_results = [
            self._create_judge_result(score=70.0, confidence=0.8),
            self._create_judge_result(score=75.0, confidence=0.85),
            self._create_judge_result(score=72.0, confidence=0.82)
        ]
        
        metrics = MetricsCalculator.calculate_confidence_metrics(
            judge_results,
            confidence_level=0.95
        )
        
        assert metrics.confidence_level == 0.95
        assert metrics.confidence_interval[0] < metrics.confidence_interval[1]
        assert 0 <= metrics.confidence_interval[0] <= 100
        assert 0 <= metrics.confidence_interval[1] <= 100
    
    def test_empty_judge_results(self):
        """Test handling of empty judge results."""
        metrics = MetricsCalculator.calculate_confidence_metrics([])
        
        assert metrics.mean_confidence == 0.0
        assert metrics.is_low_confidence
    
    def _create_judge_result(self, score: float, confidence: float) -> JudgeResult:
        """Create a mock judge result."""
        return JudgeResult(
            id=uuid4(),
            session_id=uuid4(),
            judge_name="test_judge",
            score=score,
            confidence=confidence,
            reasoning="Test reasoning"
        )


class TestInterJudgeAgreement:
    """Tests for inter-judge agreement calculation."""
    
    def test_two_judges_perfect_agreement(self):
        """Test Cohen's Kappa for 2 judges with perfect agreement."""
        judge_results = [
            self._create_judge_result(score=85.0),  # Category 4 (excellent)
            self._create_judge_result(score=87.0)   # Category 4 (excellent)
        ]
        
        metrics = MetricsCalculator.calculate_inter_judge_agreement(judge_results)
        
        assert isinstance(metrics, InterJudgeAgreement)
        assert metrics.cohens_kappa is not None
        assert metrics.cohens_kappa == 1.0  # Perfect agreement
        assert metrics.interpretation == "almost_perfect"
    
    def test_two_judges_no_agreement(self):
        """Test Cohen's Kappa for 2 judges with no agreement."""
        judge_results = [
            self._create_judge_result(score=15.0),  # Category 0 (poor)
            self._create_judge_result(score=85.0)   # Category 4 (excellent)
        ]
        
        metrics = MetricsCalculator.calculate_inter_judge_agreement(judge_results)
        
        assert metrics.cohens_kappa is not None
        assert metrics.cohens_kappa < 0  # No agreement
    
    def test_three_judges_fleiss_kappa(self):
        """Test Fleiss' Kappa for 3+ judges."""
        judge_results = [
            self._create_judge_result(score=85.0),  # Category 4
            self._create_judge_result(score=87.0),  # Category 4
            self._create_judge_result(score=82.0)   # Category 4
        ]
        
        metrics = MetricsCalculator.calculate_inter_judge_agreement(judge_results)
        
        assert metrics.fleiss_kappa is not None
        assert metrics.cohens_kappa is None  # Not calculated for 3+ judges
        assert metrics.fleiss_kappa > 0.6  # Good agreement
    
    def test_pairwise_correlations(self):
        """Test pairwise correlation calculation."""
        judge_results = [
            self._create_judge_result(score=80.0, name="judge1"),
            self._create_judge_result(score=85.0, name="judge2"),
            self._create_judge_result(score=82.0, name="judge3")
        ]
        
        metrics = MetricsCalculator.calculate_inter_judge_agreement(judge_results)
        
        assert isinstance(metrics.pairwise_correlations, dict)
        assert "judge1" in metrics.pairwise_correlations
        assert "judge2" in metrics.pairwise_correlations["judge1"]
        assert 0 <= metrics.pairwise_correlations["judge1"]["judge2"] <= 1
    
    def test_single_judge_insufficient(self):
        """Test handling of single judge (insufficient for agreement)."""
        judge_results = [
            self._create_judge_result(score=80.0)
        ]
        
        metrics = MetricsCalculator.calculate_inter_judge_agreement(judge_results)
        
        assert metrics.cohens_kappa is None
        assert metrics.fleiss_kappa is None
        assert metrics.interpretation == "insufficient_judges"
    
    def _create_judge_result(self, score: float, name: str = "test_judge") -> JudgeResult:
        """Create a mock judge result."""
        return JudgeResult(
            id=uuid4(),
            session_id=uuid4(),
            judge_name=name,
            score=score,
            confidence=0.9,
            reasoning="Test reasoning"
        )


class TestStatisticalMetrics:
    """Tests for statistical metrics calculation."""
    
    def test_basic_statistics(self):
        """Test calculation of basic statistical metrics."""
        judge_results = [
            self._create_judge_result(score=70.0),
            self._create_judge_result(score=80.0),
            self._create_judge_result(score=75.0),
            self._create_judge_result(score=85.0)
        ]
        
        metrics = MetricsCalculator.calculate_statistical_metrics(judge_results)
        
        assert 'variance' in metrics
        assert 'standard_deviation' in metrics
        assert 'mean' in metrics
        assert 'median' in metrics
        assert 'min' in metrics
        assert 'max' in metrics
        
        assert metrics['mean'] == 77.5
        assert metrics['min'] == 70.0
        assert metrics['max'] == 85.0
    
    def test_quartiles(self):
        """Test quartile calculation."""
        judge_results = [
            self._create_judge_result(score=60.0),
            self._create_judge_result(score=70.0),
            self._create_judge_result(score=80.0),
            self._create_judge_result(score=90.0)
        ]
        
        metrics = MetricsCalculator.calculate_statistical_metrics(judge_results)
        
        assert 'quartiles' in metrics
        assert len(metrics['quartiles']) == 3
        assert metrics['quartiles'][1] == metrics['median']  # Q2 = median
    
    def test_score_distribution(self):
        """Test score distribution histogram."""
        judge_results = [
            self._create_judge_result(score=15.0),   # 0-20
            self._create_judge_result(score=35.0),   # 20-40
            self._create_judge_result(score=55.0),   # 40-60
            self._create_judge_result(score=75.0),   # 60-80
            self._create_judge_result(score=95.0)    # 80-100
        ]
        
        metrics = MetricsCalculator.calculate_statistical_metrics(judge_results)
        
        assert 'score_distribution' in metrics
        distribution = metrics['score_distribution']
        
        assert distribution['0-20'] == 1
        assert distribution['20-40'] == 1
        assert distribution['40-60'] == 1
        assert distribution['60-80'] == 1
        assert distribution['80-100'] == 1
    
    def test_empty_results(self):
        """Test handling of empty results."""
        metrics = MetricsCalculator.calculate_statistical_metrics([])
        
        assert metrics['variance'] == 0.0
        assert metrics['standard_deviation'] == 0.0
        assert metrics['mean'] == 0.0
    
    def _create_judge_result(self, score: float) -> JudgeResult:
        """Create a mock judge result."""
        return JudgeResult(
            id=uuid4(),
            session_id=uuid4(),
            judge_name="test_judge",
            score=score,
            confidence=0.9,
            reasoning="Test reasoning"
        )


class TestAggregateStatistics:
    """Tests for aggregate statistics across sessions."""
    
    def test_aggregate_across_sessions(self):
        """Test calculation of aggregate statistics."""
        sessions = [
            {
                'consensus_score': 80.0,
                'hallucination_score': 15.0,
                'confidence_interval_lower': 75.0,
                'confidence_interval_upper': 85.0
            },
            {
                'consensus_score': 85.0,
                'hallucination_score': 10.0,
                'confidence_interval_lower': 80.0,
                'confidence_interval_upper': 90.0
            },
            {
                'consensus_score': 75.0,
                'hallucination_score': 20.0,
                'confidence_interval_lower': 70.0,
                'confidence_interval_upper': 80.0
            }
        ]
        
        metrics = MetricsCalculator.calculate_aggregate_statistics(sessions)
        
        assert metrics['total_sessions'] == 3
        assert metrics['average_consensus_score'] == 80.0
        assert metrics['average_hallucination_score'] == 15.0
        assert len(metrics['score_trend']) == 3
    
    def test_empty_sessions(self):
        """Test handling of empty sessions."""
        metrics = MetricsCalculator.calculate_aggregate_statistics([])
        
        assert metrics['total_sessions'] == 0
        assert metrics['average_consensus_score'] == 0.0
    
    def test_score_trend(self):
        """Test score trend calculation."""
        sessions = [
            {'consensus_score': score}
            for score in range(70, 85)  # 15 sessions
        ]
        
        metrics = MetricsCalculator.calculate_aggregate_statistics(sessions)
        
        # Should only include last 10
        assert len(metrics['score_trend']) == 10
        assert metrics['score_trend'][0] == 75  # Last 10 start from 75 (70-84, last 10 are 75-84)
