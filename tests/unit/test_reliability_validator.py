"""
Unit tests for the ReliabilityValidator component.
"""

import pytest
import numpy as np

from llm_judge_auditor.components.reliability_validator import (
    ReliabilityValidator,
    ConsistencyReport,
    AgreementReport,
    RankingCorrelationReport,
)


class TestReliabilityValidator:
    """Test suite for ReliabilityValidator."""
    
    def test_initialization(self):
        """Test validator initialization."""
        validator = ReliabilityValidator()
        assert validator.consistency_threshold == 5.0
        
        validator_custom = ReliabilityValidator(consistency_threshold=10.0)
        assert validator_custom.consistency_threshold == 10.0
    
    def test_check_consistency_valid(self):
        """Test consistency checking with valid scores."""
        validator = ReliabilityValidator()
        scores = [85.0, 87.0, 84.5, 86.0, 85.5]
        
        report = validator.check_consistency(scores)
        
        assert isinstance(report, ConsistencyReport)
        assert report.num_evaluations == 5
        assert 84.0 < report.mean_score < 88.0
        assert report.variance < 5.0  # Should be consistent
        assert report.is_consistent is True
        assert report.scores == scores
    
    def test_check_consistency_inconsistent(self):
        """Test consistency checking with inconsistent scores."""
        validator = ReliabilityValidator()
        scores = [50.0, 80.0, 30.0, 90.0, 20.0]  # High variance
        
        report = validator.check_consistency(scores)
        
        assert isinstance(report, ConsistencyReport)
        assert report.variance > 5.0  # Should be inconsistent
        assert report.is_consistent is False
    
    def test_check_consistency_empty_list(self):
        """Test consistency checking with empty list."""
        validator = ReliabilityValidator()
        
        with pytest.raises(ValueError, match="scores list cannot be empty"):
            validator.check_consistency([])
    
    def test_check_consistency_single_score(self):
        """Test consistency checking with single score."""
        validator = ReliabilityValidator()
        
        with pytest.raises(ValueError, match="at least 2 elements"):
            validator.check_consistency([85.0])
    
    def test_check_consistency_perfect(self):
        """Test consistency checking with identical scores."""
        validator = ReliabilityValidator()
        scores = [85.0, 85.0, 85.0, 85.0]
        
        report = validator.check_consistency(scores)
        
        assert report.variance == 0.0
        assert report.std_deviation == 0.0
        assert report.is_consistent is True
    
    def test_calculate_inter_model_agreement_high(self):
        """Test inter-model agreement with high agreement."""
        validator = ReliabilityValidator()
        
        # Three judges with high agreement
        judge_scores = {
            "judge_1": [85.0, 45.0, 90.0, 30.0, 75.0],
            "judge_2": [80.0, 40.0, 88.0, 35.0, 70.0],
            "judge_3": [82.0, 48.0, 92.0, 32.0, 73.0],
        }
        
        report = validator.calculate_inter_model_agreement(judge_scores, threshold=50.0)
        
        assert isinstance(report, AgreementReport)
        assert report.num_models == 3
        assert 0.0 <= report.cohens_kappa <= 1.0
        assert report.agreement_level in ["poor", "slight", "fair", "moderate", "substantial", "almost_perfect"]
        assert len(report.pairwise_agreements) == 3  # 3 choose 2 = 3 pairs
    
    def test_calculate_inter_model_agreement_perfect(self):
        """Test inter-model agreement with perfect agreement."""
        validator = ReliabilityValidator()
        
        # Three judges with identical scores
        judge_scores = {
            "judge_1": [85.0, 45.0, 90.0, 30.0],
            "judge_2": [85.0, 45.0, 90.0, 30.0],
            "judge_3": [85.0, 45.0, 90.0, 30.0],
        }
        
        report = validator.calculate_inter_model_agreement(judge_scores, threshold=50.0)
        
        assert report.cohens_kappa == 1.0
        assert report.agreement_level == "almost_perfect"
    
    def test_calculate_inter_model_agreement_low(self):
        """Test inter-model agreement with low agreement."""
        validator = ReliabilityValidator()
        
        # Three judges with low agreement
        judge_scores = {
            "judge_1": [85.0, 45.0, 90.0, 30.0],
            "judge_2": [45.0, 85.0, 30.0, 90.0],  # Opposite pattern
            "judge_3": [50.0, 50.0, 50.0, 50.0],  # All at threshold
        }
        
        report = validator.calculate_inter_model_agreement(judge_scores, threshold=50.0)
        
        assert isinstance(report, AgreementReport)
        # Low agreement expected
        assert report.cohens_kappa < 0.6
    
    def test_calculate_inter_model_agreement_empty(self):
        """Test inter-model agreement with empty dict."""
        validator = ReliabilityValidator()
        
        with pytest.raises(ValueError, match="judge_scores cannot be empty"):
            validator.calculate_inter_model_agreement({})
    
    def test_calculate_inter_model_agreement_mismatched_lengths(self):
        """Test inter-model agreement with mismatched score lengths."""
        validator = ReliabilityValidator()
        
        judge_scores = {
            "judge_1": [85.0, 45.0, 90.0],
            "judge_2": [80.0, 40.0],  # Different length
        }
        
        with pytest.raises(ValueError, match="same number of scores"):
            validator.calculate_inter_model_agreement(judge_scores)
    
    def test_calculate_inter_model_agreement_insufficient_scores(self):
        """Test inter-model agreement with insufficient scores."""
        validator = ReliabilityValidator()
        
        judge_scores = {
            "judge_1": [85.0],
            "judge_2": [80.0],
        }
        
        with pytest.raises(ValueError, match="at least 2 scores"):
            validator.calculate_inter_model_agreement(judge_scores)
    
    def test_calculate_ranking_correlation_perfect(self):
        """Test ranking correlation with perfect correlation."""
        validator = ReliabilityValidator()
        
        predicted = [("A", "B"), ("C", "D"), ("E", "F")]
        ground_truth = [("A", "B"), ("C", "D"), ("E", "F")]
        
        report = validator.calculate_ranking_correlation(predicted, ground_truth)
        
        assert isinstance(report, RankingCorrelationReport)
        assert report.num_pairs == 3
        # Perfect agreement: all rankings match
        # When all values are identical (all 1s), Kendall's tau is 0 (no variance)
        # but Spearman's rho should be 1
        assert report.spearmans_rho == 1.0
    
    def test_calculate_ranking_correlation_opposite(self):
        """Test ranking correlation with opposite rankings."""
        validator = ReliabilityValidator()
        
        predicted = [("A", "B"), ("C", "D"), ("E", "F")]
        ground_truth = [("B", "A"), ("D", "C"), ("F", "E")]  # All reversed
        
        report = validator.calculate_ranking_correlation(predicted, ground_truth)
        
        assert isinstance(report, RankingCorrelationReport)
        # Opposite rankings: all disagreements (all -1s)
        # When all values are identical (all -1s), Kendall's tau is 0 (no variance)
        # but Spearman's rho should be 1 (perfect correlation of disagreement)
        assert report.spearmans_rho == 1.0
    
    def test_calculate_ranking_correlation_mixed(self):
        """Test ranking correlation with mixed rankings."""
        validator = ReliabilityValidator()
        
        predicted = [("A", "B"), ("C", "D"), ("E", "F"), ("G", "H")]
        ground_truth = [("A", "B"), ("D", "C"), ("E", "F"), ("H", "G")]  # 50% match
        
        report = validator.calculate_ranking_correlation(predicted, ground_truth)
        
        assert isinstance(report, RankingCorrelationReport)
        assert report.num_pairs == 4
        # Mixed correlation
        assert -1.0 <= report.kendalls_tau <= 1.0
        assert -1.0 <= report.spearmans_rho <= 1.0
    
    def test_calculate_ranking_correlation_empty(self):
        """Test ranking correlation with empty lists."""
        validator = ReliabilityValidator()
        
        with pytest.raises(ValueError, match="predicted_rankings cannot be empty"):
            validator.calculate_ranking_correlation([], [])
    
    def test_calculate_ranking_correlation_mismatched_lengths(self):
        """Test ranking correlation with mismatched lengths."""
        validator = ReliabilityValidator()
        
        predicted = [("A", "B"), ("C", "D")]
        ground_truth = [("A", "B")]
        
        with pytest.raises(ValueError, match="same length"):
            validator.calculate_ranking_correlation(predicted, ground_truth)
    
    def test_calculate_ranking_correlation_mismatched_pairs(self):
        """Test ranking correlation with mismatched pairs."""
        validator = ReliabilityValidator()
        
        predicted = [("A", "B")]
        ground_truth = [("C", "D")]  # Different pair
        
        with pytest.raises(ValueError, match="Mismatched pairs"):
            validator.calculate_ranking_correlation(predicted, ground_truth)
    
    def test_cohens_kappa_calculation(self):
        """Test Cohen's kappa calculation directly."""
        validator = ReliabilityValidator()
        
        # Perfect agreement
        ratings_a = [1, 1, 0, 0, 1]
        ratings_b = [1, 1, 0, 0, 1]
        kappa = validator._calculate_cohens_kappa(ratings_a, ratings_b)
        assert kappa == 1.0
        
        # No agreement (opposite)
        ratings_a = [1, 1, 0, 0]
        ratings_b = [0, 0, 1, 1]
        kappa = validator._calculate_cohens_kappa(ratings_a, ratings_b)
        assert kappa < 0.0
        
        # Random agreement
        ratings_a = [1, 0, 1, 0]
        ratings_b = [1, 1, 0, 0]
        kappa = validator._calculate_cohens_kappa(ratings_a, ratings_b)
        assert -1.0 <= kappa <= 1.0
    
    def test_interpret_kappa(self):
        """Test kappa interpretation."""
        validator = ReliabilityValidator()
        
        assert validator._interpret_kappa(-0.1) == "poor"
        assert validator._interpret_kappa(0.1) == "slight"
        assert validator._interpret_kappa(0.3) == "fair"
        assert validator._interpret_kappa(0.5) == "moderate"
        assert validator._interpret_kappa(0.7) == "substantial"
        assert validator._interpret_kappa(0.9) == "almost_perfect"
    
    def test_convert_to_ranks(self):
        """Test rank conversion."""
        validator = ReliabilityValidator()
        
        # Simple case
        values = [10.0, 20.0, 30.0]
        ranks = validator._convert_to_ranks(values)
        assert ranks == [1.0, 2.0, 3.0]
        
        # With ties
        values = [10.0, 20.0, 20.0, 30.0]
        ranks = validator._convert_to_ranks(values)
        assert ranks == [1.0, 2.5, 2.5, 4.0]  # Tied values get average rank
        
        # Reverse order
        values = [30.0, 20.0, 10.0]
        ranks = validator._convert_to_ranks(values)
        assert ranks == [3.0, 2.0, 1.0]


class TestConsistencyReport:
    """Test ConsistencyReport dataclass."""
    
    def test_consistency_report_creation(self):
        """Test creating a ConsistencyReport."""
        report = ConsistencyReport(
            mean_score=85.0,
            variance=2.5,
            std_deviation=1.58,
            is_consistent=True,
            num_evaluations=5,
            scores=[84.0, 85.0, 86.0, 85.5, 84.5],
        )
        
        assert report.mean_score == 85.0
        assert report.variance == 2.5
        assert report.is_consistent is True


class TestAgreementReport:
    """Test AgreementReport dataclass."""
    
    def test_agreement_report_creation(self):
        """Test creating an AgreementReport."""
        report = AgreementReport(
            cohens_kappa=0.75,
            agreement_level="substantial",
            num_models=3,
            pairwise_agreements={
                ("judge_1", "judge_2"): 0.8,
                ("judge_1", "judge_3"): 0.7,
                ("judge_2", "judge_3"): 0.75,
            },
        )
        
        assert report.cohens_kappa == 0.75
        assert report.agreement_level == "substantial"
        assert report.num_models == 3


class TestRankingCorrelationReport:
    """Test RankingCorrelationReport dataclass."""
    
    def test_ranking_correlation_report_creation(self):
        """Test creating a RankingCorrelationReport."""
        report = RankingCorrelationReport(
            kendalls_tau=0.85,
            kendalls_tau_p_value=0.01,
            spearmans_rho=0.90,
            spearmans_rho_p_value=0.005,
            num_pairs=10,
            is_significant=True,
        )
        
        assert report.kendalls_tau == 0.85
        assert report.spearmans_rho == 0.90
        assert report.is_significant is True
