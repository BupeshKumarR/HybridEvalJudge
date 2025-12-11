"""Unit tests for the HallucinationMetricsCalculator component."""

import json
import pytest

from llm_judge_auditor.components.hallucination_metrics import (
    ConsensusF1Result,
    HallucinationMetricsCalculator,
    HallucinationMetricsConfig,
    HallucinationProfile,
    KappaResult,
    MaHRResult,
    MiHRResult,
    ReliabilityLevel,
    UncertaintyResult,
)
from llm_judge_auditor.models import Claim, ClaimType, Verdict, VerdictLabel


class TestHallucinationMetricsCalculator:
    """Tests for HallucinationMetricsCalculator."""

    def test_init_default_config(self):
        """Test initialization with default config."""
        calc = HallucinationMetricsCalculator()
        assert calc.config.mihr_high_risk_threshold == 0.3
        assert calc.config.kappa_low_threshold == 0.4
        assert calc.config.uncertainty_high_threshold == 0.8

    def test_init_custom_config(self):
        """Test initialization with custom config."""
        config = HallucinationMetricsConfig(
            mihr_high_risk_threshold=0.5,
            kappa_low_threshold=0.3,
            uncertainty_high_threshold=0.9,
        )
        calc = HallucinationMetricsCalculator(config=config)
        assert calc.config.mihr_high_risk_threshold == 0.5
        assert calc.config.kappa_low_threshold == 0.3
        assert calc.config.uncertainty_high_threshold == 0.9


class TestMiHRComputation:
    """Tests for MiHR computation."""

    def test_mihr_empty_verdicts(self):
        """Test MiHR with empty verdicts returns None."""
        calc = HallucinationMetricsCalculator()
        result = calc.compute_mihr([])
        
        assert result.value is None
        assert result.has_claims is False
        assert result.total_claims == 0
        assert result.unsupported_claims == 0

    def test_mihr_all_supported(self):
        """Test MiHR with all supported claims."""
        calc = HallucinationMetricsCalculator()
        verdicts = [
            Verdict(label=VerdictLabel.SUPPORTED, confidence=0.9),
            Verdict(label=VerdictLabel.SUPPORTED, confidence=0.8),
            Verdict(label=VerdictLabel.SUPPORTED, confidence=0.7),
        ]
        result = calc.compute_mihr(verdicts)
        
        assert result.value == 0.0
        assert result.has_claims is True
        assert result.total_claims == 3
        assert result.unsupported_claims == 0

    def test_mihr_all_refuted(self):
        """Test MiHR with all refuted claims."""
        calc = HallucinationMetricsCalculator()
        verdicts = [
            Verdict(label=VerdictLabel.REFUTED, confidence=0.9),
            Verdict(label=VerdictLabel.REFUTED, confidence=0.8),
        ]
        result = calc.compute_mihr(verdicts)
        
        assert result.value == 1.0
        assert result.has_claims is True
        assert result.total_claims == 2
        assert result.unsupported_claims == 2

    def test_mihr_mixed_verdicts(self):
        """Test MiHR with mixed verdicts."""
        calc = HallucinationMetricsCalculator()
        verdicts = [
            Verdict(label=VerdictLabel.SUPPORTED, confidence=0.9),
            Verdict(label=VerdictLabel.REFUTED, confidence=0.8),
            Verdict(label=VerdictLabel.NOT_ENOUGH_INFO, confidence=0.7),
            Verdict(label=VerdictLabel.SUPPORTED, confidence=0.6),
        ]
        result = calc.compute_mihr(verdicts)
        
        # 2 unsupported (REFUTED + NOT_ENOUGH_INFO) out of 4
        assert result.value == 0.5
        assert result.has_claims is True
        assert result.total_claims == 4
        assert result.unsupported_claims == 2

    def test_mihr_not_enough_info_counts_as_unsupported(self):
        """Test that NOT_ENOUGH_INFO is counted as unsupported."""
        calc = HallucinationMetricsCalculator()
        verdicts = [
            Verdict(label=VerdictLabel.NOT_ENOUGH_INFO, confidence=0.9),
        ]
        result = calc.compute_mihr(verdicts)
        
        assert result.value == 1.0
        assert result.unsupported_claims == 1



class TestMaHRComputation:
    """Tests for MaHR computation."""

    def test_mahr_empty_responses(self):
        """Test MaHR with empty responses."""
        calc = HallucinationMetricsCalculator()
        result = calc.compute_mahr([])
        
        assert result.value == 0.0
        assert result.total_responses == 0
        assert result.responses_with_hallucinations == 0

    def test_mahr_no_hallucinations(self):
        """Test MaHR with no hallucinations."""
        calc = HallucinationMetricsCalculator()
        response_verdicts = [
            [Verdict(label=VerdictLabel.SUPPORTED, confidence=0.9)],
            [Verdict(label=VerdictLabel.SUPPORTED, confidence=0.8)],
        ]
        result = calc.compute_mahr(response_verdicts)
        
        assert result.value == 0.0
        assert result.total_responses == 2
        assert result.responses_with_hallucinations == 0

    def test_mahr_all_hallucinations(self):
        """Test MaHR with all responses having hallucinations."""
        calc = HallucinationMetricsCalculator()
        response_verdicts = [
            [Verdict(label=VerdictLabel.REFUTED, confidence=0.9)],
            [Verdict(label=VerdictLabel.NOT_ENOUGH_INFO, confidence=0.8)],
        ]
        result = calc.compute_mahr(response_verdicts)
        
        assert result.value == 1.0
        assert result.total_responses == 2
        assert result.responses_with_hallucinations == 2

    def test_mahr_mixed_responses(self):
        """Test MaHR with mixed responses."""
        calc = HallucinationMetricsCalculator()
        response_verdicts = [
            [Verdict(label=VerdictLabel.SUPPORTED, confidence=0.9)],
            [Verdict(label=VerdictLabel.REFUTED, confidence=0.8)],
            [
                Verdict(label=VerdictLabel.SUPPORTED, confidence=0.7),
                Verdict(label=VerdictLabel.NOT_ENOUGH_INFO, confidence=0.6),
            ],
            [Verdict(label=VerdictLabel.SUPPORTED, confidence=0.9)],
        ]
        result = calc.compute_mahr(response_verdicts)
        
        # 2 responses with hallucinations out of 4
        assert result.value == 0.5
        assert result.total_responses == 4
        assert result.responses_with_hallucinations == 2

    def test_mahr_single_hallucination_in_response(self):
        """Test that a single hallucination marks the entire response."""
        calc = HallucinationMetricsCalculator()
        response_verdicts = [
            [
                Verdict(label=VerdictLabel.SUPPORTED, confidence=0.9),
                Verdict(label=VerdictLabel.SUPPORTED, confidence=0.8),
                Verdict(label=VerdictLabel.REFUTED, confidence=0.7),
                Verdict(label=VerdictLabel.SUPPORTED, confidence=0.6),
            ],
        ]
        result = calc.compute_mahr(response_verdicts)
        
        assert result.value == 1.0
        assert result.responses_with_hallucinations == 1


class TestHighRiskDetection:
    """Tests for high risk detection."""

    def test_high_risk_mihr_threshold(self):
        """Test high risk detection based on MiHR threshold."""
        calc = HallucinationMetricsCalculator()
        
        # Below threshold
        mihr_low = MiHRResult(value=0.2, unsupported_claims=2, total_claims=10, has_claims=True)
        assert calc.is_high_risk(mihr=mihr_low) is False
        
        # Above threshold
        mihr_high = MiHRResult(value=0.4, unsupported_claims=4, total_claims=10, has_claims=True)
        assert calc.is_high_risk(mihr=mihr_high) is True

    def test_high_risk_kappa_threshold(self):
        """Test high risk detection based on Kappa threshold."""
        calc = HallucinationMetricsCalculator()
        
        # Above threshold (good agreement)
        kappa_good = KappaResult(kappa=0.6, interpretation="moderate", observed_agreement=0.7, expected_agreement=0.3)
        assert calc.is_high_risk(kappa=kappa_good) is False
        
        # Below threshold (poor agreement)
        kappa_poor = KappaResult(kappa=0.3, interpretation="fair", observed_agreement=0.5, expected_agreement=0.3)
        assert calc.is_high_risk(kappa=kappa_poor) is True

    def test_high_risk_uncertainty_threshold(self):
        """Test high risk detection based on uncertainty threshold."""
        calc = HallucinationMetricsCalculator()
        
        # Below threshold
        uncertainty_low = UncertaintyResult(
            shannon_entropy=0.5, epistemic=0.2, aleatoric=0.2, total=0.4, is_high_uncertainty=False
        )
        assert calc.is_high_risk(uncertainty=uncertainty_low) is False
        
        # Above threshold
        uncertainty_high = UncertaintyResult(
            shannon_entropy=0.9, epistemic=0.5, aleatoric=0.4, total=0.9, is_high_uncertainty=True
        )
        assert calc.is_high_risk(uncertainty=uncertainty_high) is True

    def test_high_risk_none_values(self):
        """Test high risk detection with None values."""
        calc = HallucinationMetricsCalculator()
        
        # MiHR with None value
        mihr_none = MiHRResult(value=None, unsupported_claims=0, total_claims=0, has_claims=False)
        assert calc.is_high_risk(mihr=mihr_none) is False
        
        # Undefined Kappa
        kappa_undefined = KappaResult(
            kappa=None, interpretation="undefined", observed_agreement=0.0, 
            expected_agreement=0.0, is_undefined=True, error_message="Fewer than 2 judges"
        )
        assert calc.is_high_risk(kappa=kappa_undefined) is False


class TestReliabilityClassification:
    """Tests for reliability classification."""

    def test_reliability_high(self):
        """Test high reliability classification."""
        calc = HallucinationMetricsCalculator()
        
        mihr = MiHRResult(value=0.1, unsupported_claims=1, total_claims=10, has_claims=True)
        kappa = KappaResult(kappa=0.7, interpretation="substantial", observed_agreement=0.8, expected_agreement=0.3)
        uncertainty = UncertaintyResult(
            shannon_entropy=0.3, epistemic=0.1, aleatoric=0.1, total=0.2, is_high_uncertainty=False
        )
        
        reliability = calc.determine_reliability(mihr=mihr, kappa=kappa, uncertainty=uncertainty)
        assert reliability == ReliabilityLevel.HIGH

    def test_reliability_low_due_to_high_risk(self):
        """Test low reliability due to high risk indicators."""
        calc = HallucinationMetricsCalculator()
        
        mihr = MiHRResult(value=0.5, unsupported_claims=5, total_claims=10, has_claims=True)
        
        reliability = calc.determine_reliability(mihr=mihr)
        assert reliability == ReliabilityLevel.LOW

    def test_reliability_medium(self):
        """Test medium reliability classification."""
        calc = HallucinationMetricsCalculator()
        
        # Values that trigger medium indicators but not high risk
        mihr = MiHRResult(value=0.2, unsupported_claims=2, total_claims=10, has_claims=True)
        kappa = KappaResult(kappa=0.5, interpretation="moderate", observed_agreement=0.6, expected_agreement=0.3)
        uncertainty = UncertaintyResult(
            shannon_entropy=0.6, epistemic=0.3, aleatoric=0.3, total=0.6, is_high_uncertainty=False
        )
        
        reliability = calc.determine_reliability(mihr=mihr, kappa=kappa, uncertainty=uncertainty)
        assert reliability == ReliabilityLevel.MEDIUM


class TestDataclassSerialization:
    """Tests for dataclass serialization."""

    def test_mihr_result_fields(self):
        """Test MiHRResult dataclass fields."""
        result = MiHRResult(value=0.5, unsupported_claims=5, total_claims=10, has_claims=True)
        assert result.value == 0.5
        assert result.unsupported_claims == 5
        assert result.total_claims == 10
        assert result.has_claims is True

    def test_mahr_result_fields(self):
        """Test MaHRResult dataclass fields."""
        result = MaHRResult(value=0.3, responses_with_hallucinations=3, total_responses=10)
        assert result.value == 0.3
        assert result.responses_with_hallucinations == 3
        assert result.total_responses == 10

    def test_hallucination_profile_to_json(self):
        """Test HallucinationProfile JSON serialization."""
        profile = HallucinationProfile(
            mihr=MiHRResult(value=0.2, unsupported_claims=2, total_claims=10, has_claims=True),
            mahr=MaHRResult(value=0.3, responses_with_hallucinations=3, total_responses=10),
            factscore=0.8,
            consensus_f1=ConsensusF1Result(precision=0.7, recall=0.8, f1=0.75),
            fleiss_kappa=KappaResult(kappa=0.6, interpretation="moderate", observed_agreement=0.7, expected_agreement=0.3),
            uncertainty=UncertaintyResult(
                shannon_entropy=0.5, epistemic=0.2, aleatoric=0.2, total=0.4, is_high_uncertainty=False
            ),
            reliability=ReliabilityLevel.HIGH,
            disputed_claims=[],
            consensus_claims=[],
            is_high_risk=False,
        )
        
        json_str = profile.to_json()
        assert isinstance(json_str, str)
        
        # Verify it's valid JSON
        data = json.loads(json_str)
        assert data["reliability"] == "high"
        assert data["mihr"]["value"] == 0.2
        assert data["mahr"]["value"] == 0.3

    def test_hallucination_profile_round_trip(self):
        """Test HallucinationProfile JSON round-trip."""
        original = HallucinationProfile(
            mihr=MiHRResult(value=0.2, unsupported_claims=2, total_claims=10, has_claims=True),
            mahr=MaHRResult(value=0.3, responses_with_hallucinations=3, total_responses=10),
            factscore=0.8,
            consensus_f1=ConsensusF1Result(precision=0.7, recall=0.8, f1=0.75),
            fleiss_kappa=KappaResult(kappa=0.6, interpretation="moderate", observed_agreement=0.7, expected_agreement=0.3),
            uncertainty=UncertaintyResult(
                shannon_entropy=0.5, epistemic=0.2, aleatoric=0.2, total=0.4, is_high_uncertainty=False
            ),
            reliability=ReliabilityLevel.HIGH,
            disputed_claims=[
                Claim(text="test claim", source_span=(0, 10), claim_type=ClaimType.FACTUAL)
            ],
            consensus_claims=[],
            is_high_risk=False,
        )
        
        json_str = original.to_json()
        restored = HallucinationProfile.from_json(json_str)
        
        assert restored.mihr.value == original.mihr.value
        assert restored.mahr.value == original.mahr.value
        assert restored.factscore == original.factscore
        assert restored.reliability == original.reliability
        assert len(restored.disputed_claims) == 1
        assert restored.disputed_claims[0].text == "test claim"



class TestFactScoreComputation:
    """Tests for FactScore computation."""

    def test_factscore_empty_verdicts(self):
        """Test FactScore with empty verdicts returns None."""
        calc = HallucinationMetricsCalculator()
        result = calc.compute_factscore([])
        assert result is None

    def test_factscore_all_supported(self):
        """Test FactScore with all supported claims."""
        calc = HallucinationMetricsCalculator()
        verdicts = [
            Verdict(label=VerdictLabel.SUPPORTED, confidence=0.9),
            Verdict(label=VerdictLabel.SUPPORTED, confidence=0.8),
            Verdict(label=VerdictLabel.SUPPORTED, confidence=0.7),
        ]
        result = calc.compute_factscore(verdicts)
        assert result == 1.0

    def test_factscore_none_supported(self):
        """Test FactScore with no supported claims."""
        calc = HallucinationMetricsCalculator()
        verdicts = [
            Verdict(label=VerdictLabel.REFUTED, confidence=0.9),
            Verdict(label=VerdictLabel.NOT_ENOUGH_INFO, confidence=0.8),
        ]
        result = calc.compute_factscore(verdicts)
        assert result == 0.0

    def test_factscore_mixed_verdicts(self):
        """Test FactScore with mixed verdicts."""
        calc = HallucinationMetricsCalculator()
        verdicts = [
            Verdict(label=VerdictLabel.SUPPORTED, confidence=0.9),
            Verdict(label=VerdictLabel.REFUTED, confidence=0.8),
            Verdict(label=VerdictLabel.SUPPORTED, confidence=0.7),
            Verdict(label=VerdictLabel.NOT_ENOUGH_INFO, confidence=0.6),
        ]
        result = calc.compute_factscore(verdicts)
        # 2 supported out of 4
        assert result == 0.5


class TestClaimVerificationMatrix:
    """Tests for ClaimVerificationMatrix."""

    def test_matrix_empty(self):
        """Test empty matrix."""
        from llm_judge_auditor.components.hallucination_metrics import (
            ClaimVerificationMatrix,
        )
        matrix = ClaimVerificationMatrix(claims=[], models=[], support_matrix=[])
        assert len(matrix.claims) == 0
        assert len(matrix.models) == 0
        assert len(matrix.support_matrix) == 0

    def test_matrix_get_claim_support_count(self):
        """Test getting claim support count."""
        from llm_judge_auditor.components.hallucination_metrics import (
            ClaimVerificationMatrix,
        )
        claims = [
            Claim(text="claim1", source_span=(0, 6), claim_type=ClaimType.FACTUAL),
            Claim(text="claim2", source_span=(7, 13), claim_type=ClaimType.FACTUAL),
        ]
        models = ["model_a", "model_b", "model_c"]
        support_matrix = [
            [1, 1, 0],  # claim1 supported by model_a and model_b
            [0, 1, 1],  # claim2 supported by model_b and model_c
        ]
        matrix = ClaimVerificationMatrix(claims=claims, models=models, support_matrix=support_matrix)
        
        assert matrix.get_claim_support_count(0) == 2
        assert matrix.get_claim_support_count(1) == 2
        assert matrix.get_claim_support_count(99) == 0  # Out of bounds

    def test_matrix_get_model_claim_count(self):
        """Test getting model claim count."""
        from llm_judge_auditor.components.hallucination_metrics import (
            ClaimVerificationMatrix,
        )
        claims = [
            Claim(text="claim1", source_span=(0, 6), claim_type=ClaimType.FACTUAL),
            Claim(text="claim2", source_span=(7, 13), claim_type=ClaimType.FACTUAL),
        ]
        models = ["model_a", "model_b", "model_c"]
        support_matrix = [
            [1, 1, 0],
            [0, 1, 1],
        ]
        matrix = ClaimVerificationMatrix(claims=claims, models=models, support_matrix=support_matrix)
        
        assert matrix.get_model_claim_count(0) == 1  # model_a supports 1 claim
        assert matrix.get_model_claim_count(1) == 2  # model_b supports 2 claims
        assert matrix.get_model_claim_count(2) == 1  # model_c supports 1 claim
        assert matrix.get_model_claim_count(99) == 0  # Out of bounds

    def test_matrix_get_consensus_claims(self):
        """Test getting consensus claims."""
        from llm_judge_auditor.components.hallucination_metrics import (
            ClaimVerificationMatrix,
        )
        claims = [
            Claim(text="claim1", source_span=(0, 6), claim_type=ClaimType.FACTUAL),
            Claim(text="claim2", source_span=(7, 13), claim_type=ClaimType.FACTUAL),
            Claim(text="claim3", source_span=(14, 20), claim_type=ClaimType.FACTUAL),
        ]
        models = ["model_a", "model_b", "model_c"]
        support_matrix = [
            [1, 1, 1],  # claim1 supported by all (100%)
            [1, 1, 0],  # claim2 supported by 2/3 (67%)
            [1, 0, 0],  # claim3 supported by 1/3 (33%)
        ]
        matrix = ClaimVerificationMatrix(claims=claims, models=models, support_matrix=support_matrix)
        
        consensus = matrix.get_consensus_claims(threshold=0.5)
        assert len(consensus) == 2  # claim1 and claim2
        assert consensus[0].text == "claim1"
        assert consensus[1].text == "claim2"

    def test_matrix_get_disputed_claims(self):
        """Test getting disputed claims."""
        from llm_judge_auditor.components.hallucination_metrics import (
            ClaimVerificationMatrix,
        )
        claims = [
            Claim(text="claim1", source_span=(0, 6), claim_type=ClaimType.FACTUAL),
            Claim(text="claim2", source_span=(7, 13), claim_type=ClaimType.FACTUAL),
            Claim(text="claim3", source_span=(14, 20), claim_type=ClaimType.FACTUAL),
        ]
        models = ["model_a", "model_b", "model_c"]
        support_matrix = [
            [1, 1, 1],  # claim1 supported by all (100%)
            [1, 1, 0],  # claim2 supported by 2/3 (67%)
            [1, 0, 0],  # claim3 supported by 1/3 (33%)
        ]
        matrix = ClaimVerificationMatrix(claims=claims, models=models, support_matrix=support_matrix)
        
        disputed = matrix.get_disputed_claims(threshold=0.5)
        assert len(disputed) == 1  # Only claim3
        assert disputed[0].text == "claim3"


class TestClaimVerificationMatrixBuilder:
    """Tests for ClaimVerificationMatrixBuilder."""

    def test_builder_empty_input(self):
        """Test building matrix with empty input."""
        from llm_judge_auditor.components.hallucination_metrics import (
            ClaimVerificationMatrixBuilder,
        )
        builder = ClaimVerificationMatrixBuilder()
        matrix = builder.build_matrix({})
        
        assert len(matrix.claims) == 0
        assert len(matrix.models) == 0
        assert len(matrix.support_matrix) == 0

    def test_builder_single_model(self):
        """Test building matrix with single model."""
        from llm_judge_auditor.components.hallucination_metrics import (
            ClaimVerificationMatrixBuilder,
        )
        builder = ClaimVerificationMatrixBuilder()
        
        model_claims = {
            "model_a": [
                Claim(text="claim1", source_span=(0, 6), claim_type=ClaimType.FACTUAL),
                Claim(text="claim2", source_span=(7, 13), claim_type=ClaimType.FACTUAL),
            ]
        }
        
        matrix = builder.build_matrix(model_claims)
        
        assert len(matrix.claims) == 2
        assert len(matrix.models) == 1
        assert matrix.models[0] == "model_a"
        assert len(matrix.support_matrix) == 2
        assert matrix.support_matrix[0] == [1]
        assert matrix.support_matrix[1] == [1]

    def test_builder_multiple_models_shared_claims(self):
        """Test building matrix with multiple models sharing claims."""
        from llm_judge_auditor.components.hallucination_metrics import (
            ClaimVerificationMatrixBuilder,
        )
        builder = ClaimVerificationMatrixBuilder()
        
        model_claims = {
            "model_a": [
                Claim(text="shared claim", source_span=(0, 12), claim_type=ClaimType.FACTUAL),
                Claim(text="unique to a", source_span=(13, 24), claim_type=ClaimType.FACTUAL),
            ],
            "model_b": [
                Claim(text="shared claim", source_span=(0, 12), claim_type=ClaimType.FACTUAL),
                Claim(text="unique to b", source_span=(13, 24), claim_type=ClaimType.FACTUAL),
            ],
        }
        
        matrix = builder.build_matrix(model_claims)
        
        assert len(matrix.claims) == 3  # 1 shared + 2 unique
        assert len(matrix.models) == 2

    def test_builder_with_verdicts(self):
        """Test building matrix with verdicts."""
        from llm_judge_auditor.components.hallucination_metrics import (
            ClaimVerificationMatrixBuilder,
        )
        builder = ClaimVerificationMatrixBuilder()
        
        model_claims = {
            "model_a": [
                Claim(text="claim1", source_span=(0, 6), claim_type=ClaimType.FACTUAL),
            ],
        }
        
        # Only SUPPORTED verdicts should count as support
        model_verdicts = {
            "model_a": [
                Verdict(label=VerdictLabel.REFUTED, confidence=0.9),
            ],
        }
        
        matrix = builder.build_matrix(model_claims, model_verdicts)
        
        assert len(matrix.claims) == 1
        assert matrix.support_matrix[0][0] == 0  # Not supported because verdict is REFUTED


class TestConsensusF1Computation:
    """Tests for Consensus F1 computation."""

    def test_consensus_f1_model_not_in_matrix(self):
        """Test Consensus F1 when model is not in matrix."""
        from llm_judge_auditor.components.hallucination_metrics import (
            ClaimVerificationMatrix,
        )
        calc = HallucinationMetricsCalculator()
        
        matrix = ClaimVerificationMatrix(
            claims=[Claim(text="claim1", source_span=(0, 6), claim_type=ClaimType.FACTUAL)],
            models=["model_a"],
            support_matrix=[[1]]
        )
        
        result = calc.compute_consensus_f1(matrix, "nonexistent_model")
        
        assert result.precision == 0.0
        assert result.recall == 0.0
        assert result.f1 == 0.0

    def test_consensus_f1_empty_matrix(self):
        """Test Consensus F1 with empty matrix."""
        from llm_judge_auditor.components.hallucination_metrics import (
            ClaimVerificationMatrix,
        )
        calc = HallucinationMetricsCalculator()
        
        matrix = ClaimVerificationMatrix(claims=[], models=[], support_matrix=[])
        
        result = calc.compute_consensus_f1(matrix, "model_a")
        
        assert result.precision == 0.0
        assert result.recall == 0.0
        assert result.f1 == 0.0

    def test_consensus_f1_perfect_agreement(self):
        """Test Consensus F1 with perfect agreement."""
        from llm_judge_auditor.components.hallucination_metrics import (
            ClaimVerificationMatrix,
        )
        calc = HallucinationMetricsCalculator()
        
        claims = [
            Claim(text="claim1", source_span=(0, 6), claim_type=ClaimType.FACTUAL),
            Claim(text="claim2", source_span=(7, 13), claim_type=ClaimType.FACTUAL),
        ]
        models = ["model_a", "model_b"]
        support_matrix = [
            [1, 1],  # Both models support claim1
            [1, 1],  # Both models support claim2
        ]
        matrix = ClaimVerificationMatrix(claims=claims, models=models, support_matrix=support_matrix)
        
        result = calc.compute_consensus_f1(matrix, "model_a", consensus_threshold=0.5)
        
        # Perfect agreement: precision=1.0, recall=1.0, f1=1.0
        assert result.precision == 1.0
        assert result.recall == 1.0
        assert result.f1 == 1.0

    def test_consensus_f1_no_support(self):
        """Test Consensus F1 when model supports no claims."""
        from llm_judge_auditor.components.hallucination_metrics import (
            ClaimVerificationMatrix,
        )
        calc = HallucinationMetricsCalculator()
        
        claims = [
            Claim(text="claim1", source_span=(0, 6), claim_type=ClaimType.FACTUAL),
        ]
        models = ["model_a", "model_b"]
        support_matrix = [
            [0, 1],  # Only model_b supports claim1
        ]
        matrix = ClaimVerificationMatrix(claims=claims, models=models, support_matrix=support_matrix)
        
        result = calc.compute_consensus_f1(matrix, "model_a", consensus_threshold=0.5)
        
        # model_a supports nothing, so precision=0, recall=0, f1=0
        assert result.precision == 0.0
        assert result.recall == 0.0
        assert result.f1 == 0.0

    def test_average_consensus_f1(self):
        """Test average Consensus F1 across all models."""
        from llm_judge_auditor.components.hallucination_metrics import (
            ClaimVerificationMatrix,
        )
        calc = HallucinationMetricsCalculator()
        
        claims = [
            Claim(text="claim1", source_span=(0, 6), claim_type=ClaimType.FACTUAL),
            Claim(text="claim2", source_span=(7, 13), claim_type=ClaimType.FACTUAL),
        ]
        models = ["model_a", "model_b"]
        support_matrix = [
            [1, 1],
            [1, 1],
        ]
        matrix = ClaimVerificationMatrix(claims=claims, models=models, support_matrix=support_matrix)
        
        result = calc.compute_average_consensus_f1(matrix, consensus_threshold=0.5)
        
        # Both models have perfect agreement
        assert result.precision == 1.0
        assert result.recall == 1.0
        assert result.f1 == 1.0

    def test_average_consensus_f1_empty_matrix(self):
        """Test average Consensus F1 with empty matrix."""
        from llm_judge_auditor.components.hallucination_metrics import (
            ClaimVerificationMatrix,
        )
        calc = HallucinationMetricsCalculator()
        
        matrix = ClaimVerificationMatrix(claims=[], models=[], support_matrix=[])
        
        result = calc.compute_average_consensus_f1(matrix)
        
        assert result.precision == 0.0
        assert result.recall == 0.0
        assert result.f1 == 0.0


class TestFleissKappaComputation:
    """Tests for Fleiss' Kappa computation."""

    def test_fleiss_kappa_empty_ratings(self):
        """Test Fleiss' Kappa with empty ratings returns undefined."""
        calc = HallucinationMetricsCalculator()
        result = calc.compute_fleiss_kappa([], num_categories=3)
        
        assert result.is_undefined is True
        assert result.kappa is None
        assert result.error_message == "No ratings provided"

    def test_fleiss_kappa_fewer_than_2_judges(self):
        """Test Fleiss' Kappa with fewer than 2 judges returns undefined."""
        calc = HallucinationMetricsCalculator()
        # Only 1 rater per item
        ratings = [
            [1, 0, 0],  # Item 1: 1 rater chose category 0
            [0, 1, 0],  # Item 2: 1 rater chose category 1
        ]
        result = calc.compute_fleiss_kappa(ratings, num_categories=3)
        
        assert result.is_undefined is True
        assert result.kappa is None
        assert "Fewer than 2 judges" in result.error_message

    def test_fleiss_kappa_perfect_agreement(self):
        """Test Fleiss' Kappa with perfect agreement."""
        calc = HallucinationMetricsCalculator()
        # 3 raters, all agree on each item
        ratings = [
            [3, 0, 0],  # All 3 raters chose category 0
            [0, 3, 0],  # All 3 raters chose category 1
            [0, 0, 3],  # All 3 raters chose category 2
        ]
        result = calc.compute_fleiss_kappa(ratings, num_categories=3)
        
        assert result.is_undefined is False
        assert result.kappa is not None
        assert result.kappa == pytest.approx(1.0, abs=0.01)
        assert result.interpretation == "almost_perfect"
        assert result.observed_agreement == pytest.approx(1.0, abs=0.01)

    def test_fleiss_kappa_no_agreement(self):
        """Test Fleiss' Kappa with no agreement (worse than chance)."""
        calc = HallucinationMetricsCalculator()
        # 3 raters, each chooses a different category for each item
        # This represents systematic disagreement (worse than random chance)
        ratings = [
            [1, 1, 1],  # Each rater chose a different category
            [1, 1, 1],
            [1, 1, 1],
        ]
        result = calc.compute_fleiss_kappa(ratings, num_categories=3)
        
        assert result.is_undefined is False
        assert result.kappa is not None
        # With systematic disagreement, kappa is negative (-0.5 in this case)
        # This is mathematically correct: agreement worse than chance
        assert result.kappa < 0
        assert result.interpretation == "poor"

    def test_fleiss_kappa_moderate_agreement(self):
        """Test Fleiss' Kappa with moderate agreement."""
        calc = HallucinationMetricsCalculator()
        # 4 raters with some agreement
        ratings = [
            [3, 1, 0],  # 3 agree on category 0, 1 on category 1
            [2, 2, 0],  # 2 on each of first two categories
            [0, 3, 1],  # 3 agree on category 1, 1 on category 2
            [4, 0, 0],  # All 4 agree on category 0
        ]
        result = calc.compute_fleiss_kappa(ratings, num_categories=3)
        
        assert result.is_undefined is False
        assert result.kappa is not None
        # Should be somewhere between 0 and 1
        assert 0.0 < result.kappa < 1.0
        assert result.observed_agreement > result.expected_agreement

    def test_fleiss_kappa_mismatched_categories(self):
        """Test Fleiss' Kappa with mismatched category counts."""
        calc = HallucinationMetricsCalculator()
        ratings = [
            [2, 1, 0],  # 3 categories
            [2, 1],     # Only 2 categories - mismatch
        ]
        result = calc.compute_fleiss_kappa(ratings, num_categories=3)
        
        assert result.is_undefined is True
        assert result.kappa is None
        assert "categories" in result.error_message.lower()

    def test_fleiss_kappa_mismatched_raters(self):
        """Test Fleiss' Kappa with different number of raters per item."""
        calc = HallucinationMetricsCalculator()
        ratings = [
            [2, 1, 0],  # 3 raters
            [3, 1, 0],  # 4 raters - mismatch
        ]
        result = calc.compute_fleiss_kappa(ratings, num_categories=3)
        
        assert result.is_undefined is True
        assert result.kappa is None
        assert "raters" in result.error_message.lower()


class TestKappaInterpretation:
    """Tests for Kappa interpretation."""

    def test_interpret_kappa_poor(self):
        """Test interpretation for poor agreement (< 0.2)."""
        calc = HallucinationMetricsCalculator()
        assert calc.interpret_kappa(0.0) == "poor"
        assert calc.interpret_kappa(0.1) == "poor"
        assert calc.interpret_kappa(0.19) == "poor"
        assert calc.interpret_kappa(-0.5) == "poor"

    def test_interpret_kappa_fair(self):
        """Test interpretation for fair agreement (0.2 - 0.4)."""
        calc = HallucinationMetricsCalculator()
        assert calc.interpret_kappa(0.2) == "fair"
        assert calc.interpret_kappa(0.3) == "fair"
        assert calc.interpret_kappa(0.39) == "fair"

    def test_interpret_kappa_moderate(self):
        """Test interpretation for moderate agreement (0.4 - 0.6)."""
        calc = HallucinationMetricsCalculator()
        assert calc.interpret_kappa(0.4) == "moderate"
        assert calc.interpret_kappa(0.5) == "moderate"
        assert calc.interpret_kappa(0.59) == "moderate"

    def test_interpret_kappa_substantial(self):
        """Test interpretation for substantial agreement (0.6 - 0.8)."""
        calc = HallucinationMetricsCalculator()
        assert calc.interpret_kappa(0.6) == "substantial"
        assert calc.interpret_kappa(0.7) == "substantial"
        assert calc.interpret_kappa(0.79) == "substantial"

    def test_interpret_kappa_almost_perfect(self):
        """Test interpretation for almost perfect agreement (>= 0.8)."""
        calc = HallucinationMetricsCalculator()
        assert calc.interpret_kappa(0.8) == "almost_perfect"
        assert calc.interpret_kappa(0.9) == "almost_perfect"
        assert calc.interpret_kappa(1.0) == "almost_perfect"


class TestFleissKappaFromVerdicts:
    """Tests for computing Fleiss' Kappa from verdict lists."""

    def test_fleiss_kappa_from_verdicts_empty(self):
        """Test with empty judge verdicts."""
        calc = HallucinationMetricsCalculator()
        result = calc.compute_fleiss_kappa_from_verdicts({})
        
        assert result.is_undefined is True
        assert result.kappa is None
        assert "No judge verdicts" in result.error_message

    def test_fleiss_kappa_from_verdicts_single_judge(self):
        """Test with only one judge."""
        calc = HallucinationMetricsCalculator()
        result = calc.compute_fleiss_kappa_from_verdicts({
            "judge_1": [
                Verdict(label=VerdictLabel.SUPPORTED, confidence=0.9),
            ]
        })
        
        assert result.is_undefined is True
        assert result.kappa is None
        assert "Fewer than 2 judges" in result.error_message

    def test_fleiss_kappa_from_verdicts_perfect_agreement(self):
        """Test with perfect agreement among judges."""
        calc = HallucinationMetricsCalculator()
        result = calc.compute_fleiss_kappa_from_verdicts({
            "judge_1": [
                Verdict(label=VerdictLabel.SUPPORTED, confidence=0.9),
                Verdict(label=VerdictLabel.REFUTED, confidence=0.8),
                Verdict(label=VerdictLabel.NOT_ENOUGH_INFO, confidence=0.7),
            ],
            "judge_2": [
                Verdict(label=VerdictLabel.SUPPORTED, confidence=0.85),
                Verdict(label=VerdictLabel.REFUTED, confidence=0.75),
                Verdict(label=VerdictLabel.NOT_ENOUGH_INFO, confidence=0.65),
            ],
            "judge_3": [
                Verdict(label=VerdictLabel.SUPPORTED, confidence=0.8),
                Verdict(label=VerdictLabel.REFUTED, confidence=0.7),
                Verdict(label=VerdictLabel.NOT_ENOUGH_INFO, confidence=0.6),
            ],
        })
        
        assert result.is_undefined is False
        assert result.kappa is not None
        assert result.kappa == pytest.approx(1.0, abs=0.01)
        assert result.interpretation == "almost_perfect"

    def test_fleiss_kappa_from_verdicts_no_agreement(self):
        """Test with systematic disagreement among judges."""
        calc = HallucinationMetricsCalculator()
        result = calc.compute_fleiss_kappa_from_verdicts({
            "judge_1": [
                Verdict(label=VerdictLabel.SUPPORTED, confidence=0.9),
                Verdict(label=VerdictLabel.REFUTED, confidence=0.8),
                Verdict(label=VerdictLabel.NOT_ENOUGH_INFO, confidence=0.7),
            ],
            "judge_2": [
                Verdict(label=VerdictLabel.REFUTED, confidence=0.85),
                Verdict(label=VerdictLabel.NOT_ENOUGH_INFO, confidence=0.75),
                Verdict(label=VerdictLabel.SUPPORTED, confidence=0.65),
            ],
            "judge_3": [
                Verdict(label=VerdictLabel.NOT_ENOUGH_INFO, confidence=0.8),
                Verdict(label=VerdictLabel.SUPPORTED, confidence=0.7),
                Verdict(label=VerdictLabel.REFUTED, confidence=0.6),
            ],
        })
        
        assert result.is_undefined is False
        assert result.kappa is not None
        # With systematic disagreement (each judge choosing different categories),
        # kappa is negative (worse than chance agreement)
        assert result.kappa < 0

    def test_fleiss_kappa_from_verdicts_mismatched_lengths(self):
        """Test with mismatched verdict list lengths."""
        calc = HallucinationMetricsCalculator()
        result = calc.compute_fleiss_kappa_from_verdicts({
            "judge_1": [
                Verdict(label=VerdictLabel.SUPPORTED, confidence=0.9),
                Verdict(label=VerdictLabel.REFUTED, confidence=0.8),
            ],
            "judge_2": [
                Verdict(label=VerdictLabel.SUPPORTED, confidence=0.85),
            ],  # Only 1 verdict
        })
        
        assert result.is_undefined is True
        assert result.kappa is None
        assert "verdicts" in result.error_message.lower()

    def test_fleiss_kappa_from_verdicts_empty_items(self):
        """Test with empty verdict lists."""
        calc = HallucinationMetricsCalculator()
        result = calc.compute_fleiss_kappa_from_verdicts({
            "judge_1": [],
            "judge_2": [],
        })
        
        assert result.is_undefined is True
        assert result.kappa is None
        assert "No items" in result.error_message



class TestShannonEntropyComputation:
    """Tests for Shannon entropy computation."""

    def test_shannon_entropy_empty_probabilities(self):
        """Test Shannon entropy with empty probabilities returns 0."""
        calc = HallucinationMetricsCalculator()
        result = calc.compute_shannon_entropy([])
        assert result == 0.0

    def test_shannon_entropy_single_certain_outcome(self):
        """Test Shannon entropy with single certain outcome (p=1) returns 0."""
        calc = HallucinationMetricsCalculator()
        result = calc.compute_shannon_entropy([1.0])
        assert result == 0.0

    def test_shannon_entropy_uniform_distribution(self):
        """Test Shannon entropy with uniform distribution."""
        import math
        calc = HallucinationMetricsCalculator()
        
        # Uniform distribution over 2 outcomes: H = ln(2)
        result = calc.compute_shannon_entropy([0.5, 0.5])
        expected = math.log(2)
        assert abs(result - expected) < 0.001

    def test_shannon_entropy_uniform_distribution_three_outcomes(self):
        """Test Shannon entropy with uniform distribution over 3 outcomes."""
        import math
        calc = HallucinationMetricsCalculator()
        
        # Uniform distribution over 3 outcomes: H = ln(3)
        result = calc.compute_shannon_entropy([1/3, 1/3, 1/3])
        expected = math.log(3)
        assert abs(result - expected) < 0.001

    def test_shannon_entropy_zero_probability_handling(self):
        """Test Shannon entropy handles zero probabilities correctly."""
        calc = HallucinationMetricsCalculator()
        
        # Distribution with zero probability: [0.5, 0.5, 0.0]
        # Should be same as [0.5, 0.5] since 0*log(0) = 0 by convention
        result = calc.compute_shannon_entropy([0.5, 0.5, 0.0])
        import math
        expected = math.log(2)
        assert abs(result - expected) < 0.001

    def test_shannon_entropy_skewed_distribution(self):
        """Test Shannon entropy with skewed distribution."""
        calc = HallucinationMetricsCalculator()
        
        # Skewed distribution: [0.9, 0.1]
        result = calc.compute_shannon_entropy([0.9, 0.1])
        
        # Should be less than uniform distribution entropy
        import math
        uniform_entropy = math.log(2)
        assert result < uniform_entropy
        assert result > 0.0

    def test_shannon_entropy_non_negative(self):
        """Test Shannon entropy is always non-negative."""
        calc = HallucinationMetricsCalculator()
        
        # Various distributions
        distributions = [
            [1.0],
            [0.5, 0.5],
            [0.1, 0.2, 0.7],
            [0.25, 0.25, 0.25, 0.25],
            [0.99, 0.01],
        ]
        
        for dist in distributions:
            result = calc.compute_shannon_entropy(dist)
            assert result >= 0.0, f"Entropy should be non-negative for {dist}"


class TestEpistemicUncertaintyComputation:
    """Tests for epistemic uncertainty computation."""

    def test_epistemic_uncertainty_empty_samples(self):
        """Test epistemic uncertainty with empty samples returns 0."""
        calc = HallucinationMetricsCalculator()
        result = calc.compute_epistemic_uncertainty([])
        assert result == 0.0

    def test_epistemic_uncertainty_single_sample(self):
        """Test epistemic uncertainty with single sample returns 0."""
        calc = HallucinationMetricsCalculator()
        result = calc.compute_epistemic_uncertainty([[0.5, 0.5]])
        assert result == 0.0

    def test_epistemic_uncertainty_identical_samples(self):
        """Test epistemic uncertainty with identical samples."""
        calc = HallucinationMetricsCalculator()
        
        # All samples are identical - low epistemic uncertainty
        samples = [
            [0.5, 0.5],
            [0.5, 0.5],
            [0.5, 0.5],
        ]
        result = calc.compute_epistemic_uncertainty(samples)
        assert result == 0.0

    def test_epistemic_uncertainty_varying_samples(self):
        """Test epistemic uncertainty with varying samples."""
        calc = HallucinationMetricsCalculator()
        
        # Samples vary - higher epistemic uncertainty
        samples = [
            [0.9, 0.1],
            [0.1, 0.9],
            [0.5, 0.5],
        ]
        result = calc.compute_epistemic_uncertainty(samples)
        assert result > 0.0

    def test_epistemic_uncertainty_non_negative(self):
        """Test epistemic uncertainty is always non-negative."""
        calc = HallucinationMetricsCalculator()
        
        samples = [
            [0.3, 0.7],
            [0.4, 0.6],
            [0.5, 0.5],
        ]
        result = calc.compute_epistemic_uncertainty(samples)
        assert result >= 0.0


class TestAleatoricUncertaintyComputation:
    """Tests for aleatoric uncertainty computation."""

    def test_aleatoric_uncertainty_empty_samples(self):
        """Test aleatoric uncertainty with empty samples returns 0."""
        calc = HallucinationMetricsCalculator()
        result = calc.compute_aleatoric_uncertainty([])
        assert result == 0.0

    def test_aleatoric_uncertainty_certain_samples(self):
        """Test aleatoric uncertainty with certain samples (low variance)."""
        calc = HallucinationMetricsCalculator()
        
        # Each sample has low internal variance
        samples = [
            [1.0, 0.0],  # Certain outcome
            [1.0, 0.0],
            [1.0, 0.0],
        ]
        result = calc.compute_aleatoric_uncertainty(samples)
        # Variance within each sample is 0.25 (mean=0.5, (1-0.5)^2 + (0-0.5)^2 / 2)
        assert result > 0.0

    def test_aleatoric_uncertainty_uniform_samples(self):
        """Test aleatoric uncertainty with uniform samples."""
        calc = HallucinationMetricsCalculator()
        
        # Uniform distributions have some internal variance
        samples = [
            [0.5, 0.5],
            [0.5, 0.5],
            [0.5, 0.5],
        ]
        result = calc.compute_aleatoric_uncertainty(samples)
        # Variance within each sample is 0 (all values are 0.5)
        assert result == 0.0

    def test_aleatoric_uncertainty_non_negative(self):
        """Test aleatoric uncertainty is always non-negative."""
        calc = HallucinationMetricsCalculator()
        
        samples = [
            [0.3, 0.7],
            [0.4, 0.6],
            [0.5, 0.5],
        ]
        result = calc.compute_aleatoric_uncertainty(samples)
        assert result >= 0.0


class TestUncertaintyQuantification:
    """Tests for comprehensive uncertainty quantification."""

    def test_compute_uncertainty_basic(self):
        """Test basic uncertainty computation."""
        calc = HallucinationMetricsCalculator()
        
        probabilities = [0.5, 0.5]
        result = calc.compute_uncertainty(probabilities)
        
        assert isinstance(result, UncertaintyResult)
        assert result.shannon_entropy > 0.0
        assert result.epistemic == 0.0  # No samples provided
        assert result.aleatoric == 0.0  # No samples provided
        assert result.total == 0.0

    def test_compute_uncertainty_with_samples(self):
        """Test uncertainty computation with inference samples."""
        calc = HallucinationMetricsCalculator()
        
        probabilities = [0.5, 0.5]
        samples = [
            [0.6, 0.4],
            [0.4, 0.6],
            [0.5, 0.5],
        ]
        result = calc.compute_uncertainty(probabilities, samples)
        
        assert isinstance(result, UncertaintyResult)
        assert result.shannon_entropy > 0.0
        assert result.epistemic >= 0.0
        assert result.aleatoric >= 0.0
        assert result.total == result.epistemic + result.aleatoric

    def test_compute_uncertainty_high_uncertainty_flag(self):
        """Test high uncertainty flagging."""
        # Use low threshold to trigger high uncertainty
        config = HallucinationMetricsConfig(uncertainty_high_threshold=0.001)
        calc = HallucinationMetricsCalculator(config=config)
        
        probabilities = [0.5, 0.5]
        samples = [
            [0.9, 0.1],
            [0.1, 0.9],
        ]
        result = calc.compute_uncertainty(probabilities, samples)
        
        # With varying samples and low threshold, should be flagged
        assert result.is_high_uncertainty is True

    def test_compute_uncertainty_low_uncertainty_flag(self):
        """Test low uncertainty is not flagged."""
        # Use high threshold
        config = HallucinationMetricsConfig(uncertainty_high_threshold=10.0)
        calc = HallucinationMetricsCalculator(config=config)
        
        probabilities = [0.5, 0.5]
        samples = [
            [0.5, 0.5],
            [0.5, 0.5],
        ]
        result = calc.compute_uncertainty(probabilities, samples)
        
        assert result.is_high_uncertainty is False

    def test_compute_uncertainty_empty_probabilities(self):
        """Test uncertainty with empty probabilities."""
        calc = HallucinationMetricsCalculator()
        
        result = calc.compute_uncertainty([])
        
        assert result.shannon_entropy == 0.0
        assert result.epistemic == 0.0
        assert result.aleatoric == 0.0
        assert result.total == 0.0


class TestHighUncertaintyFlagging:
    """Tests for high uncertainty flagging."""

    def test_flag_high_uncertainty_above_threshold(self):
        """Test flagging when uncertainty is above threshold."""
        calc = HallucinationMetricsCalculator()
        
        uncertainty = UncertaintyResult(
            shannon_entropy=0.5,
            epistemic=0.5,
            aleatoric=0.4,
            total=0.9,  # Above default threshold of 0.8
            is_high_uncertainty=True
        )
        
        assert calc.flag_high_uncertainty(uncertainty) is True

    def test_flag_high_uncertainty_below_threshold(self):
        """Test not flagging when uncertainty is below threshold."""
        calc = HallucinationMetricsCalculator()
        
        uncertainty = UncertaintyResult(
            shannon_entropy=0.3,
            epistemic=0.2,
            aleatoric=0.1,
            total=0.3,  # Below default threshold of 0.8
            is_high_uncertainty=False
        )
        
        assert calc.flag_high_uncertainty(uncertainty) is False

    def test_flag_high_uncertainty_custom_threshold(self):
        """Test flagging with custom threshold."""
        calc = HallucinationMetricsCalculator()
        
        uncertainty = UncertaintyResult(
            shannon_entropy=0.3,
            epistemic=0.2,
            aleatoric=0.1,
            total=0.3,
            is_high_uncertainty=False
        )
        
        # With custom low threshold, should be flagged
        assert calc.flag_high_uncertainty(uncertainty, threshold=0.2) is True
        
        # With custom high threshold, should not be flagged
        assert calc.flag_high_uncertainty(uncertainty, threshold=0.5) is False

    def test_flag_high_uncertainty_at_threshold(self):
        """Test flagging when uncertainty equals threshold."""
        calc = HallucinationMetricsCalculator()
        
        uncertainty = UncertaintyResult(
            shannon_entropy=0.4,
            epistemic=0.4,
            aleatoric=0.4,
            total=0.8,  # Exactly at default threshold
            is_high_uncertainty=False
        )
        
        # At threshold (not above), should not be flagged
        assert calc.flag_high_uncertainty(uncertainty) is False


class TestGenerateHallucinationProfile:
    """Tests for generate_hallucination_profile method."""

    def test_generate_profile_basic(self):
        """Test basic profile generation with minimal input."""
        calc = HallucinationMetricsCalculator()
        
        verdicts = [
            Verdict(label=VerdictLabel.SUPPORTED, confidence=0.9),
            Verdict(label=VerdictLabel.REFUTED, confidence=0.8),
            Verdict(label=VerdictLabel.SUPPORTED, confidence=0.7),
        ]
        
        profile = calc.generate_hallucination_profile(verdicts=verdicts)
        
        # MiHR should be computed
        assert profile.mihr is not None
        assert profile.mihr.value == pytest.approx(1/3)  # 1 unsupported out of 3
        assert profile.mihr.total_claims == 3
        assert profile.mihr.unsupported_claims == 1
        
        # FactScore should be computed
        assert profile.factscore == pytest.approx(2/3)  # 2 supported out of 3
        
        # Optional metrics should be None
        assert profile.mahr is None
        assert profile.consensus_f1 is None
        assert profile.fleiss_kappa is None
        assert profile.uncertainty is None
        
        # Reliability should be determined
        assert profile.reliability in [ReliabilityLevel.HIGH, ReliabilityLevel.MEDIUM, ReliabilityLevel.LOW]

    def test_generate_profile_with_mahr(self):
        """Test profile generation with MaHR computation."""
        calc = HallucinationMetricsCalculator()
        
        verdicts = [
            Verdict(label=VerdictLabel.SUPPORTED, confidence=0.9),
        ]
        
        response_verdicts = [
            [Verdict(label=VerdictLabel.SUPPORTED, confidence=0.9)],  # No hallucination
            [Verdict(label=VerdictLabel.REFUTED, confidence=0.8)],    # Has hallucination
            [Verdict(label=VerdictLabel.SUPPORTED, confidence=0.7)],  # No hallucination
        ]
        
        profile = calc.generate_hallucination_profile(
            verdicts=verdicts,
            response_verdicts=response_verdicts,
        )
        
        # MaHR should be computed
        assert profile.mahr is not None
        assert profile.mahr.value == pytest.approx(1/3)  # 1 response with hallucination out of 3
        assert profile.mahr.total_responses == 3
        assert profile.mahr.responses_with_hallucinations == 1

    def test_generate_profile_with_claim_matrix(self):
        """Test profile generation with claim matrix for Consensus F1."""
        from llm_judge_auditor.components.hallucination_metrics import ClaimVerificationMatrix
        
        calc = HallucinationMetricsCalculator()
        
        verdicts = [
            Verdict(label=VerdictLabel.SUPPORTED, confidence=0.9),
        ]
        
        claims = [
            Claim(text="claim1", source_span=(0, 6), claim_type=ClaimType.FACTUAL),
            Claim(text="claim2", source_span=(7, 13), claim_type=ClaimType.FACTUAL),
        ]
        models = ["model_a", "model_b"]
        support_matrix = [
            [1, 1],  # Both models support claim1
            [1, 1],  # Both models support claim2
        ]
        claim_matrix = ClaimVerificationMatrix(claims=claims, models=models, support_matrix=support_matrix)
        
        profile = calc.generate_hallucination_profile(
            verdicts=verdicts,
            claim_matrix=claim_matrix,
        )
        
        # Consensus F1 should be computed
        assert profile.consensus_f1 is not None
        assert profile.consensus_f1.f1 == 1.0  # Perfect agreement
        
        # Claim analysis should be populated
        assert len(profile.consensus_claims) == 2  # Both claims are consensus
        assert len(profile.disputed_claims) == 0

    def test_generate_profile_with_judge_verdicts(self):
        """Test profile generation with judge verdicts for Fleiss' Kappa."""
        calc = HallucinationMetricsCalculator()
        
        verdicts = [
            Verdict(label=VerdictLabel.SUPPORTED, confidence=0.9),
            Verdict(label=VerdictLabel.SUPPORTED, confidence=0.8),
        ]
        
        judge_verdicts = {
            "judge_1": [
                Verdict(label=VerdictLabel.SUPPORTED, confidence=0.9),
                Verdict(label=VerdictLabel.SUPPORTED, confidence=0.8),
            ],
            "judge_2": [
                Verdict(label=VerdictLabel.SUPPORTED, confidence=0.9),
                Verdict(label=VerdictLabel.SUPPORTED, confidence=0.8),
            ],
        }
        
        profile = calc.generate_hallucination_profile(
            verdicts=verdicts,
            judge_verdicts=judge_verdicts,
        )
        
        # Fleiss' Kappa should be computed
        assert profile.fleiss_kappa is not None
        assert profile.fleiss_kappa.kappa == 1.0  # Perfect agreement
        assert profile.fleiss_kappa.interpretation == "almost_perfect"

    def test_generate_profile_with_uncertainty(self):
        """Test profile generation with uncertainty quantification."""
        calc = HallucinationMetricsCalculator()
        
        verdicts = [
            Verdict(label=VerdictLabel.SUPPORTED, confidence=0.9),
        ]
        
        probabilities = [0.5, 0.3, 0.2]
        inference_samples = [
            [0.5, 0.3, 0.2],
            [0.4, 0.4, 0.2],
            [0.6, 0.2, 0.2],
        ]
        
        profile = calc.generate_hallucination_profile(
            verdicts=verdicts,
            probabilities=probabilities,
            inference_samples=inference_samples,
        )
        
        # Uncertainty should be computed
        assert profile.uncertainty is not None
        assert profile.uncertainty.shannon_entropy > 0
        assert profile.uncertainty.total >= 0

    def test_generate_profile_high_risk_mihr(self):
        """Test profile flags high risk when MiHR > 0.3."""
        calc = HallucinationMetricsCalculator()
        
        # 4 out of 5 claims are unsupported (MiHR = 0.8)
        verdicts = [
            Verdict(label=VerdictLabel.REFUTED, confidence=0.9),
            Verdict(label=VerdictLabel.REFUTED, confidence=0.8),
            Verdict(label=VerdictLabel.REFUTED, confidence=0.7),
            Verdict(label=VerdictLabel.REFUTED, confidence=0.6),
            Verdict(label=VerdictLabel.SUPPORTED, confidence=0.5),
        ]
        
        profile = calc.generate_hallucination_profile(verdicts=verdicts)
        
        assert profile.mihr.value == 0.8
        assert profile.is_high_risk is True
        assert profile.reliability == ReliabilityLevel.LOW

    def test_generate_profile_high_risk_kappa(self):
        """Test profile flags high risk when Kappa < 0.4."""
        calc = HallucinationMetricsCalculator()
        
        verdicts = [
            Verdict(label=VerdictLabel.SUPPORTED, confidence=0.9),
            Verdict(label=VerdictLabel.SUPPORTED, confidence=0.8),
        ]
        
        # Judges disagree significantly
        judge_verdicts = {
            "judge_1": [
                Verdict(label=VerdictLabel.SUPPORTED, confidence=0.9),
                Verdict(label=VerdictLabel.REFUTED, confidence=0.8),
            ],
            "judge_2": [
                Verdict(label=VerdictLabel.REFUTED, confidence=0.9),
                Verdict(label=VerdictLabel.SUPPORTED, confidence=0.8),
            ],
        }
        
        profile = calc.generate_hallucination_profile(
            verdicts=verdicts,
            judge_verdicts=judge_verdicts,
        )
        
        # Kappa should be low due to disagreement
        assert profile.fleiss_kappa is not None
        assert profile.fleiss_kappa.kappa < 0.4
        assert profile.is_high_risk is True
        assert profile.reliability == ReliabilityLevel.LOW

    def test_generate_profile_empty_verdicts(self):
        """Test profile generation with empty verdicts."""
        calc = HallucinationMetricsCalculator()
        
        profile = calc.generate_hallucination_profile(verdicts=[])
        
        # MiHR should indicate no claims
        assert profile.mihr is not None
        assert profile.mihr.value is None
        assert profile.mihr.has_claims is False
        
        # FactScore should be None
        assert profile.factscore is None
        
        # Should not be high risk (no data to evaluate)
        assert profile.is_high_risk is False

    def test_generate_profile_json_round_trip(self):
        """Test profile JSON serialization round-trip."""
        from llm_judge_auditor.components.hallucination_metrics import ClaimVerificationMatrix
        
        calc = HallucinationMetricsCalculator()
        
        verdicts = [
            Verdict(label=VerdictLabel.SUPPORTED, confidence=0.9),
            Verdict(label=VerdictLabel.REFUTED, confidence=0.8),
        ]
        
        claims = [
            Claim(text="test claim", source_span=(0, 10), claim_type=ClaimType.FACTUAL),
        ]
        models = ["model_a", "model_b"]
        support_matrix = [[1, 0]]  # Disputed claim
        claim_matrix = ClaimVerificationMatrix(claims=claims, models=models, support_matrix=support_matrix)
        
        profile = calc.generate_hallucination_profile(
            verdicts=verdicts,
            claim_matrix=claim_matrix,
            probabilities=[0.5, 0.5],
        )
        
        # Serialize and deserialize
        json_str = profile.to_json()
        restored = HallucinationProfile.from_json(json_str)
        
        # Verify round-trip
        assert restored.mihr.value == profile.mihr.value
        assert restored.factscore == profile.factscore
        assert restored.reliability == profile.reliability
        assert restored.is_high_risk == profile.is_high_risk
        assert len(restored.disputed_claims) == len(profile.disputed_claims)



class TestFalseAcceptanceCalculator:
    """Tests for FalseAcceptanceCalculator."""

    def test_init_default_patterns(self):
        """Test initialization with default abstention patterns."""
        from llm_judge_auditor.components.hallucination_metrics import (
            FalseAcceptanceCalculator,
        )
        calc = FalseAcceptanceCalculator()
        assert len(calc.abstention_patterns) > 0
        assert "i don't know" in calc.abstention_patterns
        assert calc.case_sensitive is False

    def test_init_custom_patterns(self):
        """Test initialization with custom abstention patterns."""
        from llm_judge_auditor.components.hallucination_metrics import (
            FalseAcceptanceCalculator,
        )
        custom_patterns = ["cannot answer", "no data available"]
        calc = FalseAcceptanceCalculator(abstention_patterns=custom_patterns)
        assert calc.abstention_patterns == custom_patterns

    def test_detect_abstention_positive(self):
        """Test abstention detection with positive cases."""
        from llm_judge_auditor.components.hallucination_metrics import (
            FalseAcceptanceCalculator,
        )
        calc = FalseAcceptanceCalculator()
        
        # Test various abstention responses
        assert calc._detect_abstention("I don't know about that topic.") is True
        assert calc._detect_abstention("I'm not sure what you're asking.") is True
        assert calc._detect_abstention("There is no information available.") is True
        assert calc._detect_abstention("I cannot find any data on this.") is True
        assert calc._detect_abstention("That entity doesn't exist.") is True

    def test_detect_abstention_negative(self):
        """Test abstention detection with negative cases."""
        from llm_judge_auditor.components.hallucination_metrics import (
            FalseAcceptanceCalculator,
        )
        calc = FalseAcceptanceCalculator()
        
        # Test responses that should NOT be detected as abstention
        assert calc._detect_abstention("The capital of France is Paris.") is False
        assert calc._detect_abstention("Here is the information you requested.") is False
        assert calc._detect_abstention("John Smith was born in 1985.") is False

    def test_evaluate_abstention_correct_refusal(self):
        """Test evaluate_abstention with correct refusal."""
        from llm_judge_auditor.components.hallucination_metrics import (
            FalseAcceptanceCalculator,
        )
        calc = FalseAcceptanceCalculator()
        
        result = calc.evaluate_abstention(
            query="Who is Zorbax the Magnificent?",
            response="I don't know who Zorbax the Magnificent is.",
            is_nonexistent=True,
        )
        
        assert result.is_nonexistent_entity is True
        assert result.did_abstain is True
        assert result.is_false_acceptance is False

    def test_evaluate_abstention_false_acceptance(self):
        """Test evaluate_abstention with false acceptance."""
        from llm_judge_auditor.components.hallucination_metrics import (
            FalseAcceptanceCalculator,
        )
        calc = FalseAcceptanceCalculator()
        
        result = calc.evaluate_abstention(
            query="Who is Zorbax the Magnificent?",
            response="Zorbax the Magnificent was a famous wizard from the 15th century.",
            is_nonexistent=True,
        )
        
        assert result.is_nonexistent_entity is True
        assert result.did_abstain is False
        assert result.is_false_acceptance is True

    def test_evaluate_abstention_real_entity(self):
        """Test evaluate_abstention with real entity."""
        from llm_judge_auditor.components.hallucination_metrics import (
            FalseAcceptanceCalculator,
        )
        calc = FalseAcceptanceCalculator()
        
        result = calc.evaluate_abstention(
            query="Who is Albert Einstein?",
            response="Albert Einstein was a theoretical physicist.",
            is_nonexistent=False,
        )
        
        assert result.is_nonexistent_entity is False
        assert result.did_abstain is False
        assert result.is_false_acceptance is False  # Not false acceptance for real entities

    def test_compute_far_empty_results(self):
        """Test FAR computation with empty results."""
        from llm_judge_auditor.components.hallucination_metrics import (
            FalseAcceptanceCalculator,
        )
        calc = FalseAcceptanceCalculator()
        
        result = calc.compute_far([])
        
        assert result.value == 0.0
        assert result.failed_abstentions == 0
        assert result.correct_refusals == 0
        assert result.total_nonexistent_queries == 0

    def test_compute_far_all_correct_refusals(self):
        """Test FAR computation with all correct refusals."""
        from llm_judge_auditor.components.hallucination_metrics import (
            FalseAcceptanceCalculator,
            AbstentionResult,
        )
        calc = FalseAcceptanceCalculator()
        
        results = [
            AbstentionResult(
                query="Who is Fake Person 1?",
                response="I don't know who that is.",
                is_nonexistent_entity=True,
                did_abstain=True,
                is_false_acceptance=False,
            ),
            AbstentionResult(
                query="Who is Fake Person 2?",
                response="I have no information about that person.",
                is_nonexistent_entity=True,
                did_abstain=True,
                is_false_acceptance=False,
            ),
        ]
        
        far_result = calc.compute_far(results)
        
        assert far_result.value == 0.0
        assert far_result.failed_abstentions == 0
        assert far_result.correct_refusals == 2
        assert far_result.total_nonexistent_queries == 2

    def test_compute_far_all_false_acceptances(self):
        """Test FAR computation with all false acceptances."""
        from llm_judge_auditor.components.hallucination_metrics import (
            FalseAcceptanceCalculator,
            AbstentionResult,
        )
        calc = FalseAcceptanceCalculator()
        
        results = [
            AbstentionResult(
                query="Who is Fake Person 1?",
                response="Fake Person 1 was a famous inventor.",
                is_nonexistent_entity=True,
                did_abstain=False,
                is_false_acceptance=True,
            ),
            AbstentionResult(
                query="Who is Fake Person 2?",
                response="Fake Person 2 was a renowned scientist.",
                is_nonexistent_entity=True,
                did_abstain=False,
                is_false_acceptance=True,
            ),
        ]
        
        far_result = calc.compute_far(results)
        
        assert far_result.value == 1.0
        assert far_result.failed_abstentions == 2
        assert far_result.correct_refusals == 0
        assert far_result.total_nonexistent_queries == 2

    def test_compute_far_mixed_results(self):
        """Test FAR computation with mixed results."""
        from llm_judge_auditor.components.hallucination_metrics import (
            FalseAcceptanceCalculator,
            AbstentionResult,
        )
        calc = FalseAcceptanceCalculator()
        
        results = [
            AbstentionResult(
                query="Who is Fake Person 1?",
                response="I don't know.",
                is_nonexistent_entity=True,
                did_abstain=True,
                is_false_acceptance=False,
            ),
            AbstentionResult(
                query="Who is Fake Person 2?",
                response="Fake Person 2 was famous.",
                is_nonexistent_entity=True,
                did_abstain=False,
                is_false_acceptance=True,
            ),
            AbstentionResult(
                query="Who is Albert Einstein?",
                response="Albert Einstein was a physicist.",
                is_nonexistent_entity=False,
                did_abstain=False,
                is_false_acceptance=False,
            ),
        ]
        
        far_result = calc.compute_far(results)
        
        # Only 2 non-existent queries, 1 false acceptance
        assert far_result.value == 0.5
        assert far_result.failed_abstentions == 1
        assert far_result.correct_refusals == 1
        assert far_result.total_nonexistent_queries == 2

    def test_compute_far_only_real_entities(self):
        """Test FAR computation with only real entity queries."""
        from llm_judge_auditor.components.hallucination_metrics import (
            FalseAcceptanceCalculator,
            AbstentionResult,
        )
        calc = FalseAcceptanceCalculator()
        
        results = [
            AbstentionResult(
                query="Who is Albert Einstein?",
                response="Albert Einstein was a physicist.",
                is_nonexistent_entity=False,
                did_abstain=False,
                is_false_acceptance=False,
            ),
        ]
        
        far_result = calc.compute_far(results)
        
        # No non-existent queries, so FAR is 0
        assert far_result.value == 0.0
        assert far_result.total_nonexistent_queries == 0

    def test_evaluate_and_compute_far(self):
        """Test convenience method evaluate_and_compute_far."""
        from llm_judge_auditor.components.hallucination_metrics import (
            FalseAcceptanceCalculator,
        )
        calc = FalseAcceptanceCalculator()
        
        queries = [
            "Who is Fake Person?",
            "Who is Albert Einstein?",
        ]
        responses = [
            "I don't know who that is.",
            "Albert Einstein was a physicist.",
        ]
        is_nonexistent_flags = [True, False]
        
        result = calc.evaluate_and_compute_far(queries, responses, is_nonexistent_flags)
        
        assert result.value == 0.0  # Correct refusal for the fake person
        assert result.total_nonexistent_queries == 1
        assert result.correct_refusals == 1

    def test_evaluate_and_compute_far_mismatched_lengths(self):
        """Test evaluate_and_compute_far with mismatched input lengths."""
        from llm_judge_auditor.components.hallucination_metrics import (
            FalseAcceptanceCalculator,
        )
        calc = FalseAcceptanceCalculator()
        
        with pytest.raises(ValueError):
            calc.evaluate_and_compute_far(
                queries=["query1", "query2"],
                responses=["response1"],
                is_nonexistent_flags=[True, False],
            )

    def test_add_abstention_pattern(self):
        """Test adding custom abstention pattern."""
        from llm_judge_auditor.components.hallucination_metrics import (
            FalseAcceptanceCalculator,
        )
        calc = FalseAcceptanceCalculator()
        
        original_count = len(calc.abstention_patterns)
        calc.add_abstention_pattern("custom pattern")
        
        assert len(calc.abstention_patterns) == original_count + 1
        assert "custom pattern" in calc.abstention_patterns

    def test_remove_abstention_pattern(self):
        """Test removing abstention pattern."""
        from llm_judge_auditor.components.hallucination_metrics import (
            FalseAcceptanceCalculator,
        )
        calc = FalseAcceptanceCalculator()
        
        # Add and then remove
        calc.add_abstention_pattern("test pattern")
        assert calc.remove_abstention_pattern("test pattern") is True
        assert "test pattern" not in calc.abstention_patterns
        
        # Try to remove non-existent pattern
        assert calc.remove_abstention_pattern("nonexistent") is False

    def test_case_sensitive_matching(self):
        """Test case-sensitive pattern matching."""
        from llm_judge_auditor.components.hallucination_metrics import (
            FalseAcceptanceCalculator,
        )
        calc = FalseAcceptanceCalculator(case_sensitive=True)
        
        # Should not match due to case difference
        assert calc._detect_abstention("I DON'T KNOW") is False
        
        # Should match with exact case
        assert calc._detect_abstention("i don't know") is True
