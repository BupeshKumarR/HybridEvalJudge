"""
Hallucination Metrics Calculator for quantifying hallucination rates.

This module provides research-backed hallucination quantification metrics including:
- MiHR (Micro Hallucination Rate): unsupported_claims / total_claims
- MaHR (Macro Hallucination Rate): responses_with_hallucinations / total_responses
- FactScore: verified_claims / total_claims
- Consensus F1: Cross-model agreement metric
- Fleiss' Kappa: Inter-judge agreement statistic
- Uncertainty quantification: Shannon entropy with epistemic/aleatoric decomposition

Validates: Requirements 15.1, 15.2, 15.3, 15.4, 15.5, 19.1, 19.2
"""

import json
import math
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from llm_judge_auditor.models import Claim, Verdict, VerdictLabel


class ReliabilityLevel(str, Enum):
    """Reliability classification levels for hallucination profiles."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class ConsensusF1Result:
    """
    Result from Consensus F1 computation.
    
    Attributes:
        precision: Fraction of model claims supported by other models
        recall: Fraction of consensus claims included by the model
        f1: Harmonic mean of precision and recall
    """
    precision: float
    recall: float
    f1: float


@dataclass
class KappaResult:
    """
    Result from Fleiss' Kappa computation.
    
    Attributes:
        kappa: The kappa statistic value
        interpretation: Human-readable interpretation (poor, fair, moderate, substantial, almost_perfect)
        observed_agreement: Po - the observed agreement among raters
        expected_agreement: Pe - the expected agreement by chance
        is_undefined: True if kappa cannot be computed (e.g., fewer than 2 judges)
        error_message: Error message if kappa is undefined
    """
    kappa: Optional[float]
    interpretation: str
    observed_agreement: float
    expected_agreement: float
    is_undefined: bool = False
    error_message: Optional[str] = None


@dataclass
class UncertaintyResult:
    """
    Result from uncertainty quantification.
    
    Attributes:
        shannon_entropy: H(p) = -Σ pᵢ log pᵢ
        epistemic: Model uncertainty from lack of knowledge (variance across samples)
        aleatoric: Inherent data noise (expected variance within samples)
        total: Total uncertainty (epistemic + aleatoric)
        is_high_uncertainty: True if uncertainty exceeds threshold
    """
    shannon_entropy: float
    epistemic: float
    aleatoric: float
    total: float
    is_high_uncertainty: bool


@dataclass
class MiHRResult:
    """
    Result from MiHR (Micro Hallucination Rate) computation.
    
    Attributes:
        value: The MiHR value (unsupported_claims / total_claims), or None if no claims
        unsupported_claims: Count of unsupported claims
        total_claims: Total number of claims
        has_claims: True if there were claims to evaluate
    """
    value: Optional[float]
    unsupported_claims: int
    total_claims: int
    has_claims: bool


@dataclass
class MaHRResult:
    """
    Result from MaHR (Macro Hallucination Rate) computation.
    
    Attributes:
        value: The MaHR value (responses_with_hallucinations / total_responses)
        responses_with_hallucinations: Count of responses containing hallucinations
        total_responses: Total number of responses evaluated
    """
    value: float
    responses_with_hallucinations: int
    total_responses: int


@dataclass
class AbstentionResult:
    """
    Result from evaluating a single abstention query.
    
    Attributes:
        query: The original query about a non-existent entity
        response: The model's response to the query
        is_nonexistent_entity: True if the query is about a non-existent entity
        did_abstain: True if the model correctly abstained from answering
        is_false_acceptance: True if the model generated content for a non-existent entity
    
    Validates: Requirements 20.1, 20.3, 20.4
    """
    query: str
    response: str
    is_nonexistent_entity: bool
    did_abstain: bool
    is_false_acceptance: bool


@dataclass
class FalseAcceptanceRateResult:
    """
    Result from False Acceptance Rate (FAR) computation.
    
    Attributes:
        value: The FAR value (failed_abstentions / total_nonexistent_queries)
        failed_abstentions: Count of queries where model failed to abstain
        correct_refusals: Count of queries where model correctly refused
        total_nonexistent_queries: Total number of non-existent entity queries
        abstention_results: List of individual abstention evaluation results
    
    Validates: Requirements 20.2
    """
    value: float
    failed_abstentions: int
    correct_refusals: int
    total_nonexistent_queries: int
    abstention_results: List[AbstentionResult] = field(default_factory=list)



@dataclass
class ClaimVerificationMatrix:
    """
    Matrix tracking which claims are supported by which models.
    
    Attributes:
        claims: List of unique claims
        models: List of model names
        support_matrix: 2D list of shape (num_claims, num_models) with binary values
                       1 = claim supported by model, 0 = claim not supported
    """
    claims: List[Claim]
    models: List[str]
    support_matrix: List[List[int]]  # shape: (num_claims, num_models)
    
    def get_claim_support_count(self, claim_index: int) -> int:
        """Get the number of models that support a specific claim."""
        if 0 <= claim_index < len(self.claims):
            return sum(self.support_matrix[claim_index])
        return 0
    
    def get_model_claim_count(self, model_index: int) -> int:
        """Get the number of claims supported by a specific model."""
        if 0 <= model_index < len(self.models):
            return sum(row[model_index] for row in self.support_matrix)
        return 0
    
    def get_consensus_claims(self, threshold: float = 0.5) -> List[Claim]:
        """Get claims supported by at least threshold fraction of models."""
        consensus = []
        num_models = len(self.models)
        if num_models == 0:
            return consensus
        for i, claim in enumerate(self.claims):
            support_ratio = self.get_claim_support_count(i) / num_models
            if support_ratio >= threshold:
                consensus.append(claim)
        return consensus
    
    def get_disputed_claims(self, threshold: float = 0.5) -> List[Claim]:
        """Get claims supported by less than threshold fraction of models."""
        disputed = []
        num_models = len(self.models)
        if num_models == 0:
            return disputed
        for i, claim in enumerate(self.claims):
            support_ratio = self.get_claim_support_count(i) / num_models
            if support_ratio < threshold:
                disputed.append(claim)
        return disputed


@dataclass
class ClaimConsensus:
    """
    Consensus information for a single claim.
    
    Attributes:
        claim: The claim being analyzed
        support_count: Number of models supporting this claim
        total_models: Total number of models
        consensus_ratio: Fraction of models supporting this claim
        is_consensus: True if majority of models agree
    """
    claim: Claim
    support_count: int
    total_models: int
    consensus_ratio: float
    is_consensus: bool


@dataclass
class HallucinationProfile:
    """
    Comprehensive hallucination profile with all quantified metrics.
    
    Attributes:
        mihr: Micro Hallucination Rate result
        mahr: Macro Hallucination Rate result (optional, requires multiple responses)
        factscore: FactScore value (verified_claims / total_claims)
        consensus_f1: Consensus F1 result (optional, requires multiple models)
        fleiss_kappa: Fleiss' Kappa result (optional, requires multiple judges)
        uncertainty: Uncertainty quantification result (optional)
        reliability: Reliability classification (high, medium, low)
        disputed_claims: Claims with low agreement across models
        consensus_claims: Claims with high agreement across models
        is_high_risk: True if MiHR > 0.3 or Kappa < 0.4 or uncertainty > 0.8
    """
    mihr: Optional[MiHRResult]
    mahr: Optional[MaHRResult]
    factscore: Optional[float]
    consensus_f1: Optional[ConsensusF1Result]
    fleiss_kappa: Optional[KappaResult]
    uncertainty: Optional[UncertaintyResult]
    reliability: ReliabilityLevel
    disputed_claims: List[Claim] = field(default_factory=list)
    consensus_claims: List[Claim] = field(default_factory=list)
    is_high_risk: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return _dataclass_to_dict(self)
    
    def to_json(self, indent: Optional[int] = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HallucinationProfile":
        """Create HallucinationProfile from dictionary."""
        # Convert nested structures
        mihr = MiHRResult(**data["mihr"]) if data.get("mihr") else None
        mahr = MaHRResult(**data["mahr"]) if data.get("mahr") else None
        consensus_f1 = ConsensusF1Result(**data["consensus_f1"]) if data.get("consensus_f1") else None
        fleiss_kappa = KappaResult(**data["fleiss_kappa"]) if data.get("fleiss_kappa") else None
        uncertainty = UncertaintyResult(**data["uncertainty"]) if data.get("uncertainty") else None
        
        # Convert claims
        disputed_claims = []
        for c in data.get("disputed_claims", []):
            from llm_judge_auditor.models import ClaimType
            disputed_claims.append(Claim(
                text=c["text"],
                source_span=tuple(c["source_span"]),
                claim_type=ClaimType(c.get("claim_type", "factual"))
            ))
        
        consensus_claims = []
        for c in data.get("consensus_claims", []):
            from llm_judge_auditor.models import ClaimType
            consensus_claims.append(Claim(
                text=c["text"],
                source_span=tuple(c["source_span"]),
                claim_type=ClaimType(c.get("claim_type", "factual"))
            ))
        
        return cls(
            mihr=mihr,
            mahr=mahr,
            factscore=data.get("factscore"),
            consensus_f1=consensus_f1,
            fleiss_kappa=fleiss_kappa,
            uncertainty=uncertainty,
            reliability=ReliabilityLevel(data["reliability"]),
            disputed_claims=disputed_claims,
            consensus_claims=consensus_claims,
            is_high_risk=data.get("is_high_risk", False)
        )
    
    @classmethod
    def from_json(cls, json_str: str) -> "HallucinationProfile":
        """Create HallucinationProfile from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)


class HallucinationMetricsConfig:
    """Configuration for hallucination metrics thresholds."""
    
    def __init__(
        self,
        mihr_high_risk_threshold: float = 0.3,
        kappa_low_threshold: float = 0.4,
        uncertainty_high_threshold: float = 0.8,
    ):
        """
        Initialize configuration.
        
        Args:
            mihr_high_risk_threshold: MiHR threshold for high risk (default 0.3)
            kappa_low_threshold: Kappa threshold below which is low agreement (default 0.4)
            uncertainty_high_threshold: Uncertainty threshold for high uncertainty (default 0.8)
        """
        self.mihr_high_risk_threshold = mihr_high_risk_threshold
        self.kappa_low_threshold = kappa_low_threshold
        self.uncertainty_high_threshold = uncertainty_high_threshold


def _dataclass_to_dict(obj: Any) -> Any:
    """
    Convert a dataclass to a dictionary, handling enums and nested structures.
    
    Args:
        obj: Object to convert
        
    Returns:
        Dictionary representation suitable for JSON serialization
    """
    if isinstance(obj, Enum):
        return obj.value
    elif hasattr(obj, "__dataclass_fields__"):
        result = {}
        for field_name in obj.__dataclass_fields__:
            field_value = getattr(obj, field_name)
            result[field_name] = _dataclass_to_dict(field_value)
        return result
    elif isinstance(obj, dict):
        return {key: _dataclass_to_dict(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_dataclass_to_dict(item) for item in obj]
    else:
        return obj



class HallucinationMetricsCalculator:
    """
    Calculator for research-backed hallucination quantification metrics.
    
    Computes:
    - MiHR (Micro Hallucination Rate): unsupported_claims / total_claims
    - MaHR (Macro Hallucination Rate): responses_with_hallucinations / total_responses
    - FactScore: verified_claims / total_claims
    - Consensus F1: Cross-model agreement metric
    - Fleiss' Kappa: Inter-judge agreement statistic
    - Uncertainty quantification: Shannon entropy with epistemic/aleatoric decomposition
    
    Validates: Requirements 15.1, 15.2, 15.3, 15.4, 15.5, 18.1, 18.2, 18.3, 18.4, 18.5, 19.1, 19.2, 19.5
    """
    
    def __init__(self, config: Optional[HallucinationMetricsConfig] = None):
        """
        Initialize the calculator.
        
        Args:
            config: Configuration for thresholds. Uses defaults if not provided.
        """
        self.config = config or HallucinationMetricsConfig()
    
    def compute_shannon_entropy(
        self,
        probabilities: List[float],
    ) -> float:
        """
        Compute Shannon entropy for a probability distribution.
        
        H(p) = -Σ pᵢ log pᵢ
        
        Uses natural logarithm (base e). For a uniform distribution over n outcomes,
        the maximum entropy is ln(n).
        
        Args:
            probabilities: List of probabilities that should sum to 1.0
                          (or close to 1.0 within tolerance)
            
        Returns:
            Shannon entropy value (non-negative float)
            
        Validates: Requirements 18.1
        """
        if not probabilities:
            return 0.0
        
        entropy = 0.0
        for p in probabilities:
            # Handle edge case: zero probabilities
            # By convention, 0 * log(0) = 0 in entropy calculations
            if p > 0:
                entropy -= p * math.log(p)
        
        # Entropy should always be non-negative
        return max(0.0, entropy)
    
    def compute_epistemic_uncertainty(
        self,
        inference_samples: List[List[float]],
    ) -> float:
        """
        Compute epistemic uncertainty (model uncertainty from lack of knowledge).
        
        Epistemic uncertainty = Var(E[p]) across inference samples
        
        This measures the variance of the probability predictions across multiple
        inference runs for each outcome, then averages across outcomes.
        High epistemic uncertainty indicates the model is uncertain about its predictions.
        
        Args:
            inference_samples: List of probability distributions from multiple
                              inference runs. Each inner list is a probability
                              distribution over outcomes.
            
        Returns:
            Epistemic uncertainty value (non-negative float)
            
        Validates: Requirements 18.2
        """
        if not inference_samples or len(inference_samples) < 2:
            return 0.0
        
        num_samples = len(inference_samples)
        num_outcomes = len(inference_samples[0])
        
        if num_outcomes == 0:
            return 0.0
        
        # For each outcome, compute the variance of probabilities across samples
        # This captures how much the model's predictions vary for each outcome
        outcome_variances = []
        
        for outcome_idx in range(num_outcomes):
            outcome_probs = [
                sample[outcome_idx] 
                for sample in inference_samples 
                if outcome_idx < len(sample)
            ]
            if len(outcome_probs) >= 2:
                # Compute variance of this outcome's probability across samples
                mean_prob = sum(outcome_probs) / len(outcome_probs)
                variance = sum((p - mean_prob) ** 2 for p in outcome_probs) / len(outcome_probs)
                outcome_variances.append(variance)
        
        if not outcome_variances:
            return 0.0
        
        # Average variance across all outcomes
        # This is the epistemic uncertainty: how much predictions vary across samples
        epistemic = sum(outcome_variances) / len(outcome_variances)
        
        return max(0.0, epistemic)
    
    def compute_aleatoric_uncertainty(
        self,
        inference_samples: List[List[float]],
    ) -> float:
        """
        Compute aleatoric uncertainty (inherent data noise).
        
        Aleatoric uncertainty = E[Var(p)] within inference samples
        
        This measures the expected variance within each inference sample,
        capturing inherent randomness in the data.
        
        Args:
            inference_samples: List of probability distributions from multiple
                              inference runs. Each inner list is a probability
                              distribution over outcomes.
            
        Returns:
            Aleatoric uncertainty value (non-negative float)
            
        Validates: Requirements 18.3
        """
        if not inference_samples:
            return 0.0
        
        # Compute variance within each sample, then average
        sample_variances = []
        
        for sample in inference_samples:
            if not sample:
                continue
            
            # Compute variance of probabilities within this sample
            mean_p = sum(sample) / len(sample)
            variance = sum((p - mean_p) ** 2 for p in sample) / len(sample)
            sample_variances.append(variance)
        
        if not sample_variances:
            return 0.0
        
        # Expected variance (average of variances)
        expected_variance = sum(sample_variances) / len(sample_variances)
        
        return max(0.0, expected_variance)
    
    def compute_uncertainty(
        self,
        probabilities: List[float],
        inference_samples: Optional[List[List[float]]] = None,
    ) -> UncertaintyResult:
        """
        Compute comprehensive uncertainty quantification.
        
        Computes:
        - Shannon entropy: H(p) = -Σ pᵢ log pᵢ
        - Epistemic uncertainty: Var(E[p]) across inference samples
        - Aleatoric uncertainty: E[Var(p)] within inference samples
        - Total uncertainty: epistemic + aleatoric
        
        Args:
            probabilities: Primary probability distribution for entropy calculation
            inference_samples: Optional list of probability distributions from
                              multiple inference runs for epistemic/aleatoric
                              decomposition. If not provided, epistemic and
                              aleatoric will be 0.0.
            
        Returns:
            UncertaintyResult with all uncertainty metrics
            
        Validates: Requirements 18.1, 18.2, 18.3, 18.4, 18.5
        """
        # Compute Shannon entropy (Requirement 18.1)
        shannon_entropy = self.compute_shannon_entropy(probabilities)
        
        # Compute epistemic and aleatoric uncertainty if samples provided
        if inference_samples and len(inference_samples) >= 2:
            # Requirement 18.2: Epistemic = Var(E[p]) across samples
            epistemic = self.compute_epistemic_uncertainty(inference_samples)
            
            # Requirement 18.3: Aleatoric = E[Var(p)] within samples
            aleatoric = self.compute_aleatoric_uncertainty(inference_samples)
        else:
            epistemic = 0.0
            aleatoric = 0.0
        
        # Requirement 18.5: Total = epistemic + aleatoric
        total = epistemic + aleatoric
        
        # Requirement 18.4: Flag high uncertainty
        is_high_uncertainty = total > self.config.uncertainty_high_threshold
        
        return UncertaintyResult(
            shannon_entropy=shannon_entropy,
            epistemic=epistemic,
            aleatoric=aleatoric,
            total=total,
            is_high_uncertainty=is_high_uncertainty
        )
    
    def flag_high_uncertainty(
        self,
        uncertainty: UncertaintyResult,
        threshold: Optional[float] = None,
    ) -> bool:
        """
        Determine if uncertainty exceeds the threshold.
        
        Args:
            uncertainty: UncertaintyResult to evaluate
            threshold: Optional custom threshold. If not provided, uses
                      config.uncertainty_high_threshold
            
        Returns:
            True if uncertainty exceeds threshold
            
        Validates: Requirements 18.4
        """
        if threshold is None:
            threshold = self.config.uncertainty_high_threshold
        
        return uncertainty.total > threshold
    
    def compute_mihr(
        self,
        verdicts: List[Verdict],
    ) -> MiHRResult:
        """
        Compute Micro Hallucination Rate (MiHR).
        
        MiHR = unsupported_claims / total_claims
        
        A claim is considered unsupported if its verdict is REFUTED or NOT_ENOUGH_INFO.
        
        Args:
            verdicts: List of verdicts for each claim
            
        Returns:
            MiHRResult with the computed rate and metadata
            
        Validates: Requirements 15.1, 15.3, 15.4, 15.5
        """
        total_claims = len(verdicts)
        
        # Handle zero claims edge case (Requirement 15.5)
        if total_claims == 0:
            return MiHRResult(
                value=None,
                unsupported_claims=0,
                total_claims=0,
                has_claims=False
            )
        
        # Count unsupported claims (REFUTED or NOT_ENOUGH_INFO)
        # Requirement 15.3: Classify each claim as supported, refuted, or unverifiable
        unsupported_claims = sum(
            1 for v in verdicts 
            if v.label in (VerdictLabel.REFUTED, VerdictLabel.NOT_ENOUGH_INFO)
        )
        
        # Compute MiHR (Requirement 15.1)
        mihr_value = unsupported_claims / total_claims
        
        # Ensure output in range [0.0, 1.0] (Requirement 15.4)
        mihr_value = max(0.0, min(1.0, mihr_value))
        
        return MiHRResult(
            value=mihr_value,
            unsupported_claims=unsupported_claims,
            total_claims=total_claims,
            has_claims=True
        )
    
    def compute_mahr(
        self,
        response_verdicts: List[List[Verdict]],
    ) -> MaHRResult:
        """
        Compute Macro Hallucination Rate (MaHR).
        
        MaHR = responses_with_hallucinations / total_responses
        
        A response is considered to have hallucinations if any of its claims
        are REFUTED or NOT_ENOUGH_INFO.
        
        Args:
            response_verdicts: List of verdict lists, one per response
            
        Returns:
            MaHRResult with the computed rate and metadata
            
        Validates: Requirements 15.2, 15.4
        """
        total_responses = len(response_verdicts)
        
        if total_responses == 0:
            return MaHRResult(
                value=0.0,
                responses_with_hallucinations=0,
                total_responses=0
            )
        
        # Count responses with at least one hallucination
        responses_with_hallucinations = 0
        for verdicts in response_verdicts:
            has_hallucination = any(
                v.label in (VerdictLabel.REFUTED, VerdictLabel.NOT_ENOUGH_INFO)
                for v in verdicts
            )
            if has_hallucination:
                responses_with_hallucinations += 1
        
        # Compute MaHR (Requirement 15.2)
        mahr_value = responses_with_hallucinations / total_responses
        
        # Ensure output in range [0.0, 1.0] (Requirement 15.4)
        mahr_value = max(0.0, min(1.0, mahr_value))
        
        return MaHRResult(
            value=mahr_value,
            responses_with_hallucinations=responses_with_hallucinations,
            total_responses=total_responses
        )
    
    def is_high_risk(
        self,
        mihr: Optional[MiHRResult] = None,
        kappa: Optional[KappaResult] = None,
        uncertainty: Optional[UncertaintyResult] = None,
    ) -> bool:
        """
        Determine if the evaluation is high risk based on thresholds.
        
        High risk if:
        - MiHR > 0.3 (configurable)
        - Kappa < 0.4 (configurable)
        - Uncertainty > 0.8 (configurable)
        
        Args:
            mihr: MiHR result
            kappa: Kappa result
            uncertainty: Uncertainty result
            
        Returns:
            True if any threshold is exceeded
            
        Validates: Requirements 19.5
        """
        # Check MiHR threshold
        if mihr and mihr.value is not None:
            if mihr.value > self.config.mihr_high_risk_threshold:
                return True
        
        # Check Kappa threshold
        if kappa and kappa.kappa is not None and not kappa.is_undefined:
            if kappa.kappa < self.config.kappa_low_threshold:
                return True
        
        # Check uncertainty threshold
        if uncertainty:
            if uncertainty.total > self.config.uncertainty_high_threshold:
                return True
        
        return False
    
    def compute_factscore(
        self,
        verdicts: List[Verdict],
    ) -> Optional[float]:
        """
        Compute FactScore (factual precision metric).
        
        FactScore = verified_claims / total_claims
        
        A claim is considered verified if its verdict is SUPPORTED.
        
        Args:
            verdicts: List of verdicts for each claim
            
        Returns:
            FactScore value in range [0.0, 1.0], or None if no claims
            
        Validates: Requirements 16.1
        """
        total_claims = len(verdicts)
        
        # Handle zero claims edge case
        if total_claims == 0:
            return None
        
        # Count verified (SUPPORTED) claims
        verified_claims = sum(
            1 for v in verdicts 
            if v.label == VerdictLabel.SUPPORTED
        )
        
        # Compute FactScore (Requirement 16.1)
        factscore = verified_claims / total_claims
        
        # Ensure output in range [0.0, 1.0]
        return max(0.0, min(1.0, factscore))

    def compute_consensus_f1(
        self,
        matrix: "ClaimVerificationMatrix",
        model_name: str,
        consensus_threshold: float = 0.5,
    ) -> ConsensusF1Result:
        """
        Compute Consensus F1 for a specific model.
        
        Precision = model_claims_supported_by_others / model_claims
        Recall = consensus_claims_included / total_consensus_claims
        F1 = 2 × (precision × recall) / (precision + recall)
        
        Args:
            matrix: The claim verification matrix
            model_name: Name of the model to compute F1 for
            model_claims: Claims made by this model
            consensus_threshold: Fraction of models required for consensus
            
        Returns:
            ConsensusF1Result with precision, recall, and F1
            
        Validates: Requirements 16.3, 16.4, 16.5
        """
        # Handle edge case: model not in matrix
        if model_name not in matrix.models:
            return ConsensusF1Result(precision=0.0, recall=0.0, f1=0.0)
        
        model_idx = matrix.models.index(model_name)
        num_models = len(matrix.models)
        
        if num_models == 0:
            return ConsensusF1Result(precision=0.0, recall=0.0, f1=0.0)
        
        # Count model's claims and claims supported by others
        model_claim_count = 0
        model_claims_supported_by_others = 0
        
        # Count consensus claims and how many the model includes
        consensus_claims_total = 0
        consensus_claims_included_by_model = 0
        
        for claim_idx in range(len(matrix.claims)):
            support_count = matrix.get_claim_support_count(claim_idx)
            model_supports = matrix.support_matrix[claim_idx][model_idx] == 1
            
            # Check if this is a consensus claim (supported by >= threshold of models)
            consensus_ratio = support_count / num_models
            is_consensus = consensus_ratio >= consensus_threshold
            
            if is_consensus:
                consensus_claims_total += 1
                if model_supports:
                    consensus_claims_included_by_model += 1
            
            if model_supports:
                model_claim_count += 1
                # Count how many OTHER models support this claim
                other_support = support_count - 1  # Exclude this model
                other_models = num_models - 1
                if other_models > 0 and other_support > 0:
                    model_claims_supported_by_others += 1
        
        # Compute precision (Requirement 16.3)
        # precision = model_claims_supported_by_others / model_claims
        if model_claim_count == 0:
            precision = 0.0
        else:
            precision = model_claims_supported_by_others / model_claim_count
        
        # Compute recall (Requirement 16.4)
        # recall = consensus_claims_included / total_consensus_claims
        if consensus_claims_total == 0:
            recall = 0.0
        else:
            recall = consensus_claims_included_by_model / consensus_claims_total
        
        # Compute F1 (Requirement 16.5)
        # F1 = 2 × (precision × recall) / (precision + recall)
        # Handle zero division (return 0.0)
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)
        
        # Ensure all values are in range [0.0, 1.0]
        precision = max(0.0, min(1.0, precision))
        recall = max(0.0, min(1.0, recall))
        f1 = max(0.0, min(1.0, f1))
        
        return ConsensusF1Result(precision=precision, recall=recall, f1=f1)

    def compute_average_consensus_f1(
        self,
        matrix: "ClaimVerificationMatrix",
        consensus_threshold: float = 0.5,
    ) -> ConsensusF1Result:
        """
        Compute average Consensus F1 across all models.
        
        Args:
            matrix: The claim verification matrix
            consensus_threshold: Fraction of models required for consensus
            
        Returns:
            ConsensusF1Result with averaged precision, recall, and F1
        """
        if not matrix.models:
            return ConsensusF1Result(precision=0.0, recall=0.0, f1=0.0)
        
        total_precision = 0.0
        total_recall = 0.0
        total_f1 = 0.0
        
        for model_name in matrix.models:
            result = self.compute_consensus_f1(matrix, model_name, consensus_threshold)
            total_precision += result.precision
            total_recall += result.recall
            total_f1 += result.f1
        
        num_models = len(matrix.models)
        return ConsensusF1Result(
            precision=total_precision / num_models,
            recall=total_recall / num_models,
            f1=total_f1 / num_models
        )

    def compute_fleiss_kappa(
        self,
        ratings: List[List[int]],
        num_categories: int,
    ) -> KappaResult:
        """
        Compute Fleiss' Kappa for inter-judge agreement.
        
        Fleiss' Kappa measures agreement among multiple raters when assigning
        categorical ratings to a fixed number of items.
        
        Formula: κ = (Po - Pe) / (1 - Pe)
        where:
        - Po = observed agreement (proportion of agreeing pairs)
        - Pe = expected agreement by chance
        
        Args:
            ratings: 2D list where ratings[i][j] is the number of raters who
                    assigned category j to item i. Shape: (num_items, num_categories)
            num_categories: Number of possible categories (k)
            
        Returns:
            KappaResult with kappa value, interpretation, Po, and Pe
            
        Validates: Requirements 17.1, 17.2, 17.4
        """
        # Validate input
        if not ratings:
            return KappaResult(
                kappa=None,
                interpretation="undefined",
                observed_agreement=0.0,
                expected_agreement=0.0,
                is_undefined=True,
                error_message="No ratings provided"
            )
        
        num_items = len(ratings)
        
        # Check that all rows have the correct number of categories
        for i, row in enumerate(ratings):
            if len(row) != num_categories:
                return KappaResult(
                    kappa=None,
                    interpretation="undefined",
                    observed_agreement=0.0,
                    expected_agreement=0.0,
                    is_undefined=True,
                    error_message=f"Row {i} has {len(row)} categories, expected {num_categories}"
                )
        
        # Calculate number of raters (n) from the first item
        # All items should have the same number of raters
        num_raters = sum(ratings[0])
        
        # Requirement 17.4: Handle fewer than 2 judges
        if num_raters < 2:
            return KappaResult(
                kappa=None,
                interpretation="undefined",
                observed_agreement=0.0,
                expected_agreement=0.0,
                is_undefined=True,
                error_message="Fewer than 2 judges provided ratings"
            )
        
        # Verify all items have the same number of raters
        for i, row in enumerate(ratings):
            if sum(row) != num_raters:
                return KappaResult(
                    kappa=None,
                    interpretation="undefined",
                    observed_agreement=0.0,
                    expected_agreement=0.0,
                    is_undefined=True,
                    error_message=f"Item {i} has {sum(row)} raters, expected {num_raters}"
                )
        
        # Step 1: Compute P_i for each item (proportion of agreeing pairs)
        # P_i = (1 / (n * (n-1))) * sum_j(n_ij * (n_ij - 1))
        # where n_ij is the number of raters who assigned category j to item i
        
        p_i_values = []
        denominator = num_raters * (num_raters - 1)
        
        for row in ratings:
            sum_pairs = sum(n_ij * (n_ij - 1) for n_ij in row)
            p_i = sum_pairs / denominator if denominator > 0 else 0.0
            p_i_values.append(p_i)
        
        # Step 2: Compute Po (observed agreement)
        # Po = (1/N) * sum_i(P_i)
        po = sum(p_i_values) / num_items if num_items > 0 else 0.0
        
        # Step 3: Compute p_j for each category (proportion of all assignments to category j)
        # p_j = (1 / (N * n)) * sum_i(n_ij)
        total_assignments = num_items * num_raters
        
        p_j_values = []
        for j in range(num_categories):
            sum_category = sum(ratings[i][j] for i in range(num_items))
            p_j = sum_category / total_assignments if total_assignments > 0 else 0.0
            p_j_values.append(p_j)
        
        # Step 4: Compute Pe (expected agreement by chance)
        # Pe = sum_j(p_j^2)
        pe = sum(p_j ** 2 for p_j in p_j_values)
        
        # Step 5: Compute Kappa
        # κ = (Po - Pe) / (1 - Pe)
        if abs(1 - pe) < 1e-10:
            # Perfect expected agreement - kappa is undefined or 1
            if abs(po - pe) < 1e-10:
                kappa = 1.0  # Perfect agreement
            else:
                kappa = 0.0  # Cannot compute
        else:
            kappa = (po - pe) / (1 - pe)
        
        # Clamp kappa to valid range [-1, 1]
        kappa = max(-1.0, min(1.0, kappa))
        
        # Get interpretation
        interpretation = self.interpret_kappa(kappa)
        
        return KappaResult(
            kappa=kappa,
            interpretation=interpretation,
            observed_agreement=po,
            expected_agreement=pe,
            is_undefined=False,
            error_message=None
        )
    
    def interpret_kappa(self, kappa: float) -> str:
        """
        Interpret Fleiss' Kappa value according to standard thresholds.
        
        Interpretation scale:
        - κ < 0.2: poor agreement
        - 0.2 ≤ κ < 0.4: fair agreement
        - 0.4 ≤ κ < 0.6: moderate agreement
        - 0.6 ≤ κ < 0.8: substantial agreement
        - κ ≥ 0.8: almost perfect agreement
        
        Args:
            kappa: The kappa value to interpret
            
        Returns:
            String interpretation of the kappa value
            
        Validates: Requirements 17.3
        """
        if kappa < 0.2:
            return "poor"
        elif kappa < 0.4:
            return "fair"
        elif kappa < 0.6:
            return "moderate"
        elif kappa < 0.8:
            return "substantial"
        else:
            return "almost_perfect"
    
    def compute_fleiss_kappa_from_verdicts(
        self,
        judge_verdicts: Dict[str, List[Verdict]],
    ) -> KappaResult:
        """
        Compute Fleiss' Kappa from judge verdicts.
        
        This is a convenience method that converts verdict lists from multiple
        judges into the rating matrix format required by compute_fleiss_kappa.
        
        Args:
            judge_verdicts: Dictionary mapping judge names to their verdict lists.
                           All judges must have evaluated the same items in the same order.
            
        Returns:
            KappaResult with kappa value, interpretation, Po, and Pe
            
        Validates: Requirements 17.1, 17.2, 17.3, 17.4
        """
        if not judge_verdicts:
            return KappaResult(
                kappa=None,
                interpretation="undefined",
                observed_agreement=0.0,
                expected_agreement=0.0,
                is_undefined=True,
                error_message="No judge verdicts provided"
            )
        
        judges = list(judge_verdicts.keys())
        num_judges = len(judges)
        
        # Requirement 17.4: Handle fewer than 2 judges
        if num_judges < 2:
            return KappaResult(
                kappa=None,
                interpretation="undefined",
                observed_agreement=0.0,
                expected_agreement=0.0,
                is_undefined=True,
                error_message="Fewer than 2 judges provided ratings"
            )
        
        # Get the number of items from the first judge
        first_judge_verdicts = judge_verdicts[judges[0]]
        num_items = len(first_judge_verdicts)
        
        if num_items == 0:
            return KappaResult(
                kappa=None,
                interpretation="undefined",
                observed_agreement=0.0,
                expected_agreement=0.0,
                is_undefined=True,
                error_message="No items to rate"
            )
        
        # Verify all judges have the same number of verdicts
        for judge_name, verdicts in judge_verdicts.items():
            if len(verdicts) != num_items:
                return KappaResult(
                    kappa=None,
                    interpretation="undefined",
                    observed_agreement=0.0,
                    expected_agreement=0.0,
                    is_undefined=True,
                    error_message=f"Judge {judge_name} has {len(verdicts)} verdicts, expected {num_items}"
                )
        
        # Categories: SUPPORTED=0, REFUTED=1, NOT_ENOUGH_INFO=2
        category_map = {
            VerdictLabel.SUPPORTED: 0,
            VerdictLabel.REFUTED: 1,
            VerdictLabel.NOT_ENOUGH_INFO: 2,
        }
        num_categories = 3
        
        # Build the rating matrix
        # ratings[i][j] = number of judges who assigned category j to item i
        ratings: List[List[int]] = []
        
        for item_idx in range(num_items):
            row = [0] * num_categories
            for judge_name in judges:
                verdict = judge_verdicts[judge_name][item_idx]
                category = category_map.get(verdict.label, 0)
                row[category] += 1
            ratings.append(row)
        
        return self.compute_fleiss_kappa(ratings, num_categories)

    def determine_reliability(
        self,
        mihr: Optional[MiHRResult] = None,
        kappa: Optional[KappaResult] = None,
        uncertainty: Optional[UncertaintyResult] = None,
    ) -> ReliabilityLevel:
        """
        Determine reliability classification based on metrics.
        
        Args:
            mihr: MiHR result
            kappa: Kappa result
            uncertainty: Uncertainty result
            
        Returns:
            ReliabilityLevel (HIGH, MEDIUM, or LOW)
            
        Validates: Requirements 19.2
        """
        # High risk = low reliability
        if self.is_high_risk(mihr, kappa, uncertainty):
            return ReliabilityLevel.LOW
        
        # Check for medium reliability indicators
        medium_indicators = 0
        
        if mihr and mihr.value is not None:
            if mihr.value > 0.15:  # Half of high risk threshold
                medium_indicators += 1
        
        if kappa and kappa.kappa is not None and not kappa.is_undefined:
            if kappa.kappa < 0.6:  # Moderate agreement threshold
                medium_indicators += 1
        
        if uncertainty:
            if uncertainty.total > 0.5:  # Moderate uncertainty
                medium_indicators += 1
        
        if medium_indicators >= 2:
            return ReliabilityLevel.MEDIUM
        
        return ReliabilityLevel.HIGH

    def generate_hallucination_profile(
        self,
        verdicts: List[Verdict],
        response_verdicts: Optional[List[List[Verdict]]] = None,
        claim_matrix: Optional["ClaimVerificationMatrix"] = None,
        judge_verdicts: Optional[Dict[str, List[Verdict]]] = None,
        probabilities: Optional[List[float]] = None,
        inference_samples: Optional[List[List[float]]] = None,
        consensus_threshold: float = 0.5,
    ) -> HallucinationProfile:
        """
        Generate a comprehensive hallucination profile with all quantified metrics.
        
        Compiles MiHR, MaHR, FactScore, Consensus F1, Fleiss' Kappa, and uncertainty
        into a single profile with reliability classification and claim-level analysis.
        
        Args:
            verdicts: List of verdicts for the primary response (for MiHR and FactScore)
            response_verdicts: Optional list of verdict lists for multiple responses (for MaHR)
            claim_matrix: Optional claim verification matrix (for Consensus F1 and claim analysis)
            judge_verdicts: Optional dictionary of judge verdicts (for Fleiss' Kappa)
            probabilities: Optional probability distribution (for uncertainty)
            inference_samples: Optional inference samples (for epistemic/aleatoric decomposition)
            consensus_threshold: Threshold for consensus claims (default 0.5)
            
        Returns:
            HallucinationProfile with all metrics and analysis
            
        Validates: Requirements 19.1, 19.2, 19.3, 19.4, 19.5
        """
        # Compute MiHR (Requirement 15.1)
        mihr = self.compute_mihr(verdicts)
        
        # Compute MaHR if multiple responses provided (Requirement 15.2)
        mahr = None
        if response_verdicts:
            mahr = self.compute_mahr(response_verdicts)
        
        # Compute FactScore (Requirement 16.1)
        factscore = self.compute_factscore(verdicts)
        
        # Compute Consensus F1 if claim matrix provided (Requirements 16.3, 16.4, 16.5)
        consensus_f1 = None
        if claim_matrix and claim_matrix.models:
            consensus_f1 = self.compute_average_consensus_f1(claim_matrix, consensus_threshold)
        
        # Compute Fleiss' Kappa if judge verdicts provided (Requirements 17.1, 17.2, 17.3)
        fleiss_kappa = None
        if judge_verdicts:
            fleiss_kappa = self.compute_fleiss_kappa_from_verdicts(judge_verdicts)
        
        # Compute uncertainty if probabilities provided (Requirements 18.1-18.5)
        uncertainty = None
        if probabilities:
            uncertainty = self.compute_uncertainty(probabilities, inference_samples)
        
        # Get disputed and consensus claims from matrix (Requirement 19.3)
        disputed_claims: List[Claim] = []
        consensus_claims: List[Claim] = []
        if claim_matrix:
            disputed_claims = claim_matrix.get_disputed_claims(consensus_threshold)
            consensus_claims = claim_matrix.get_consensus_claims(consensus_threshold)
        
        # Determine reliability classification (Requirement 19.2)
        reliability = self.determine_reliability(mihr, fleiss_kappa, uncertainty)
        
        # Determine if high risk (Requirement 19.5)
        high_risk = self.is_high_risk(mihr, fleiss_kappa, uncertainty)
        
        return HallucinationProfile(
            mihr=mihr,
            mahr=mahr,
            factscore=factscore,
            consensus_f1=consensus_f1,
            fleiss_kappa=fleiss_kappa,
            uncertainty=uncertainty,
            reliability=reliability,
            disputed_claims=disputed_claims,
            consensus_claims=consensus_claims,
            is_high_risk=high_risk,
        )


class ClaimVerificationMatrixBuilder:
    """
    Builder for constructing claim verification matrices for cross-model consensus analysis.
    
    Tracks which claims appear in which model responses and computes consensus metrics.
    
    Validates: Requirements 16.2
    """
    
    def build_matrix(
        self,
        model_claims: Dict[str, List[Claim]],
        model_verdicts: Optional[Dict[str, List[Verdict]]] = None,
    ) -> ClaimVerificationMatrix:
        """
        Build a claim verification matrix from model responses.
        
        The matrix has dimensions (num_unique_claims × num_models) with binary values
        indicating whether each model supports each claim.
        
        Args:
            model_claims: Dictionary mapping model names to their extracted claims
            model_verdicts: Optional dictionary mapping model names to verdicts for their claims.
                           If provided, only SUPPORTED verdicts count as support.
                           If not provided, presence of claim counts as support.
            
        Returns:
            ClaimVerificationMatrix with the constructed matrix
            
        Validates: Requirements 16.2
        """
        models = list(model_claims.keys())
        
        if not models:
            return ClaimVerificationMatrix(
                claims=[],
                models=[],
                support_matrix=[]
            )
        
        # Collect all unique claims across models
        # Use claim text as the key for uniqueness
        unique_claims_dict: Dict[str, Claim] = {}
        for model_name, claims in model_claims.items():
            for claim in claims:
                # Normalize claim text for comparison
                normalized_text = claim.text.strip().lower()
                if normalized_text not in unique_claims_dict:
                    unique_claims_dict[normalized_text] = claim
        
        unique_claims = list(unique_claims_dict.values())
        claim_text_to_index = {
            claim.text.strip().lower(): i 
            for i, claim in enumerate(unique_claims)
        }
        
        # Build the support matrix
        # Dimensions: (num_unique_claims × num_models)
        num_claims = len(unique_claims)
        num_models = len(models)
        
        support_matrix: List[List[int]] = [
            [0] * num_models for _ in range(num_claims)
        ]
        
        for model_idx, model_name in enumerate(models):
            claims = model_claims[model_name]
            verdicts = model_verdicts.get(model_name, []) if model_verdicts else []
            
            for claim_idx, claim in enumerate(claims):
                normalized_text = claim.text.strip().lower()
                if normalized_text in claim_text_to_index:
                    matrix_claim_idx = claim_text_to_index[normalized_text]
                    
                    # Determine if this model supports this claim
                    if verdicts and claim_idx < len(verdicts):
                        # Use verdict to determine support
                        if verdicts[claim_idx].label == VerdictLabel.SUPPORTED:
                            support_matrix[matrix_claim_idx][model_idx] = 1
                    else:
                        # No verdicts provided - presence counts as support
                        support_matrix[matrix_claim_idx][model_idx] = 1
        
        return ClaimVerificationMatrix(
            claims=unique_claims,
            models=models,
            support_matrix=support_matrix
        )
    
    def compute_claim_consensus(
        self,
        matrix: ClaimVerificationMatrix,
        threshold: float = 0.5,
    ) -> List[ClaimConsensus]:
        """
        Compute consensus information for each claim in the matrix.
        
        Args:
            matrix: The claim verification matrix
            threshold: Fraction of models required for consensus (default 0.5)
            
        Returns:
            List of ClaimConsensus objects for each claim
        """
        results = []
        num_models = len(matrix.models)
        
        if num_models == 0:
            return results
        
        for i, claim in enumerate(matrix.claims):
            support_count = matrix.get_claim_support_count(i)
            consensus_ratio = support_count / num_models
            is_consensus = consensus_ratio >= threshold
            
            results.append(ClaimConsensus(
                claim=claim,
                support_count=support_count,
                total_models=num_models,
                consensus_ratio=consensus_ratio,
                is_consensus=is_consensus
            ))
        
        return results
    
    def identify_disputed_claims(
        self,
        matrix: ClaimVerificationMatrix,
        threshold: float = 0.5,
    ) -> List[Claim]:
        """
        Identify claims with low agreement across models.
        
        Args:
            matrix: The claim verification matrix
            threshold: Fraction below which a claim is considered disputed
            
        Returns:
            List of disputed claims
        """
        return matrix.get_disputed_claims(threshold)
    
    def identify_consensus_claims(
        self,
        matrix: ClaimVerificationMatrix,
        threshold: float = 0.5,
    ) -> List[Claim]:
        """
        Identify claims with high agreement across models.
        
        Args:
            matrix: The claim verification matrix
            threshold: Fraction above which a claim is considered consensus
            
        Returns:
            List of consensus claims
        """
        return matrix.get_consensus_claims(threshold)


class FalseAcceptanceCalculator:
    """
    Calculator for False Acceptance Rate (FAR) in abstention testing.
    
    Measures how often a model fails to abstain when asked about non-existent
    entities. A lower FAR indicates better model behavior.
    
    FAR = failed_abstentions / total_nonexistent_queries
    
    Validates: Requirements 20.1, 20.2, 20.3, 20.4
    """
    
    # Default patterns that indicate abstention/refusal
    DEFAULT_ABSTENTION_PATTERNS = [
        # Direct refusals
        "i don't know",
        "i do not know",
        "i'm not sure",
        "i am not sure",
        "i cannot find",
        "i can't find",
        "no information",
        "no data",
        "not found",
        "doesn't exist",
        "does not exist",
        "no such",
        "unknown",
        "unfamiliar",
        "not aware",
        "cannot provide",
        "can't provide",
        "unable to find",
        "unable to provide",
        "no record",
        "no evidence",
        "not in my knowledge",
        "outside my knowledge",
        "beyond my knowledge",
        "i have no information",
        "there is no",
        "there are no",
        "i couldn't find",
        "i could not find",
        # Hedging/uncertainty
        "i'm not certain",
        "i am not certain",
        "i'm unable to",
        "i am unable to",
        "i cannot confirm",
        "i can't confirm",
        "no reliable information",
        "fictional",
        "made up",
        "invented",
        "fabricated",
        "not a real",
        "not real",
        "doesn't appear to exist",
        "does not appear to exist",
    ]
    
    def __init__(
        self,
        abstention_patterns: Optional[List[str]] = None,
        case_sensitive: bool = False,
    ):
        """
        Initialize the False Acceptance Calculator.
        
        Args:
            abstention_patterns: Custom patterns that indicate abstention.
                               If not provided, uses DEFAULT_ABSTENTION_PATTERNS.
            case_sensitive: Whether pattern matching should be case-sensitive.
                          Default is False (case-insensitive).
        """
        self.abstention_patterns = abstention_patterns or self.DEFAULT_ABSTENTION_PATTERNS
        self.case_sensitive = case_sensitive
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for pattern matching."""
        if not self.case_sensitive:
            return text.lower()
        return text
    
    def _detect_abstention(self, response: str) -> bool:
        """
        Detect if a response indicates abstention/refusal.
        
        Args:
            response: The model's response text
            
        Returns:
            True if the response indicates abstention
        """
        normalized_response = self._normalize_text(response)
        
        for pattern in self.abstention_patterns:
            normalized_pattern = self._normalize_text(pattern)
            if normalized_pattern in normalized_response:
                return True
        
        return False
    
    def evaluate_abstention(
        self,
        query: str,
        response: str,
        is_nonexistent: bool,
    ) -> AbstentionResult:
        """
        Evaluate a single query-response pair for abstention behavior.
        
        Args:
            query: The query about a potentially non-existent entity
            response: The model's response to the query
            is_nonexistent: True if the query is about a non-existent entity
            
        Returns:
            AbstentionResult with classification of the response
            
        Validates: Requirements 20.1, 20.3, 20.4
        """
        # Detect if the model abstained
        did_abstain = self._detect_abstention(response)
        
        # Determine if this is a false acceptance
        # False acceptance = non-existent entity query + model did NOT abstain
        is_false_acceptance = is_nonexistent and not did_abstain
        
        return AbstentionResult(
            query=query,
            response=response,
            is_nonexistent_entity=is_nonexistent,
            did_abstain=did_abstain,
            is_false_acceptance=is_false_acceptance,
        )
    
    def compute_far(
        self,
        results: List[AbstentionResult],
    ) -> FalseAcceptanceRateResult:
        """
        Compute False Acceptance Rate from abstention results.
        
        FAR = failed_abstentions / total_nonexistent_queries
        
        Args:
            results: List of AbstentionResult objects
            
        Returns:
            FalseAcceptanceRateResult with the computed FAR
            
        Validates: Requirements 20.2
        """
        # Filter to only non-existent entity queries
        nonexistent_results = [r for r in results if r.is_nonexistent_entity]
        total_nonexistent = len(nonexistent_results)
        
        if total_nonexistent == 0:
            return FalseAcceptanceRateResult(
                value=0.0,
                failed_abstentions=0,
                correct_refusals=0,
                total_nonexistent_queries=0,
                abstention_results=results,
            )
        
        # Count failed abstentions (false acceptances)
        failed_abstentions = sum(1 for r in nonexistent_results if r.is_false_acceptance)
        
        # Count correct refusals
        correct_refusals = sum(1 for r in nonexistent_results if r.did_abstain)
        
        # Compute FAR
        far_value = failed_abstentions / total_nonexistent
        
        # Ensure value is in range [0.0, 1.0]
        far_value = max(0.0, min(1.0, far_value))
        
        return FalseAcceptanceRateResult(
            value=far_value,
            failed_abstentions=failed_abstentions,
            correct_refusals=correct_refusals,
            total_nonexistent_queries=total_nonexistent,
            abstention_results=results,
        )
    
    def evaluate_and_compute_far(
        self,
        queries: List[str],
        responses: List[str],
        is_nonexistent_flags: List[bool],
    ) -> FalseAcceptanceRateResult:
        """
        Convenience method to evaluate multiple query-response pairs and compute FAR.
        
        Args:
            queries: List of queries
            responses: List of model responses (same length as queries)
            is_nonexistent_flags: List of flags indicating if each query is about
                                 a non-existent entity (same length as queries)
            
        Returns:
            FalseAcceptanceRateResult with the computed FAR
            
        Raises:
            ValueError: If input lists have different lengths
        """
        if not (len(queries) == len(responses) == len(is_nonexistent_flags)):
            raise ValueError(
                f"Input lists must have the same length. "
                f"Got queries={len(queries)}, responses={len(responses)}, "
                f"is_nonexistent_flags={len(is_nonexistent_flags)}"
            )
        
        results = []
        for query, response, is_nonexistent in zip(queries, responses, is_nonexistent_flags):
            result = self.evaluate_abstention(query, response, is_nonexistent)
            results.append(result)
        
        return self.compute_far(results)
    
    def add_abstention_pattern(self, pattern: str) -> None:
        """
        Add a custom abstention pattern.
        
        Args:
            pattern: The pattern to add
        """
        if pattern not in self.abstention_patterns:
            self.abstention_patterns.append(pattern)
    
    def remove_abstention_pattern(self, pattern: str) -> bool:
        """
        Remove an abstention pattern.
        
        Args:
            pattern: The pattern to remove
            
        Returns:
            True if the pattern was removed, False if it wasn't found
        """
        try:
            self.abstention_patterns.remove(pattern)
            return True
        except ValueError:
            return False
    
    def get_abstention_patterns(self) -> List[str]:
        """
        Get the current list of abstention patterns.
        
        Returns:
            List of abstention patterns
        """
        return self.abstention_patterns.copy()
