"""
Metrics calculation module for evaluation results.

This module provides comprehensive metric calculations including:
- Hallucination score with breakdown by issue type
- Confidence metrics with bootstrap confidence intervals
- Inter-judge agreement (Cohen's Kappa, Fleiss' Kappa)
- Statistical metrics (variance, standard deviation, distributions)
"""
import logging
from typing import List, Dict, Any, Tuple, Optional
from collections import Counter
import numpy as np
from scipy import stats

from ..models import JudgeResult, VerifierVerdict, FlaggedIssue
from ..schemas import IssueType, IssueSeverity, VerifierLabel

logger = logging.getLogger(__name__)


class HallucinationMetrics:
    """Container for hallucination metrics."""
    
    def __init__(
        self,
        overall_score: float,
        breakdown_by_type: Dict[str, float],
        affected_text_spans: List[Tuple[int, int, str]],
        severity_distribution: Dict[str, int]
    ):
        self.overall_score = overall_score
        self.breakdown_by_type = breakdown_by_type
        self.affected_text_spans = affected_text_spans
        self.severity_distribution = severity_distribution
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'overall_score': self.overall_score,
            'breakdown_by_type': self.breakdown_by_type,
            'affected_text_spans': self.affected_text_spans,
            'severity_distribution': self.severity_distribution
        }


class ConfidenceMetrics:
    """Container for confidence metrics."""
    
    def __init__(
        self,
        mean_confidence: float,
        confidence_interval: Tuple[float, float],
        confidence_level: float,
        is_low_confidence: bool
    ):
        self.mean_confidence = mean_confidence
        self.confidence_interval = confidence_interval
        self.confidence_level = confidence_level
        self.is_low_confidence = is_low_confidence
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'mean_confidence': self.mean_confidence,
            'confidence_interval': list(self.confidence_interval),
            'confidence_level': self.confidence_level,
            'is_low_confidence': self.is_low_confidence
        }


class InterJudgeAgreement:
    """Container for inter-judge agreement metrics."""
    
    def __init__(
        self,
        cohens_kappa: Optional[float],
        fleiss_kappa: Optional[float],
        krippendorff_alpha: Optional[float],
        pairwise_correlations: Dict[str, Dict[str, float]],
        interpretation: str
    ):
        self.cohens_kappa = cohens_kappa
        self.fleiss_kappa = fleiss_kappa
        self.krippendorff_alpha = krippendorff_alpha
        self.pairwise_correlations = pairwise_correlations
        self.interpretation = interpretation
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'cohens_kappa': self.cohens_kappa,
            'fleiss_kappa': self.fleiss_kappa,
            'krippendorff_alpha': self.krippendorff_alpha,
            'pairwise_correlations': self.pairwise_correlations,
            'interpretation': self.interpretation
        }


class MetricsCalculator:
    """Calculator for evaluation metrics."""
    
    # Issue severity weights for hallucination calculation
    SEVERITY_WEIGHTS = {
        IssueSeverity.LOW: 0.25,
        IssueSeverity.MEDIUM: 0.5,
        IssueSeverity.HIGH: 0.75,
        IssueSeverity.CRITICAL: 1.0
    }
    
    # Hallucination score component weights
    HALLUCINATION_WEIGHTS = {
        'inverse_consensus': 0.4,
        'verifier_refutation': 0.3,
        'issue_severity': 0.2,
        'confidence_penalty': 0.1
    }
    
    @staticmethod
    def calculate_hallucination_score(
        judge_results: List[JudgeResult],
        verifier_verdicts: List[VerifierVerdict],
        consensus_score: float
    ) -> HallucinationMetrics:
        """
        Calculate comprehensive hallucination score.
        
        The hallucination score is a composite metric (0-100) calculated using:
        1. Inverse of consensus score (40% weight)
        2. Verifier refutation rate (30% weight)
        3. Judge-flagged issues severity (20% weight)
        4. Confidence penalty (10% weight)
        
        Args:
            judge_results: List of judge results
            verifier_verdicts: List of verifier verdicts
            consensus_score: Consensus score from judges
            
        Returns:
            HallucinationMetrics object with overall score and breakdown
        """
        # Component 1: Inverse consensus (lower score = more hallucination)
        inverse_consensus = (100 - consensus_score) * MetricsCalculator.HALLUCINATION_WEIGHTS['inverse_consensus']
        
        # Component 2: Verifier refutation rate
        verifier_component = 0.0
        if verifier_verdicts:
            refuted_count = sum(1 for v in verifier_verdicts if v.label == VerifierLabel.REFUTED)
            refutation_rate = (refuted_count / len(verifier_verdicts)) * 100
            verifier_component = refutation_rate * MetricsCalculator.HALLUCINATION_WEIGHTS['verifier_refutation']
        
        # Component 3: Weighted issue severity
        all_issues = [issue for jr in judge_results for issue in jr.flagged_issues]
        severity_score = 0.0
        
        if all_issues:
            weighted_severity = sum(
                MetricsCalculator.SEVERITY_WEIGHTS[issue.severity]
                for issue in all_issues
            )
            max_possible = len(all_issues)
            severity_score = (weighted_severity / max_possible) * 100 * MetricsCalculator.HALLUCINATION_WEIGHTS['issue_severity']
        
        # Component 4: Confidence penalty (low confidence suggests uncertainty/hallucination)
        if judge_results:
            mean_confidence = sum(jr.confidence for jr in judge_results) / len(judge_results)
            confidence_penalty = (1 - mean_confidence) * 100 * MetricsCalculator.HALLUCINATION_WEIGHTS['confidence_penalty']
        else:
            confidence_penalty = 0.0
        
        # Final score
        hallucination_score = (
            inverse_consensus +
            verifier_component +
            severity_score +
            confidence_penalty
        )
        
        # Calculate breakdown by type
        breakdown = MetricsCalculator._calculate_hallucination_breakdown(all_issues)
        
        # Extract affected text spans
        affected_spans = [
            (issue.text_span_start, issue.text_span_end, issue.issue_type)
            for issue in all_issues
            if issue.text_span_start is not None and issue.text_span_end is not None
        ]
        
        # Severity distribution
        severity_dist = {
            severity.value: sum(1 for i in all_issues if i.severity == severity)
            for severity in IssueSeverity
        }
        
        return HallucinationMetrics(
            overall_score=min(100.0, max(0.0, hallucination_score)),
            breakdown_by_type=breakdown,
            affected_text_spans=affected_spans,
            severity_distribution=severity_dist
        )
    
    @staticmethod
    def _calculate_hallucination_breakdown(issues: List[FlaggedIssue]) -> Dict[str, float]:
        """
        Calculate hallucination score breakdown by issue type.
        
        Args:
            issues: List of flagged issues
            
        Returns:
            Dictionary mapping issue type to score contribution
        """
        breakdown = {}
        
        for issue_type in IssueType:
            type_issues = [i for i in issues if i.issue_type == issue_type]
            
            if type_issues:
                type_score = sum(
                    MetricsCalculator.SEVERITY_WEIGHTS[i.severity]
                    for i in type_issues
                )
                breakdown[issue_type.value] = (type_score / len(type_issues)) * 100
            else:
                breakdown[issue_type.value] = 0.0
        
        return breakdown
    
    @staticmethod
    def calculate_confidence_metrics(
        judge_results: List[JudgeResult],
        confidence_level: float = 0.95
    ) -> ConfidenceMetrics:
        """
        Calculate confidence interval for consensus score using bootstrap resampling.
        
        Args:
            judge_results: List of judge results
            confidence_level: Confidence level for interval (default 0.95)
            
        Returns:
            ConfidenceMetrics object with mean confidence and intervals
        """
        if not judge_results:
            return ConfidenceMetrics(
                mean_confidence=0.0,
                confidence_interval=(0.0, 0.0),
                confidence_level=confidence_level,
                is_low_confidence=True
            )
        
        scores = np.array([jr.score for jr in judge_results])
        confidences = np.array([jr.confidence for jr in judge_results])
        
        # Mean confidence
        mean_conf = float(np.mean(confidences))
        
        # Bootstrap confidence interval for consensus score
        n_bootstrap = 10000
        bootstrap_means = []
        
        rng = np.random.default_rng(42)  # Fixed seed for reproducibility
        
        for _ in range(n_bootstrap):
            sample_indices = rng.choice(len(scores), size=len(scores), replace=True)
            sample = scores[sample_indices]
            bootstrap_means.append(np.mean(sample))
        
        # Calculate percentile-based CI
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        ci_lower = float(np.percentile(bootstrap_means, lower_percentile))
        ci_upper = float(np.percentile(bootstrap_means, upper_percentile))
        
        # Determine if confidence is low
        ci_width = ci_upper - ci_lower
        is_low_confidence = ci_width > 20 or mean_conf < 0.7
        
        return ConfidenceMetrics(
            mean_confidence=mean_conf,
            confidence_interval=(ci_lower, ci_upper),
            confidence_level=confidence_level,
            is_low_confidence=is_low_confidence
        )
    
    @staticmethod
    def calculate_inter_judge_agreement(
        judge_results: List[JudgeResult]
    ) -> InterJudgeAgreement:
        """
        Calculate inter-judge agreement using multiple metrics.
        
        For 2 judges: Cohen's Kappa
        For 3+ judges: Fleiss' Kappa
        
        Args:
            judge_results: List of judge results
            
        Returns:
            InterJudgeAgreement object with kappa values and interpretation
        """
        n_judges = len(judge_results)
        
        if n_judges < 2:
            return InterJudgeAgreement(
                cohens_kappa=None,
                fleiss_kappa=None,
                krippendorff_alpha=None,
                pairwise_correlations={},
                interpretation="insufficient_judges"
            )
        
        # Convert scores to categorical ratings for kappa calculation
        # Bins: 0-20 (poor), 20-40 (fair), 40-60 (moderate), 60-80 (good), 80-100 (excellent)
        categories = [MetricsCalculator._score_to_category(jr.score) for jr in judge_results]
        
        # Cohen's Kappa (for 2 judges)
        cohens_kappa = None
        if n_judges == 2:
            cohens_kappa = MetricsCalculator._calculate_cohens_kappa(categories)
        
        # Fleiss' Kappa (for 3+ judges)
        fleiss_kappa = None
        if n_judges >= 3:
            fleiss_kappa = MetricsCalculator._calculate_fleiss_kappa(categories)
        
        # Pairwise correlations
        pairwise_corr = MetricsCalculator._calculate_pairwise_correlations(judge_results)
        
        # Interpretation
        kappa_value = fleiss_kappa if fleiss_kappa is not None else cohens_kappa
        interpretation = MetricsCalculator._interpret_kappa(kappa_value)
        
        return InterJudgeAgreement(
            cohens_kappa=cohens_kappa,
            fleiss_kappa=fleiss_kappa,
            krippendorff_alpha=None,  # Can be added if needed
            pairwise_correlations=pairwise_corr,
            interpretation=interpretation
        )
    
    @staticmethod
    def _score_to_category(score: float) -> int:
        """
        Convert score to categorical rating.
        
        Args:
            score: Score value (0-100)
            
        Returns:
            Category index (0-4)
        """
        if score < 20:
            return 0  # poor
        elif score < 40:
            return 1  # fair
        elif score < 60:
            return 2  # moderate
        elif score < 80:
            return 3  # good
        else:
            return 4  # excellent
    
    @staticmethod
    def _calculate_cohens_kappa(categories: List[int]) -> float:
        """
        Calculate Cohen's Kappa for 2 raters.
        
        Args:
            categories: List of 2 category ratings
            
        Returns:
            Cohen's Kappa value
        """
        if len(categories) != 2:
            return 0.0
        
        # For 2 raters with single rating each, simplified calculation
        if categories[0] == categories[1]:
            return 1.0  # Perfect agreement
        
        # Calculate observed agreement
        p_o = 0.0  # No agreement
        
        # Calculate expected agreement (random chance)
        # With 5 categories, random chance is 1/5 = 0.2
        p_e = 0.2
        
        # Kappa = (P_o - P_e) / (1 - P_e)
        kappa = (p_o - p_e) / (1 - p_e)
        
        return float(kappa)
    
    @staticmethod
    def _calculate_fleiss_kappa(categories: List[int]) -> float:
        """
        Calculate Fleiss' Kappa for multiple raters.
        
        Args:
            categories: List of category ratings
            
        Returns:
            Fleiss' Kappa value
        """
        n = len(categories)
        if n < 3:
            return 0.0
        
        # Count frequency of each category
        category_counts = Counter(categories)
        
        # Calculate proportion of agreement
        mode_category = max(category_counts, key=category_counts.get)
        p_o = category_counts[mode_category] / n
        
        # Expected agreement (random chance with 5 categories)
        p_e = 1 / 5
        
        # Kappa = (P_o - P_e) / (1 - P_e)
        if p_e >= 1.0:
            return 0.0
        
        kappa = (p_o - p_e) / (1 - p_e)
        
        return float(kappa)
    
    @staticmethod
    def _calculate_pairwise_correlations(
        judge_results: List[JudgeResult]
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate pairwise correlations between judges.
        
        Args:
            judge_results: List of judge results
            
        Returns:
            Dictionary of pairwise correlations
        """
        pairwise_corr = {}
        
        for i, jr1 in enumerate(judge_results):
            pairwise_corr[jr1.judge_name] = {}
            
            for j, jr2 in enumerate(judge_results):
                if i != j:
                    # Simplified correlation based on score difference
                    # In production, would use actual correlation if multiple evaluations
                    corr = 1.0 - abs(jr1.score - jr2.score) / 100
                    pairwise_corr[jr1.judge_name][jr2.judge_name] = float(corr)
        
        return pairwise_corr
    
    @staticmethod
    def _interpret_kappa(kappa: Optional[float]) -> str:
        """
        Interpret kappa value according to Landis & Koch scale.
        
        Args:
            kappa: Kappa value
            
        Returns:
            Interpretation string
        """
        if kappa is None:
            return "insufficient_data"
        elif kappa < 0:
            return "poor"
        elif kappa < 0.20:
            return "slight"
        elif kappa < 0.40:
            return "fair"
        elif kappa < 0.60:
            return "moderate"
        elif kappa < 0.80:
            return "substantial"
        else:
            return "almost_perfect"
    
    @staticmethod
    def calculate_statistical_metrics(
        judge_results: List[JudgeResult]
    ) -> Dict[str, Any]:
        """
        Calculate statistical metrics for judge scores.
        
        Args:
            judge_results: List of judge results
            
        Returns:
            Dictionary of statistical metrics
        """
        if not judge_results:
            return {
                'variance': 0.0,
                'standard_deviation': 0.0,
                'mean': 0.0,
                'median': 0.0,
                'min': 0.0,
                'max': 0.0,
                'quartiles': [0.0, 0.0, 0.0],
                'score_distribution': {}
            }
        
        scores = np.array([jr.score for jr in judge_results])
        
        # Basic statistics
        variance = float(np.var(scores))
        std_dev = float(np.std(scores))
        mean = float(np.mean(scores))
        median = float(np.median(scores))
        min_score = float(np.min(scores))
        max_score = float(np.max(scores))
        
        # Quartiles
        q1 = float(np.percentile(scores, 25))
        q2 = float(np.percentile(scores, 50))  # Same as median
        q3 = float(np.percentile(scores, 75))
        
        # Score distribution (histogram bins)
        bins = [0, 20, 40, 60, 80, 100]
        hist, _ = np.histogram(scores, bins=bins)
        
        distribution = {
            '0-20': int(hist[0]),
            '20-40': int(hist[1]),
            '40-60': int(hist[2]),
            '60-80': int(hist[3]),
            '80-100': int(hist[4])
        }
        
        return {
            'variance': variance,
            'standard_deviation': std_dev,
            'mean': mean,
            'median': median,
            'min': min_score,
            'max': max_score,
            'quartiles': [q1, q2, q3],
            'score_distribution': distribution
        }
    
    @staticmethod
    def calculate_aggregate_statistics(
        sessions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Calculate aggregate statistics across multiple sessions.
        
        Args:
            sessions: List of session data dictionaries
            
        Returns:
            Dictionary of aggregate statistics
        """
        if not sessions:
            return {
                'total_sessions': 0,
                'average_consensus_score': 0.0,
                'average_hallucination_score': 0.0,
                'average_confidence': 0.0,
                'score_trend': [],
                'judge_performance': {}
            }
        
        consensus_scores = [s.get('consensus_score', 0) for s in sessions if s.get('consensus_score') is not None]
        hallucination_scores = [s.get('hallucination_score', 0) for s in sessions if s.get('hallucination_score') is not None]
        
        # Calculate averages
        avg_consensus = float(np.mean(consensus_scores)) if consensus_scores else 0.0
        avg_hallucination = float(np.mean(hallucination_scores)) if hallucination_scores else 0.0
        
        # Calculate confidence intervals
        if len(consensus_scores) >= 2:
            avg_confidence_lower = float(np.mean([s.get('confidence_interval_lower', 0) for s in sessions if s.get('confidence_interval_lower') is not None]))
            avg_confidence_upper = float(np.mean([s.get('confidence_interval_upper', 0) for s in sessions if s.get('confidence_interval_upper') is not None]))
            avg_confidence = (avg_confidence_lower + avg_confidence_upper) / 2
        else:
            avg_confidence = 0.0
        
        # Score trend (last 10 sessions)
        recent_scores = consensus_scores[-10:] if len(consensus_scores) > 10 else consensus_scores
        
        return {
            'total_sessions': len(sessions),
            'average_consensus_score': avg_consensus,
            'average_hallucination_score': avg_hallucination,
            'average_confidence': avg_confidence,
            'score_trend': recent_scores,
            'judge_performance': {}  # Can be expanded with per-judge statistics
        }
