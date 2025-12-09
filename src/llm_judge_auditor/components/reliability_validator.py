"""
Reliability Validator for the LLM Judge Auditor.

This module provides reliability validation features including:
- Evaluation consistency checking (variance < 5 points)
- Cohen's kappa calculation for inter-model agreement
- Kendall's Tau and Spearman correlation for ranking validation

These metrics help ensure the evaluation system produces consistent,
reliable results across multiple evaluations and judge models.
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ConsistencyReport:
    """Report on evaluation consistency across multiple runs."""
    
    mean_score: float
    variance: float
    std_deviation: float
    is_consistent: bool  # True if variance < 5 points
    num_evaluations: int
    scores: List[float]


@dataclass
class AgreementReport:
    """Report on inter-model agreement using Cohen's kappa."""
    
    cohens_kappa: float
    agreement_level: str  # "poor", "fair", "moderate", "substantial", "almost_perfect"
    num_models: int
    pairwise_agreements: Dict[Tuple[str, str], float]


@dataclass
class RankingCorrelationReport:
    """Report on ranking correlation metrics."""
    
    kendalls_tau: float
    kendalls_tau_p_value: float
    spearmans_rho: float
    spearmans_rho_p_value: float
    num_pairs: int
    is_significant: bool  # True if p-value < 0.05


class ReliabilityValidator:
    """
    Validator for checking reliability and consistency of evaluations.
    
    This class provides methods to validate that the evaluation system
    produces consistent, reliable results. It implements:
    
    1. Consistency checking: Ensures repeated evaluations have low variance
    2. Inter-model agreement: Measures agreement between judge models using Cohen's kappa
    3. Ranking correlation: Validates pairwise rankings using Kendall's Tau and Spearman's rho
    
    Example:
        >>> validator = ReliabilityValidator()
        >>> # Check consistency across multiple evaluations
        >>> scores = [85.0, 87.0, 84.5, 86.0, 85.5]
        >>> consistency = validator.check_consistency(scores)
        >>> print(f"Consistent: {consistency.is_consistent}")
        >>> 
        >>> # Check inter-model agreement
        >>> judge_scores = {
        ...     "judge_1": [1, 0, 1, 1, 0],
        ...     "judge_2": [1, 0, 1, 0, 0],
        ...     "judge_3": [1, 1, 1, 1, 0],
        ... }
        >>> agreement = validator.calculate_inter_model_agreement(judge_scores)
        >>> print(f"Cohen's kappa: {agreement.cohens_kappa:.3f}")
    """
    
    def __init__(self, consistency_threshold: float = 5.0):
        """
        Initialize the ReliabilityValidator.
        
        Args:
            consistency_threshold: Maximum variance for consistent evaluations (default: 5.0)
        """
        self.consistency_threshold = consistency_threshold
        logger.info(f"ReliabilityValidator initialized with consistency_threshold={consistency_threshold}")
    
    def check_consistency(self, scores: List[float]) -> ConsistencyReport:
        """
        Check evaluation consistency across multiple runs.
        
        Evaluates whether repeated evaluations of the same input produce
        consistent scores. Consistency is defined as variance < threshold
        (default 5 points).
        
        Args:
            scores: List of consensus scores from multiple evaluations
        
        Returns:
            ConsistencyReport with variance, mean, and consistency status
        
        Raises:
            ValueError: If scores list is empty or has fewer than 2 elements
        
        Example:
            >>> validator = ReliabilityValidator()
            >>> scores = [85.0, 87.0, 84.5, 86.0, 85.5]
            >>> report = validator.check_consistency(scores)
            >>> print(f"Variance: {report.variance:.2f}")
            >>> print(f"Is consistent: {report.is_consistent}")
        """
        if not scores:
            raise ValueError("scores list cannot be empty")
        
        if len(scores) < 2:
            raise ValueError("scores list must contain at least 2 elements")
        
        logger.info(f"Checking consistency for {len(scores)} evaluations")
        
        # Calculate statistics
        scores_array = np.array(scores)
        mean_score = float(np.mean(scores_array))
        variance = float(np.var(scores_array, ddof=1))  # Sample variance
        std_deviation = float(np.std(scores_array, ddof=1))  # Sample std
        
        # Check if consistent (variance < threshold)
        is_consistent = variance < self.consistency_threshold
        
        logger.info(f"  Mean: {mean_score:.2f}, Variance: {variance:.2f}, Std: {std_deviation:.2f}")
        logger.info(f"  Consistent: {is_consistent} (threshold: {self.consistency_threshold})")
        
        return ConsistencyReport(
            mean_score=mean_score,
            variance=variance,
            std_deviation=std_deviation,
            is_consistent=is_consistent,
            num_evaluations=len(scores),
            scores=scores,
        )
    
    def calculate_inter_model_agreement(
        self,
        judge_scores: Dict[str, List[float]],
        threshold: float = 50.0,
    ) -> AgreementReport:
        """
        Calculate Cohen's kappa for inter-model agreement.
        
        Measures agreement between judge models by converting continuous scores
        to binary classifications (pass/fail based on threshold) and computing
        Cohen's kappa for each pair of judges.
        
        Cohen's kappa interpretation:
        - < 0.00: Poor agreement
        - 0.00-0.20: Slight agreement
        - 0.21-0.40: Fair agreement
        - 0.41-0.60: Moderate agreement
        - 0.61-0.80: Substantial agreement
        - 0.81-1.00: Almost perfect agreement
        
        Args:
            judge_scores: Dictionary mapping judge names to lists of scores
            threshold: Score threshold for binary classification (default: 50.0)
        
        Returns:
            AgreementReport with Cohen's kappa and agreement level
        
        Raises:
            ValueError: If judge_scores is empty or judges have different numbers of scores
        
        Example:
            >>> validator = ReliabilityValidator()
            >>> judge_scores = {
            ...     "llama-3": [85.0, 45.0, 90.0, 30.0],
            ...     "mistral": [80.0, 40.0, 88.0, 35.0],
            ...     "phi-3": [82.0, 48.0, 92.0, 32.0],
            ... }
            >>> report = validator.calculate_inter_model_agreement(judge_scores)
            >>> print(f"Cohen's kappa: {report.cohens_kappa:.3f}")
            >>> print(f"Agreement level: {report.agreement_level}")
        """
        if not judge_scores:
            raise ValueError("judge_scores cannot be empty")
        
        # Validate all judges have same number of scores
        score_lengths = [len(scores) for scores in judge_scores.values()]
        if len(set(score_lengths)) > 1:
            raise ValueError("All judges must have the same number of scores")
        
        if score_lengths[0] < 2:
            raise ValueError("Each judge must have at least 2 scores")
        
        judge_names = list(judge_scores.keys())
        num_models = len(judge_names)
        
        logger.info(f"Calculating inter-model agreement for {num_models} judges")
        logger.info(f"  Using threshold: {threshold}")
        
        # Convert continuous scores to binary classifications
        binary_classifications = {}
        for judge_name, scores in judge_scores.items():
            binary_classifications[judge_name] = [1 if score >= threshold else 0 for score in scores]
        
        # Calculate pairwise Cohen's kappa
        pairwise_kappas = {}
        kappa_values = []
        
        for i in range(num_models):
            for j in range(i + 1, num_models):
                judge_i = judge_names[i]
                judge_j = judge_names[j]
                
                ratings_i = binary_classifications[judge_i]
                ratings_j = binary_classifications[judge_j]
                
                kappa = self._calculate_cohens_kappa(ratings_i, ratings_j)
                pairwise_kappas[(judge_i, judge_j)] = kappa
                kappa_values.append(kappa)
                
                logger.info(f"  {judge_i} vs {judge_j}: κ = {kappa:.3f}")
        
        # Average Cohen's kappa across all pairs
        mean_kappa = float(np.mean(kappa_values)) if kappa_values else 0.0
        
        # Determine agreement level
        agreement_level = self._interpret_kappa(mean_kappa)
        
        logger.info(f"  Mean Cohen's kappa: {mean_kappa:.3f} ({agreement_level})")
        
        return AgreementReport(
            cohens_kappa=mean_kappa,
            agreement_level=agreement_level,
            num_models=num_models,
            pairwise_agreements=pairwise_kappas,
        )
    
    def _calculate_cohens_kappa(self, ratings_a: List[int], ratings_b: List[int]) -> float:
        """
        Calculate Cohen's kappa for two raters.
        
        Args:
            ratings_a: Binary ratings from rater A
            ratings_b: Binary ratings from rater B
        
        Returns:
            Cohen's kappa coefficient
        """
        n = len(ratings_a)
        
        # Build confusion matrix
        agree_positive = sum(1 for a, b in zip(ratings_a, ratings_b) if a == 1 and b == 1)
        agree_negative = sum(1 for a, b in zip(ratings_a, ratings_b) if a == 0 and b == 0)
        disagree_a1_b0 = sum(1 for a, b in zip(ratings_a, ratings_b) if a == 1 and b == 0)
        disagree_a0_b1 = sum(1 for a, b in zip(ratings_a, ratings_b) if a == 0 and b == 1)
        
        # Observed agreement
        p_o = (agree_positive + agree_negative) / n
        
        # Expected agreement by chance
        p_a_positive = (agree_positive + disagree_a1_b0) / n
        p_b_positive = (agree_positive + disagree_a0_b1) / n
        p_a_negative = (agree_negative + disagree_a0_b1) / n
        p_b_negative = (agree_negative + disagree_a1_b0) / n
        
        p_e = (p_a_positive * p_b_positive) + (p_a_negative * p_b_negative)
        
        # Cohen's kappa
        if p_e == 1.0:
            # Perfect expected agreement (degenerate case)
            return 1.0 if p_o == 1.0 else 0.0
        
        kappa = (p_o - p_e) / (1 - p_e)
        
        return kappa
    
    def _interpret_kappa(self, kappa: float) -> str:
        """
        Interpret Cohen's kappa value.
        
        Args:
            kappa: Cohen's kappa coefficient
        
        Returns:
            Agreement level description
        """
        if kappa < 0.0:
            return "poor"
        elif kappa < 0.21:
            return "slight"
        elif kappa < 0.41:
            return "fair"
        elif kappa < 0.61:
            return "moderate"
        elif kappa < 0.81:
            return "substantial"
        else:
            return "almost_perfect"
    
    def calculate_ranking_correlation(
        self,
        predicted_rankings: List[Tuple[str, str]],
        ground_truth_rankings: List[Tuple[str, str]],
    ) -> RankingCorrelationReport:
        """
        Calculate Kendall's Tau and Spearman's rho for ranking validation.
        
        Validates pairwise rankings by comparing predicted rankings against
        ground truth. Both Kendall's Tau and Spearman's rho measure the
        correlation between two rankings.
        
        Kendall's Tau measures the proportion of concordant pairs minus
        discordant pairs. Spearman's rho is the Pearson correlation of
        rank values.
        
        Args:
            predicted_rankings: List of (winner, loser) tuples from system
            ground_truth_rankings: List of (winner, loser) tuples from ground truth
        
        Returns:
            RankingCorrelationReport with correlation coefficients and p-values
        
        Raises:
            ValueError: If rankings lists are empty or have different lengths
        
        Example:
            >>> validator = ReliabilityValidator()
            >>> predicted = [("A", "B"), ("C", "D"), ("A", "C")]
            >>> ground_truth = [("A", "B"), ("C", "D"), ("C", "A")]
            >>> report = validator.calculate_ranking_correlation(predicted, ground_truth)
            >>> print(f"Kendall's Tau: {report.kendalls_tau:.3f}")
            >>> print(f"Spearman's rho: {report.spearmans_rho:.3f}")
        """
        if not predicted_rankings:
            raise ValueError("predicted_rankings cannot be empty")
        
        if not ground_truth_rankings:
            raise ValueError("ground_truth_rankings cannot be empty")
        
        if len(predicted_rankings) != len(ground_truth_rankings):
            raise ValueError("predicted_rankings and ground_truth_rankings must have the same length")
        
        num_pairs = len(predicted_rankings)
        logger.info(f"Calculating ranking correlation for {num_pairs} pairs")
        
        # Convert rankings to agreement scores
        # 1 if rankings agree (same winner), -1 if they disagree
        agreement_scores = []
        
        for pred, truth in zip(predicted_rankings, ground_truth_rankings):
            pred_winner, pred_loser = pred
            truth_winner, truth_loser = truth
            
            # Ensure we're comparing the same pair
            pred_pair = tuple(sorted([pred_winner, pred_loser]))
            truth_pair = tuple(sorted([truth_winner, truth_loser]))
            
            if pred_pair != truth_pair:
                raise ValueError(f"Mismatched pairs: {pred_pair} vs {truth_pair}")
            
            # Check if rankings agree
            if pred_winner == truth_winner:
                agreement_scores.append(1)
            else:
                agreement_scores.append(-1)
        
        # For correlation calculation, we need two sequences
        # We'll use the agreement scores as one sequence and a perfect agreement sequence
        predicted_sequence = agreement_scores
        ground_truth_sequence = [1] * len(agreement_scores)  # Perfect agreement baseline
        
        # Calculate Kendall's Tau
        kendalls_tau, kendalls_p = self._calculate_kendalls_tau(
            predicted_sequence, ground_truth_sequence
        )
        
        # Calculate Spearman's rho
        spearmans_rho, spearmans_p = self._calculate_spearmans_rho(
            predicted_sequence, ground_truth_sequence
        )
        
        # Check if significant (p < 0.05)
        is_significant = kendalls_p < 0.05 and spearmans_p < 0.05
        
        logger.info(f"  Kendall's Tau: {kendalls_tau:.3f} (p={kendalls_p:.4f})")
        logger.info(f"  Spearman's rho: {spearmans_rho:.3f} (p={spearmans_p:.4f})")
        logger.info(f"  Significant: {is_significant}")
        
        return RankingCorrelationReport(
            kendalls_tau=kendalls_tau,
            kendalls_tau_p_value=kendalls_p,
            spearmans_rho=spearmans_rho,
            spearmans_rho_p_value=spearmans_p,
            num_pairs=num_pairs,
            is_significant=is_significant,
        )
    
    def _calculate_kendalls_tau(
        self, x: List[float], y: List[float]
    ) -> Tuple[float, float]:
        """
        Calculate Kendall's Tau correlation coefficient.
        
        Args:
            x: First ranking
            y: Second ranking
        
        Returns:
            Tuple of (tau, p_value)
        """
        n = len(x)
        
        # Count concordant and discordant pairs
        concordant = 0
        discordant = 0
        
        for i in range(n):
            for j in range(i + 1, n):
                x_diff = x[i] - x[j]
                y_diff = y[i] - y[j]
                
                if x_diff * y_diff > 0:
                    concordant += 1
                elif x_diff * y_diff < 0:
                    discordant += 1
                # If x_diff * y_diff == 0, it's a tie (neither concordant nor discordant)
        
        # Calculate Kendall's Tau
        total_pairs = n * (n - 1) / 2
        tau = (concordant - discordant) / total_pairs if total_pairs > 0 else 0.0
        
        # Calculate approximate p-value using normal approximation
        # For large n, tau is approximately normally distributed
        if n > 10:
            var_tau = (2 * (2 * n + 5)) / (9 * n * (n - 1))
            z_score = tau / np.sqrt(var_tau)
            # Two-tailed p-value
            from scipy import stats
            p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
        else:
            # For small n, use a conservative p-value
            p_value = 0.5
        
        return float(tau), float(p_value)
    
    def _calculate_spearmans_rho(
        self, x: List[float], y: List[float]
    ) -> Tuple[float, float]:
        """
        Calculate Spearman's rho correlation coefficient.
        
        Args:
            x: First ranking
            y: Second ranking
        
        Returns:
            Tuple of (rho, p_value)
        """
        n = len(x)
        
        # Convert to ranks
        x_ranks = self._convert_to_ranks(x)
        y_ranks = self._convert_to_ranks(y)
        
        # Calculate Spearman's rho using the formula:
        # rho = 1 - (6 * sum(d^2)) / (n * (n^2 - 1))
        # where d is the difference between ranks
        
        d_squared_sum = sum((rx - ry) ** 2 for rx, ry in zip(x_ranks, y_ranks))
        rho = 1 - (6 * d_squared_sum) / (n * (n ** 2 - 1)) if n > 1 else 0.0
        
        # Calculate p-value using t-distribution approximation
        if n > 2:
            t_stat = rho * np.sqrt((n - 2) / (1 - rho ** 2 + 1e-10))
            from scipy import stats
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
        else:
            p_value = 0.5
        
        return float(rho), float(p_value)
    
    def _convert_to_ranks(self, values: List[float]) -> List[float]:
        """
        Convert values to ranks (1-indexed).
        
        Args:
            values: List of values to rank
        
        Returns:
            List of ranks
        """
        # Create list of (value, original_index) tuples
        indexed_values = [(v, i) for i, v in enumerate(values)]
        
        # Sort by value
        sorted_values = sorted(indexed_values, key=lambda x: x[0])
        
        # Assign ranks (handle ties by averaging)
        ranks = [0.0] * len(values)
        i = 0
        while i < len(sorted_values):
            # Find all values equal to current value (ties)
            j = i
            while j < len(sorted_values) and sorted_values[j][0] == sorted_values[i][0]:
                j += 1
            
            # Average rank for ties
            avg_rank = (i + j + 1) / 2  # +1 for 1-indexing
            
            # Assign average rank to all tied values
            for k in range(i, j):
                original_index = sorted_values[k][1]
                ranks[original_index] = avg_rank
            
            i = j
        
        return ranks
