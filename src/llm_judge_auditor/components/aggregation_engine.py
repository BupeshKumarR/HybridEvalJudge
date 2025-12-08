"""
Aggregation Engine for combining results from multiple evaluation components.

This module provides the AggregationEngine class that combines results from
specialized verifiers and judge model ensembles using configurable strategies
(mean, median, weighted average, majority vote).
"""

import logging
import statistics
from enum import Enum
from typing import Dict, List, Optional

from llm_judge_auditor.models import (
    AggregationMetadata,
    JudgeResult,
    Verdict,
    VerdictLabel,
)

logger = logging.getLogger(__name__)


class AggregationStrategy(str, Enum):
    """Strategies for aggregating judge scores."""

    MEAN = "mean"
    MEDIAN = "median"
    WEIGHTED_AVERAGE = "weighted_average"
    MAJORITY_VOTE = "majority_vote"


class AggregationEngine:
    """
    Combines results from specialized verifiers and judge ensembles.

    This class implements multiple aggregation strategies for combining
    scores from different evaluation components, with disagreement detection
    and confidence assessment.
    """

    def __init__(
        self,
        strategy: AggregationStrategy = AggregationStrategy.MEAN,
        disagreement_threshold: float = 20.0,
        weights: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize the AggregationEngine.

        Args:
            strategy: Aggregation strategy to use
            disagreement_threshold: Variance threshold for flagging low confidence
            weights: Optional weights for weighted average (model_name -> weight)
        """
        self.strategy = strategy
        self.disagreement_threshold = disagreement_threshold
        self.weights = weights or {}

        logger.info(
            f"AggregationEngine initialized with strategy={strategy.value}, "
            f"disagreement_threshold={disagreement_threshold}"
        )

    def set_strategy(self, strategy: AggregationStrategy) -> None:
        """
        Set the aggregation strategy.

        Args:
            strategy: New aggregation strategy to use
        """
        self.strategy = strategy
        logger.info(f"Aggregation strategy changed to: {strategy.value}")

    def set_weights(self, weights: Dict[str, float]) -> None:
        """
        Set weights for weighted average aggregation.

        Args:
            weights: Dictionary mapping model names to weights
        """
        self.weights = weights
        logger.info(f"Aggregation weights updated: {weights}")

    def _calculate_variance(self, scores: List[float]) -> float:
        """
        Calculate variance of scores.

        Args:
            scores: List of scores

        Returns:
            Variance of the scores
        """
        if len(scores) < 2:
            return 0.0

        return statistics.variance(scores)

    def _aggregate_mean(self, scores: List[float]) -> float:
        """
        Aggregate scores using mean.

        Args:
            scores: List of scores

        Returns:
            Mean of the scores
        """
        if not scores:
            return 0.0

        return statistics.mean(scores)

    def _aggregate_median(self, scores: List[float]) -> float:
        """
        Aggregate scores using median.

        Args:
            scores: List of scores

        Returns:
            Median of the scores
        """
        if not scores:
            return 0.0

        return statistics.median(scores)

    def _aggregate_weighted_average(
        self, judge_results: List[JudgeResult]
    ) -> float:
        """
        Aggregate scores using weighted average.

        Args:
            judge_results: List of JudgeResult objects

        Returns:
            Weighted average of the scores
        """
        if not judge_results:
            return 0.0

        # If no weights specified, fall back to mean
        if not self.weights:
            logger.warning(
                "No weights specified for weighted average, falling back to mean"
            )
            return self._aggregate_mean([jr.score for jr in judge_results])

        # Calculate weighted average
        total_weight = 0.0
        weighted_sum = 0.0

        for judge_result in judge_results:
            model_name = judge_result.model_name
            weight = self.weights.get(model_name, 1.0)  # Default weight of 1.0
            weighted_sum += judge_result.score * weight
            total_weight += weight

        if total_weight == 0:
            logger.warning("Total weight is zero, returning 0.0")
            return 0.0

        return weighted_sum / total_weight

    def _aggregate_majority_vote(self, scores: List[float]) -> float:
        """
        Aggregate scores using majority vote (most common score range).

        Scores are binned into ranges: [0-33], [34-66], [67-100]
        The median of the most common bin is returned.

        Args:
            scores: List of scores

        Returns:
            Representative score from the majority bin
        """
        if not scores:
            return 0.0

        # Bin scores into three categories
        low = [s for s in scores if s <= 33]
        medium = [s for s in scores if 34 <= s <= 66]
        high = [s for s in scores if s >= 67]

        # Find the bin with most votes
        bins = [
            (low, 16.5),  # Midpoint of [0-33]
            (medium, 50.0),  # Midpoint of [34-66]
            (high, 83.5),  # Midpoint of [67-100]
        ]

        majority_bin, default_value = max(bins, key=lambda x: len(x[0]))

        # Return median of majority bin, or default if empty
        if majority_bin:
            return statistics.median(majority_bin)
        else:
            return default_value

    def aggregate_scores(
        self,
        judge_results: List[JudgeResult],
        verifier_verdicts: Optional[List[Verdict]] = None,
    ) -> tuple[float, AggregationMetadata]:
        """
        Aggregate scores from judge results and optionally verifier verdicts.

        This method combines scores from multiple judges using the configured
        strategy and detects disagreement among judges.

        Args:
            judge_results: List of JudgeResult objects from the ensemble
            verifier_verdicts: Optional list of Verdict objects from specialized verifier

        Returns:
            Tuple of (consensus_score, aggregation_metadata)

        Raises:
            ValueError: If judge_results is empty

        Example:
            >>> engine = AggregationEngine(strategy=AggregationStrategy.MEAN)
            >>> judge_results = [
            ...     JudgeResult("judge1", 80.0, "Good", [], 0.9),
            ...     JudgeResult("judge2", 85.0, "Very good", [], 0.95),
            ... ]
            >>> score, metadata = engine.aggregate_scores(judge_results)
            >>> print(f"Consensus: {score}")
        """
        if not judge_results:
            raise ValueError("Cannot aggregate empty judge results")

        # Extract scores and build individual scores dict
        scores = [jr.score for jr in judge_results]
        individual_scores = {jr.model_name: jr.score for jr in judge_results}

        # Calculate variance
        variance = self._calculate_variance(scores)

        # Detect disagreement
        is_low_confidence = variance > self.disagreement_threshold

        if is_low_confidence:
            logger.warning(
                f"High disagreement detected: variance={variance:.2f} > "
                f"threshold={self.disagreement_threshold}"
            )

        # Apply aggregation strategy
        if self.strategy == AggregationStrategy.MEAN:
            consensus_score = self._aggregate_mean(scores)
        elif self.strategy == AggregationStrategy.MEDIAN:
            consensus_score = self._aggregate_median(scores)
        elif self.strategy == AggregationStrategy.WEIGHTED_AVERAGE:
            consensus_score = self._aggregate_weighted_average(judge_results)
        elif self.strategy == AggregationStrategy.MAJORITY_VOTE:
            consensus_score = self._aggregate_majority_vote(scores)
        else:
            logger.error(f"Unknown strategy: {self.strategy}, falling back to mean")
            consensus_score = self._aggregate_mean(scores)

        # Adjust consensus score based on verifier verdicts if provided
        if verifier_verdicts:
            consensus_score = self._incorporate_verifier_verdicts(
                consensus_score, verifier_verdicts
            )

        # Create metadata
        metadata = AggregationMetadata(
            strategy=self.strategy.value,
            individual_scores=individual_scores,
            variance=variance,
            is_low_confidence=is_low_confidence,
            weights=self.weights if self.strategy == AggregationStrategy.WEIGHTED_AVERAGE else None,
        )

        logger.info(
            f"Aggregation complete: consensus_score={consensus_score:.2f}, "
            f"variance={variance:.2f}, low_confidence={is_low_confidence}"
        )

        return consensus_score, metadata

    def _incorporate_verifier_verdicts(
        self, consensus_score: float, verifier_verdicts: List[Verdict]
    ) -> float:
        """
        Adjust consensus score based on verifier verdicts.

        If verifier finds refuted claims, lower the score.
        If verifier finds all claims supported, potentially boost the score.

        Args:
            consensus_score: Current consensus score from judges
            verifier_verdicts: List of verdicts from specialized verifier

        Returns:
            Adjusted consensus score
        """
        if not verifier_verdicts:
            return consensus_score

        # Count verdict types
        supported = sum(
            1 for v in verifier_verdicts if v.label == VerdictLabel.SUPPORTED
        )
        refuted = sum(
            1 for v in verifier_verdicts if v.label == VerdictLabel.REFUTED
        )
        not_enough_info = sum(
            1 for v in verifier_verdicts if v.label == VerdictLabel.NOT_ENOUGH_INFO
        )

        total = len(verifier_verdicts)

        # Calculate adjustment factor based on verifier results
        # Refuted claims should lower the score
        # Supported claims should maintain or slightly boost the score
        if total > 0:
            refuted_ratio = refuted / total
            supported_ratio = supported / total

            # If many claims are refuted, penalize the score
            if refuted_ratio > 0.3:  # More than 30% refuted
                penalty = refuted_ratio * 20  # Up to 20 point penalty
                adjusted_score = consensus_score - penalty
                logger.info(
                    f"Applied verifier penalty: {penalty:.2f} points "
                    f"({refuted}/{total} claims refuted)"
                )
            # If most claims are supported, small boost
            elif supported_ratio > 0.8:  # More than 80% supported
                boost = (supported_ratio - 0.8) * 10  # Up to 2 point boost
                adjusted_score = consensus_score + boost
                logger.info(
                    f"Applied verifier boost: {boost:.2f} points "
                    f"({supported}/{total} claims supported)"
                )
            else:
                adjusted_score = consensus_score
                logger.info(
                    f"No verifier adjustment: {supported} supported, "
                    f"{refuted} refuted, {not_enough_info} NEI"
                )

            # Clamp to valid range [0, 100]
            adjusted_score = max(0.0, min(100.0, adjusted_score))

            return adjusted_score

        return consensus_score

    def detect_disagreement(
        self, judge_results: List[JudgeResult]
    ) -> Dict[str, any]:
        """
        Detect and analyze disagreement among judge results.

        Args:
            judge_results: List of JudgeResult objects

        Returns:
            Dictionary with disagreement analysis including:
            - has_disagreement: bool
            - variance: float
            - score_range: tuple (min, max)
            - outliers: list of model names with outlier scores

        Example:
            >>> engine = AggregationEngine()
            >>> results = [
            ...     JudgeResult("judge1", 80.0, "Good", [], 0.9),
            ...     JudgeResult("judge2", 30.0, "Poor", [], 0.9),
            ... ]
            >>> disagreement = engine.detect_disagreement(results)
            >>> print(disagreement['has_disagreement'])  # True
        """
        if not judge_results:
            return {
                "has_disagreement": False,
                "variance": 0.0,
                "score_range": (0.0, 0.0),
                "outliers": [],
            }

        scores = [jr.score for jr in judge_results]
        variance = self._calculate_variance(scores)
        has_disagreement = variance > self.disagreement_threshold

        # Calculate score range
        min_score = min(scores)
        max_score = max(scores)
        score_range = (min_score, max_score)

        # Detect outliers (scores more than 1.5 * IQR from quartiles)
        outliers = []
        if len(scores) >= 4:
            sorted_scores = sorted(scores)
            q1 = statistics.quantiles(sorted_scores, n=4)[0]
            q3 = statistics.quantiles(sorted_scores, n=4)[2]
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            for jr in judge_results:
                if jr.score < lower_bound or jr.score > upper_bound:
                    outliers.append(jr.model_name)

        result = {
            "has_disagreement": has_disagreement,
            "variance": variance,
            "score_range": score_range,
            "outliers": outliers,
        }

        if has_disagreement:
            logger.warning(
                f"Disagreement detected: variance={variance:.2f}, "
                f"range={score_range}, outliers={outliers}"
            )

        return result
