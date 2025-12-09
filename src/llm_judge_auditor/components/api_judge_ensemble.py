"""
API Judge Ensemble for coordinating multiple API-based judge clients.

This module implements the APIJudgeEnsemble class that manages multiple
API judge clients (Groq, Gemini) and coordinates their parallel execution
for comprehensive LLM output evaluation.
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

from llm_judge_auditor.components.api_key_manager import APIKeyManager
from llm_judge_auditor.components.base_judge_client import BaseJudgeClient, JudgeVerdict
from llm_judge_auditor.components.gemini_judge_client import GeminiJudgeClient
from llm_judge_auditor.components.groq_judge_client import GroqJudgeClient
from llm_judge_auditor.config import ToolkitConfig

logger = logging.getLogger(__name__)


class APIJudgeEnsemble:
    """
    Ensemble of API-based judge clients for LLM evaluation.
    
    This class coordinates multiple API judge clients (Groq, Gemini) to
    evaluate LLM outputs. It handles:
    - Initialization of judges based on available API keys
    - Parallel execution of judge evaluations
    - Graceful handling of partial failures
    - Score aggregation and disagreement detection
    
    Attributes:
        judges: List of initialized judge clients
        config: Toolkit configuration
        parallel_execution: Whether to execute judges in parallel
    """
    
    def __init__(
        self,
        config: ToolkitConfig,
        api_key_manager: APIKeyManager,
        parallel_execution: bool = True
    ):
        """
        Initialize the API Judge Ensemble.
        
        Args:
            config: Toolkit configuration
            api_key_manager: Manager for API keys
            parallel_execution: Whether to execute judges in parallel (default: True)
        """
        self.config = config
        self.api_key_manager = api_key_manager
        self.parallel_execution = parallel_execution
        self.judges: List[BaseJudgeClient] = []
        
        # Initialize judges based on available API keys
        self._initialize_judges()
        
        logger.info(
            f"APIJudgeEnsemble initialized with {len(self.judges)} judges "
            f"(parallel={parallel_execution})"
        )
    
    def _initialize_judges(self) -> None:
        """
        Initialize judge clients based on available API keys.
        
        Creates judge clients for each service with an available API key.
        Logs warnings for services without keys.
        """
        # Initialize Groq judge if key is available
        if self.api_key_manager.groq_key:
            try:
                groq_judge = GroqJudgeClient(
                    api_key=self.api_key_manager.groq_key,
                    model="llama-3.3-70b-versatile",
                    max_retries=2,
                    base_delay=1.0,
                    timeout=30
                )
                self.judges.append(groq_judge)
                logger.info("Groq judge initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Groq judge: {e}")
        else:
            logger.warning("Groq API key not available, skipping Groq judge")
        
        # Initialize Gemini judge if key is available
        if self.api_key_manager.gemini_key:
            try:
                gemini_judge = GeminiJudgeClient(
                    api_key=self.api_key_manager.gemini_key,
                    model="gemini-2.0-flash-exp",
                    max_retries=2,
                    base_delay=1.0,
                    timeout=30
                )
                self.judges.append(gemini_judge)
                logger.info("Gemini judge initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Gemini judge: {e}")
        else:
            logger.warning("Gemini API key not available, skipping Gemini judge")
        
        if not self.judges:
            logger.error("No judges initialized! At least one API key is required.")
    
    def evaluate(
        self,
        source_text: str,
        candidate_output: str,
        task: str = "factual_accuracy"
    ) -> List[JudgeVerdict]:
        """
        Evaluate candidate output using all available judges.
        
        Executes all judges either in parallel or sequentially, depending
        on the parallel_execution setting. Handles individual judge failures
        gracefully and returns all successful verdicts.
        
        Args:
            source_text: Reference document or context
            candidate_output: Text to be evaluated
            task: Evaluation task type (e.g., "factual_accuracy", "bias_detection")
        
        Returns:
            List of JudgeVerdict objects from successful judges
        
        Raises:
            RuntimeError: If no judges are available or all judges fail
        
        Example:
            >>> ensemble = APIJudgeEnsemble(config, api_key_manager)
            >>> verdicts = ensemble.evaluate(
            ...     source_text="The sky is blue.",
            ...     candidate_output="The sky is green.",
            ...     task="factual_accuracy"
            ... )
            >>> for verdict in verdicts:
            ...     print(f"{verdict.judge_name}: {verdict.score}")
        """
        if not self.judges:
            raise RuntimeError(
                "No judges available for evaluation. "
                "Please configure at least one API key (GROQ_API_KEY or GEMINI_API_KEY)."
            )
        
        logger.info(
            f"Starting evaluation with {len(self.judges)} judges "
            f"(task={task}, parallel={self.parallel_execution})"
        )
        
        if self.parallel_execution:
            verdicts = self._evaluate_parallel(source_text, candidate_output, task)
        else:
            verdicts = self._evaluate_sequential(source_text, candidate_output, task)
        
        if not verdicts:
            raise RuntimeError(
                "All judges failed during evaluation. "
                "Please check API keys and network connectivity."
            )
        
        logger.info(
            f"Evaluation complete: {len(verdicts)}/{len(self.judges)} judges succeeded"
        )
        
        return verdicts
    
    def _evaluate_parallel(
        self,
        source_text: str,
        candidate_output: str,
        task: str
    ) -> List[JudgeVerdict]:
        """
        Evaluate using all judges in parallel.
        
        Uses ThreadPoolExecutor to call all judges simultaneously,
        reducing total evaluation time.
        
        Args:
            source_text: Reference document or context
            candidate_output: Text to be evaluated
            task: Evaluation task type
        
        Returns:
            List of successful JudgeVerdict objects
        """
        verdicts = []
        
        with ThreadPoolExecutor(max_workers=len(self.judges)) as executor:
            # Submit all judge evaluations
            future_to_judge = {
                executor.submit(
                    judge.evaluate,
                    source_text,
                    candidate_output,
                    task
                ): judge
                for judge in self.judges
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_judge):
                judge = future_to_judge[future]
                judge_name = judge.get_judge_name()
                
                try:
                    verdict = future.result()
                    verdicts.append(verdict)
                    logger.info(
                        f"Judge {judge_name} completed successfully "
                        f"(score: {verdict.score:.1f})"
                    )
                except Exception as e:
                    logger.error(
                        f"Judge {judge_name} failed: {e}",
                        exc_info=True
                    )
                    # Continue with other judges
        
        return verdicts
    
    def _evaluate_sequential(
        self,
        source_text: str,
        candidate_output: str,
        task: str
    ) -> List[JudgeVerdict]:
        """
        Evaluate using all judges sequentially.
        
        Calls each judge one at a time. Useful for debugging or
        when parallel execution is not desired.
        
        Args:
            source_text: Reference document or context
            candidate_output: Text to be evaluated
            task: Evaluation task type
        
        Returns:
            List of successful JudgeVerdict objects
        """
        verdicts = []
        
        for judge in self.judges:
            judge_name = judge.get_judge_name()
            
            try:
                verdict = judge.evaluate(source_text, candidate_output, task)
                verdicts.append(verdict)
                logger.info(
                    f"Judge {judge_name} completed successfully "
                    f"(score: {verdict.score:.1f})"
                )
            except Exception as e:
                logger.error(
                    f"Judge {judge_name} failed: {e}",
                    exc_info=True
                )
                # Continue with other judges
        
        return verdicts
    
    def get_judge_count(self) -> int:
        """
        Get the number of active judges in the ensemble.
        
        Returns:
            Number of initialized judges
        """
        return len(self.judges)
    
    def get_judge_names(self) -> List[str]:
        """
        Get the names of all active judges.
        
        Returns:
            List of judge names
        """
        return [judge.get_judge_name() for judge in self.judges]
    
    def aggregate_verdicts(
        self,
        verdicts: List[JudgeVerdict]
    ) -> Tuple[float, Dict[str, float], float]:
        """
        Aggregate verdicts from multiple judges.
        
        Calculates:
        - Consensus score (mean of all scores)
        - Individual scores by judge
        - Disagreement level (variance)
        
        Args:
            verdicts: List of JudgeVerdict objects
        
        Returns:
            Tuple of (consensus_score, individual_scores_dict, disagreement_level)
        
        Example:
            >>> verdicts = ensemble.evaluate(source, candidate)
            >>> consensus, individual, disagreement = ensemble.aggregate_verdicts(verdicts)
            >>> print(f"Consensus: {consensus:.1f}, Disagreement: {disagreement:.1f}")
        """
        if not verdicts:
            return 0.0, {}, 0.0
        
        # Extract scores
        scores = [v.score for v in verdicts]
        individual_scores = {v.judge_name: v.score for v in verdicts}
        
        # Calculate consensus (mean)
        consensus_score = sum(scores) / len(scores)
        
        # Calculate disagreement (variance)
        if len(scores) > 1:
            mean = consensus_score
            variance = sum((s - mean) ** 2 for s in scores) / len(scores)
            disagreement_level = variance
        else:
            disagreement_level = 0.0
        
        logger.info(
            f"Aggregation: consensus={consensus_score:.1f}, "
            f"disagreement={disagreement_level:.1f}"
        )
        
        return consensus_score, individual_scores, disagreement_level
    
    def identify_disagreements(
        self,
        verdicts: List[JudgeVerdict],
        threshold: float = 20.0
    ) -> Dict[str, any]:
        """
        Identify and analyze disagreements between judges.
        
        Detects when judges have significantly different scores and
        provides analysis of the disagreement.
        
        Args:
            verdicts: List of JudgeVerdict objects
            threshold: Variance threshold for flagging disagreement (default: 20.0)
        
        Returns:
            Dictionary with disagreement analysis:
            - has_disagreement: bool
            - variance: float
            - score_range: tuple (min, max)
            - outliers: list of judge names with outlier scores
            - reasoning_summary: dict of judge_name -> reasoning
        
        Example:
            >>> verdicts = ensemble.evaluate(source, candidate)
            >>> disagreement = ensemble.identify_disagreements(verdicts)
            >>> if disagreement['has_disagreement']:
            ...     print(f"Judges disagree! Variance: {disagreement['variance']:.1f}")
        """
        if not verdicts:
            return {
                "has_disagreement": False,
                "variance": 0.0,
                "score_range": (0.0, 0.0),
                "outliers": [],
                "reasoning_summary": {}
            }
        
        scores = [v.score for v in verdicts]
        
        # Calculate variance
        if len(scores) > 1:
            mean = sum(scores) / len(scores)
            variance = sum((s - mean) ** 2 for s in scores) / len(scores)
        else:
            variance = 0.0
        
        has_disagreement = variance > threshold
        
        # Calculate score range
        min_score = min(scores)
        max_score = max(scores)
        score_range = (min_score, max_score)
        
        # Detect outliers (scores more than 1.5 standard deviations from mean)
        outliers = []
        if len(scores) >= 2:
            mean = sum(scores) / len(scores)
            std_dev = (variance ** 0.5)
            
            for verdict in verdicts:
                if abs(verdict.score - mean) > 1.5 * std_dev:
                    outliers.append(verdict.judge_name)
        
        # Collect reasoning from all judges
        reasoning_summary = {
            v.judge_name: v.reasoning for v in verdicts
        }
        
        result = {
            "has_disagreement": has_disagreement,
            "variance": variance,
            "score_range": score_range,
            "outliers": outliers,
            "reasoning_summary": reasoning_summary
        }
        
        if has_disagreement:
            logger.warning(
                f"Disagreement detected: variance={variance:.2f} > threshold={threshold}, "
                f"range={score_range}, outliers={outliers}"
            )
        
        return result
