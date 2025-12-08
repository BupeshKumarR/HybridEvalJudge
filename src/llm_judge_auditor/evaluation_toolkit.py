"""
Main Evaluation Toolkit orchestrator for the LLM Judge Auditor.

This module provides the EvaluationToolkit class, which serves as the main entry
point for evaluating LLM outputs. It orchestrates the multi-stage pipeline:
retrieval → verifier → ensemble → aggregation → reporting.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional

from llm_judge_auditor.components.aggregation_engine import (
    AggregationEngine,
    AggregationStrategy as EngineAggregationStrategy,
)
from llm_judge_auditor.components.judge_ensemble import JudgeEnsemble
from llm_judge_auditor.components.model_manager import ModelManager
from llm_judge_auditor.components.prompt_manager import PromptManager
from llm_judge_auditor.components.retrieval_component import RetrievalComponent
from llm_judge_auditor.components.specialized_verifier import SpecializedVerifier
from llm_judge_auditor.config import AggregationStrategy, ToolkitConfig
from llm_judge_auditor.models import (
    BatchResult,
    Claim,
    EvaluationRequest,
    EvaluationResult,
    Issue,
    IssueType,
    IssueSeverity,
    Passage,
    Report,
    Verdict,
    VerdictLabel,
)

logger = logging.getLogger(__name__)


class EvaluationToolkit:
    """
    Main orchestrator for the hybrid LLM evaluation system.

    This class integrates all components (retrieval, specialized verifier,
    judge ensemble, aggregation) into a unified evaluation pipeline. It
    provides the primary interface for evaluating LLM outputs for factual
    accuracy, hallucinations, and bias.

    The evaluation follows a multi-stage pipeline:
    1. Retrieval Stage (optional): Extract claims and retrieve relevant passages
    2. Verification Stage: Specialized verifier performs statement-level fact-checking
    3. Ensemble Stage: Multiple judge LLMs evaluate the output
    4. Aggregation Stage: Combine results using configured strategy
    5. Reporting Stage: Generate comprehensive evaluation report

    Example:
        >>> config = ToolkitConfig.from_preset("balanced")
        >>> toolkit = EvaluationToolkit(config)
        >>> result = toolkit.evaluate(
        ...     source_text="Paris is the capital of France.",
        ...     candidate_output="Paris is the capital of Germany."
        ... )
        >>> print(f"Score: {result.consensus_score}")
    """

    def __init__(self, config: ToolkitConfig):
        """
        Initialize the EvaluationToolkit with the given configuration.

        Args:
            config: ToolkitConfig instance with all settings

        Raises:
            RuntimeError: If initialization of any component fails
        """
        self.config = config
        logger.info("Initializing EvaluationToolkit...")

        try:
            # Initialize model manager
            logger.info("Initializing ModelManager...")
            self.model_manager = ModelManager(
                device=config.device.value,
                quantize=config.quantize,
                cache_dir=str(config.cache_dir),
            )

            # Load verifier model
            logger.info(f"Loading verifier model: {config.verifier_model}")
            self.model_manager.load_verifier(config.verifier_model)

            # Load judge models
            logger.info(f"Loading {len(config.judge_models)} judge models...")
            for judge_model in config.judge_models:
                logger.info(f"  - Loading judge: {judge_model}")
                self.model_manager.load_judge(judge_model)

            # Verify all models are ready
            if not self.model_manager.verify_models_ready():
                raise RuntimeError("Not all models loaded successfully")

            logger.info("All models loaded and verified")

            # Initialize retrieval component
            logger.info("Initializing RetrievalComponent...")
            self.retrieval = RetrievalComponent(
                embedding_model="all-MiniLM-L6-v2",
                top_k=config.retrieval_top_k,
                device=config.device.value if config.device != "auto" else "cpu",
            )

            # Initialize knowledge base if configured
            if config.enable_retrieval and config.knowledge_base_path:
                logger.info(f"Loading knowledge base from {config.knowledge_base_path}")
                self.retrieval.initialize_knowledge_base(
                    kb_path=str(config.knowledge_base_path),
                    index_type="faiss",
                )
            else:
                logger.info("Retrieval disabled or no knowledge base configured")

            # Initialize specialized verifier
            logger.info("Initializing SpecializedVerifier...")
            verifier_model, verifier_tokenizer = self.model_manager.get_verifier()
            self.verifier = SpecializedVerifier(
                model=verifier_model,
                tokenizer=verifier_tokenizer,
                device=config.device.value if config.device != "auto" else "cpu",
                max_length=config.max_length,
                batch_size=config.batch_size,
            )

            # Initialize prompt manager
            logger.info("Initializing PromptManager...")
            self.prompt_manager = PromptManager()
            if config.prompt_template_path:
                self.prompt_manager.load_templates(str(config.prompt_template_path))

            # Initialize judge ensemble
            logger.info("Initializing JudgeEnsemble...")
            self.judge_ensemble = JudgeEnsemble(
                model_manager=self.model_manager,
                prompt_manager=self.prompt_manager,
                max_length=config.max_length,
                temperature=0.1,
            )

            # Initialize aggregation engine
            logger.info("Initializing AggregationEngine...")
            # Convert config AggregationStrategy to engine AggregationStrategy
            engine_strategy = EngineAggregationStrategy(config.aggregation_strategy.value)
            self.aggregation = AggregationEngine(
                strategy=engine_strategy,
                disagreement_threshold=config.disagreement_threshold,
                weights=config.judge_weights,
            )

            logger.info("EvaluationToolkit initialization complete")

        except Exception as e:
            error_msg = f"Failed to initialize EvaluationToolkit: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def evaluate(
        self,
        source_text: str,
        candidate_output: str,
        task: str = "factual_accuracy",
        criteria: Optional[List[str]] = None,
        use_retrieval: Optional[bool] = None,
    ) -> EvaluationResult:
        """
        Evaluate a candidate output against a source text.

        This method executes the full multi-stage evaluation pipeline:
        1. Retrieval (if enabled): Extract claims and retrieve relevant passages
        2. Verification: Specialized verifier checks each statement
        3. Ensemble: Multiple judges evaluate the output
        4. Aggregation: Combine results into consensus score
        5. Reporting: Generate comprehensive report

        Args:
            source_text: Reference document or context
            candidate_output: Text to be evaluated
            task: Evaluation task type (default: "factual_accuracy")
            criteria: Optional evaluation criteria (default: ["correctness"])
            use_retrieval: Override config retrieval setting (default: use config)

        Returns:
            EvaluationResult with consensus score, verdicts, judge results, and report

        Raises:
            ValueError: If inputs are invalid
            RuntimeError: If evaluation fails

        Example:
            >>> toolkit = EvaluationToolkit(config)
            >>> result = toolkit.evaluate(
            ...     source_text="The Eiffel Tower was completed in 1889.",
            ...     candidate_output="The Eiffel Tower was built in 1889."
            ... )
            >>> print(f"Consensus Score: {result.consensus_score}")
            >>> print(f"Confidence: {result.report.confidence}")
        """
        # Validate inputs
        if not source_text or not source_text.strip():
            raise ValueError("source_text cannot be empty")
        if not candidate_output or not candidate_output.strip():
            raise ValueError("candidate_output cannot be empty")

        # Create evaluation request
        request = EvaluationRequest(
            source_text=source_text,
            candidate_output=candidate_output,
            task=task,
            criteria=criteria or ["correctness"],
            use_retrieval=use_retrieval if use_retrieval is not None else self.config.enable_retrieval,
        )

        logger.info(f"Starting evaluation (task={task}, use_retrieval={request.use_retrieval})")

        try:
            # Stage 1: Retrieval (if enabled)
            retrieved_passages: List[Passage] = []
            claims: List[Claim] = []

            if request.use_retrieval and not self.retrieval.fallback_mode():
                logger.info("Stage 1: Retrieval - Extracting claims and retrieving passages")
                claims = self.retrieval.extract_claims(candidate_output)
                logger.info(f"  Extracted {len(claims)} claims")

                # Retrieve passages for each claim
                for claim in claims:
                    passages = self.retrieval.retrieve_passages(claim)
                    retrieved_passages.extend(passages)

                logger.info(f"  Retrieved {len(retrieved_passages)} passages")
            else:
                logger.info("Stage 1: Retrieval - Skipped (disabled or fallback mode)")

            # Stage 2: Specialized Verification
            logger.info("Stage 2: Specialized Verification - Verifying statements")
            verifier_verdicts = self.verifier.verify_text(
                candidate_text=candidate_output,
                source_context=source_text,
                passages=retrieved_passages if retrieved_passages else None,
            )
            logger.info(f"  Generated {len(verifier_verdicts)} verdicts")

            # Log verdict summary
            supported = sum(1 for v in verifier_verdicts if v.label == VerdictLabel.SUPPORTED)
            refuted = sum(1 for v in verifier_verdicts if v.label == VerdictLabel.REFUTED)
            nei = sum(1 for v in verifier_verdicts if v.label == VerdictLabel.NOT_ENOUGH_INFO)
            logger.info(f"  Verdicts: {supported} supported, {refuted} refuted, {nei} NEI")

            # Stage 3: Judge Ensemble Evaluation
            logger.info("Stage 3: Judge Ensemble - Evaluating with all judges")

            # Format retrieved context for judges
            retrieved_context = ""
            if retrieved_passages:
                context_parts = []
                for i, passage in enumerate(retrieved_passages[:5], 1):  # Limit to top 5
                    context_parts.append(f"Evidence {i}: {passage.text}")
                retrieved_context = "\n".join(context_parts)

            judge_results = self.judge_ensemble.evaluate_all(
                source_text=source_text,
                candidate_output=candidate_output,
                retrieved_context=retrieved_context,
            )
            logger.info(f"  Received {len(judge_results)} judge evaluations")

            # Stage 4: Aggregation
            logger.info("Stage 4: Aggregation - Combining results")
            consensus_score, aggregation_metadata = self.aggregation.aggregate_scores(
                judge_results=judge_results,
                verifier_verdicts=verifier_verdicts,
            )
            logger.info(f"  Consensus score: {consensus_score:.2f}")

            # Stage 5: Reporting
            logger.info("Stage 5: Reporting - Generating comprehensive report")
            report = self._generate_report(
                request=request,
                consensus_score=consensus_score,
                verifier_verdicts=verifier_verdicts,
                judge_results=judge_results,
                retrieved_passages=retrieved_passages,
                aggregation_metadata=aggregation_metadata,
            )

            # Compile all flagged issues
            flagged_issues = self._compile_flagged_issues(
                verifier_verdicts=verifier_verdicts,
                judge_results=judge_results,
            )

            # Create evaluation result
            result = EvaluationResult(
                request=request,
                consensus_score=consensus_score,
                verifier_verdicts=verifier_verdicts,
                judge_results=judge_results,
                aggregation_metadata=aggregation_metadata,
                report=report,
                flagged_issues=flagged_issues,
            )

            logger.info("Evaluation complete")
            return result

        except Exception as e:
            error_msg = f"Evaluation failed: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def batch_evaluate(
        self,
        requests: List[EvaluationRequest],
        continue_on_error: bool = True,
    ) -> BatchResult:
        """
        Evaluate multiple candidate outputs in batch.

        This method processes a list of evaluation requests sequentially,
        providing error resilience and batch statistics. If an evaluation
        fails, the error is logged and processing continues with remaining
        requests (if continue_on_error is True).

        Args:
            requests: List of EvaluationRequest objects to process
            continue_on_error: If True, continue processing after errors (default: True)

        Returns:
            BatchResult with successful results, errors, and statistics

        Raises:
            ValueError: If requests list is empty
            RuntimeError: If continue_on_error is False and an evaluation fails

        Example:
            >>> toolkit = EvaluationToolkit.from_preset("balanced")
            >>> requests = [
            ...     EvaluationRequest(
            ...         source_text="Paris is the capital of France.",
            ...         candidate_output="Paris is in France."
            ...     ),
            ...     EvaluationRequest(
            ...         source_text="The Earth orbits the Sun.",
            ...         candidate_output="The Sun orbits the Earth."
            ...     ),
            ... ]
            >>> batch_result = toolkit.batch_evaluate(requests)
            >>> print(f"Processed: {len(batch_result.results)}/{len(requests)}")
            >>> print(f"Mean score: {batch_result.statistics['mean']:.2f}")
        """
        if not requests:
            raise ValueError("requests list cannot be empty")

        logger.info(f"Starting batch evaluation of {len(requests)} requests")

        results: List[EvaluationResult] = []
        errors: List[Dict[str, Any]] = []

        # Process each request sequentially
        for idx, request in enumerate(requests):
            logger.info(f"Processing request {idx + 1}/{len(requests)}")

            try:
                # Evaluate using the standard evaluate method
                result = self.evaluate(
                    source_text=request.source_text,
                    candidate_output=request.candidate_output,
                    task=request.task,
                    criteria=request.criteria,
                    use_retrieval=request.use_retrieval,
                )
                results.append(result)
                logger.info(f"  Request {idx + 1} completed successfully (score: {result.consensus_score:.2f})")

            except Exception as e:
                error_info = {
                    "request_index": idx,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "source_text_preview": request.source_text[:100] + "..." if len(request.source_text) > 100 else request.source_text,
                    "candidate_output_preview": request.candidate_output[:100] + "..." if len(request.candidate_output) > 100 else request.candidate_output,
                }
                errors.append(error_info)
                logger.error(f"  Request {idx + 1} failed: {str(e)}")

                if not continue_on_error:
                    raise RuntimeError(f"Batch evaluation failed at request {idx + 1}: {str(e)}") from e

        # Calculate batch statistics
        logger.info("Calculating batch statistics")
        statistics = self._calculate_batch_statistics(results)

        # Build metadata
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "total_requests": len(requests),
            "successful_evaluations": len(results),
            "failed_evaluations": len(errors),
            "success_rate": len(results) / len(requests) if requests else 0.0,
            "config": {
                "verifier_model": self.config.verifier_model,
                "judge_models": self.config.judge_models,
                "aggregation_strategy": self.config.aggregation_strategy.value,
            },
        }

        batch_result = BatchResult(
            results=results,
            errors=errors,
            statistics=statistics,
            metadata=metadata,
        )

        logger.info(f"Batch evaluation complete: {len(results)}/{len(requests)} successful")
        logger.info(f"  Mean score: {statistics.get('mean', 0):.2f}")
        logger.info(f"  Median score: {statistics.get('median', 0):.2f}")

        return batch_result

    def _calculate_batch_statistics(self, results: List[EvaluationResult]) -> Dict[str, float]:
        """
        Calculate summary statistics for batch evaluation results.

        Args:
            results: List of successful evaluation results

        Returns:
            Dictionary with statistical measures (mean, median, std, min, max)
        """
        if not results:
            return {
                "mean": 0.0,
                "median": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
                "count": 0,
            }

        scores = [result.consensus_score for result in results]
        scores_sorted = sorted(scores)

        # Calculate mean
        mean_score = sum(scores) / len(scores)

        # Calculate median
        n = len(scores_sorted)
        if n % 2 == 0:
            median_score = (scores_sorted[n // 2 - 1] + scores_sorted[n // 2]) / 2
        else:
            median_score = scores_sorted[n // 2]

        # Calculate standard deviation
        variance = sum((s - mean_score) ** 2 for s in scores) / len(scores)
        std_score = variance ** 0.5

        # Min and max
        min_score = min(scores)
        max_score = max(scores)

        # Calculate percentiles
        def percentile(sorted_list, p):
            """Calculate the p-th percentile of a sorted list."""
            if not sorted_list:
                return 0.0
            k = (len(sorted_list) - 1) * p
            f = int(k)
            c = k - f
            if f + 1 < len(sorted_list):
                return sorted_list[f] + c * (sorted_list[f + 1] - sorted_list[f])
            return sorted_list[f]

        p25 = percentile(scores_sorted, 0.25)
        p75 = percentile(scores_sorted, 0.75)

        return {
            "mean": mean_score,
            "median": median_score,
            "std": std_score,
            "min": min_score,
            "max": max_score,
            "count": len(results),
            "p25": p25,
            "p75": p75,
        }

    def _generate_report(
        self,
        request: EvaluationRequest,
        consensus_score: float,
        verifier_verdicts: List[Verdict],
        judge_results: List,
        retrieved_passages: List[Passage],
        aggregation_metadata,
    ) -> Report:
        """
        Generate a comprehensive evaluation report.

        Args:
            request: Original evaluation request
            consensus_score: Final aggregated score
            verifier_verdicts: Verdicts from specialized verifier
            judge_results: Results from judge ensemble
            retrieved_passages: Retrieved passages from knowledge base
            aggregation_metadata: Metadata from aggregation process

        Returns:
            Report object with all evaluation details
        """
        # Build metadata
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "task": request.task,
            "criteria": request.criteria,
            "retrieval_enabled": request.use_retrieval,
            "verifier_model": self.config.verifier_model,
            "judge_models": self.config.judge_models,
            "aggregation_strategy": self.config.aggregation_strategy.value,
            "num_retrieved_passages": len(retrieved_passages),
            "num_verifier_verdicts": len(verifier_verdicts),
            "num_judge_results": len(judge_results),
        }

        # Extract individual scores
        individual_scores = {jr.model_name: jr.score for jr in judge_results}

        # Extract reasoning from each judge
        reasoning = {jr.model_name: jr.reasoning for jr in judge_results}

        # Calculate overall confidence
        # Based on aggregation metadata and judge confidences
        judge_confidences = [jr.confidence for jr in judge_results]
        avg_judge_confidence = sum(judge_confidences) / len(judge_confidences) if judge_confidences else 0.5

        # Lower confidence if high disagreement
        if aggregation_metadata.is_low_confidence:
            confidence = avg_judge_confidence * 0.7  # Reduce by 30%
        else:
            confidence = avg_judge_confidence

        # Calculate disagreement level (variance)
        disagreement_level = aggregation_metadata.variance

        # Compile all flagged issues
        flagged_issues = self._compile_flagged_issues(verifier_verdicts, judge_results)

        # Categorize hallucinations by type
        hallucination_categories = self._categorize_hallucinations(flagged_issues)

        report = Report(
            metadata=metadata,
            consensus_score=consensus_score,
            individual_scores=individual_scores,
            verifier_verdicts=verifier_verdicts,
            retrieval_provenance=retrieved_passages,
            reasoning=reasoning,
            confidence=confidence,
            disagreement_level=disagreement_level,
            flagged_issues=flagged_issues,
            hallucination_categories=hallucination_categories,
        )

        return report

    def _compile_flagged_issues(
        self,
        verifier_verdicts: List[Verdict],
        judge_results: List,
    ) -> List[Issue]:
        """
        Compile all flagged issues from verifier and judges.

        Args:
            verifier_verdicts: Verdicts from specialized verifier
            judge_results: Results from judge ensemble

        Returns:
            List of all unique flagged issues
        """
        issues = []

        # Add issues from verifier (refuted statements are hallucinations)
        for verdict in verifier_verdicts:
            if verdict.label == VerdictLabel.REFUTED:
                issues.append(
                    Issue(
                        type=IssueType.HALLUCINATION,
                        severity=IssueSeverity.HIGH,
                        description=f"Refuted claim: {verdict.reasoning}",
                        evidence=verdict.evidence,
                    )
                )
            elif verdict.label == VerdictLabel.NOT_ENOUGH_INFO and verdict.confidence < 0.3:
                issues.append(
                    Issue(
                        type=IssueType.UNSUPPORTED_CLAIM,
                        severity=IssueSeverity.MEDIUM,
                        description=f"Unverifiable claim: {verdict.reasoning}",
                        evidence=verdict.evidence,
                    )
                )

        # Add issues from judges
        for judge_result in judge_results:
            for issue in judge_result.flagged_issues:
                # Avoid duplicates by checking description
                if not any(i.description == issue.description for i in issues):
                    issues.append(issue)

        return issues

    def _categorize_hallucinations(self, issues: List[Issue]) -> Dict[str, int]:
        """
        Categorize hallucinations by type.

        Args:
            issues: List of flagged issues

        Returns:
            Dictionary mapping issue types to counts
        """
        categories = {
            "factual_error": 0,
            "unsupported_claim": 0,
            "temporal_inconsistency": 0,
            "numerical_error": 0,
            "bias": 0,
            "inconsistency": 0,
            "other": 0,
        }

        for issue in issues:
            if issue.type == IssueType.HALLUCINATION:
                categories["factual_error"] += 1
            elif issue.type == IssueType.UNSUPPORTED_CLAIM:
                categories["unsupported_claim"] += 1
            elif issue.type == IssueType.TEMPORAL_INCONSISTENCY:
                categories["temporal_inconsistency"] += 1
            elif issue.type == IssueType.NUMERICAL_ERROR:
                categories["numerical_error"] += 1
            elif issue.type == IssueType.BIAS:
                categories["bias"] += 1
            elif issue.type == IssueType.INCONSISTENCY:
                categories["inconsistency"] += 1
            else:
                categories["other"] += 1

        return categories

    @classmethod
    def from_preset(cls, preset_name: str) -> "EvaluationToolkit":
        """
        Create an EvaluationToolkit from a preset configuration.

        Args:
            preset_name: Name of the preset ("fast", "balanced", "strict", "research")

        Returns:
            EvaluationToolkit instance configured with the preset

        Raises:
            ValueError: If preset_name is not recognized

        Example:
            >>> toolkit = EvaluationToolkit.from_preset("balanced")
            >>> result = toolkit.evaluate(source, candidate)
        """
        config = ToolkitConfig.from_preset(preset_name)
        return cls(config)

    @classmethod
    def from_config_file(cls, config_path: str) -> "EvaluationToolkit":
        """
        Create an EvaluationToolkit from a YAML configuration file.

        Args:
            config_path: Path to YAML configuration file

        Returns:
            EvaluationToolkit instance configured from the file

        Example:
            >>> toolkit = EvaluationToolkit.from_config_file("config.yaml")
            >>> result = toolkit.evaluate(source, candidate)
        """
        from pathlib import Path

        config = ToolkitConfig.from_yaml(Path(config_path))
        return cls(config)

    def get_stats(self) -> Dict:
        """
        Get statistics about the toolkit components.

        Returns:
            Dictionary with component statistics

        Example:
            >>> toolkit = EvaluationToolkit.from_preset("balanced")
            >>> stats = toolkit.get_stats()
            >>> print(f"Loaded judges: {stats['num_judges']}")
        """
        model_info = self.model_manager.get_model_info()
        retrieval_stats = self.retrieval.get_stats()

        return {
            "config": {
                "preset": "custom",
                "verifier_model": self.config.verifier_model,
                "judge_models": self.config.judge_models,
                "retrieval_enabled": self.config.enable_retrieval,
                "aggregation_strategy": self.config.aggregation_strategy.value,
            },
            "models": model_info,
            "retrieval": retrieval_stats,
            "num_judges": len(self.config.judge_models),
            "verifier_loaded": model_info.get("verifier") is not None,
        }
