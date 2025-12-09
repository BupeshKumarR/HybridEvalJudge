"""
Streaming Evaluator for processing large documents incrementally.

This module provides the StreamingEvaluator class, which enables evaluation
of large documents that cannot fit into memory at once. It chunks documents
into manageable segments and processes them incrementally through the
multi-stage evaluation pipeline.
"""

import logging
from dataclasses import dataclass, field
from typing import Iterator, List, Optional

from llm_judge_auditor.models import (
    EvaluationResult,
    Issue,
    JudgeResult,
    Passage,
    Verdict,
    VerdictLabel,
)

logger = logging.getLogger(__name__)


@dataclass
class PartialResult:
    """
    Result from evaluating a single chunk of a document.

    Attributes:
        chunk_index: Index of the chunk in the document
        chunk_text: The text of this chunk
        consensus_score: Score for this chunk
        verifier_verdicts: Verdicts from verifier for this chunk
        judge_results: Judge results for this chunk
        retrieved_passages: Passages retrieved for this chunk
        flagged_issues: Issues detected in this chunk
    """

    chunk_index: int
    chunk_text: str
    consensus_score: float
    verifier_verdicts: List[Verdict] = field(default_factory=list)
    judge_results: List[JudgeResult] = field(default_factory=list)
    retrieved_passages: List[Passage] = field(default_factory=list)
    flagged_issues: List[Issue] = field(default_factory=list)


class StreamingEvaluator:
    """
    Evaluator for processing large documents incrementally.

    This class wraps an EvaluationToolkit and provides streaming evaluation
    capabilities. It chunks large documents into manageable segments, processes
    each segment through the full evaluation pipeline, and aggregates results
    across all segments.

    The streaming approach reduces memory footprint and enables evaluation of
    documents that would otherwise be too large to process at once.

    Example:
        >>> from llm_judge_auditor import EvaluationToolkit
        >>> from llm_judge_auditor.components.streaming_evaluator import StreamingEvaluator
        >>> 
        >>> toolkit = EvaluationToolkit.from_preset("balanced")
        >>> streaming = StreamingEvaluator(toolkit, chunk_size=512)
        >>> 
        >>> with open("large_source.txt") as source_file, \\
        ...      open("large_candidate.txt") as candidate_file:
        ...     result = streaming.evaluate_stream(
        ...         source_stream=source_file,
        ...         candidate_stream=candidate_file
        ...     )
        >>> print(f"Final Score: {result.consensus_score}")
    """

    def __init__(self, toolkit, chunk_size: int = 512, overlap: int = 50):
        """
        Initialize the StreamingEvaluator.

        Args:
            toolkit: EvaluationToolkit instance to use for evaluation
            chunk_size: Number of characters per chunk (default: 512)
            overlap: Number of characters to overlap between chunks (default: 50)
                    This helps maintain context across chunk boundaries

        Raises:
            ValueError: If chunk_size <= overlap or if either is negative
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if overlap < 0:
            raise ValueError("overlap cannot be negative")
        if chunk_size <= overlap:
            raise ValueError("chunk_size must be greater than overlap")

        self.toolkit = toolkit
        self.chunk_size = chunk_size
        self.overlap = overlap

        logger.info(
            f"StreamingEvaluator initialized with chunk_size={chunk_size}, overlap={overlap}"
        )

    def evaluate_stream(
        self,
        source_stream: Iterator[str],
        candidate_stream: Iterator[str],
        task: str = "factual_accuracy",
        criteria: Optional[List[str]] = None,
        use_retrieval: Optional[bool] = None,
    ) -> EvaluationResult:
        """
        Evaluate large documents from streams.

        This method reads source and candidate documents from iterators (e.g., file
        objects), chunks them into manageable segments, evaluates each segment, and
        aggregates results into a final evaluation.

        Args:
            source_stream: Iterator yielding source text (e.g., open file)
            candidate_stream: Iterator yielding candidate text (e.g., open file)
            task: Evaluation task type (default: "factual_accuracy")
            criteria: Optional evaluation criteria (default: ["correctness"])
            use_retrieval: Override config retrieval setting (default: use config)

        Returns:
            EvaluationResult with aggregated scores and results from all chunks

        Raises:
            ValueError: If streams are invalid
            RuntimeError: If evaluation fails

        Example:
            >>> with open("source.txt") as src, open("candidate.txt") as cand:
            ...     result = streaming.evaluate_stream(src, cand)
            >>> print(f"Score: {result.consensus_score}")
        """
        logger.info("Starting streaming evaluation")

        # Read full text from streams
        logger.info("Reading source stream...")
        source_text = self._read_stream(source_stream)
        logger.info(f"  Source text length: {len(source_text)} characters")

        logger.info("Reading candidate stream...")
        candidate_text = self._read_stream(candidate_stream)
        logger.info(f"  Candidate text length: {len(candidate_text)} characters")

        # Validate inputs
        if not source_text or not source_text.strip():
            raise ValueError("source_stream produced empty text")
        if not candidate_text or not candidate_text.strip():
            raise ValueError("candidate_stream produced empty text")

        # Chunk the candidate text (source is used as context for all chunks)
        logger.info("Chunking candidate text...")
        chunks = self._chunk_text(candidate_text)
        logger.info(f"  Created {len(chunks)} chunks")

        # Process each chunk
        partial_results = []
        for chunk_index, chunk_text in enumerate(chunks):
            logger.info(f"Processing chunk {chunk_index + 1}/{len(chunks)}")
            partial_result = self._evaluate_chunk(
                source_text=source_text,
                chunk_text=chunk_text,
                chunk_index=chunk_index,
                task=task,
                criteria=criteria,
                use_retrieval=use_retrieval,
            )
            partial_results.append(partial_result)

        # Aggregate results from all chunks
        logger.info("Aggregating results from all chunks...")
        final_result = self._aggregate_stream_results(
            partial_results=partial_results,
            source_text=source_text,
            candidate_text=candidate_text,
            task=task,
            criteria=criteria,
            use_retrieval=use_retrieval,
        )

        logger.info(
            f"Streaming evaluation complete. Final score: {final_result.consensus_score:.2f}"
        )
        return final_result

    def _read_stream(self, stream: Iterator[str]) -> str:
        """
        Read all text from a stream iterator.

        Args:
            stream: Iterator yielding text chunks (e.g., file object)

        Returns:
            Complete text as a single string
        """
        chunks = []
        for chunk in stream:
            chunks.append(chunk)
        return "".join(chunks)

    def _chunk_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks.

        This method creates chunks of size chunk_size with overlap between
        consecutive chunks to maintain context across boundaries.

        Args:
            text: Text to chunk

        Returns:
            List of text chunks

        Example:
            >>> evaluator = StreamingEvaluator(toolkit, chunk_size=100, overlap=20)
            >>> chunks = evaluator._chunk_text("A" * 250)
            >>> len(chunks)  # Should be 3 chunks
            3
        """
        if len(text) <= self.chunk_size:
            # Text fits in a single chunk
            return [text]

        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]

            # Try to break at sentence boundary if possible
            if end < len(text):
                # Look for sentence endings near the chunk boundary
                last_period = chunk.rfind(". ")
                last_question = chunk.rfind("? ")
                last_exclamation = chunk.rfind("! ")

                # Find the latest sentence ending
                sentence_end = max(last_period, last_question, last_exclamation)

                # If we found a sentence ending in the last 20% of the chunk, use it
                if sentence_end > self.chunk_size * 0.8:
                    chunk = chunk[: sentence_end + 2]  # Include the punctuation and space
                    end = start + len(chunk)

            chunks.append(chunk)

            # Move start position, accounting for overlap
            if end >= len(text):
                break
            start = end - self.overlap

        logger.debug(f"Created {len(chunks)} chunks from text of length {len(text)}")
        return chunks

    def _evaluate_chunk(
        self,
        source_text: str,
        chunk_text: str,
        chunk_index: int,
        task: str,
        criteria: Optional[List[str]],
        use_retrieval: Optional[bool],
    ) -> PartialResult:
        """
        Evaluate a single chunk of text.

        Args:
            source_text: Full source text (used as context)
            chunk_text: Text chunk to evaluate
            chunk_index: Index of this chunk
            task: Evaluation task type
            criteria: Evaluation criteria
            use_retrieval: Whether to use retrieval

        Returns:
            PartialResult for this chunk

        Raises:
            RuntimeError: If chunk evaluation fails
        """
        try:
            # Evaluate this chunk using the toolkit
            result = self.toolkit.evaluate(
                source_text=source_text,
                candidate_output=chunk_text,
                task=task,
                criteria=criteria,
                use_retrieval=use_retrieval,
            )

            # Create partial result
            partial = PartialResult(
                chunk_index=chunk_index,
                chunk_text=chunk_text,
                consensus_score=result.consensus_score,
                verifier_verdicts=result.verifier_verdicts,
                judge_results=result.judge_results,
                retrieved_passages=result.report.retrieval_provenance,
                flagged_issues=result.flagged_issues,
            )

            logger.debug(
                f"  Chunk {chunk_index} score: {partial.consensus_score:.2f}, "
                f"issues: {len(partial.flagged_issues)}"
            )

            return partial

        except Exception as e:
            error_msg = f"Failed to evaluate chunk {chunk_index}: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def _aggregate_stream_results(
        self,
        partial_results: List[PartialResult],
        source_text: str,
        candidate_text: str,
        task: str,
        criteria: Optional[List[str]],
        use_retrieval: Optional[bool],
    ) -> EvaluationResult:
        """
        Aggregate partial results from all chunks into a final evaluation.

        This method combines scores, verdicts, and issues from all chunks using
        weighted averaging based on chunk size. Larger chunks have more influence
        on the final score.

        Args:
            partial_results: List of PartialResult from each chunk
            source_text: Full source text
            candidate_text: Full candidate text
            task: Evaluation task type
            criteria: Evaluation criteria
            use_retrieval: Whether retrieval was used

        Returns:
            EvaluationResult with aggregated results

        Raises:
            ValueError: If partial_results is empty
        """
        if not partial_results:
            raise ValueError("Cannot aggregate empty partial_results")

        logger.info(f"Aggregating {len(partial_results)} partial results")

        # Calculate weighted average score based on chunk sizes
        total_chars = sum(len(pr.chunk_text) for pr in partial_results)
        weighted_score = 0.0

        for pr in partial_results:
            weight = len(pr.chunk_text) / total_chars
            weighted_score += pr.consensus_score * weight

        logger.info(f"  Weighted consensus score: {weighted_score:.2f}")

        # Combine all verdicts (deduplicate by reasoning)
        all_verdicts = []
        seen_reasoning = set()
        for pr in partial_results:
            for verdict in pr.verifier_verdicts:
                if verdict.reasoning not in seen_reasoning:
                    all_verdicts.append(verdict)
                    seen_reasoning.add(verdict.reasoning)

        logger.info(f"  Combined {len(all_verdicts)} unique verdicts")

        # Combine all judge results (average scores per judge across chunks)
        judge_scores = {}
        judge_reasonings = {}
        judge_issues = {}
        judge_confidences = {}

        for pr in partial_results:
            for jr in pr.judge_results:
                if jr.model_name not in judge_scores:
                    judge_scores[jr.model_name] = []
                    judge_reasonings[jr.model_name] = []
                    judge_issues[jr.model_name] = []
                    judge_confidences[jr.model_name] = []

                judge_scores[jr.model_name].append(jr.score)
                judge_reasonings[jr.model_name].append(jr.reasoning)
                judge_issues[jr.model_name].extend(jr.flagged_issues)
                judge_confidences[jr.model_name].append(jr.confidence)

        # Create aggregated judge results
        aggregated_judge_results = []
        for model_name in judge_scores:
            avg_score = sum(judge_scores[model_name]) / len(judge_scores[model_name])
            avg_confidence = sum(judge_confidences[model_name]) / len(
                judge_confidences[model_name]
            )

            # Combine reasoning from all chunks
            combined_reasoning = " | ".join(
                f"Chunk {i + 1}: {r}"
                for i, r in enumerate(judge_reasonings[model_name])
            )

            aggregated_judge_results.append(
                JudgeResult(
                    model_name=model_name,
                    score=avg_score,
                    reasoning=combined_reasoning,
                    flagged_issues=judge_issues[model_name],
                    confidence=avg_confidence,
                )
            )

        logger.info(f"  Aggregated results from {len(aggregated_judge_results)} judges")

        # Combine all retrieved passages (deduplicate by source)
        all_passages = []
        seen_sources = set()
        for pr in partial_results:
            for passage in pr.retrieved_passages:
                if passage.source not in seen_sources:
                    all_passages.append(passage)
                    seen_sources.add(passage.source)

        logger.info(f"  Combined {len(all_passages)} unique passages")

        # Combine all flagged issues (deduplicate by description)
        all_issues = []
        seen_descriptions = set()
        for pr in partial_results:
            for issue in pr.flagged_issues:
                if issue.description not in seen_descriptions:
                    all_issues.append(issue)
                    seen_descriptions.add(issue.description)

        logger.info(f"  Combined {len(all_issues)} unique issues")

        # Create a synthetic EvaluationResult
        # We need to import these here to avoid circular imports
        from llm_judge_auditor.models import (
            AggregationMetadata,
            EvaluationRequest,
            Report,
        )
        from datetime import datetime

        # Create request
        request = EvaluationRequest(
            source_text=source_text,
            candidate_output=candidate_text,
            task=task,
            criteria=criteria or ["correctness"],
            use_retrieval=use_retrieval
            if use_retrieval is not None
            else self.toolkit.config.enable_retrieval,
        )

        # Calculate variance across chunks
        chunk_scores = [pr.consensus_score for pr in partial_results]
        mean_score = sum(chunk_scores) / len(chunk_scores)
        variance = sum((s - mean_score) ** 2 for s in chunk_scores) / len(chunk_scores)

        # Create aggregation metadata
        individual_scores = {jr.model_name: jr.score for jr in aggregated_judge_results}
        aggregation_metadata = AggregationMetadata(
            strategy="weighted_average_streaming",
            individual_scores=individual_scores,
            variance=variance,
            is_low_confidence=variance > self.toolkit.config.disagreement_threshold,
            weights=None,
        )

        # Calculate confidence
        avg_confidence = sum(jr.confidence for jr in aggregated_judge_results) / len(
            aggregated_judge_results
        )
        if aggregation_metadata.is_low_confidence:
            confidence = avg_confidence * 0.7
        else:
            confidence = avg_confidence

        # Categorize hallucinations
        hallucination_categories = self._categorize_issues(all_issues)

        # Create report
        report = Report(
            metadata={
                "timestamp": datetime.now().isoformat(),
                "task": task,
                "criteria": criteria or ["correctness"],
                "retrieval_enabled": request.use_retrieval,
                "verifier_model": self.toolkit.config.verifier_model,
                "judge_models": self.toolkit.config.judge_models,
                "aggregation_strategy": "weighted_average_streaming",
                "num_chunks": len(partial_results),
                "total_characters": total_chars,
                "chunk_size": self.chunk_size,
                "overlap": self.overlap,
                "num_retrieved_passages": len(all_passages),
                "num_verifier_verdicts": len(all_verdicts),
                "num_judge_results": len(aggregated_judge_results),
            },
            consensus_score=weighted_score,
            individual_scores=individual_scores,
            verifier_verdicts=all_verdicts,
            retrieval_provenance=all_passages,
            reasoning={jr.model_name: jr.reasoning for jr in aggregated_judge_results},
            confidence=confidence,
            disagreement_level=variance,
            flagged_issues=all_issues,
            hallucination_categories=hallucination_categories,
        )

        # Create final result
        result = EvaluationResult(
            request=request,
            consensus_score=weighted_score,
            verifier_verdicts=all_verdicts,
            judge_results=aggregated_judge_results,
            aggregation_metadata=aggregation_metadata,
            report=report,
            flagged_issues=all_issues,
        )

        return result

    def _categorize_issues(self, issues: List[Issue]) -> dict:
        """
        Categorize issues by type.

        Args:
            issues: List of issues to categorize

        Returns:
            Dictionary mapping issue type names to counts
        """
        from llm_judge_auditor.models import IssueType

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
            if issue.type == IssueType.HALLUCINATION or issue.type == IssueType.FACTUAL_ERROR:
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
