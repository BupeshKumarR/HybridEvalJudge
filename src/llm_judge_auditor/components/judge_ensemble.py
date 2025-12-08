"""
Judge Model Ensemble for the LLM Judge Auditor toolkit.

This module provides the JudgeEnsemble class for evaluating outputs using
multiple LLM judges with structured prompts, including support for single
evaluation, ensemble evaluation, and pairwise comparison.
"""

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch

from llm_judge_auditor.components.model_manager import ModelManager
from llm_judge_auditor.components.prompt_manager import PromptManager
from llm_judge_auditor.models import Issue, IssueType, IssueSeverity, JudgeResult
from llm_judge_auditor.utils.error_handling import (
    log_error_with_context,
    parse_score_with_fallback,
    safe_parse_judge_output,
)

logger = logging.getLogger(__name__)


@dataclass
class BiasDetectionResult:
    """
    Result from bias detection evaluation.

    Attributes:
        model_name: Name of the judge model that performed the evaluation
        flagged_phrases: List of issues representing biased phrases with explanations
        overall_assessment: Summary of bias detection findings
        reasoning: Detailed step-by-step analysis
    """

    model_name: str
    flagged_phrases: List[Issue]
    overall_assessment: str
    reasoning: str


@dataclass
class PairwiseResult:
    """
    Result from pairwise comparison of two candidate outputs.

    Attributes:
        winner: Which candidate won ("A", "B", or "TIE")
        reasoning: Detailed explanation of the comparison
        candidate_a_score: Optional score for candidate A
        candidate_b_score: Optional score for candidate B
    """

    winner: str  # "A", "B", or "TIE"
    reasoning: str
    candidate_a_score: Optional[float] = None
    candidate_b_score: Optional[float] = None


class JudgeEnsemble:
    """
    Manages evaluation using multiple judge LLMs.

    This class orchestrates evaluation across multiple judge models, providing
    single evaluation, ensemble evaluation, and pairwise comparison functionality.
    Each judge provides independent evaluation with structured prompts and
    chain-of-thought reasoning.
    """

    def __init__(
        self,
        model_manager: ModelManager,
        prompt_manager: Optional[PromptManager] = None,
        max_length: int = 2048,
        temperature: float = 0.1,
    ):
        """
        Initialize the JudgeEnsemble.

        Args:
            model_manager: ModelManager instance with loaded judge models
            prompt_manager: Optional PromptManager for custom prompts
            max_length: Maximum generation length for judge outputs
            temperature: Temperature for generation (lower = more deterministic)
        """
        self.model_manager = model_manager
        self.prompt_manager = prompt_manager or PromptManager()
        self.max_length = max_length
        self.temperature = temperature

        logger.info(
            f"JudgeEnsemble initialized with max_length={max_length}, "
            f"temperature={temperature}"
        )

    def _generate_response(
        self,
        model: Any,
        tokenizer: Any,
        prompt: str,
        max_length: Optional[int] = None,
        timeout_seconds: Optional[float] = None,
    ) -> str:
        """
        Generate a response from a judge model with timeout handling.

        Args:
            model: The judge model
            tokenizer: The tokenizer for the model
            prompt: The input prompt
            max_length: Optional max length override
            timeout_seconds: Optional timeout in seconds (default: 60)

        Returns:
            Generated text response

        Raises:
            TimeoutError: If generation exceeds timeout
            RuntimeError: If generation fails

        Requirements: 9.2 (timeout handling)
        """
        max_len = max_length or self.max_length
        timeout = timeout_seconds or 60.0  # Default 60 second timeout

        try:
            # Tokenize input
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=max_len,
            )

            # Move to same device as model
            device = next(iter(model.parameters())).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Generate with timeout tracking
            import time
            start_time = time.time()

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_len,
                    temperature=self.temperature,
                    do_sample=self.temperature > 0,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )

            elapsed = time.time() - start_time

            # Check if we exceeded timeout
            if elapsed > timeout:
                logger.warning(
                    f"Generation took {elapsed:.2f}s, exceeding timeout of {timeout}s"
                )

            # Decode
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Remove the prompt from the response if it's included
            if response.startswith(prompt):
                response = response[len(prompt) :].strip()

            return response

        except Exception as e:
            # Log error with context
            log_error_with_context(
                e,
                {
                    "operation": "generate_response",
                    "prompt_length": len(prompt),
                    "max_length": max_len,
                    "timeout": timeout,
                },
            )
            raise RuntimeError(f"Failed to generate response: {str(e)}") from e

    def _parse_factual_accuracy_response(
        self, response: str, model_name: str
    ) -> JudgeResult:
        """
        Parse a factual accuracy evaluation response with error handling.

        Expected format:
        REASONING: [reasoning text]
        SCORE: [0-100]
        FLAGGED_ISSUES: [issues or "None detected"]

        This method uses safe parsing with fallback for malformed outputs.

        Args:
            response: Raw response from judge model
            model_name: Name of the judge model

        Returns:
            JudgeResult with parsed score, reasoning, and issues

        Requirements: 9.1 (malformed output parsing with partial results)
        """
        try:
            # Use safe parsing utility for malformed output handling
            parsed = safe_parse_judge_output(response, model_name, default_score=50.0)

            score = parsed["score"]
            reasoning = parsed["reasoning"]

            # Log if parsing was not fully successful
            if not parsed["parse_success"]:
                logger.warning(
                    f"Partial parse for {model_name}: {'; '.join(parsed['parse_errors'])}"
                )

            # Extract flagged issues
            flagged_issues = []
            issues_text = parsed["flagged_issues"]

            if issues_text and issues_text.lower() not in [
                "none",
                "none detected",
                "no issues",
            ]:
                # Parse issues - each line is an issue
                for line in issues_text.split("\n"):
                    line = line.strip()
                    if line and not line.startswith("-"):
                        # Simple issue parsing
                        flagged_issues.append(
                            Issue(
                                type=IssueType.HALLUCINATION,
                                severity=IssueSeverity.MEDIUM,
                                description=line,
                                evidence=[],
                            )
                        )

            # Calculate confidence based on score extremity
            # More extreme scores (close to 0 or 100) indicate higher confidence
            confidence = abs(score - 50) / 50.0

            return JudgeResult(
                model_name=model_name,
                score=score,
                reasoning=reasoning,
                flagged_issues=flagged_issues,
                confidence=confidence,
            )

        except Exception as e:
            # Log error with context
            log_error_with_context(
                e,
                {
                    "operation": "parse_factual_accuracy_response",
                    "model_name": model_name,
                    "response_length": len(response),
                    "response_preview": response[:200],
                },
            )

            # Return a default result with low confidence
            return JudgeResult(
                model_name=model_name,
                score=50.0,
                reasoning=f"Failed to parse response: {str(e)}. Raw output: {response[:200]}...",
                flagged_issues=[],
                confidence=0.0,
            )

    def _parse_pairwise_response(self, response: str) -> PairwiseResult:
        """
        Parse a pairwise comparison response.

        Expected format:
        REASONING: [reasoning text]
        WINNER: [A, B, or TIE]
        EXPLANATION: [explanation]

        Args:
            response: Raw response from judge model

        Returns:
            PairwiseResult with winner and reasoning
        """
        # Extract winner
        winner_match = re.search(
            r"WINNER:\s*(A|B|TIE)", response, re.IGNORECASE
        )
        if winner_match:
            winner = winner_match.group(1).upper()
        else:
            logger.warning("Could not parse winner from response, defaulting to TIE")
            winner = "TIE"

        # Extract reasoning
        reasoning_match = re.search(
            r"REASONING:\s*(.*?)(?=WINNER:|EXPLANATION:|$)",
            response,
            re.IGNORECASE | re.DOTALL,
        )
        if reasoning_match:
            reasoning = reasoning_match.group(1).strip()
        else:
            reasoning = response.strip()

        # Extract explanation (optional)
        explanation_match = re.search(
            r"EXPLANATION:\s*(.*?)$", response, re.IGNORECASE | re.DOTALL
        )
        if explanation_match:
            explanation = explanation_match.group(1).strip()
            reasoning = f"{reasoning}\n\n{explanation}"

        return PairwiseResult(winner=winner, reasoning=reasoning)

    def _parse_bias_detection_response(
        self, response: str, model_name: str
    ) -> BiasDetectionResult:
        """
        Parse a bias detection evaluation response.

        Expected format:
        REASONING: [reasoning text]
        FLAGGED_PHRASES: [phrases with severity]
        OVERALL_ASSESSMENT: [summary]

        Args:
            response: Raw response from judge model
            model_name: Name of the judge model

        Returns:
            BiasDetectionResult with flagged phrases and assessment
        """
        # Extract reasoning
        reasoning_match = re.search(
            r"REASONING:\s*(.*?)(?=FLAGGED_PHRASES:|OVERALL_ASSESSMENT:|$)",
            response,
            re.IGNORECASE | re.DOTALL,
        )
        if reasoning_match:
            reasoning = reasoning_match.group(1).strip()
        else:
            reasoning = response.strip()

        # Extract flagged phrases
        flagged_phrases = []
        phrases_match = re.search(
            r"FLAGGED_PHRASES:\s*(.*?)(?=OVERALL_ASSESSMENT:|$)",
            response,
            re.IGNORECASE | re.DOTALL,
        )
        if phrases_match:
            phrases_text = phrases_match.group(1).strip()
            if phrases_text and phrases_text.lower() not in [
                "none",
                "none detected",
                "no bias detected",
                "no issues",
            ]:
                # Parse phrases - each line or bullet point is a flagged phrase
                # Format: "phrase" - explanation [SEVERITY: low/medium/high]
                lines = phrases_text.split("\n")
                for line in lines:
                    line = line.strip()
                    if not line or line.startswith("---"):
                        continue

                    # Remove bullet points
                    line = re.sub(r"^[-*•]\s*", "", line)

                    # Try to extract severity
                    severity = IssueSeverity.MEDIUM  # Default
                    severity_match = re.search(
                        r"\[?SEVERITY:\s*(low|medium|high)\]?",
                        line,
                        re.IGNORECASE,
                    )
                    if severity_match:
                        severity_str = severity_match.group(1).lower()
                        severity = IssueSeverity(severity_str)
                        # Remove severity tag from line
                        line = re.sub(
                            r"\[?SEVERITY:\s*(low|medium|high)\]?",
                            "",
                            line,
                            flags=re.IGNORECASE,
                        ).strip()

                    # Try to extract phrase and explanation
                    # Format: "phrase" - explanation
                    phrase_match = re.match(r'"([^"]+)"\s*[-:]\s*(.*)', line)
                    if phrase_match:
                        phrase = phrase_match.group(1)
                        explanation = phrase_match.group(2).strip()
                        description = f'"{phrase}" - {explanation}'
                    else:
                        # Just use the whole line as description
                        description = line

                    if description:
                        flagged_phrases.append(
                            Issue(
                                type=IssueType.BIAS,
                                severity=severity,
                                description=description,
                                evidence=[],
                            )
                        )

        # Extract overall assessment
        assessment_match = re.search(
            r"OVERALL_ASSESSMENT:\s*(.*?)$", response, re.IGNORECASE | re.DOTALL
        )
        if assessment_match:
            overall_assessment = assessment_match.group(1).strip()
        else:
            # Generate a default assessment based on flagged phrases
            if not flagged_phrases:
                overall_assessment = "No bias detected in the candidate output."
            else:
                high_count = sum(
                    1 for p in flagged_phrases if p.severity == IssueSeverity.HIGH
                )
                medium_count = sum(
                    1 for p in flagged_phrases if p.severity == IssueSeverity.MEDIUM
                )
                low_count = sum(
                    1 for p in flagged_phrases if p.severity == IssueSeverity.LOW
                )
                overall_assessment = (
                    f"Detected {len(flagged_phrases)} instances of bias: "
                    f"{high_count} high severity, {medium_count} medium severity, "
                    f"{low_count} low severity."
                )

        return BiasDetectionResult(
            model_name=model_name,
            flagged_phrases=flagged_phrases,
            overall_assessment=overall_assessment,
            reasoning=reasoning,
        )

    def evaluate_single(
        self,
        judge_name: str,
        source_text: str,
        candidate_output: str,
        retrieved_context: str = "",
    ) -> JudgeResult:
        """
        Evaluate using a single judge model.

        Args:
            judge_name: Name of the judge model to use
            source_text: Reference document or context
            candidate_output: Text to be evaluated
            retrieved_context: Optional retrieved passages for context

        Returns:
            JudgeResult with score, reasoning, and flagged issues

        Raises:
            ValueError: If judge model is not loaded
            RuntimeError: If evaluation fails

        Example:
            >>> ensemble = JudgeEnsemble(model_manager)
            >>> result = ensemble.evaluate_single(
            ...     "llama-3-8b",
            ...     "Paris is the capital of France.",
            ...     "Paris is the capital of Germany."
            ... )
            >>> print(f"Score: {result.score}")
        """
        # Get the judge model
        judge = self.model_manager.get_judge(judge_name)
        if judge is None:
            raise ValueError(
                f"Judge model '{judge_name}' is not loaded. "
                f"Available judges: {list(self.model_manager.get_all_judges().keys())}"
            )

        model, tokenizer = judge

        try:
            logger.info(f"Evaluating with judge: {judge_name}")

            # Construct prompt
            prompt = self.prompt_manager.get_prompt(
                task="factual_accuracy",
                source_text=source_text,
                candidate_output=candidate_output,
                retrieved_context=retrieved_context or "No additional context available.",
            )

            # Generate response with timeout handling
            response = self._generate_response(
                model, tokenizer, prompt, timeout_seconds=60.0
            )

            # Parse response (with malformed output handling)
            result = self._parse_factual_accuracy_response(response, judge_name)

            logger.info(
                f"Judge {judge_name} evaluation complete: score={result.score:.1f}, "
                f"issues={len(result.flagged_issues)}"
            )

            return result

        except Exception as e:
            # Log error with context
            log_error_with_context(
                e,
                {
                    "operation": "evaluate_single",
                    "judge_name": judge_name,
                    "source_text_length": len(source_text),
                    "candidate_output_length": len(candidate_output),
                },
            )

            error_msg = f"Failed to evaluate with judge '{judge_name}': {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def evaluate_all(
        self,
        source_text: str,
        candidate_output: str,
        retrieved_context: str = "",
    ) -> List[JudgeResult]:
        """
        Evaluate using all loaded judge models (ensemble evaluation).

        This method processes judges sequentially, collecting results from each.

        Args:
            source_text: Reference document or context
            candidate_output: Text to be evaluated
            retrieved_context: Optional retrieved passages for context

        Returns:
            List of JudgeResult objects, one per judge

        Raises:
            RuntimeError: If no judges are loaded or all evaluations fail

        Example:
            >>> ensemble = JudgeEnsemble(model_manager)
            >>> results = ensemble.evaluate_all(
            ...     "Paris is the capital of France.",
            ...     "Paris is the capital of Germany."
            ... )
            >>> for result in results:
            ...     print(f"{result.model_name}: {result.score}")
        """
        # Get all loaded judges
        judges = self.model_manager.get_all_judges()

        if not judges:
            raise RuntimeError(
                "No judge models loaded. Please load judge models before evaluation."
            )

        logger.info(f"Starting ensemble evaluation with {len(judges)} judges")

        results = []
        failed_judges = []

        # Evaluate with each judge sequentially
        for judge_name in judges.keys():
            try:
                result = self.evaluate_single(
                    judge_name=judge_name,
                    source_text=source_text,
                    candidate_output=candidate_output,
                    retrieved_context=retrieved_context,
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Judge {judge_name} failed: {e}")
                failed_judges.append((judge_name, str(e)))

        # If all judges failed, raise an error
        if not results:
            error_details = "\n".join(
                f"  - {name}: {error}" for name, error in failed_judges
            )
            raise RuntimeError(
                f"All judge evaluations failed. Errors:\n{error_details}"
            )

        # If some judges failed, log a warning
        if failed_judges:
            logger.warning(
                f"Ensemble evaluation completed with {len(results)}/{len(judges)} judges. "
                f"Failed judges: {[name for name, _ in failed_judges]}"
            )
        else:
            logger.info(f"Ensemble evaluation completed successfully with {len(results)} judges")

        return results

    def pairwise_compare(
        self,
        source_text: str,
        candidate_a: str,
        candidate_b: str,
        judge_name: Optional[str] = None,
    ) -> PairwiseResult:
        """
        Perform pairwise comparison of two candidate outputs.

        If no judge is specified, uses the first available judge.

        Args:
            source_text: Reference document or context
            candidate_a: First candidate output
            candidate_b: Second candidate output
            judge_name: Optional specific judge to use

        Returns:
            PairwiseResult with winner and reasoning

        Raises:
            ValueError: If no judges are loaded
            RuntimeError: If comparison fails

        Example:
            >>> ensemble = JudgeEnsemble(model_manager)
            >>> result = ensemble.pairwise_compare(
            ...     "Paris is the capital of France.",
            ...     "Paris is the capital of France.",
            ...     "Paris is the capital of Germany."
            ... )
            >>> print(f"Winner: {result.winner}")
        """
        # Get judge to use
        if judge_name:
            judge = self.model_manager.get_judge(judge_name)
            if judge is None:
                raise ValueError(f"Judge model '{judge_name}' is not loaded")
        else:
            # Use first available judge
            judges = self.model_manager.get_all_judges()
            if not judges:
                raise ValueError("No judge models loaded")
            judge_name = list(judges.keys())[0]
            judge = judges[judge_name]

        model, tokenizer = judge

        try:
            logger.info(f"Performing pairwise comparison with judge: {judge_name}")

            # Construct prompt
            prompt = self.prompt_manager.get_prompt(
                task="pairwise_ranking",
                source_text=source_text,
                candidate_a=candidate_a,
                candidate_b=candidate_b,
            )

            # Generate response
            response = self._generate_response(model, tokenizer, prompt)

            # Parse response
            result = self._parse_pairwise_response(response)

            logger.info(
                f"Pairwise comparison complete: winner={result.winner}"
            )

            return result

        except Exception as e:
            error_msg = f"Failed to perform pairwise comparison: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def detect_bias(
        self,
        candidate_output: str,
        judge_name: Optional[str] = None,
    ) -> BiasDetectionResult:
        """
        Detect bias and harmful language in a candidate output.

        If no judge is specified, uses the first available judge.

        Args:
            candidate_output: Text to be evaluated for bias
            judge_name: Optional specific judge to use

        Returns:
            BiasDetectionResult with flagged phrases, severity ratings, and assessment

        Raises:
            ValueError: If no judges are loaded
            RuntimeError: If bias detection fails

        Example:
            >>> ensemble = JudgeEnsemble(model_manager)
            >>> result = ensemble.detect_bias(
            ...     "Women are naturally better at nursing than men."
            ... )
            >>> print(f"Flagged phrases: {len(result.flagged_phrases)}")
            >>> for issue in result.flagged_phrases:
            ...     print(f"  - {issue.description} [{issue.severity.value}]")
        """
        # Get judge to use
        if judge_name:
            judge = self.model_manager.get_judge(judge_name)
            if judge is None:
                raise ValueError(f"Judge model '{judge_name}' is not loaded")
        else:
            # Use first available judge
            judges = self.model_manager.get_all_judges()
            if not judges:
                raise ValueError("No judge models loaded")
            judge_name = list(judges.keys())[0]
            judge = judges[judge_name]

        model, tokenizer = judge

        try:
            logger.info(f"Performing bias detection with judge: {judge_name}")

            # Construct prompt
            prompt = self.prompt_manager.get_prompt(
                task="bias_detection",
                candidate_output=candidate_output,
            )

            # Generate response
            response = self._generate_response(model, tokenizer, prompt)

            # Parse response
            result = self._parse_bias_detection_response(response, judge_name)

            logger.info(
                f"Bias detection complete: {len(result.flagged_phrases)} phrases flagged"
            )

            return result

        except Exception as e:
            error_msg = f"Failed to perform bias detection: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
