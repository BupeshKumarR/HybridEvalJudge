"""
Specialized Verifier for statement-level fact-checking.

This module provides the SpecializedVerifier class that performs fine-grained
fact-checking using small fine-tuned models (e.g., MiniCheck, HHEM) with
three-way classification: SUPPORTED, REFUTED, NOT_ENOUGH_INFO.
"""

import logging
import re
from typing import Any, List, Optional, Tuple

import torch

from llm_judge_auditor.models import Passage, Verdict, VerdictLabel

logger = logging.getLogger(__name__)


class SpecializedVerifier:
    """
    Performs statement-level fact-checking using specialized models.

    This class uses fine-tuned small models (< 1B parameters) to verify
    individual statements against source text and retrieved passages.
    It provides three-way classification with confidence scoring.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        device: str = "cpu",
        max_length: int = 512,
        batch_size: int = 8,
    ):
        """
        Initialize the SpecializedVerifier.

        Args:
            model: The loaded verifier model
            tokenizer: The tokenizer for the model
            device: Device to run inference on ("cpu", "cuda", "mps")
            max_length: Maximum sequence length for tokenization
            batch_size: Batch size for batch verification
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size

        logger.info(
            f"SpecializedVerifier initialized (device={device}, "
            f"max_length={max_length}, batch_size={batch_size})"
        )

    def _format_input(
        self,
        statement: str,
        context: str,
        passages: Optional[List[Passage]] = None,
    ) -> str:
        """
        Format the input for the verifier model.

        Args:
            statement: The statement to verify
            context: The source context
            passages: Optional retrieved passages for additional context

        Returns:
            Formatted input string for the model
        """
        # Build the input with context and passages
        parts = []

        # Add source context
        if context:
            parts.append(f"Context: {context}")

        # Add retrieved passages if available
        if passages:
            for i, passage in enumerate(passages[:3], 1):  # Limit to top 3
                parts.append(f"Evidence {i}: {passage.text}")

        # Add the statement to verify
        parts.append(f"Statement: {statement}")

        # Join all parts
        formatted_input = " ".join(parts)

        return formatted_input

    def _parse_model_output(
        self, output_text: str, logits: Optional[torch.Tensor] = None
    ) -> Tuple[VerdictLabel, float]:
        """
        Parse the model output to extract verdict label and confidence.

        Args:
            output_text: The generated text from the model
            logits: Optional logits for confidence calculation

        Returns:
            Tuple of (VerdictLabel, confidence_score)
        """
        # Normalize output text
        output_lower = output_text.lower().strip()

        # Try to extract label from output
        label = VerdictLabel.NOT_ENOUGH_INFO  # Default

        if "supported" in output_lower or "support" in output_lower:
            label = VerdictLabel.SUPPORTED
        elif "refuted" in output_lower or "refute" in output_lower or "contradict" in output_lower:
            label = VerdictLabel.REFUTED
        elif "not enough" in output_lower or "insufficient" in output_lower or "nei" in output_lower:
            label = VerdictLabel.NOT_ENOUGH_INFO

        # Calculate confidence from logits if available
        confidence = 0.5  # Default confidence
        if logits is not None:
            try:
                # Use softmax to get probabilities
                probs = torch.softmax(logits, dim=-1)
                # Take the max probability as confidence
                confidence = float(probs.max().item())
            except Exception as e:
                logger.warning(f"Could not calculate confidence from logits: {e}")

        return label, confidence

    def verify_statement(
        self,
        statement: str,
        context: str,
        passages: Optional[List[Passage]] = None,
    ) -> Verdict:
        """
        Verify a single statement against context and passages.

        This method performs statement-level fact-checking using the specialized
        verifier model. It returns a three-way classification (SUPPORTED, REFUTED,
        NOT_ENOUGH_INFO) with confidence scoring and evidence.

        Args:
            statement: The statement to verify
            context: The source context to verify against
            passages: Optional retrieved passages for additional evidence

        Returns:
            Verdict containing label, confidence, evidence, and reasoning

        Example:
            >>> verifier = SpecializedVerifier(model, tokenizer)
            >>> verdict = verifier.verify_statement(
            ...     "The Eiffel Tower was built in 1889.",
            ...     "The Eiffel Tower was completed in 1889 for the World's Fair.",
            ...     passages=[]
            ... )
            >>> print(verdict.label)  # VerdictLabel.SUPPORTED
        """
        try:
            # Format input for the model
            model_input = self._format_input(statement, context, passages)

            # Tokenize
            inputs = self.tokenizer(
                model_input,
                max_length=self.max_length,
                truncation=True,
                padding=True,
                return_tensors="pt",
            )

            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Generate prediction
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=50,
                    num_beams=1,
                    do_sample=False,
                    return_dict_in_generate=True,
                    output_scores=True,
                )

            # Decode output
            output_text = self.tokenizer.decode(
                outputs.sequences[0], skip_special_tokens=True
            )

            # Get logits for confidence calculation
            logits = None
            if hasattr(outputs, "scores") and outputs.scores:
                # Use the first token's logits for confidence
                logits = outputs.scores[0][0]

            # Parse output to get label and confidence
            label, confidence = self._parse_model_output(output_text, logits)

            # Collect evidence
            evidence = []
            if passages:
                evidence = [p.text for p in passages[:3]]  # Top 3 passages
            elif context:
                # Use a snippet of context as evidence
                evidence = [context[:200] + "..." if len(context) > 200 else context]

            # Generate reasoning
            reasoning = f"Model output: {output_text}. "
            if label == VerdictLabel.SUPPORTED:
                reasoning += "The statement is supported by the provided context."
            elif label == VerdictLabel.REFUTED:
                reasoning += "The statement contradicts the provided context."
            else:
                reasoning += "There is not enough information to verify the statement."

            verdict = Verdict(
                label=label,
                confidence=confidence,
                evidence=evidence,
                reasoning=reasoning,
            )

            logger.debug(
                f"Verified statement: '{statement[:50]}...' -> {label.value} "
                f"(confidence={confidence:.2f})"
            )

            return verdict

        except Exception as e:
            logger.error(f"Error verifying statement: {e}")
            # Return a low-confidence NOT_ENOUGH_INFO verdict on error
            return Verdict(
                label=VerdictLabel.NOT_ENOUGH_INFO,
                confidence=0.0,
                evidence=[],
                reasoning=f"Error during verification: {str(e)}",
            )

    def batch_verify(
        self,
        statements: List[str],
        contexts: List[str],
        passages_list: Optional[List[List[Passage]]] = None,
    ) -> List[Verdict]:
        """
        Verify multiple statements in batch.

        This method processes multiple statements efficiently by batching
        the inference calls to the model.

        Args:
            statements: List of statements to verify
            contexts: List of source contexts (one per statement)
            passages_list: Optional list of passage lists (one per statement)

        Returns:
            List of Verdict objects, one per statement

        Raises:
            ValueError: If input lists have mismatched lengths

        Example:
            >>> verifier = SpecializedVerifier(model, tokenizer)
            >>> statements = ["Statement 1", "Statement 2"]
            >>> contexts = ["Context 1", "Context 2"]
            >>> verdicts = verifier.batch_verify(statements, contexts)
            >>> print(len(verdicts))  # 2
        """
        # Validate input lengths
        if len(statements) != len(contexts):
            raise ValueError(
                f"Mismatched lengths: {len(statements)} statements, "
                f"{len(contexts)} contexts"
            )

        if passages_list is not None and len(passages_list) != len(statements):
            raise ValueError(
                f"Mismatched lengths: {len(statements)} statements, "
                f"{len(passages_list)} passage lists"
            )

        # If no passages provided, create empty lists
        if passages_list is None:
            passages_list = [None] * len(statements)

        # Process in batches
        verdicts = []
        for i in range(0, len(statements), self.batch_size):
            batch_statements = statements[i : i + self.batch_size]
            batch_contexts = contexts[i : i + self.batch_size]
            batch_passages = passages_list[i : i + self.batch_size]

            # Verify each statement in the batch
            # Note: For simplicity, we process sequentially within batch
            # A more optimized version would batch the model calls
            for statement, context, passages in zip(
                batch_statements, batch_contexts, batch_passages
            ):
                verdict = self.verify_statement(statement, context, passages)
                verdicts.append(verdict)

        logger.info(f"Batch verified {len(statements)} statements")
        return verdicts

    def extract_statements(self, text: str) -> List[str]:
        """
        Extract individual statements from text for verification.

        This is a simple sentence-splitting approach. More sophisticated
        claim extraction can be implemented using NLP libraries.

        Args:
            text: The text to extract statements from

        Returns:
            List of extracted statements

        Example:
            >>> verifier = SpecializedVerifier(model, tokenizer)
            >>> text = "The sky is blue. Water is wet."
            >>> statements = verifier.extract_statements(text)
            >>> print(len(statements))  # 2
        """
        # Simple sentence splitting using regex
        # Split on periods, exclamation marks, and question marks
        sentences = re.split(r"[.!?]+", text)

        # Clean up and filter
        statements = []
        for sentence in sentences:
            sentence = sentence.strip()
            # Filter out very short sentences (likely not meaningful claims)
            if len(sentence) > 10:
                statements.append(sentence)

        logger.debug(f"Extracted {len(statements)} statements from text")
        return statements

    def verify_text(
        self,
        candidate_text: str,
        source_context: str,
        passages: Optional[List[Passage]] = None,
    ) -> List[Verdict]:
        """
        Verify all statements in a candidate text.

        This is a convenience method that extracts statements from the
        candidate text and verifies each one.

        Args:
            candidate_text: The text to verify
            source_context: The source context to verify against
            passages: Optional retrieved passages for evidence

        Returns:
            List of Verdict objects, one per extracted statement

        Example:
            >>> verifier = SpecializedVerifier(model, tokenizer)
            >>> verdicts = verifier.verify_text(
            ...     "The Eiffel Tower was built in 1889. It is in Paris.",
            ...     "The Eiffel Tower was completed in 1889 for the World's Fair in Paris.",
            ... )
            >>> print(len(verdicts))  # 2
        """
        # Extract statements
        statements = self.extract_statements(candidate_text)

        if not statements:
            logger.warning("No statements extracted from candidate text")
            return []

        # Verify all statements with the same context and passages
        contexts = [source_context] * len(statements)
        passages_list = [passages] * len(statements) if passages else None

        verdicts = self.batch_verify(statements, contexts, passages_list)

        logger.info(
            f"Verified {len(verdicts)} statements from candidate text. "
            f"Results: {sum(1 for v in verdicts if v.label == VerdictLabel.SUPPORTED)} supported, "
            f"{sum(1 for v in verdicts if v.label == VerdictLabel.REFUTED)} refuted, "
            f"{sum(1 for v in verdicts if v.label == VerdictLabel.NOT_ENOUGH_INFO)} NEI"
        )

        return verdicts
