"""
Unit tests for the SpecializedVerifier component.

Tests cover statement verification, batch processing, statement extraction,
and integration with retrieval passages.
"""

import pytest
import torch
from unittest.mock import MagicMock, Mock, patch

from llm_judge_auditor.components.specialized_verifier import SpecializedVerifier
from llm_judge_auditor.models import Passage, Verdict, VerdictLabel


@pytest.fixture
def mock_model():
    """Create a mock verifier model."""
    model = MagicMock()
    model.generate = MagicMock()
    return model


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer."""
    tokenizer = MagicMock()
    tokenizer.return_value = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "attention_mask": torch.tensor([[1, 1, 1]]),
    }
    tokenizer.decode = MagicMock(return_value="SUPPORTED")
    return tokenizer


@pytest.fixture
def verifier(mock_model, mock_tokenizer):
    """Create a SpecializedVerifier instance with mocked components."""
    return SpecializedVerifier(
        model=mock_model,
        tokenizer=mock_tokenizer,
        device="cpu",
        max_length=512,
        batch_size=4,
    )


class TestSpecializedVerifierInit:
    """Tests for SpecializedVerifier initialization."""

    def test_init_with_defaults(self, mock_model, mock_tokenizer):
        """Test initialization with default parameters."""
        verifier = SpecializedVerifier(mock_model, mock_tokenizer)
        assert verifier.model == mock_model
        assert verifier.tokenizer == mock_tokenizer
        assert verifier.device == "cpu"
        assert verifier.max_length == 512
        assert verifier.batch_size == 8

    def test_init_with_custom_params(self, mock_model, mock_tokenizer):
        """Test initialization with custom parameters."""
        verifier = SpecializedVerifier(
            mock_model,
            mock_tokenizer,
            device="cuda",
            max_length=1024,
            batch_size=16,
        )
        assert verifier.device == "cuda"
        assert verifier.max_length == 1024
        assert verifier.batch_size == 16


class TestFormatInput:
    """Tests for input formatting."""

    def test_format_input_with_context_only(self, verifier):
        """Test formatting with only context."""
        result = verifier._format_input(
            statement="The sky is blue.",
            context="The sky appears blue during the day.",
            passages=None,
        )
        assert "Context: The sky appears blue during the day." in result
        assert "Statement: The sky is blue." in result

    def test_format_input_with_passages(self, verifier):
        """Test formatting with context and passages."""
        passages = [
            Passage("Evidence 1 text", "source1", 0.9),
            Passage("Evidence 2 text", "source2", 0.8),
        ]
        result = verifier._format_input(
            statement="The sky is blue.",
            context="Context text",
            passages=passages,
        )
        assert "Context: Context text" in result
        assert "Evidence 1: Evidence 1 text" in result
        assert "Evidence 2: Evidence 2 text" in result
        assert "Statement: The sky is blue." in result

    def test_format_input_limits_passages(self, verifier):
        """Test that only top 3 passages are included."""
        passages = [
            Passage(f"Evidence {i}", f"source{i}", 1.0 - i * 0.1)
            for i in range(5)
        ]
        result = verifier._format_input(
            statement="Test statement",
            context="Test context",
            passages=passages,
        )
        # Should only include first 3 passages
        assert "Evidence 1:" in result
        assert "Evidence 2:" in result
        assert "Evidence 3:" in result
        assert "Evidence 4:" not in result


class TestParseModelOutput:
    """Tests for parsing model output."""

    def test_parse_supported(self, verifier):
        """Test parsing SUPPORTED verdict."""
        label, confidence = verifier._parse_model_output("supported", None)
        assert label == VerdictLabel.SUPPORTED
        assert confidence == 0.5  # Default when no logits

    def test_parse_refuted(self, verifier):
        """Test parsing REFUTED verdict."""
        label, confidence = verifier._parse_model_output("refuted", None)
        assert label == VerdictLabel.REFUTED

    def test_parse_not_enough_info(self, verifier):
        """Test parsing NOT_ENOUGH_INFO verdict."""
        label, confidence = verifier._parse_model_output("not enough info", None)
        assert label == VerdictLabel.NOT_ENOUGH_INFO

    def test_parse_with_logits(self, verifier):
        """Test confidence calculation from logits."""
        logits = torch.tensor([1.0, 2.0, 0.5])
        label, confidence = verifier._parse_model_output("supported", logits)
        assert label == VerdictLabel.SUPPORTED
        assert 0.0 <= confidence <= 1.0
        assert confidence > 0.5  # Should be high for clear logits

    def test_parse_case_insensitive(self, verifier):
        """Test that parsing is case-insensitive."""
        label1, _ = verifier._parse_model_output("SUPPORTED", None)
        label2, _ = verifier._parse_model_output("supported", None)
        label3, _ = verifier._parse_model_output("Supported", None)
        assert label1 == label2 == label3 == VerdictLabel.SUPPORTED


class TestVerifyStatement:
    """Tests for single statement verification."""

    def test_verify_statement_basic(self, verifier, mock_model, mock_tokenizer):
        """Test basic statement verification."""
        # Setup mock
        mock_output = MagicMock()
        mock_output.sequences = torch.tensor([[1, 2, 3]])
        mock_output.scores = [torch.tensor([[1.0, 2.0, 0.5]])]
        mock_model.generate.return_value = mock_output
        mock_tokenizer.decode.return_value = "SUPPORTED"

        # Verify statement
        verdict = verifier.verify_statement(
            statement="The Eiffel Tower is in Paris.",
            context="The Eiffel Tower is located in Paris, France.",
        )

        # Assertions
        assert isinstance(verdict, Verdict)
        assert verdict.label == VerdictLabel.SUPPORTED
        assert 0.0 <= verdict.confidence <= 1.0
        assert len(verdict.reasoning) > 0

    def test_verify_statement_with_passages(self, verifier, mock_model, mock_tokenizer):
        """Test verification with retrieved passages."""
        # Setup mock
        mock_output = MagicMock()
        mock_output.sequences = torch.tensor([[1, 2, 3]])
        mock_output.scores = [torch.tensor([[1.0, 2.0, 0.5]])]
        mock_model.generate.return_value = mock_output
        mock_tokenizer.decode.return_value = "SUPPORTED"

        passages = [
            Passage("The Eiffel Tower is in Paris.", "Wikipedia:Eiffel_Tower", 0.95),
        ]

        verdict = verifier.verify_statement(
            statement="The Eiffel Tower is in Paris.",
            context="Context about Paris.",
            passages=passages,
        )

        assert isinstance(verdict, Verdict)
        assert len(verdict.evidence) > 0
        assert passages[0].text in verdict.evidence

    def test_verify_statement_error_handling(self, verifier, mock_model):
        """Test error handling during verification."""
        # Make model raise an error
        mock_model.generate.side_effect = RuntimeError("Model error")

        verdict = verifier.verify_statement(
            statement="Test statement",
            context="Test context",
        )

        # Should return NOT_ENOUGH_INFO with zero confidence on error
        assert verdict.label == VerdictLabel.NOT_ENOUGH_INFO
        assert verdict.confidence == 0.0
        assert "Error during verification" in verdict.reasoning


class TestBatchVerify:
    """Tests for batch verification."""

    def test_batch_verify_basic(self, verifier, mock_model, mock_tokenizer):
        """Test basic batch verification."""
        # Setup mock
        mock_output = MagicMock()
        mock_output.sequences = torch.tensor([[1, 2, 3]])
        mock_output.scores = [torch.tensor([[1.0, 2.0, 0.5]])]
        mock_model.generate.return_value = mock_output
        mock_tokenizer.decode.return_value = "SUPPORTED"

        statements = ["Statement 1", "Statement 2", "Statement 3"]
        contexts = ["Context 1", "Context 2", "Context 3"]

        verdicts = verifier.batch_verify(statements, contexts)

        assert len(verdicts) == 3
        assert all(isinstance(v, Verdict) for v in verdicts)

    def test_batch_verify_with_passages(self, verifier, mock_model, mock_tokenizer):
        """Test batch verification with passages."""
        # Setup mock
        mock_output = MagicMock()
        mock_output.sequences = torch.tensor([[1, 2, 3]])
        mock_output.scores = [torch.tensor([[1.0, 2.0, 0.5]])]
        mock_model.generate.return_value = mock_output
        mock_tokenizer.decode.return_value = "SUPPORTED"

        statements = ["Statement 1", "Statement 2"]
        contexts = ["Context 1", "Context 2"]
        passages_list = [
            [Passage("Passage 1", "source1", 0.9)],
            [Passage("Passage 2", "source2", 0.8)],
        ]

        verdicts = verifier.batch_verify(statements, contexts, passages_list)

        assert len(verdicts) == 2

    def test_batch_verify_mismatched_lengths(self, verifier):
        """Test error handling for mismatched input lengths."""
        statements = ["Statement 1", "Statement 2"]
        contexts = ["Context 1"]  # Mismatched length

        with pytest.raises(ValueError, match="Mismatched lengths"):
            verifier.batch_verify(statements, contexts)

    def test_batch_verify_empty_lists(self, verifier):
        """Test batch verification with empty lists."""
        verdicts = verifier.batch_verify([], [])
        assert len(verdicts) == 0


class TestExtractStatements:
    """Tests for statement extraction."""

    def test_extract_statements_basic(self, verifier):
        """Test basic statement extraction."""
        text = "The sky is blue. Water is wet. Fire is hot."
        statements = verifier.extract_statements(text)

        assert len(statements) == 3
        assert "The sky is blue" in statements
        assert "Water is wet" in statements
        assert "Fire is hot" in statements

    def test_extract_statements_multiple_punctuation(self, verifier):
        """Test extraction with different punctuation."""
        text = "Is the sky blue? Yes it is! The water is wet."
        statements = verifier.extract_statements(text)

        assert len(statements) >= 2

    def test_extract_statements_filters_short(self, verifier):
        """Test that very short sentences are filtered out."""
        text = "The sky is blue. Hi. Water is wet."
        statements = verifier.extract_statements(text)

        # "Hi." should be filtered out (too short)
        assert len(statements) == 2
        assert all(len(s) > 10 for s in statements)

    def test_extract_statements_empty_text(self, verifier):
        """Test extraction from empty text."""
        statements = verifier.extract_statements("")
        assert len(statements) == 0


class TestVerifyText:
    """Tests for full text verification."""

    def test_verify_text_basic(self, verifier, mock_model, mock_tokenizer):
        """Test verifying a full text."""
        # Setup mock
        mock_output = MagicMock()
        mock_output.sequences = torch.tensor([[1, 2, 3]])
        mock_output.scores = [torch.tensor([[1.0, 2.0, 0.5]])]
        mock_model.generate.return_value = mock_output
        mock_tokenizer.decode.return_value = "SUPPORTED"

        candidate_text = "The Eiffel Tower is in Paris. It was built in 1889."
        source_context = "The Eiffel Tower is located in Paris, France. It was completed in 1889."

        verdicts = verifier.verify_text(candidate_text, source_context)

        assert len(verdicts) == 2
        assert all(isinstance(v, Verdict) for v in verdicts)

    def test_verify_text_with_passages(self, verifier, mock_model, mock_tokenizer):
        """Test verifying text with passages."""
        # Setup mock
        mock_output = MagicMock()
        mock_output.sequences = torch.tensor([[1, 2, 3]])
        mock_output.scores = [torch.tensor([[1.0, 2.0, 0.5]])]
        mock_model.generate.return_value = mock_output
        mock_tokenizer.decode.return_value = "SUPPORTED"

        passages = [Passage("Evidence text", "source", 0.9)]

        verdicts = verifier.verify_text(
            "The sky is blue. Water is wet.",
            "Context about nature.",
            passages=passages,
        )

        assert len(verdicts) == 2

    def test_verify_text_no_statements(self, verifier):
        """Test verifying text with no extractable statements."""
        verdicts = verifier.verify_text("", "Context")
        assert len(verdicts) == 0


class TestIntegration:
    """Integration tests for SpecializedVerifier."""

    def test_end_to_end_verification(self, verifier, mock_model, mock_tokenizer):
        """Test end-to-end verification workflow."""
        # Setup mock for multiple calls
        mock_output = MagicMock()
        mock_output.sequences = torch.tensor([[1, 2, 3]])
        mock_output.scores = [torch.tensor([[1.0, 2.0, 0.5]])]
        mock_model.generate.return_value = mock_output

        # Alternate between SUPPORTED and REFUTED
        mock_tokenizer.decode.side_effect = ["SUPPORTED", "REFUTED"]

        candidate = "Paris is the capital of France. London is the capital of Germany."
        source = "Paris is the capital of France. London is the capital of the UK."

        verdicts = verifier.verify_text(candidate, source)

        assert len(verdicts) == 2
        # First statement should be supported
        assert verdicts[0].label == VerdictLabel.SUPPORTED
        # Second statement should be refuted
        assert verdicts[1].label == VerdictLabel.REFUTED
