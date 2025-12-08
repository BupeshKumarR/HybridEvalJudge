"""
Unit tests for the CLI module.

Tests the command-line interface functionality including argument parsing,
command execution, and file I/O operations.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from llm_judge_auditor.cli import (
    command_batch_evaluate,
    command_evaluate,
    create_parser,
    load_batch_requests,
    load_text_from_file,
)
from llm_judge_auditor.models import EvaluationRequest


class TestArgumentParser:
    """Test argument parser creation and validation."""

    def test_create_parser(self):
        """Test that parser is created with correct subcommands."""
        parser = create_parser()
        assert parser is not None
        assert parser.prog == "llm-judge-auditor"

    def test_evaluate_command_required_args(self):
        """Test that evaluate command requires source and candidate."""
        parser = create_parser()

        # Should fail without required arguments
        with pytest.raises(SystemExit):
            parser.parse_args(["evaluate"])

    def test_evaluate_command_with_args(self):
        """Test evaluate command with valid arguments."""
        parser = create_parser()
        args = parser.parse_args([
            "evaluate",
            "--source", "Source text",
            "--candidate", "Candidate text",
        ])

        assert args.command == "evaluate"
        assert args.source == "Source text"
        assert args.candidate == "Candidate text"
        assert args.preset == "balanced"  # default

    def test_evaluate_command_with_preset(self):
        """Test evaluate command with preset selection."""
        parser = create_parser()
        args = parser.parse_args([
            "evaluate",
            "--source", "Source text",
            "--candidate", "Candidate text",
            "--preset", "fast",
        ])

        assert args.preset == "fast"

    def test_batch_evaluate_command_required_args(self):
        """Test that batch-evaluate command requires input and output."""
        parser = create_parser()

        # Should fail without required arguments
        with pytest.raises(SystemExit):
            parser.parse_args(["batch-evaluate"])

    def test_batch_evaluate_command_with_args(self):
        """Test batch-evaluate command with valid arguments."""
        parser = create_parser()
        args = parser.parse_args([
            "batch-evaluate",
            "--input", "input.json",
            "--output", "output.json",
        ])

        assert args.command == "batch-evaluate"
        assert args.input == "input.json"
        assert args.output == "output.json"
        assert args.preset == "balanced"  # default


class TestFileLoading:
    """Test file loading utilities."""

    def test_load_text_from_file(self):
        """Test loading text from a file."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write("Test content")
            temp_path = f.name

        try:
            content = load_text_from_file(temp_path)
            assert content == "Test content"
        finally:
            Path(temp_path).unlink()

    def test_load_text_from_nonexistent_file(self):
        """Test loading from nonexistent file raises error."""
        with pytest.raises(FileNotFoundError):
            load_text_from_file("nonexistent_file.txt")

    def test_load_batch_requests(self):
        """Test loading batch requests from JSON file."""
        requests_data = [
            {
                "source_text": "Source 1",
                "candidate_output": "Candidate 1",
                "task": "factual_accuracy",
                "criteria": ["correctness"],
                "use_retrieval": False,
            },
            {
                "source_text": "Source 2",
                "candidate_output": "Candidate 2",
            },
        ]

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            json.dump(requests_data, f)
            temp_path = f.name

        try:
            requests = load_batch_requests(temp_path)
            assert len(requests) == 2
            assert isinstance(requests[0], EvaluationRequest)
            assert requests[0].source_text == "Source 1"
            assert requests[0].candidate_output == "Candidate 1"
            assert requests[1].source_text == "Source 2"
        finally:
            Path(temp_path).unlink()

    def test_load_batch_requests_invalid_format(self):
        """Test loading batch requests with invalid format."""
        # Not a list
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            json.dump({"not": "a list"}, f)
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="must contain a JSON array"):
                load_batch_requests(temp_path)
        finally:
            Path(temp_path).unlink()

    def test_load_batch_requests_missing_fields(self):
        """Test loading batch requests with missing required fields."""
        requests_data = [
            {
                "source_text": "Source 1",
                # Missing candidate_output
            },
        ]

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            json.dump(requests_data, f)
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="missing required field"):
                load_batch_requests(temp_path)
        finally:
            Path(temp_path).unlink()


class TestCommandEvaluate:
    """Test the evaluate command execution."""

    @patch("llm_judge_auditor.cli.EvaluationToolkit")
    @patch("llm_judge_auditor.cli.ReportGenerator")
    def test_command_evaluate_basic(self, mock_report_gen, mock_toolkit):
        """Test basic evaluate command execution."""
        # Setup mocks
        mock_result = MagicMock()
        mock_result.consensus_score = 85.5
        mock_result.report.confidence = 0.9
        mock_result.report.disagreement_level = 5.0
        mock_result.report.individual_scores = {"judge1": 85.0, "judge2": 86.0}
        mock_result.flagged_issues = []

        mock_toolkit_instance = MagicMock()
        mock_toolkit_instance.evaluate.return_value = mock_result
        mock_toolkit.return_value = mock_toolkit_instance

        # Create args
        parser = create_parser()
        args = parser.parse_args([
            "evaluate",
            "--source", "Source text",
            "--candidate", "Candidate text",
            "--preset", "fast",
        ])

        # Execute command
        exit_code = command_evaluate(args)

        # Verify
        assert exit_code == 0
        mock_toolkit_instance.evaluate.assert_called_once()

    @patch("llm_judge_auditor.cli.EvaluationToolkit")
    @patch("llm_judge_auditor.cli.ReportGenerator")
    def test_command_evaluate_with_output(self, mock_report_gen, mock_toolkit):
        """Test evaluate command with output file."""
        # Setup mocks
        mock_result = MagicMock()
        mock_result.consensus_score = 85.5
        mock_result.report.confidence = 0.9
        mock_result.report.disagreement_level = 5.0
        mock_result.report.individual_scores = {"judge1": 85.0}
        mock_result.flagged_issues = []

        mock_toolkit_instance = MagicMock()
        mock_toolkit_instance.evaluate.return_value = mock_result
        mock_toolkit.return_value = mock_toolkit_instance

        mock_generator = MagicMock()
        mock_report_gen.return_value = mock_generator

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.json"

            # Create args
            parser = create_parser()
            args = parser.parse_args([
                "evaluate",
                "--source", "Source text",
                "--candidate", "Candidate text",
                "--output", str(output_path),
            ])

            # Execute command
            exit_code = command_evaluate(args)

            # Verify
            assert exit_code == 0
            mock_generator.export_json.assert_called_once()


class TestCommandBatchEvaluate:
    """Test the batch-evaluate command execution."""

    @patch("llm_judge_auditor.cli.EvaluationToolkit")
    def test_command_batch_evaluate_basic(self, mock_toolkit):
        """Test basic batch-evaluate command execution."""
        # Create test input file
        requests_data = [
            {
                "source_text": "Source 1",
                "candidate_output": "Candidate 1",
            },
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.json"
            output_path = Path(tmpdir) / "output.json"

            with open(input_path, "w") as f:
                json.dump(requests_data, f)

            # Setup mocks
            mock_batch_result = MagicMock()
            mock_batch_result.metadata = {
                "total_requests": 1,
                "successful_evaluations": 1,
                "failed_evaluations": 0,
                "success_rate": 1.0,
            }
            mock_batch_result.statistics = {
                "mean": 85.0,
                "median": 85.0,
                "std": 0.0,
                "min": 85.0,
                "max": 85.0,
            }
            mock_batch_result.errors = []

            mock_toolkit_instance = MagicMock()
            mock_toolkit_instance.batch_evaluate.return_value = mock_batch_result
            mock_toolkit.return_value = mock_toolkit_instance

            # Create args
            parser = create_parser()
            args = parser.parse_args([
                "batch-evaluate",
                "--input", str(input_path),
                "--output", str(output_path),
            ])

            # Execute command
            exit_code = command_batch_evaluate(args)

            # Verify
            assert exit_code == 0
            mock_toolkit_instance.batch_evaluate.assert_called_once()
            mock_batch_result.save_to_file.assert_called_once_with(str(output_path))
