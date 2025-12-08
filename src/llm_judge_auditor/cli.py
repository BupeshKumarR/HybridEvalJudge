"""
Command-line interface for the LLM Judge Auditor toolkit.

This module provides CLI commands for:
- Single evaluations (evaluate)
- Batch processing (batch-evaluate)
- Configuration management (preset selection, config files)
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from llm_judge_auditor.components.report_generator import ReportGenerator
from llm_judge_auditor.config import ToolkitConfig
from llm_judge_auditor.evaluation_toolkit import EvaluationToolkit
from llm_judge_auditor.models import EvaluationRequest


def create_parser() -> argparse.ArgumentParser:
    """
    Create the argument parser for the CLI.

    Returns:
        Configured ArgumentParser instance
    """
    parser = argparse.ArgumentParser(
        prog="llm-judge-auditor",
        description="Hybrid LLM Evaluation Toolkit - Evaluate LLM outputs for factual accuracy, hallucinations, and bias",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Add subcommands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Evaluate command
    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate a single candidate output against a source text",
    )
    evaluate_parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Source text (reference document or context)",
    )
    evaluate_parser.add_argument(
        "--candidate",
        type=str,
        required=True,
        help="Candidate output to evaluate",
    )
    evaluate_parser.add_argument(
        "--source-file",
        type=str,
        help="Path to file containing source text (alternative to --source)",
    )
    evaluate_parser.add_argument(
        "--candidate-file",
        type=str,
        help="Path to file containing candidate output (alternative to --candidate)",
    )
    evaluate_parser.add_argument(
        "--config",
        type=str,
        help="Path to YAML configuration file",
    )
    evaluate_parser.add_argument(
        "--preset",
        type=str,
        choices=["fast", "balanced", "strict", "research"],
        default="balanced",
        help="Preset configuration to use (default: balanced)",
    )
    evaluate_parser.add_argument(
        "--output",
        type=str,
        help="Path to save evaluation report (JSON format)",
    )
    evaluate_parser.add_argument(
        "--output-format",
        type=str,
        choices=["json", "markdown", "text"],
        default="json",
        help="Output format for the report (default: json)",
    )
    evaluate_parser.add_argument(
        "--no-retrieval",
        action="store_true",
        help="Disable retrieval-augmented verification",
    )

    # Batch evaluate command
    batch_parser = subparsers.add_parser(
        "batch-evaluate",
        help="Evaluate multiple candidate outputs in batch",
    )
    batch_parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to JSON file containing batch evaluation requests",
    )
    batch_parser.add_argument(
        "--config",
        type=str,
        help="Path to YAML configuration file",
    )
    batch_parser.add_argument(
        "--preset",
        type=str,
        choices=["fast", "balanced", "strict", "research"],
        default="balanced",
        help="Preset configuration to use (default: balanced)",
    )
    batch_parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save batch evaluation results (JSON format)",
    )
    batch_parser.add_argument(
        "--continue-on-error",
        action="store_true",
        default=True,
        help="Continue processing if an evaluation fails (default: True)",
    )

    return parser


def load_text_from_file(filepath: str) -> str:
    """
    Load text content from a file.

    Args:
        filepath: Path to the text file

    Returns:
        Text content from the file

    Raises:
        FileNotFoundError: If file doesn't exist
        IOError: If file cannot be read
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load_batch_requests(filepath: str) -> List[EvaluationRequest]:
    """
    Load batch evaluation requests from a JSON file.

    Expected JSON format:
    [
        {
            "source_text": "...",
            "candidate_output": "...",
            "task": "factual_accuracy",
            "criteria": ["correctness"],
            "use_retrieval": false
        },
        ...
    ]

    Args:
        filepath: Path to JSON file containing requests

    Returns:
        List of EvaluationRequest objects

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If JSON format is invalid
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Batch input file must contain a JSON array of requests")

    requests = []
    for idx, item in enumerate(data):
        try:
            request = EvaluationRequest(
                source_text=item["source_text"],
                candidate_output=item["candidate_output"],
                task=item.get("task", "factual_accuracy"),
                criteria=item.get("criteria", ["correctness"]),
                use_retrieval=item.get("use_retrieval", False),
            )
            requests.append(request)
        except KeyError as e:
            raise ValueError(f"Request {idx} missing required field: {e}")

    return requests


def command_evaluate(args: argparse.Namespace) -> int:
    """
    Execute the evaluate command.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success, 1 for error)
    """
    try:
        # Load source text
        if args.source_file:
            source_text = load_text_from_file(args.source_file)
        else:
            source_text = args.source

        # Load candidate output
        if args.candidate_file:
            candidate_output = load_text_from_file(args.candidate_file)
        else:
            candidate_output = args.candidate

        # Load configuration
        if args.config:
            print(f"Loading configuration from: {args.config}")
            config = ToolkitConfig.from_yaml(Path(args.config))
        else:
            print(f"Using preset: {args.preset}")
            config = ToolkitConfig.from_preset(args.preset)

        # Override retrieval setting if specified
        if args.no_retrieval:
            config.enable_retrieval = False

        # Initialize toolkit
        print("Initializing evaluation toolkit...")
        toolkit = EvaluationToolkit(config)

        # Run evaluation
        print("Running evaluation...")
        result = toolkit.evaluate(
            source_text=source_text,
            candidate_output=candidate_output,
            use_retrieval=config.enable_retrieval,
        )

        # Display results
        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)
        print(f"Consensus Score: {result.consensus_score:.2f}/100")
        print(f"Confidence: {result.report.confidence:.2f}")
        print(f"Disagreement Level: {result.report.disagreement_level:.2f}")
        print("\nIndividual Judge Scores:")
        for model_name, score in result.report.individual_scores.items():
            print(f"  - {model_name}: {score:.2f}/100")

        if result.flagged_issues:
            print(f"\nFlagged Issues: {len(result.flagged_issues)}")
            for idx, issue in enumerate(result.flagged_issues[:5], 1):  # Show first 5
                print(f"  {idx}. [{issue.severity.value.upper()}] {issue.type.value}: {issue.description[:80]}...")

        # Save report if output path specified
        if args.output:
            print(f"\nSaving report to: {args.output}")
            generator = ReportGenerator()

            if args.output_format == "json":
                generator.export_json(result.report, args.output)
            elif args.output_format == "markdown":
                generator.export_markdown(result.report, args.output)
            elif args.output_format == "text":
                generator.export_text(result.report, args.output)

            print("Report saved successfully")

        print("\nEvaluation complete!")
        return 0

    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        return 1


def command_batch_evaluate(args: argparse.Namespace) -> int:
    """
    Execute the batch-evaluate command.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success, 1 for error)
    """
    try:
        # Load batch requests
        print(f"Loading batch requests from: {args.input}")
        requests = load_batch_requests(args.input)
        print(f"Loaded {len(requests)} evaluation requests")

        # Load configuration
        if args.config:
            print(f"Loading configuration from: {args.config}")
            config = ToolkitConfig.from_yaml(Path(args.config))
        else:
            print(f"Using preset: {args.preset}")
            config = ToolkitConfig.from_preset(args.preset)

        # Initialize toolkit
        print("Initializing evaluation toolkit...")
        toolkit = EvaluationToolkit(config)

        # Run batch evaluation
        print(f"Running batch evaluation ({len(requests)} requests)...")
        batch_result = toolkit.batch_evaluate(
            requests=requests,
            continue_on_error=args.continue_on_error,
        )

        # Display results
        print("\n" + "=" * 60)
        print("BATCH EVALUATION RESULTS")
        print("=" * 60)
        print(f"Total Requests: {batch_result.metadata['total_requests']}")
        print(f"Successful: {batch_result.metadata['successful_evaluations']}")
        print(f"Failed: {batch_result.metadata['failed_evaluations']}")
        print(f"Success Rate: {batch_result.metadata['success_rate']:.1%}")
        print("\nStatistics:")
        print(f"  Mean Score: {batch_result.statistics['mean']:.2f}")
        print(f"  Median Score: {batch_result.statistics['median']:.2f}")
        print(f"  Std Dev: {batch_result.statistics['std']:.2f}")
        print(f"  Min Score: {batch_result.statistics['min']:.2f}")
        print(f"  Max Score: {batch_result.statistics['max']:.2f}")

        if batch_result.errors:
            print(f"\nErrors ({len(batch_result.errors)}):")
            for error in batch_result.errors[:5]:  # Show first 5 errors
                print(f"  - Request {error['request_index']}: {error['error_type']} - {error['error_message'][:80]}...")

        # Save results
        print(f"\nSaving batch results to: {args.output}")
        batch_result.save_to_file(args.output)
        print("Batch results saved successfully")

        print("\nBatch evaluation complete!")
        return 0

    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        return 1


def main():
    """Main entry point for the CLI."""
    parser = create_parser()
    args = parser.parse_args()

    # If no command specified, show help
    if not args.command:
        parser.print_help()
        return 0

    # Execute the appropriate command
    if args.command == "evaluate":
        return command_evaluate(args)
    elif args.command == "batch-evaluate":
        return command_batch_evaluate(args)
    else:
        print(f"Unknown command: {args.command}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
