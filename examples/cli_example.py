"""
Example demonstrating CLI usage for the LLM Judge Auditor toolkit.

This script shows how to use the command-line interface for:
1. Single evaluations
2. Batch evaluations
3. Different output formats
"""

import json
import subprocess
import tempfile
from pathlib import Path


def example_single_evaluation():
    """
    Example: Single evaluation using CLI.

    This demonstrates evaluating a single candidate output against a source text.
    """
    print("=" * 60)
    print("Example 1: Single Evaluation")
    print("=" * 60)

    source = "Paris is the capital of France. It is located on the Seine River."
    candidate = "Paris is the capital of France and is situated on the Seine."

    # Using the CLI with direct text input
    cmd = [
        "llm-judge",
        "evaluate",
        "--source", source,
        "--candidate", candidate,
        "--preset", "fast",
        "--no-retrieval",  # Disable retrieval for faster demo
    ]

    print("\nCommand:")
    print(" ".join(cmd))
    print("\nNote: This would normally run the evaluation, but requires models to be downloaded.")
    print("For actual usage, run the command above in your terminal.")


def example_single_evaluation_with_files():
    """
    Example: Single evaluation using file inputs.

    This demonstrates using files for source and candidate text.
    """
    print("\n" + "=" * 60)
    print("Example 2: Single Evaluation with Files")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create source file
        source_file = Path(tmpdir) / "source.txt"
        source_file.write_text("The Eiffel Tower was completed in 1889 for the World's Fair.")

        # Create candidate file
        candidate_file = Path(tmpdir) / "candidate.txt"
        candidate_file.write_text("The Eiffel Tower was built in 1889 for the Paris Exposition.")

        # Create output path
        output_file = Path(tmpdir) / "report.json"

        cmd = [
            "llm-judge",
            "evaluate",
            "--source-file", str(source_file),
            "--candidate-file", str(candidate_file),
            "--preset", "balanced",
            "--output", str(output_file),
            "--output-format", "json",
        ]

        print("\nCommand:")
        print(" ".join(cmd))
        print(f"\nSource file: {source_file}")
        print(f"Candidate file: {candidate_file}")
        print(f"Output file: {output_file}")


def example_batch_evaluation():
    """
    Example: Batch evaluation using CLI.

    This demonstrates evaluating multiple candidate outputs in batch.
    """
    print("\n" + "=" * 60)
    print("Example 3: Batch Evaluation")
    print("=" * 60)

    # Create batch input file
    batch_requests = [
        {
            "source_text": "Paris is the capital of France.",
            "candidate_output": "Paris is the capital of Germany.",
            "task": "factual_accuracy",
            "criteria": ["correctness"],
            "use_retrieval": False,
        },
        {
            "source_text": "The Earth orbits the Sun.",
            "candidate_output": "The Earth revolves around the Sun in an elliptical orbit.",
            "task": "factual_accuracy",
            "criteria": ["correctness"],
            "use_retrieval": False,
        },
        {
            "source_text": "Water boils at 100 degrees Celsius at sea level.",
            "candidate_output": "Water boils at 100°C under standard atmospheric pressure.",
            "task": "factual_accuracy",
            "criteria": ["correctness"],
            "use_retrieval": False,
        },
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create input file
        input_file = Path(tmpdir) / "batch_input.json"
        with open(input_file, "w") as f:
            json.dump(batch_requests, f, indent=2)

        # Create output path
        output_file = Path(tmpdir) / "batch_results.json"

        cmd = [
            "llm-judge",
            "batch-evaluate",
            "--input", str(input_file),
            "--output", str(output_file),
            "--preset", "fast",
            "--continue-on-error",
        ]

        print("\nCommand:")
        print(" ".join(cmd))
        print(f"\nInput file: {input_file}")
        print(f"Output file: {output_file}")
        print(f"\nBatch contains {len(batch_requests)} evaluation requests")


def example_with_config_file():
    """
    Example: Using a custom configuration file.

    This demonstrates using a YAML configuration file instead of presets.
    """
    print("\n" + "=" * 60)
    print("Example 4: Using Custom Configuration File")
    print("=" * 60)

    config_content = """
verifier_model: "MiniCheck/flan-t5-base-finetuned"
judge_models:
  - "microsoft/Phi-3-mini-4k-instruct"
quantize: true
device: "auto"
enable_retrieval: false
aggregation_strategy: "mean"
batch_size: 1
max_length: 512
"""

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create config file
        config_file = Path(tmpdir) / "custom_config.yaml"
        config_file.write_text(config_content)

        cmd = [
            "llm-judge",
            "evaluate",
            "--source", "Source text here",
            "--candidate", "Candidate text here",
            "--config", str(config_file),
        ]

        print("\nCommand:")
        print(" ".join(cmd))
        print(f"\nConfig file: {config_file}")
        print("\nConfig content:")
        print(config_content)


def example_output_formats():
    """
    Example: Different output formats.

    This demonstrates exporting reports in JSON, Markdown, and text formats.
    """
    print("\n" + "=" * 60)
    print("Example 5: Different Output Formats")
    print("=" * 60)

    formats = ["json", "markdown", "text"]

    for fmt in formats:
        cmd = [
            "llm-judge",
            "evaluate",
            "--source", "Source text",
            "--candidate", "Candidate text",
            "--preset", "fast",
            "--output", f"report.{fmt}",
            "--output-format", fmt,
        ]

        print(f"\n{fmt.upper()} format:")
        print(" ".join(cmd))


def main():
    """Run all CLI examples."""
    print("\n" + "=" * 60)
    print("LLM Judge Auditor - CLI Examples")
    print("=" * 60)
    print("\nThese examples demonstrate how to use the command-line interface.")
    print("Note: Actual execution requires models to be downloaded first.")
    print()

    example_single_evaluation()
    example_single_evaluation_with_files()
    example_batch_evaluation()
    example_with_config_file()
    example_output_formats()

    print("\n" + "=" * 60)
    print("For more information, run: llm-judge --help")
    print("=" * 60)


if __name__ == "__main__":
    main()
