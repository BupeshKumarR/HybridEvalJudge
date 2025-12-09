"""
Example: Fine-tuning a Specialized Verifier

This example demonstrates how to fine-tune a small model for fact-checking
using the VerifierTrainer class. It shows:
1. Loading training data in FEVER format
2. Fine-tuning a base model
3. Evaluating the trained model
4. Saving and loading the model
"""

import json
import tempfile
from pathlib import Path

from llm_judge_auditor.components.verifier_trainer import (
    VerifierTrainer,
    TrainingExample,
)
from llm_judge_auditor.models import VerdictLabel


def create_sample_fever_data(filepath: str, num_examples: int = 100):
    """
    Create sample FEVER-format training data.

    Args:
        filepath: Path to save the JSONL file
        num_examples: Number of examples to generate
    """
    print(f"Creating {num_examples} sample training examples...")

    examples = [
        {
            "claim": "The Eiffel Tower is located in Paris, France.",
            "evidence": "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France.",
            "label": "SUPPORTS",
        },
        {
            "claim": "The Eiffel Tower was completed in 1900.",
            "evidence": "The Eiffel Tower was completed in 1889 as the entrance arch to the 1889 World's Fair.",
            "label": "REFUTES",
        },
        {
            "claim": "The Eiffel Tower is 500 meters tall.",
            "evidence": "The Eiffel Tower is a famous landmark in Paris.",
            "label": "NOT ENOUGH INFO",
        },
        {
            "claim": "Python is a programming language.",
            "evidence": "Python is an interpreted, high-level, general-purpose programming language.",
            "label": "SUPPORTS",
        },
        {
            "claim": "Python was created in 1995.",
            "evidence": "Python was created by Guido van Rossum and first released in 1991.",
            "label": "REFUTES",
        },
    ]

    with open(filepath, "w", encoding="utf-8") as f:
        for i in range(num_examples):
            # Cycle through examples
            example = examples[i % len(examples)]
            f.write(json.dumps(example) + "\n")

    print(f"Sample data saved to {filepath}")


def example_basic_training():
    """Example: Basic model training."""
    print("\n" + "=" * 60)
    print("Example 1: Basic Model Training")
    print("=" * 60)

    # Create temporary directory for this example
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)

        # Create sample training data
        train_file = temp_dir / "train.jsonl"
        eval_file = temp_dir / "eval.jsonl"

        create_sample_fever_data(str(train_file), num_examples=50)
        create_sample_fever_data(str(eval_file), num_examples=20)

        # Initialize trainer
        print("\nInitializing VerifierTrainer...")
        trainer = VerifierTrainer(
            base_model="google/flan-t5-small",  # Small model for demo
            output_dir=str(temp_dir / "model_output"),
        )

        # Load training data
        print("\nLoading training data...")
        train_data = trainer.load_fever_dataset(str(train_file))
        eval_data = trainer.load_fever_dataset(str(eval_file))

        print(f"Loaded {len(train_data)} training examples")
        print(f"Loaded {len(eval_data)} evaluation examples")

        # Train the model
        print("\nStarting training...")
        print("Note: This is a demo with minimal epochs. Real training needs more data and epochs.")

        metrics = trainer.train(
            train_data=train_data,
            eval_data=eval_data,
            num_epochs=1,  # Minimal for demo
            batch_size=4,
            learning_rate=5e-5,
            save_steps=100,
            eval_steps=50,
        )

        print("\nTraining completed!")
        print(f"Training metrics: {metrics}")

        # Evaluate the model
        print("\nEvaluating trained model...")
        eval_metrics = trainer.evaluate(eval_data, batch_size=4)

        print("\nEvaluation Results:")
        print(f"  Accuracy:  {eval_metrics['eval_accuracy']:.3f}")
        print(f"  Precision: {eval_metrics['eval_precision']:.3f}")
        print(f"  Recall:    {eval_metrics['eval_recall']:.3f}")
        print(f"  F1 Score:  {eval_metrics['eval_f1']:.3f}")


def example_custom_data_format():
    """Example: Training with custom data format."""
    print("\n" + "=" * 60)
    print("Example 2: Training with Custom Data Format")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)

        # Create custom JSON format data
        custom_data = [
            {
                "claim": "Water boils at 100 degrees Celsius at sea level.",
                "evidence": "At standard atmospheric pressure, water boils at 100°C (212°F).",
                "label": "SUPPORTED",
            },
            {
                "claim": "Water freezes at 10 degrees Celsius.",
                "evidence": "Water freezes at 0 degrees Celsius (32°F) at standard pressure.",
                "label": "REFUTED",
            },
            {
                "claim": "Ice is less dense than liquid water.",
                "evidence": "Water is a unique substance with many interesting properties.",
                "label": "NOT_ENOUGH_INFO",
            },
        ]

        custom_file = temp_dir / "custom_data.json"
        with open(custom_file, "w", encoding="utf-8") as f:
            json.dump(custom_data, f, indent=2)

        print(f"\nCreated custom data file: {custom_file}")

        # Initialize trainer
        trainer = VerifierTrainer(
            base_model="google/flan-t5-small",
            output_dir=str(temp_dir / "custom_model"),
        )

        # Load custom data
        print("\nLoading custom format data...")
        train_data = trainer.load_custom_dataset(str(custom_file), format="json")

        print(f"Loaded {len(train_data)} examples")
        for i, example in enumerate(train_data[:3], 1):
            print(f"\nExample {i}:")
            print(f"  Claim: {example.claim[:60]}...")
            print(f"  Label: {example.label.value}")


def example_save_and_load():
    """Example: Saving and loading trained models."""
    print("\n" + "=" * 60)
    print("Example 3: Saving and Loading Models")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)

        # Create minimal training data
        train_file = temp_dir / "train.jsonl"
        create_sample_fever_data(str(train_file), num_examples=20)

        # Train a model
        print("\nTraining a model...")
        trainer = VerifierTrainer(
            base_model="google/flan-t5-small",
            output_dir=str(temp_dir / "model_output"),
        )

        train_data = trainer.load_fever_dataset(str(train_file))
        trainer.train(
            train_data=train_data,
            num_epochs=1,
            batch_size=4,
        )

        # Save the model
        save_path = temp_dir / "saved_verifier"
        print(f"\nSaving model to {save_path}...")
        trainer.save_model(str(save_path))

        print("Model saved successfully!")

        # Load the model in a new trainer
        print("\nLoading model in a new trainer...")
        new_trainer = VerifierTrainer(output_dir=str(temp_dir / "new_output"))
        new_trainer.load_model(str(save_path))

        print("Model loaded successfully!")
        print(f"Model device: {new_trainer.device}")


def example_programmatic_data():
    """Example: Creating training data programmatically."""
    print("\n" + "=" * 60)
    print("Example 4: Creating Training Data Programmatically")
    print("=" * 60)

    # Create training examples directly
    training_examples = [
        TrainingExample(
            claim="The Earth orbits the Sun.",
            evidence="The Earth revolves around the Sun in an elliptical orbit.",
            label=VerdictLabel.SUPPORTED,
        ),
        TrainingExample(
            claim="The Sun orbits the Earth.",
            evidence="The Earth revolves around the Sun, not the other way around.",
            label=VerdictLabel.REFUTED,
        ),
        TrainingExample(
            claim="The Earth has multiple moons.",
            evidence="The Earth is the third planet from the Sun.",
            label=VerdictLabel.NOT_ENOUGH_INFO,
        ),
    ]

    print(f"\nCreated {len(training_examples)} training examples programmatically")

    for i, example in enumerate(training_examples, 1):
        print(f"\nExample {i}:")
        print(f"  Claim:    {example.claim}")
        print(f"  Evidence: {example.evidence[:50]}...")
        print(f"  Label:    {example.label.value}")

    print("\nThese examples can be used directly with trainer.train()")


def example_label_distribution():
    """Example: Analyzing label distribution in training data."""
    print("\n" + "=" * 60)
    print("Example 5: Analyzing Label Distribution")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)

        # Create training data
        train_file = temp_dir / "train.jsonl"
        create_sample_fever_data(str(train_file), num_examples=100)

        # Load data
        trainer = VerifierTrainer()
        train_data = trainer.load_fever_dataset(str(train_file))

        # Analyze distribution
        label_counts = {
            VerdictLabel.SUPPORTED: 0,
            VerdictLabel.REFUTED: 0,
            VerdictLabel.NOT_ENOUGH_INFO: 0,
        }

        for example in train_data:
            label_counts[example.label] += 1

        print(f"\nLabel Distribution (total: {len(train_data)} examples):")
        for label, count in label_counts.items():
            percentage = (count / len(train_data)) * 100
            print(f"  {label.value:20s}: {count:3d} ({percentage:5.1f}%)")

        print("\nNote: Balanced datasets typically perform better.")
        print("Consider balancing your data if one class is over-represented.")


def main():
    """Run all examples."""
    print("=" * 60)
    print("VerifierTrainer Examples")
    print("=" * 60)
    print("\nThese examples demonstrate fine-tuning specialized verifiers")
    print("for fact-checking tasks using FEVER and custom data formats.")

    try:
        # Run examples
        example_basic_training()
        example_custom_data_format()
        example_save_and_load()
        example_programmatic_data()
        example_label_distribution()

        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
