"""
Verifier Trainer for fine-tuning specialized fact-checking models.

This module provides the VerifierTrainer class for fine-tuning small models
(< 1B parameters) on fact-checking datasets like FEVER or custom data for
binary/ternary classification tasks.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EvalPrediction,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from llm_judge_auditor.models import VerdictLabel

logger = logging.getLogger(__name__)


@dataclass
class TrainingExample:
    """
    A single training example for verifier fine-tuning.

    Attributes:
        claim: The claim/statement to verify
        evidence: Context or evidence text
        label: Ground truth label (SUPPORTED, REFUTED, NOT_ENOUGH_INFO)
    """

    claim: str
    evidence: str
    label: VerdictLabel


class VerifierDataset(Dataset):
    """
    PyTorch Dataset for verifier training.

    This dataset handles tokenization and formatting of training examples
    for seq2seq models used in fact verification.
    """

    def __init__(
        self,
        examples: List[TrainingExample],
        tokenizer: Any,
        max_length: int = 512,
        target_max_length: int = 10,
    ):
        """
        Initialize the dataset.

        Args:
            examples: List of training examples
            tokenizer: Tokenizer for the model
            max_length: Maximum input sequence length
            target_max_length: Maximum target sequence length
        """
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.target_max_length = target_max_length

        # Create label mapping
        self.label_to_id = {
            VerdictLabel.SUPPORTED: 0,
            VerdictLabel.REFUTED: 1,
            VerdictLabel.NOT_ENOUGH_INFO: 2,
        }
        self.id_to_label = {v: k for k, v in self.label_to_id.items()}

    def __len__(self) -> int:
        """Return the number of examples."""
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single tokenized example.

        Args:
            idx: Index of the example

        Returns:
            Dictionary with input_ids, attention_mask, and labels
        """
        example = self.examples[idx]

        # Format input: "Evidence: {evidence} Claim: {claim}"
        input_text = f"Evidence: {example.evidence} Claim: {example.claim}"

        # Format target: label name
        target_text = example.label.value

        # Tokenize input
        inputs = self.tokenizer(
            input_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        # Tokenize target
        targets = self.tokenizer(
            target_text,
            max_length=self.target_max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "labels": targets["input_ids"].squeeze(),
        }


class VerifierTrainer:
    """
    Trainer for fine-tuning specialized verifier models.

    This class handles fine-tuning small models (< 1B parameters) on
    fact-checking datasets for binary or ternary classification.
    Supports FEVER format and custom training data.
    """

    def __init__(
        self,
        base_model: str = "google/flan-t5-base",
        output_dir: str = "models/custom_verifier",
        device: Optional[str] = None,
    ):
        """
        Initialize the VerifierTrainer.

        Args:
            base_model: HuggingFace model identifier to fine-tune
            output_dir: Directory to save trained models
            device: Device to train on (auto-detected if None)
        """
        self.base_model = base_model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        # Model and tokenizer (loaded lazily)
        self.model: Optional[Any] = None
        self.tokenizer: Optional[Any] = None

        logger.info(
            f"VerifierTrainer initialized (base_model={base_model}, "
            f"output_dir={output_dir}, device={self.device})"
        )

    def _load_model_and_tokenizer(self) -> None:
        """Load the base model and tokenizer."""
        if self.model is not None and self.tokenizer is not None:
            return

        logger.info(f"Loading base model: {self.base_model}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model,
            trust_remote_code=True,
        )

        # Load model
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.base_model,
            trust_remote_code=True,
        )

        logger.info(
            f"Model loaded: {sum(p.numel() for p in self.model.parameters()) / 1e6:.1f}M parameters"
        )

    def load_fever_dataset(self, filepath: str) -> List[TrainingExample]:
        """
        Load training data from FEVER format.

        FEVER format is JSONL with fields:
        - claim: The claim text
        - evidence: Evidence text (can be concatenated from multiple sentences)
        - label: "SUPPORTS", "REFUTES", or "NOT ENOUGH INFO"

        Args:
            filepath: Path to FEVER JSONL file

        Returns:
            List of TrainingExample objects

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is invalid
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Training data file not found: {filepath}")

        examples = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())

                    # Extract fields
                    claim = data.get("claim", "")
                    evidence = data.get("evidence", "")
                    label_str = data.get("label", "")

                    # Map FEVER labels to VerdictLabel (support various formats)
                    label_str_upper = label_str.upper()
                    if label_str_upper in ["SUPPORTS", "SUPPORTED", "SUPPORT"]:
                        label = VerdictLabel.SUPPORTED
                    elif label_str_upper in ["REFUTES", "REFUTED", "REFUTE"]:
                        label = VerdictLabel.REFUTED
                    elif label_str_upper in ["NOT ENOUGH INFO", "NOT_ENOUGH_INFO", "NOTENOUGHINFO", "NEI"]:
                        label = VerdictLabel.NOT_ENOUGH_INFO
                    else:
                        logger.warning(
                            f"Unknown label '{label_str}' at line {line_num}, skipping"
                        )
                        continue

                    # Validate required fields
                    if not claim or not evidence:
                        logger.warning(
                            f"Missing claim or evidence at line {line_num}, skipping"
                        )
                        continue

                    examples.append(
                        TrainingExample(claim=claim, evidence=evidence, label=label)
                    )

                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON at line {line_num}: {e}")
                    continue
                except Exception as e:
                    logger.warning(f"Error processing line {line_num}: {e}")
                    continue

        logger.info(f"Loaded {len(examples)} examples from {filepath}")
        return examples

    def load_custom_dataset(
        self, filepath: str, format: str = "jsonl"
    ) -> List[TrainingExample]:
        """
        Load training data from custom format.

        Custom format is JSONL with fields:
        - claim: The claim text
        - evidence: Evidence text
        - label: "SUPPORTED", "REFUTED", or "NOT_ENOUGH_INFO"

        Args:
            filepath: Path to custom data file
            format: File format ("jsonl" or "json")

        Returns:
            List of TrainingExample objects

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If format is unsupported
        """
        if format == "jsonl":
            # Same as FEVER format
            return self.load_fever_dataset(filepath)
        elif format == "json":
            # Load as single JSON array
            filepath = Path(filepath)
            if not filepath.exists():
                raise FileNotFoundError(f"Training data file not found: {filepath}")

            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            if not isinstance(data, list):
                raise ValueError("JSON file must contain an array of examples")

            examples = []
            for i, item in enumerate(data):
                try:
                    claim = item.get("claim", "")
                    evidence = item.get("evidence", "")
                    label_str = item.get("label", "")

                    # Map labels (support various formats)
                    label_str_upper = label_str.upper()
                    if label_str_upper in ["SUPPORTED", "SUPPORTS", "SUPPORT"]:
                        label = VerdictLabel.SUPPORTED
                    elif label_str_upper in ["REFUTED", "REFUTES", "REFUTE"]:
                        label = VerdictLabel.REFUTED
                    elif label_str_upper in ["NOT_ENOUGH_INFO", "NOT ENOUGH INFO", "NEI", "NOTENOUGHINFO"]:
                        label = VerdictLabel.NOT_ENOUGH_INFO
                    else:
                        logger.warning(f"Unknown label '{label_str}' at index {i}, skipping")
                        continue

                    if not claim or not evidence:
                        logger.warning(f"Missing claim or evidence at index {i}, skipping")
                        continue

                    examples.append(
                        TrainingExample(claim=claim, evidence=evidence, label=label)
                    )

                except Exception as e:
                    logger.warning(f"Error processing example {i}: {e}")
                    continue

            logger.info(f"Loaded {len(examples)} examples from {filepath}")
            return examples
        else:
            raise ValueError(f"Unsupported format: {format}")

    def _compute_metrics(self, eval_pred: EvalPrediction) -> Dict[str, float]:
        """
        Compute evaluation metrics.

        Args:
            eval_pred: Predictions and labels from evaluation

        Returns:
            Dictionary of metrics (accuracy, precision, recall, f1)
        """
        predictions, labels = eval_pred.predictions, eval_pred.label_ids

        # Decode predictions
        if isinstance(predictions, tuple):
            predictions = predictions[0]

        # Get predicted token IDs (argmax over vocabulary)
        pred_ids = predictions.argmax(axis=-1)

        # Decode to text
        pred_texts = self.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_texts = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Map to label IDs
        label_map = {
            "SUPPORTED": 0,
            "REFUTED": 1,
            "NOT_ENOUGH_INFO": 2,
            "NOT ENOUGH INFO": 2,
        }

        pred_labels = []
        true_labels = []

        for pred_text, label_text in zip(pred_texts, label_texts):
            pred_text = pred_text.strip().upper()
            label_text = label_text.strip().upper()

            # Map to IDs
            pred_id = label_map.get(pred_text, 2)  # Default to NEI
            true_id = label_map.get(label_text, 2)

            pred_labels.append(pred_id)
            true_labels.append(true_id)

        # Compute metrics
        accuracy = accuracy_score(true_labels, pred_labels)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, pred_labels, average="macro", zero_division=0
        )

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    def train(
        self,
        train_data: List[TrainingExample],
        eval_data: Optional[List[TrainingExample]] = None,
        num_epochs: int = 3,
        batch_size: int = 16,
        learning_rate: float = 5e-5,
        warmup_steps: int = 500,
        save_steps: int = 1000,
        eval_steps: int = 500,
        logging_steps: int = 100,
    ) -> Dict[str, Any]:
        """
        Fine-tune the verifier model.

        Args:
            train_data: Training examples
            eval_data: Optional evaluation examples
            num_epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate
            warmup_steps: Number of warmup steps
            save_steps: Save checkpoint every N steps
            eval_steps: Evaluate every N steps
            logging_steps: Log every N steps

        Returns:
            Training metrics and history

        Raises:
            ValueError: If train_data is empty
        """
        if not train_data:
            raise ValueError("Training data cannot be empty")

        # Load model and tokenizer
        self._load_model_and_tokenizer()

        logger.info(
            f"Starting training with {len(train_data)} examples "
            f"({len(eval_data) if eval_data else 0} eval examples)"
        )

        # Create datasets
        train_dataset = VerifierDataset(
            train_data, self.tokenizer, max_length=512, target_max_length=10
        )

        eval_dataset = None
        if eval_data:
            eval_dataset = VerifierDataset(
                eval_data, self.tokenizer, max_length=512, target_max_length=10
            )

        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            save_steps=save_steps,
            eval_steps=eval_steps if eval_dataset else None,
            logging_steps=logging_steps,
            eval_strategy="steps" if eval_dataset else "no",  # Changed from evaluation_strategy
            save_strategy="steps",
            load_best_model_at_end=True if eval_dataset else False,
            metric_for_best_model="f1" if eval_dataset else None,
            greater_is_better=True,
            save_total_limit=3,
            report_to="none",  # Disable wandb/tensorboard
            logging_dir=str(self.output_dir / "logs"),
        )

        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=self._compute_metrics if eval_dataset else None,
        )

        # Train
        logger.info("Starting training...")
        train_result = trainer.train()

        # Save final model
        logger.info(f"Saving final model to {self.output_dir}")
        trainer.save_model(str(self.output_dir / "final"))
        self.tokenizer.save_pretrained(str(self.output_dir / "final"))

        # Get training metrics
        metrics = train_result.metrics
        logger.info(f"Training completed. Metrics: {metrics}")

        return metrics

    def evaluate(
        self, eval_data: List[TrainingExample], batch_size: int = 16
    ) -> Dict[str, float]:
        """
        Evaluate the trained verifier on held-out data.

        Args:
            eval_data: Evaluation examples
            batch_size: Evaluation batch size

        Returns:
            Dictionary with accuracy, precision, recall, and f1 metrics

        Raises:
            ValueError: If eval_data is empty
            RuntimeError: If model is not loaded
        """
        if not eval_data:
            raise ValueError("Evaluation data cannot be empty")

        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call train() or load_model() first.")

        logger.info(f"Evaluating on {len(eval_data)} examples")

        # Create dataset
        eval_dataset = VerifierDataset(
            eval_data, self.tokenizer, max_length=512, target_max_length=10
        )

        # Create trainer for evaluation
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            per_device_eval_batch_size=batch_size,
            report_to="none",
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            compute_metrics=self._compute_metrics,
        )

        # Evaluate
        metrics = trainer.evaluate(eval_dataset)

        logger.info(f"Evaluation completed. Metrics: {metrics}")
        return metrics

    def save_model(self, save_path: str) -> None:
        """
        Save the trained model to disk.

        Args:
            save_path: Path to save the model

        Raises:
            RuntimeError: If model is not loaded
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call train() first.")

        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving model to {save_path}")
        self.model.save_pretrained(str(save_path))
        self.tokenizer.save_pretrained(str(save_path))

        logger.info("Model saved successfully")

    def load_model(self, model_path: str) -> None:
        """
        Load a trained model from disk.

        Args:
            model_path: Path to the saved model

        Raises:
            FileNotFoundError: If model path doesn't exist
        """
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model path not found: {model_path}")

        logger.info(f"Loading model from {model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            str(model_path),
            trust_remote_code=True,
        )

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            str(model_path),
            trust_remote_code=True,
        )

        self.model = self.model.to(self.device)
        self.model.eval()

        logger.info("Model loaded successfully")
