"""
Unit tests for VerifierTrainer.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import pytest
import torch

from llm_judge_auditor.components.verifier_trainer import (
    VerifierTrainer,
    TrainingExample,
    VerifierDataset,
)
from llm_judge_auditor.models import VerdictLabel


@pytest.fixture
def sample_training_data():
    """Create sample training data."""
    return [
        TrainingExample(
            claim="The Eiffel Tower is in Paris.",
            evidence="The Eiffel Tower is located in Paris, France.",
            label=VerdictLabel.SUPPORTED,
        ),
        TrainingExample(
            claim="The Eiffel Tower was built in 1900.",
            evidence="The Eiffel Tower was completed in 1889.",
            label=VerdictLabel.REFUTED,
        ),
        TrainingExample(
            claim="The Eiffel Tower is 500 meters tall.",
            evidence="The Eiffel Tower is a famous landmark in Paris.",
            label=VerdictLabel.NOT_ENOUGH_INFO,
        ),
    ]


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer."""
    tokenizer = Mock()
    tokenizer.return_value = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "attention_mask": torch.tensor([[1, 1, 1]]),
    }
    tokenizer.batch_decode = Mock(
        return_value=["SUPPORTED", "REFUTED", "NOT_ENOUGH_INFO"]
    )
    tokenizer.pad_token = "<pad>"
    tokenizer.eos_token = "</s>"
    return tokenizer


@pytest.fixture
def mock_model():
    """Create a mock model."""
    model = Mock()
    model.parameters = Mock(return_value=[torch.tensor([1.0])])
    model.eval = Mock()
    model.to = Mock(return_value=model)
    model.save_pretrained = Mock()
    return model


class TestTrainingExample:
    """Tests for TrainingExample dataclass."""

    def test_training_example_creation(self):
        """Test creating a training example."""
        example = TrainingExample(
            claim="Test claim",
            evidence="Test evidence",
            label=VerdictLabel.SUPPORTED,
        )

        assert example.claim == "Test claim"
        assert example.evidence == "Test evidence"
        assert example.label == VerdictLabel.SUPPORTED


class TestVerifierDataset:
    """Tests for VerifierDataset."""

    def test_dataset_length(self, sample_training_data, mock_tokenizer):
        """Test dataset length."""
        dataset = VerifierDataset(sample_training_data, mock_tokenizer)
        assert len(dataset) == 3

    def test_dataset_getitem(self, sample_training_data, mock_tokenizer):
        """Test getting an item from dataset."""
        dataset = VerifierDataset(sample_training_data, mock_tokenizer)

        # Mock tokenizer to return proper tensors
        mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3, 0, 0]]),
            "attention_mask": torch.tensor([[1, 1, 1, 0, 0]]),
        }

        item = dataset[0]

        assert "input_ids" in item
        assert "attention_mask" in item
        assert "labels" in item
        assert isinstance(item["input_ids"], torch.Tensor)

    def test_label_mapping(self, sample_training_data, mock_tokenizer):
        """Test label to ID mapping."""
        dataset = VerifierDataset(sample_training_data, mock_tokenizer)

        assert dataset.label_to_id[VerdictLabel.SUPPORTED] == 0
        assert dataset.label_to_id[VerdictLabel.REFUTED] == 1
        assert dataset.label_to_id[VerdictLabel.NOT_ENOUGH_INFO] == 2

        assert dataset.id_to_label[0] == VerdictLabel.SUPPORTED
        assert dataset.id_to_label[1] == VerdictLabel.REFUTED
        assert dataset.id_to_label[2] == VerdictLabel.NOT_ENOUGH_INFO


class TestVerifierTrainer:
    """Tests for VerifierTrainer."""

    def test_initialization(self):
        """Test trainer initialization."""
        trainer = VerifierTrainer(
            base_model="google/flan-t5-base",
            output_dir="test_output",
        )

        assert trainer.base_model == "google/flan-t5-base"
        assert trainer.output_dir == Path("test_output")
        assert trainer.device in ["cpu", "cuda", "mps"]

    def test_initialization_with_device(self):
        """Test trainer initialization with specific device."""
        trainer = VerifierTrainer(
            base_model="google/flan-t5-base",
            output_dir="test_output",
            device="cpu",
        )

        assert trainer.device == "cpu"

    def test_load_fever_dataset_file_not_found(self):
        """Test loading FEVER dataset with non-existent file."""
        trainer = VerifierTrainer()

        with pytest.raises(FileNotFoundError):
            trainer.load_fever_dataset("nonexistent_file.jsonl")

    def test_load_fever_dataset_valid(self):
        """Test loading valid FEVER dataset."""
        trainer = VerifierTrainer()

        # Create temporary FEVER file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(
                json.dumps(
                    {
                        "claim": "Test claim 1",
                        "evidence": "Test evidence 1",
                        "label": "SUPPORTS",
                    }
                )
                + "\n"
            )
            f.write(
                json.dumps(
                    {
                        "claim": "Test claim 2",
                        "evidence": "Test evidence 2",
                        "label": "REFUTES",
                    }
                )
                + "\n"
            )
            f.write(
                json.dumps(
                    {
                        "claim": "Test claim 3",
                        "evidence": "Test evidence 3",
                        "label": "NOT ENOUGH INFO",
                    }
                )
                + "\n"
            )
            temp_path = f.name

        try:
            examples = trainer.load_fever_dataset(temp_path)

            assert len(examples) == 3
            assert examples[0].claim == "Test claim 1"
            assert examples[0].label == VerdictLabel.SUPPORTED
            assert examples[1].label == VerdictLabel.REFUTED
            assert examples[2].label == VerdictLabel.NOT_ENOUGH_INFO
        finally:
            Path(temp_path).unlink()

    def test_load_fever_dataset_invalid_json(self):
        """Test loading FEVER dataset with invalid JSON."""
        trainer = VerifierTrainer()

        # Create temporary file with invalid JSON
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write("invalid json\n")
            f.write(
                json.dumps(
                    {
                        "claim": "Valid claim",
                        "evidence": "Valid evidence",
                        "label": "SUPPORTS",
                    }
                )
                + "\n"
            )
            temp_path = f.name

        try:
            examples = trainer.load_fever_dataset(temp_path)
            # Should skip invalid line and load valid one
            assert len(examples) == 1
        finally:
            Path(temp_path).unlink()

    def test_load_fever_dataset_missing_fields(self):
        """Test loading FEVER dataset with missing fields."""
        trainer = VerifierTrainer()

        # Create temporary file with missing fields
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps({"claim": "Test claim"}) + "\n")  # Missing evidence
            f.write(
                json.dumps(
                    {
                        "claim": "Valid claim",
                        "evidence": "Valid evidence",
                        "label": "SUPPORTS",
                    }
                )
                + "\n"
            )
            temp_path = f.name

        try:
            examples = trainer.load_fever_dataset(temp_path)
            # Should skip invalid line and load valid one
            assert len(examples) == 1
        finally:
            Path(temp_path).unlink()

    def test_load_custom_dataset_jsonl(self):
        """Test loading custom dataset in JSONL format."""
        trainer = VerifierTrainer()

        # Create temporary JSONL file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(
                json.dumps(
                    {
                        "claim": "Custom claim",
                        "evidence": "Custom evidence",
                        "label": "SUPPORTED",
                    }
                )
                + "\n"
            )
            temp_path = f.name

        try:
            examples = trainer.load_custom_dataset(temp_path, format="jsonl")
            assert len(examples) == 1
            assert examples[0].claim == "Custom claim"
        finally:
            Path(temp_path).unlink()

    def test_load_custom_dataset_json(self):
        """Test loading custom dataset in JSON format."""
        trainer = VerifierTrainer()

        # Create temporary JSON file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(
                [
                    {
                        "claim": "Claim 1",
                        "evidence": "Evidence 1",
                        "label": "SUPPORTED",
                    },
                    {
                        "claim": "Claim 2",
                        "evidence": "Evidence 2",
                        "label": "REFUTED",
                    },
                ],
                f,
            )
            temp_path = f.name

        try:
            examples = trainer.load_custom_dataset(temp_path, format="json")
            assert len(examples) == 2
            assert examples[0].claim == "Claim 1"
            assert examples[1].claim == "Claim 2"
        finally:
            Path(temp_path).unlink()

    def test_load_custom_dataset_unsupported_format(self):
        """Test loading custom dataset with unsupported format."""
        trainer = VerifierTrainer()

        with pytest.raises(ValueError, match="Unsupported format"):
            trainer.load_custom_dataset("dummy.txt", format="txt")

    @patch("llm_judge_auditor.components.verifier_trainer.AutoTokenizer")
    @patch("llm_judge_auditor.components.verifier_trainer.AutoModelForSeq2SeqLM")
    def test_load_model_and_tokenizer(
        self, mock_model_class, mock_tokenizer_class, mock_model, mock_tokenizer
    ):
        """Test loading model and tokenizer."""
        mock_tokenizer_class.from_pretrained = Mock(return_value=mock_tokenizer)
        mock_model_class.from_pretrained = Mock(return_value=mock_model)

        trainer = VerifierTrainer()
        trainer._load_model_and_tokenizer()

        assert trainer.model is not None
        assert trainer.tokenizer is not None
        mock_tokenizer_class.from_pretrained.assert_called_once()
        mock_model_class.from_pretrained.assert_called_once()

    @patch("llm_judge_auditor.components.verifier_trainer.TrainingArguments")
    @patch("llm_judge_auditor.components.verifier_trainer.Trainer")
    @patch("llm_judge_auditor.components.verifier_trainer.AutoTokenizer")
    @patch("llm_judge_auditor.components.verifier_trainer.AutoModelForSeq2SeqLM")
    def test_train(
        self,
        mock_model_class,
        mock_tokenizer_class,
        mock_trainer_class,
        mock_training_args_class,
        sample_training_data,
        mock_model,
        mock_tokenizer,
    ):
        """Test training the model."""
        mock_tokenizer_class.from_pretrained = Mock(return_value=mock_tokenizer)
        mock_model_class.from_pretrained = Mock(return_value=mock_model)

        # Mock training arguments
        mock_training_args = Mock()
        mock_training_args_class.return_value = mock_training_args

        # Mock trainer
        mock_trainer_instance = Mock()
        mock_trainer_instance.train = Mock(
            return_value=Mock(metrics={"train_loss": 0.5})
        )
        mock_trainer_instance.save_model = Mock()
        mock_trainer_class.return_value = mock_trainer_instance

        trainer = VerifierTrainer()

        with tempfile.TemporaryDirectory() as temp_dir:
            trainer.output_dir = Path(temp_dir)
            metrics = trainer.train(
                train_data=sample_training_data,
                num_epochs=1,
                batch_size=2,
            )

            assert "train_loss" in metrics
            mock_trainer_instance.train.assert_called_once()

    def test_train_empty_data(self):
        """Test training with empty data."""
        trainer = VerifierTrainer()

        with pytest.raises(ValueError, match="Training data cannot be empty"):
            trainer.train(train_data=[])

    @patch("llm_judge_auditor.components.verifier_trainer.TrainingArguments")
    @patch("llm_judge_auditor.components.verifier_trainer.Trainer")
    def test_evaluate(
        self, mock_trainer_class, mock_training_args_class, sample_training_data, mock_model, mock_tokenizer
    ):
        """Test evaluating the model."""
        trainer = VerifierTrainer()
        trainer.model = mock_model
        trainer.tokenizer = mock_tokenizer

        # Mock training arguments
        mock_training_args = Mock()
        mock_training_args_class.return_value = mock_training_args

        # Mock trainer
        mock_trainer_instance = Mock()
        mock_trainer_instance.evaluate = Mock(
            return_value={"eval_accuracy": 0.8, "eval_f1": 0.75}
        )
        mock_trainer_class.return_value = mock_trainer_instance

        metrics = trainer.evaluate(sample_training_data)

        assert "eval_accuracy" in metrics
        assert "eval_f1" in metrics
        mock_trainer_instance.evaluate.assert_called_once()

    def test_evaluate_empty_data(self, mock_model, mock_tokenizer):
        """Test evaluating with empty data."""
        trainer = VerifierTrainer()
        trainer.model = mock_model
        trainer.tokenizer = mock_tokenizer

        with pytest.raises(ValueError, match="Evaluation data cannot be empty"):
            trainer.evaluate(eval_data=[])

    def test_evaluate_model_not_loaded(self):
        """Test evaluating without loading model."""
        trainer = VerifierTrainer()

        with pytest.raises(RuntimeError, match="Model not loaded"):
            trainer.evaluate(eval_data=[Mock()])

    def test_save_model(self, mock_model, mock_tokenizer):
        """Test saving the model."""
        trainer = VerifierTrainer()
        trainer.model = mock_model
        trainer.tokenizer = mock_tokenizer

        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "saved_model"
            trainer.save_model(str(save_path))

            mock_model.save_pretrained.assert_called_once()
            mock_tokenizer.save_pretrained.assert_called_once()

    def test_save_model_not_loaded(self):
        """Test saving model without loading it first."""
        trainer = VerifierTrainer()

        with pytest.raises(RuntimeError, match="Model not loaded"):
            trainer.save_model("dummy_path")

    @patch("llm_judge_auditor.components.verifier_trainer.AutoTokenizer")
    @patch("llm_judge_auditor.components.verifier_trainer.AutoModelForSeq2SeqLM")
    def test_load_model(
        self, mock_model_class, mock_tokenizer_class, mock_model, mock_tokenizer
    ):
        """Test loading a saved model."""
        mock_tokenizer_class.from_pretrained = Mock(return_value=mock_tokenizer)
        mock_model_class.from_pretrained = Mock(return_value=mock_model)

        trainer = VerifierTrainer()

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a dummy model directory
            model_path = Path(temp_dir) / "model"
            model_path.mkdir()

            trainer.load_model(str(model_path))

            assert trainer.model is not None
            assert trainer.tokenizer is not None
            mock_model.eval.assert_called_once()

    def test_load_model_not_found(self):
        """Test loading model from non-existent path."""
        trainer = VerifierTrainer()

        with pytest.raises(FileNotFoundError):
            trainer.load_model("nonexistent_path")

    def test_compute_metrics(self, mock_tokenizer):
        """Test computing evaluation metrics."""
        trainer = VerifierTrainer()
        trainer.tokenizer = mock_tokenizer

        # Mock predictions and labels
        mock_tokenizer.batch_decode = Mock(
            side_effect=[
                ["SUPPORTED", "REFUTED", "NOT_ENOUGH_INFO"],  # predictions
                ["SUPPORTED", "REFUTED", "NOT_ENOUGH_INFO"],  # labels
            ]
        )

        # Create mock EvalPrediction
        eval_pred = Mock()
        eval_pred.predictions = torch.randn(3, 10, 100)  # (batch, seq_len, vocab)
        eval_pred.label_ids = torch.randint(0, 100, (3, 10))

        metrics = trainer._compute_metrics(eval_pred)

        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert 0 <= metrics["accuracy"] <= 1
