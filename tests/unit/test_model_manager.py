"""
Unit tests for ModelManager.

Tests the model manager including model loading, lazy loading,
quantization, and error handling.
"""

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch

from llm_judge_auditor.components.model_manager import ModelInfo, ModelManager
from llm_judge_auditor.config import DeviceType, ToolkitConfig


class TestModelManager:
    """Test suite for ModelManager class."""

    @pytest.fixture
    def mock_device_manager(self):
        """Create a mock DeviceManager."""
        mock_dm = Mock()
        mock_dm.auto_configure = Mock(side_effect=lambda config: config)
        return mock_dm

    @pytest.fixture
    def mock_model_downloader(self, tmp_path):
        """Create a mock ModelDownloader."""
        mock_md = Mock()
        mock_md.get_model_path = Mock(return_value=tmp_path / "model")
        mock_md.download_model = Mock(return_value=tmp_path / "model")
        return mock_md

    @pytest.fixture
    def test_config(self):
        """Create a test configuration."""
        return ToolkitConfig(
            verifier_model="test/verifier",
            judge_models=["test/judge1", "test/judge2"],
            quantize=False,
            device=DeviceType.CPU,
        )

    def test_init(self, test_config, mock_device_manager, mock_model_downloader):
        """Test ModelManager initialization."""
        manager = ModelManager(
            config=test_config,
            device_manager=mock_device_manager,
            model_downloader=mock_model_downloader,
        )
        
        assert manager.config == test_config
        assert manager.device_manager == mock_device_manager
        assert manager.model_downloader == mock_model_downloader
        assert manager._verifier_model is None
        assert manager._verifier_tokenizer is None
        assert len(manager._judge_models) == 0
        assert len(manager._judge_tokenizers) == 0

    def test_init_auto_configure_device(self, mock_device_manager, mock_model_downloader):
        """Test that AUTO device triggers auto-configuration."""
        config = ToolkitConfig(
            verifier_model="test/verifier",
            judge_models=["test/judge1"],
            device=DeviceType.AUTO,
        )
        
        manager = ModelManager(
            config=config,
            device_manager=mock_device_manager,
            model_downloader=mock_model_downloader,
        )
        
        mock_device_manager.auto_configure.assert_called_once()

    def test_get_device_string_cuda(self, test_config, mock_device_manager, mock_model_downloader):
        """Test device string for CUDA."""
        test_config.device = DeviceType.CUDA
        manager = ModelManager(test_config, mock_device_manager, mock_model_downloader)
        
        assert manager._get_device_string() == "cuda"

    def test_get_device_string_mps(self, test_config, mock_device_manager, mock_model_downloader):
        """Test device string for MPS."""
        test_config.device = DeviceType.MPS
        manager = ModelManager(test_config, mock_device_manager, mock_model_downloader)
        
        assert manager._get_device_string() == "mps"

    def test_get_device_string_cpu(self, test_config, mock_device_manager, mock_model_downloader):
        """Test device string for CPU."""
        test_config.device = DeviceType.CPU
        manager = ModelManager(test_config, mock_device_manager, mock_model_downloader)
        
        assert manager._get_device_string() == "cpu"

    def test_get_quantization_config_disabled(self, test_config, mock_device_manager, mock_model_downloader):
        """Test quantization config when disabled."""
        test_config.quantize = False
        manager = ModelManager(test_config, mock_device_manager, mock_model_downloader)
        
        config = manager._get_quantization_config()
        
        assert config is None

    def test_get_quantization_config_enabled(self, test_config, mock_device_manager, mock_model_downloader):
        """Test quantization config when enabled."""
        test_config.quantize = True
        manager = ModelManager(test_config, mock_device_manager, mock_model_downloader)
        
        config = manager._get_quantization_config()
        
        assert config is not None
        assert config.load_in_8bit is True

    def test_estimate_model_size(self, test_config, mock_device_manager, mock_model_downloader):
        """Test model size estimation."""
        manager = ModelManager(test_config, mock_device_manager, mock_model_downloader)
        
        # Test various model sizes
        assert manager._estimate_model_size("model-3b") == 3 * 1024**3
        assert manager._estimate_model_size("model-7b") == 7 * 1024**3
        assert manager._estimate_model_size("model-8b") == 8 * 1024**3
        assert manager._estimate_model_size("model-base") == 1 * 1024**3
        assert manager._estimate_model_size("model-large") == 3 * 1024**3
        assert manager._estimate_model_size("unknown-model") == 5 * 1024**3

    def test_load_verifier_no_model_specified(self, mock_device_manager, mock_model_downloader):
        """Test loading verifier with no model specified."""
        config = ToolkitConfig(
            verifier_model="",
            judge_models=["test/judge1"],
        )
        manager = ModelManager(config, mock_device_manager, mock_model_downloader)
        
        with pytest.raises(ValueError, match="No verifier model specified"):
            manager.load_verifier()

    @patch("llm_judge_auditor.components.model_manager.AutoTokenizer")
    @patch("llm_judge_auditor.components.model_manager.AutoModelForSeq2SeqLM")
    def test_load_verifier_success(
        self,
        mock_model_class,
        mock_tokenizer_class,
        test_config,
        mock_device_manager,
        mock_model_downloader,
        tmp_path,
    ):
        """Test successful verifier loading."""
        # Setup mocks
        mock_model = Mock()
        mock_model.eval = Mock(return_value=mock_model)
        mock_model.to = Mock(return_value=mock_model)
        mock_model.parameters = Mock(return_value=[torch.zeros(1000)])
        mock_model_class.from_pretrained = Mock(return_value=mock_model)
        
        mock_tokenizer = Mock()
        mock_tokenizer_class.from_pretrained = Mock(return_value=mock_tokenizer)
        
        mock_model_downloader.get_model_path = Mock(return_value=tmp_path / "model")
        
        manager = ModelManager(test_config, mock_device_manager, mock_model_downloader)
        
        model, tokenizer = manager.load_verifier()
        
        assert model == mock_model
        assert tokenizer == mock_tokenizer
        assert manager._verifier_model == mock_model
        assert manager._verifier_tokenizer == mock_tokenizer
        mock_model.eval.assert_called_once()

    @patch("llm_judge_auditor.components.model_manager.AutoTokenizer")
    @patch("llm_judge_auditor.components.model_manager.AutoModelForSeq2SeqLM")
    def test_load_verifier_cached(
        self,
        mock_model_class,
        mock_tokenizer_class,
        test_config,
        mock_device_manager,
        mock_model_downloader,
    ):
        """Test that cached verifier is returned without reloading."""
        # Setup mocks
        mock_model = Mock()
        mock_tokenizer = Mock()
        
        manager = ModelManager(test_config, mock_device_manager, mock_model_downloader)
        manager._verifier_model = mock_model
        manager._verifier_tokenizer = mock_tokenizer
        
        model, tokenizer = manager.load_verifier()
        
        assert model == mock_model
        assert tokenizer == mock_tokenizer
        # Should not call from_pretrained since model is cached
        mock_model_class.from_pretrained.assert_not_called()
        mock_tokenizer_class.from_pretrained.assert_not_called()

    @patch("llm_judge_auditor.components.model_manager.AutoTokenizer")
    @patch("llm_judge_auditor.components.model_manager.AutoModelForSeq2SeqLM")
    def test_load_verifier_download_if_not_cached(
        self,
        mock_model_class,
        mock_tokenizer_class,
        test_config,
        mock_device_manager,
        mock_model_downloader,
        tmp_path,
    ):
        """Test that model is downloaded if not cached."""
        # Setup mocks
        mock_model = Mock()
        mock_model.eval = Mock(return_value=mock_model)
        mock_model.to = Mock(return_value=mock_model)
        mock_model.parameters = Mock(return_value=[torch.zeros(1000)])
        mock_model_class.from_pretrained = Mock(return_value=mock_model)
        
        mock_tokenizer = Mock()
        mock_tokenizer_class.from_pretrained = Mock(return_value=mock_tokenizer)
        
        # Model not cached initially
        mock_model_downloader.get_model_path = Mock(return_value=None)
        mock_model_downloader.download_model = Mock(return_value=tmp_path / "model")
        
        manager = ModelManager(test_config, mock_device_manager, mock_model_downloader)
        
        model, tokenizer = manager.load_verifier()
        
        # Should call download since model not cached
        mock_model_downloader.download_model.assert_called_once()

    @patch("llm_judge_auditor.components.model_manager.AutoTokenizer")
    @patch("llm_judge_auditor.components.model_manager.AutoModelForCausalLM")
    def test_load_judge_ensemble_success(
        self,
        mock_model_class,
        mock_tokenizer_class,
        test_config,
        mock_device_manager,
        mock_model_downloader,
        tmp_path,
    ):
        """Test successful judge ensemble loading."""
        # Setup mocks
        mock_model = Mock()
        mock_model.eval = Mock(return_value=mock_model)
        mock_model.to = Mock(return_value=mock_model)
        mock_model.parameters = Mock(return_value=[torch.zeros(1000)])
        mock_model_class.from_pretrained = Mock(return_value=mock_model)
        
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "<eos>"
        mock_tokenizer_class.from_pretrained = Mock(return_value=mock_tokenizer)
        
        mock_model_downloader.get_model_path = Mock(return_value=tmp_path / "model")
        
        manager = ModelManager(test_config, mock_device_manager, mock_model_downloader)
        
        ensemble = manager.load_judge_ensemble()
        
        assert len(ensemble) == 2
        assert "test/judge1" in ensemble
        assert "test/judge2" in ensemble
        # Should set pad_token if not set
        assert mock_tokenizer.pad_token == "<eos>"

    def test_load_judge_ensemble_no_models_specified(self, mock_device_manager, mock_model_downloader):
        """Test loading judge ensemble with no models specified."""
        config = ToolkitConfig(
            verifier_model="test/verifier",
            judge_models=[],
        )
        manager = ModelManager(config, mock_device_manager, mock_model_downloader)
        
        with pytest.raises(ValueError, match="No judge models specified"):
            manager.load_judge_ensemble()

    @patch("llm_judge_auditor.components.model_manager.AutoTokenizer")
    @patch("llm_judge_auditor.components.model_manager.AutoModelForCausalLM")
    def test_load_judge_ensemble_partial_failure(
        self,
        mock_model_class,
        mock_tokenizer_class,
        test_config,
        mock_device_manager,
        mock_model_downloader,
        tmp_path,
    ):
        """Test judge ensemble loading with partial failures."""
        # First model succeeds, second fails
        def from_pretrained_side_effect(*args, **kwargs):
            if mock_model_class.from_pretrained.call_count == 1:
                mock_model = Mock()
                mock_model.eval = Mock(return_value=mock_model)
                mock_model.to = Mock(return_value=mock_model)
                mock_model.parameters = Mock(return_value=[torch.zeros(1000)])
                return mock_model
            else:
                raise RuntimeError("Model load failed")
        
        mock_model_class.from_pretrained = Mock(side_effect=from_pretrained_side_effect)
        
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = "<pad>"
        mock_tokenizer_class.from_pretrained = Mock(return_value=mock_tokenizer)
        
        mock_model_downloader.get_model_path = Mock(return_value=tmp_path / "model")
        
        manager = ModelManager(test_config, mock_device_manager, mock_model_downloader)
        
        ensemble = manager.load_judge_ensemble()
        
        # Should have one successful model
        assert len(ensemble) == 1
        assert "test/judge1" in ensemble

    @patch("llm_judge_auditor.components.model_manager.AutoTokenizer")
    @patch("llm_judge_auditor.components.model_manager.AutoModelForCausalLM")
    def test_load_judge_ensemble_all_fail(
        self,
        mock_model_class,
        mock_tokenizer_class,
        test_config,
        mock_device_manager,
        mock_model_downloader,
    ):
        """Test judge ensemble loading when all models fail."""
        mock_model_class.from_pretrained = Mock(side_effect=RuntimeError("Model load failed"))
        mock_tokenizer_class.from_pretrained = Mock(return_value=Mock())
        
        manager = ModelManager(test_config, mock_device_manager, mock_model_downloader)
        
        with pytest.raises(RuntimeError, match="Failed to load any judge models"):
            manager.load_judge_ensemble()

    def test_verify_models_ready_no_models_loaded(self, test_config, mock_device_manager, mock_model_downloader):
        """Test verify_models_ready when no models are loaded."""
        manager = ModelManager(test_config, mock_device_manager, mock_model_downloader)
        
        result = manager.verify_models_ready()
        
        assert result is False

    def test_verify_models_ready_all_loaded(self, test_config, mock_device_manager, mock_model_downloader):
        """Test verify_models_ready when all models are loaded."""
        manager = ModelManager(test_config, mock_device_manager, mock_model_downloader)
        
        # Mock loaded models
        manager._verifier_model = Mock()
        manager._verifier_tokenizer = Mock()
        manager._judge_models = {"test/judge1": Mock(), "test/judge2": Mock()}
        manager._judge_tokenizers = {"test/judge1": Mock(), "test/judge2": Mock()}
        
        # Add model info
        manager._model_info = {
            "test/verifier": ModelInfo(
                name="test/verifier",
                model_type="verifier",
                device="cpu",
                quantized=False,
                is_ready=True,
            ),
            "test/judge1": ModelInfo(
                name="test/judge1",
                model_type="judge",
                device="cpu",
                quantized=False,
                is_ready=True,
            ),
            "test/judge2": ModelInfo(
                name="test/judge2",
                model_type="judge",
                device="cpu",
                quantized=False,
                is_ready=True,
            ),
        }
        
        result = manager.verify_models_ready()
        
        assert result is True

    def test_get_model_info(self, test_config, mock_device_manager, mock_model_downloader):
        """Test getting model info."""
        manager = ModelManager(test_config, mock_device_manager, mock_model_downloader)
        
        # Add some model info
        test_info = ModelInfo(
            name="test/model",
            model_type="verifier",
            device="cpu",
            quantized=False,
            parameters=1000000,
            is_ready=True,
        )
        manager._model_info["test/model"] = test_info
        
        info = manager.get_model_info()
        
        assert "test/model" in info
        assert info["test/model"] == test_info

    def test_unload_models(self, test_config, mock_device_manager, mock_model_downloader):
        """Test unloading all models."""
        manager = ModelManager(test_config, mock_device_manager, mock_model_downloader)
        
        # Mock loaded models
        manager._verifier_model = Mock()
        manager._verifier_tokenizer = Mock()
        manager._judge_models = {"test/judge1": Mock()}
        manager._judge_tokenizers = {"test/judge1": Mock()}
        manager._model_info = {
            "test/model": ModelInfo(
                name="test/model",
                model_type="verifier",
                device="cpu",
                quantized=False,
                is_ready=True,
            )
        }
        
        manager.unload_models()
        
        assert manager._verifier_model is None
        assert manager._verifier_tokenizer is None
        assert len(manager._judge_models) == 0
        assert len(manager._judge_tokenizers) == 0
        assert manager._model_info["test/model"].is_ready is False

    def test_get_verifier_loaded(self, test_config, mock_device_manager, mock_model_downloader):
        """Test getting loaded verifier."""
        manager = ModelManager(test_config, mock_device_manager, mock_model_downloader)
        
        mock_model = Mock()
        mock_tokenizer = Mock()
        manager._verifier_model = mock_model
        manager._verifier_tokenizer = mock_tokenizer
        
        result = manager.get_verifier()
        
        assert result == (mock_model, mock_tokenizer)

    def test_get_verifier_not_loaded(self, test_config, mock_device_manager, mock_model_downloader):
        """Test getting verifier when not loaded."""
        manager = ModelManager(test_config, mock_device_manager, mock_model_downloader)
        
        result = manager.get_verifier()
        
        assert result is None

    def test_get_judge_loaded(self, test_config, mock_device_manager, mock_model_downloader):
        """Test getting specific loaded judge."""
        manager = ModelManager(test_config, mock_device_manager, mock_model_downloader)
        
        mock_model = Mock()
        mock_tokenizer = Mock()
        manager._judge_models["test/judge1"] = mock_model
        manager._judge_tokenizers["test/judge1"] = mock_tokenizer
        
        result = manager.get_judge("test/judge1")
        
        assert result == (mock_model, mock_tokenizer)

    def test_get_judge_not_loaded(self, test_config, mock_device_manager, mock_model_downloader):
        """Test getting judge when not loaded."""
        manager = ModelManager(test_config, mock_device_manager, mock_model_downloader)
        
        result = manager.get_judge("test/judge1")
        
        assert result is None

    def test_get_all_judges(self, test_config, mock_device_manager, mock_model_downloader):
        """Test getting all loaded judges."""
        manager = ModelManager(test_config, mock_device_manager, mock_model_downloader)
        
        mock_model1 = Mock()
        mock_tokenizer1 = Mock()
        mock_model2 = Mock()
        mock_tokenizer2 = Mock()
        
        manager._judge_models["test/judge1"] = mock_model1
        manager._judge_tokenizers["test/judge1"] = mock_tokenizer1
        manager._judge_models["test/judge2"] = mock_model2
        manager._judge_tokenizers["test/judge2"] = mock_tokenizer2
        
        result = manager.get_all_judges()
        
        assert len(result) == 2
        assert result["test/judge1"] == (mock_model1, mock_tokenizer1)
        assert result["test/judge2"] == (mock_model2, mock_tokenizer2)
