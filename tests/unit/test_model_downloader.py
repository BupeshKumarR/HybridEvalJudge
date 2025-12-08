"""
Unit tests for ModelDownloader.

Tests the model download utility including disk space checking,
SHA256 verification, and caching behavior.
"""

import hashlib
import shutil
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from llm_judge_auditor.components.model_downloader import ModelDownloader


class TestModelDownloader:
    """Test suite for ModelDownloader class."""

    def test_init_creates_cache_dir(self, tmp_path):
        """Test that initialization creates the cache directory."""
        cache_dir = tmp_path / "test_cache"
        downloader = ModelDownloader(cache_dir=cache_dir)
        
        assert cache_dir.exists()
        assert downloader.cache_dir == cache_dir

    def test_init_default_cache_dir(self):
        """Test that default cache directory is set correctly."""
        downloader = ModelDownloader()
        
        expected_dir = Path.home() / ".cache" / "llm-judge-auditor"
        assert downloader.cache_dir == expected_dir

    def test_check_disk_space_sufficient(self, tmp_path):
        """Test disk space check when sufficient space is available."""
        downloader = ModelDownloader(cache_dir=tmp_path)
        
        # Request a small amount of space (should always pass)
        result = downloader._check_disk_space(1024)  # 1KB
        
        assert result is True

    def test_check_disk_space_insufficient(self, tmp_path):
        """Test disk space check when insufficient space is available."""
        downloader = ModelDownloader(cache_dir=tmp_path)
        
        # Request an impossibly large amount of space
        result = downloader._check_disk_space(10**20)  # Exabytes
        
        assert result is False

    def test_compute_sha256(self, tmp_path):
        """Test SHA256 hash computation."""
        downloader = ModelDownloader(cache_dir=tmp_path)
        
        # Create a test file
        test_file = tmp_path / "test.txt"
        test_content = b"Hello, World!"
        test_file.write_bytes(test_content)
        
        # Compute hash
        computed_hash = downloader._compute_sha256(test_file)
        
        # Verify against expected hash
        expected_hash = hashlib.sha256(test_content).hexdigest()
        assert computed_hash == expected_hash

    def test_verify_model_integrity_missing_path(self, tmp_path):
        """Test model integrity verification with missing path."""
        downloader = ModelDownloader(cache_dir=tmp_path)
        
        non_existent_path = tmp_path / "non_existent"
        result = downloader._verify_model_integrity(non_existent_path)
        
        assert result is False

    def test_verify_model_integrity_missing_config(self, tmp_path):
        """Test model integrity verification with missing config.json."""
        downloader = ModelDownloader(cache_dir=tmp_path)
        
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        
        result = downloader._verify_model_integrity(model_dir)
        
        assert result is False

    def test_verify_model_integrity_missing_model_file(self, tmp_path):
        """Test model integrity verification with missing model files."""
        downloader = ModelDownloader(cache_dir=tmp_path)
        
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "config.json").write_text("{}")
        
        result = downloader._verify_model_integrity(model_dir)
        
        assert result is False

    def test_verify_model_integrity_valid_model(self, tmp_path):
        """Test model integrity verification with valid model."""
        downloader = ModelDownloader(cache_dir=tmp_path)
        
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "config.json").write_text("{}")
        (model_dir / "pytorch_model.bin").write_bytes(b"fake model")
        
        result = downloader._verify_model_integrity(model_dir)
        
        assert result is True

    def test_verify_model_integrity_with_hash_match(self, tmp_path):
        """Test model integrity verification with matching hash."""
        downloader = ModelDownloader(cache_dir=tmp_path)
        
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        
        config_content = b"{}"
        (model_dir / "config.json").write_bytes(config_content)
        (model_dir / "pytorch_model.bin").write_bytes(b"fake model")
        
        expected_hash = hashlib.sha256(config_content).hexdigest()
        result = downloader._verify_model_integrity(model_dir, expected_hash)
        
        assert result is True

    def test_verify_model_integrity_with_hash_mismatch(self, tmp_path):
        """Test model integrity verification with mismatched hash."""
        downloader = ModelDownloader(cache_dir=tmp_path)
        
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "config.json").write_bytes(b"{}")
        (model_dir / "pytorch_model.bin").write_bytes(b"fake model")
        
        wrong_hash = "0" * 64
        result = downloader._verify_model_integrity(model_dir, wrong_hash)
        
        assert result is False

    def test_get_model_path_exists(self, tmp_path):
        """Test getting model path when model exists."""
        downloader = ModelDownloader(cache_dir=tmp_path)
        
        # Create a valid model
        model_name = "test/model"
        model_dir = tmp_path / "test--model"
        model_dir.mkdir()
        (model_dir / "config.json").write_text("{}")
        (model_dir / "pytorch_model.bin").write_bytes(b"fake model")
        
        result = downloader.get_model_path(model_name)
        
        assert result == model_dir

    def test_get_model_path_not_exists(self, tmp_path):
        """Test getting model path when model doesn't exist."""
        downloader = ModelDownloader(cache_dir=tmp_path)
        
        result = downloader.get_model_path("non/existent")
        
        assert result is None

    def test_list_cached_models_empty(self, tmp_path):
        """Test listing cached models when cache is empty."""
        downloader = ModelDownloader(cache_dir=tmp_path)
        
        models = downloader.list_cached_models()
        
        assert models == []

    def test_list_cached_models_with_models(self, tmp_path):
        """Test listing cached models with valid models."""
        downloader = ModelDownloader(cache_dir=tmp_path)
        
        # Create two valid models
        for model_name in ["test--model1", "test--model2"]:
            model_dir = tmp_path / model_name
            model_dir.mkdir()
            (model_dir / "config.json").write_text("{}")
            (model_dir / "pytorch_model.bin").write_bytes(b"fake model")
        
        models = downloader.list_cached_models()
        
        assert len(models) == 2
        assert "test/model1" in models
        assert "test/model2" in models

    def test_clear_cache_specific_model(self, tmp_path):
        """Test clearing cache for a specific model."""
        downloader = ModelDownloader(cache_dir=tmp_path)
        
        # Create a model
        model_name = "test/model"
        model_dir = tmp_path / "test--model"
        model_dir.mkdir()
        (model_dir / "config.json").write_text("{}")
        
        downloader.clear_cache(model_name)
        
        assert not model_dir.exists()

    def test_clear_cache_all_models(self, tmp_path):
        """Test clearing all cached models."""
        downloader = ModelDownloader(cache_dir=tmp_path)
        
        # Create multiple models
        for i in range(3):
            model_dir = tmp_path / f"model{i}"
            model_dir.mkdir()
            (model_dir / "config.json").write_text("{}")
        
        downloader.clear_cache()
        
        # Cache dir should exist but be empty
        assert tmp_path.exists()
        assert len(list(tmp_path.iterdir())) == 0

    @patch("llm_judge_auditor.components.model_downloader.snapshot_download")
    def test_download_model_success(self, mock_snapshot, tmp_path):
        """Test successful model download."""
        downloader = ModelDownloader(cache_dir=tmp_path)
        
        model_name = "test/model"
        model_dir = tmp_path / "test--model"
        
        # Mock the download to create the model files
        def create_model_files(*args, **kwargs):
            model_dir.mkdir(exist_ok=True)
            (model_dir / "config.json").write_text("{}")
            (model_dir / "pytorch_model.bin").write_bytes(b"fake model")
            return str(model_dir)
        
        mock_snapshot.side_effect = create_model_files
        
        result = downloader.download_model(model_name)
        
        assert result == model_dir
        assert model_dir.exists()
        mock_snapshot.assert_called_once()

    @patch("llm_judge_auditor.components.model_downloader.snapshot_download")
    def test_download_model_already_cached(self, mock_snapshot, tmp_path):
        """Test that cached models are not re-downloaded."""
        downloader = ModelDownloader(cache_dir=tmp_path)
        
        model_name = "test/model"
        model_dir = tmp_path / "test--model"
        
        # Create a valid cached model
        model_dir.mkdir()
        (model_dir / "config.json").write_text("{}")
        (model_dir / "pytorch_model.bin").write_bytes(b"fake model")
        
        result = downloader.download_model(model_name)
        
        assert result == model_dir
        # Should not call download since model is cached
        mock_snapshot.assert_not_called()

    @patch("llm_judge_auditor.components.model_downloader.snapshot_download")
    def test_download_model_force_download(self, mock_snapshot, tmp_path):
        """Test force re-download of cached model."""
        downloader = ModelDownloader(cache_dir=tmp_path)
        
        model_name = "test/model"
        model_dir = tmp_path / "test--model"
        
        # Create a valid cached model
        model_dir.mkdir()
        (model_dir / "config.json").write_text("{}")
        (model_dir / "pytorch_model.bin").write_bytes(b"fake model")
        
        # Mock the download
        def create_model_files(*args, **kwargs):
            return str(model_dir)
        
        mock_snapshot.side_effect = create_model_files
        
        result = downloader.download_model(model_name, force_download=True)
        
        assert result == model_dir
        # Should call download even though model is cached
        mock_snapshot.assert_called_once()

    @patch("llm_judge_auditor.components.model_downloader.snapshot_download")
    def test_download_model_insufficient_disk_space(self, mock_snapshot, tmp_path):
        """Test download failure due to insufficient disk space."""
        downloader = ModelDownloader(cache_dir=tmp_path)
        
        # Mock disk space check to fail
        with patch.object(downloader, "_check_disk_space", return_value=False):
            with pytest.raises(ValueError, match="Insufficient disk space"):
                downloader.download_model("test/model")
        
        mock_snapshot.assert_not_called()

    @patch("llm_judge_auditor.components.model_downloader.snapshot_download")
    def test_download_model_network_error_retry(self, mock_snapshot, tmp_path):
        """Test download retry on network error."""
        from huggingface_hub.utils import HfHubHTTPError
        
        downloader = ModelDownloader(cache_dir=tmp_path)
        
        model_name = "test/model"
        model_dir = tmp_path / "test--model"
        
        # First two calls fail, third succeeds
        def side_effect(*args, **kwargs):
            if mock_snapshot.call_count < 3:
                raise HfHubHTTPError("Network error")
            model_dir.mkdir(exist_ok=True)
            (model_dir / "config.json").write_text("{}")
            (model_dir / "pytorch_model.bin").write_bytes(b"fake model")
            return str(model_dir)
        
        mock_snapshot.side_effect = side_effect
        
        result = downloader.download_model(model_name, max_retries=3)
        
        assert result == model_dir
        assert mock_snapshot.call_count == 3

    @patch("llm_judge_auditor.components.model_downloader.snapshot_download")
    def test_download_model_max_retries_exceeded(self, mock_snapshot, tmp_path):
        """Test download failure after max retries."""
        from huggingface_hub.utils import HfHubHTTPError
        
        downloader = ModelDownloader(cache_dir=tmp_path)
        
        # Always fail
        mock_snapshot.side_effect = HfHubHTTPError("Network error")
        
        with pytest.raises(RuntimeError, match="Failed to download model"):
            downloader.download_model("test/model", max_retries=2)
        
        assert mock_snapshot.call_count == 2
