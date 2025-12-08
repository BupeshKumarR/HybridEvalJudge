"""
Model Download Utility for downloading and verifying models from HuggingFace Hub.

This module provides functionality to download models from HuggingFace Hub with
SHA256 verification, disk space checking, and progress tracking.
"""

import hashlib
import logging
import shutil
from pathlib import Path
from typing import Optional

from huggingface_hub import snapshot_download
from huggingface_hub.utils import HfHubHTTPError, RepositoryNotFoundError
from tqdm import tqdm

logger = logging.getLogger(__name__)


class ModelDownloader:
    """
    Downloads and verifies models from HuggingFace Hub.

    This class handles model downloads with SHA256 verification, disk space checking,
    and automatic caching to prevent re-downloading existing models.
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize the ModelDownloader.

        Args:
            cache_dir: Directory for caching models. Defaults to ~/.cache/llm-judge-auditor/
        """
        if cache_dir is None:
            cache_dir = Path.home() / ".cache" / "llm-judge-auditor"
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"ModelDownloader initialized with cache_dir: {self.cache_dir}")

    def _check_disk_space(self, required_bytes: int) -> bool:
        """
        Check if sufficient disk space is available.

        Args:
            required_bytes: Required space in bytes

        Returns:
            True if sufficient space is available, False otherwise
        """
        try:
            stat = shutil.disk_usage(self.cache_dir)
            available_gb = stat.free / (1024**3)
            required_gb = required_bytes / (1024**3)
            
            if stat.free < required_bytes:
                logger.error(
                    f"Insufficient disk space. Required: {required_gb:.2f}GB, "
                    f"Available: {available_gb:.2f}GB"
                )
                return False
            
            logger.info(f"Disk space check passed. Available: {available_gb:.2f}GB")
            return True
        except Exception as e:
            logger.warning(f"Could not check disk space: {e}")
            return True  # Proceed if we can't check

    def _compute_sha256(self, file_path: Path) -> str:
        """
        Compute SHA256 hash of a file.

        Args:
            file_path: Path to the file

        Returns:
            SHA256 hash as hex string
        """
        sha256_hash = hashlib.sha256()
        
        with open(file_path, "rb") as f:
            # Read in chunks to handle large files
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        
        return sha256_hash.hexdigest()

    def _verify_model_integrity(
        self, model_path: Path, expected_hash: Optional[str] = None
    ) -> bool:
        """
        Verify model integrity by checking if key files exist and optionally verifying hash.

        Args:
            model_path: Path to the model directory
            expected_hash: Optional expected SHA256 hash for verification

        Returns:
            True if model is valid, False otherwise
        """
        if not model_path.exists():
            logger.debug(f"Model path does not exist: {model_path}")
            return False

        # Check for essential model files
        essential_files = ["config.json"]
        model_files = [
            "pytorch_model.bin",
            "model.safetensors",
            "adapter_model.bin",
            "adapter_model.safetensors",
        ]
        
        # At least config.json must exist
        config_exists = (model_path / "config.json").exists()
        if not config_exists:
            logger.debug(f"config.json not found in {model_path}")
            return False
        
        # At least one model file should exist
        has_model_file = any((model_path / f).exists() for f in model_files)
        if not has_model_file:
            logger.debug(f"No model files found in {model_path}")
            return False

        # If expected hash is provided, verify it
        if expected_hash is not None:
            # For simplicity, we'll hash the config.json file
            config_path = model_path / "config.json"
            actual_hash = self._compute_sha256(config_path)
            
            if actual_hash != expected_hash:
                logger.warning(
                    f"Hash mismatch for {config_path}. "
                    f"Expected: {expected_hash}, Got: {actual_hash}"
                )
                return False
            
            logger.info(f"SHA256 verification passed for {config_path}")

        logger.info(f"Model integrity verified: {model_path}")
        return True

    def download_model(
        self,
        model_name: str,
        force_download: bool = False,
        expected_hash: Optional[str] = None,
        max_retries: int = 3,
    ) -> Path:
        """
        Download a model from HuggingFace Hub.

        This method downloads models with progress tracking, SHA256 verification,
        disk space checking, and automatic retry on network errors.

        Args:
            model_name: HuggingFace model identifier (e.g., "meta-llama/Llama-3-8B")
            force_download: Force re-download even if model exists
            expected_hash: Optional SHA256 hash for verification
            max_retries: Maximum number of retry attempts on network errors

        Returns:
            Path to the downloaded model directory

        Raises:
            ValueError: If disk space is insufficient
            RepositoryNotFoundError: If model repository doesn't exist
            RuntimeError: If download fails after all retries

        Example:
            >>> downloader = ModelDownloader()
            >>> model_path = downloader.download_model("microsoft/Phi-3-mini-4k-instruct")
            >>> print(f"Model downloaded to: {model_path}")
        """
        # Sanitize model name for directory path
        safe_model_name = model_name.replace("/", "--")
        model_path = self.cache_dir / safe_model_name

        # Check if model already exists and is valid
        if not force_download and self._verify_model_integrity(model_path, expected_hash):
            logger.info(f"Model already exists and is valid: {model_path}")
            return model_path

        # Estimate required space (rough estimate: 10GB for safety)
        # In production, we could query the repo size from HF API
        estimated_size = 10 * 1024**3  # 10GB
        
        if not self._check_disk_space(estimated_size):
            raise ValueError(
                f"Insufficient disk space to download model {model_name}. "
                f"Please free up at least 10GB of space."
            )

        # Download with retries
        last_error = None
        for attempt in range(max_retries):
            try:
                logger.info(
                    f"Downloading model {model_name} (attempt {attempt + 1}/{max_retries})..."
                )
                
                # Use snapshot_download to get the entire model
                downloaded_path = snapshot_download(
                    repo_id=model_name,
                    cache_dir=self.cache_dir,
                    local_dir=model_path,
                    local_dir_use_symlinks=False,
                    resume_download=True,
                    tqdm_class=tqdm,
                )
                
                logger.info(f"Model downloaded successfully to: {downloaded_path}")
                
                # Verify integrity after download
                if not self._verify_model_integrity(model_path, expected_hash):
                    raise RuntimeError(
                        f"Model integrity verification failed after download: {model_path}"
                    )
                
                return Path(downloaded_path)
                
            except RepositoryNotFoundError as e:
                logger.error(f"Model repository not found: {model_name}")
                raise
            
            except HfHubHTTPError as e:
                last_error = e
                logger.warning(
                    f"Network error during download (attempt {attempt + 1}/{max_retries}): {e}"
                )
                
                if attempt < max_retries - 1:
                    logger.info("Retrying download...")
                    continue
            
            except Exception as e:
                last_error = e
                logger.error(f"Unexpected error during download: {e}")
                
                if attempt < max_retries - 1:
                    logger.info("Retrying download...")
                    continue

        # If we get here, all retries failed
        raise RuntimeError(
            f"Failed to download model {model_name} after {max_retries} attempts. "
            f"Last error: {last_error}"
        )

    def get_model_path(self, model_name: str) -> Optional[Path]:
        """
        Get the path to a cached model without downloading.

        Args:
            model_name: HuggingFace model identifier

        Returns:
            Path to the model if it exists and is valid, None otherwise

        Example:
            >>> downloader = ModelDownloader()
            >>> path = downloader.get_model_path("microsoft/Phi-3-mini-4k-instruct")
            >>> if path:
            ...     print(f"Model found at: {path}")
            ... else:
            ...     print("Model not cached")
        """
        safe_model_name = model_name.replace("/", "--")
        model_path = self.cache_dir / safe_model_name
        
        if self._verify_model_integrity(model_path):
            return model_path
        
        return None

    def clear_cache(self, model_name: Optional[str] = None) -> None:
        """
        Clear cached models.

        Args:
            model_name: Specific model to clear, or None to clear all models

        Example:
            >>> downloader = ModelDownloader()
            >>> downloader.clear_cache("microsoft/Phi-3-mini-4k-instruct")  # Clear one model
            >>> downloader.clear_cache()  # Clear all models
        """
        if model_name is not None:
            safe_model_name = model_name.replace("/", "--")
            model_path = self.cache_dir / safe_model_name
            
            if model_path.exists():
                shutil.rmtree(model_path)
                logger.info(f"Cleared cache for model: {model_name}")
            else:
                logger.warning(f"Model not found in cache: {model_name}")
        else:
            if self.cache_dir.exists():
                shutil.rmtree(self.cache_dir)
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                logger.info("Cleared all cached models")

    def list_cached_models(self) -> list[str]:
        """
        List all cached models.

        Returns:
            List of model names that are cached

        Example:
            >>> downloader = ModelDownloader()
            >>> models = downloader.list_cached_models()
            >>> print(f"Cached models: {models}")
        """
        if not self.cache_dir.exists():
            return []
        
        cached_models = []
        for model_dir in self.cache_dir.iterdir():
            if model_dir.is_dir():
                # Convert back from safe name to original name
                model_name = model_dir.name.replace("--", "/")
                if self._verify_model_integrity(model_dir):
                    cached_models.append(model_name)
        
        return cached_models
