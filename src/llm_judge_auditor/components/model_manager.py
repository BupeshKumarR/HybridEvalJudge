"""
Model Manager for loading and initializing models.

This module provides the ModelManager class for loading specialized verifiers
and judge model ensembles with support for lazy loading, quantization, and
comprehensive error handling.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from llm_judge_auditor.components.device_manager import DeviceManager
from llm_judge_auditor.components.model_downloader import ModelDownloader
from llm_judge_auditor.config import DeviceType, ToolkitConfig

logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """
    Information about a loaded model.

    Attributes:
        name: Model identifier
        model_type: Type of model (verifier or judge)
        device: Device the model is loaded on
        quantized: Whether the model is quantized
        parameters: Approximate number of parameters
        memory_usage: Approximate memory usage in bytes
        is_ready: Whether the model is ready for inference
    """

    name: str
    model_type: str  # "verifier" or "judge"
    device: str
    quantized: bool
    parameters: Optional[int] = None
    memory_usage: Optional[int] = None
    is_ready: bool = False


class ModelManager:
    """
    Manages loading and initialization of models.

    This class handles loading specialized verifiers and judge model ensembles
    with support for lazy loading, 8-bit quantization, and comprehensive error handling.
    """

    def __init__(
        self,
        config: ToolkitConfig,
        device_manager: Optional[DeviceManager] = None,
        model_downloader: Optional[ModelDownloader] = None,
    ):
        """
        Initialize the ModelManager.

        Args:
            config: Toolkit configuration
            device_manager: Optional DeviceManager instance
            model_downloader: Optional ModelDownloader instance
        """
        self.config = config
        self.device_manager = device_manager or DeviceManager()
        self.model_downloader = model_downloader or ModelDownloader(
            cache_dir=config.cache_dir
        )

        # Auto-configure device if needed
        if config.device == DeviceType.AUTO:
            self.config = self.device_manager.auto_configure(config)

        # Storage for loaded models
        self._verifier_model: Optional[Any] = None
        self._verifier_tokenizer: Optional[Any] = None
        self._judge_models: Dict[str, Any] = {}
        self._judge_tokenizers: Dict[str, Any] = {}
        
        # Model info tracking
        self._model_info: Dict[str, ModelInfo] = {}
        
        logger.info(f"ModelManager initialized with device: {self.config.device}")

    def _get_device_string(self) -> str:
        """
        Get the device string for PyTorch.

        Returns:
            Device string (e.g., "cuda:0", "mps", "cpu")
        """
        if self.config.device == DeviceType.CUDA:
            return "cuda"
        elif self.config.device == DeviceType.MPS:
            return "mps"
        else:
            return "cpu"

    def _get_quantization_config(self) -> Optional[BitsAndBytesConfig]:
        """
        Get the quantization configuration for 8-bit loading.

        Returns:
            BitsAndBytesConfig if quantization is enabled, None otherwise
        """
        if not self.config.quantize:
            return None

        # 8-bit quantization config
        return BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
        )

    def _estimate_model_size(self, model_name: str) -> int:
        """
        Estimate model size in bytes based on name.

        Args:
            model_name: Model identifier

        Returns:
            Estimated size in bytes
        """
        # Rough estimates based on common model sizes
        if "3b" in model_name.lower() or "3-" in model_name.lower():
            return 3 * 1024**3  # 3GB
        elif "7b" in model_name.lower() or "7-" in model_name.lower():
            return 7 * 1024**3  # 7GB
        elif "8b" in model_name.lower() or "8-" in model_name.lower():
            return 8 * 1024**3  # 8GB
        elif "13b" in model_name.lower() or "13-" in model_name.lower():
            return 13 * 1024**3  # 13GB
        elif "base" in model_name.lower() or "small" in model_name.lower():
            return 1 * 1024**3  # 1GB
        elif "large" in model_name.lower():
            return 3 * 1024**3  # 3GB
        else:
            return 5 * 1024**3  # Default 5GB

    def load_verifier(
        self,
        model_name: Optional[str] = None,
        quantize: Optional[bool] = None,
    ) -> tuple[Any, Any]:
        """
        Load the specialized verifier model.

        This method loads a specialized fact-checking model (e.g., MiniCheck, HHEM)
        with support for quantization and lazy loading.

        Args:
            model_name: Model identifier (uses config if not provided)
            quantize: Whether to quantize (uses config if not provided)

        Returns:
            Tuple of (model, tokenizer)

        Raises:
            RuntimeError: If model loading fails
            ValueError: If model name is invalid

        Example:
            >>> manager = ModelManager(config)
            >>> model, tokenizer = manager.load_verifier()
            >>> print(f"Verifier loaded: {model.config.model_type}")
        """
        # Use config values if not provided
        model_name = model_name or self.config.verifier_model
        quantize = quantize if quantize is not None else self.config.quantize

        if not model_name:
            raise ValueError("No verifier model specified in configuration")

        # Return cached model if already loaded
        if self._verifier_model is not None and self._verifier_tokenizer is not None:
            logger.info(f"Using cached verifier model: {model_name}")
            return self._verifier_model, self._verifier_tokenizer

        try:
            logger.info(f"Loading verifier model: {model_name}")
            
            # Download model if needed
            model_path = self.model_downloader.get_model_path(model_name)
            if model_path is None:
                logger.info(f"Model not cached, downloading: {model_name}")
                model_path = self.model_downloader.download_model(model_name)

            # Get device and quantization config
            device = self._get_device_string()
            quantization_config = self._get_quantization_config() if quantize else None

            # Load tokenizer
            logger.info(f"Loading tokenizer for {model_name}")
            tokenizer = AutoTokenizer.from_pretrained(
                str(model_path),
                trust_remote_code=True,
            )

            # Determine model class based on architecture
            # Most verifiers are seq2seq models (T5, BART)
            logger.info(f"Loading model from {model_path}")
            
            try:
                # Try seq2seq first (common for verifiers)
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    str(model_path),
                    quantization_config=quantization_config,
                    device_map=device if quantization_config else None,
                    trust_remote_code=True,
                    torch_dtype=torch.float16 if device != "cpu" else torch.float32,
                )
            except Exception:
                # Fall back to causal LM
                logger.info("Seq2Seq loading failed, trying CausalLM")
                model = AutoModelForCausalLM.from_pretrained(
                    str(model_path),
                    quantization_config=quantization_config,
                    device_map=device if quantization_config else None,
                    trust_remote_code=True,
                    torch_dtype=torch.float16 if device != "cpu" else torch.float32,
                )

            # Move to device if not using quantization (quantization handles device placement)
            if not quantization_config:
                model = model.to(device)

            # Set to eval mode
            model.eval()

            # Cache the model
            self._verifier_model = model
            self._verifier_tokenizer = tokenizer

            # Store model info
            num_params = sum(p.numel() for p in model.parameters())
            self._model_info[model_name] = ModelInfo(
                name=model_name,
                model_type="verifier",
                device=device,
                quantized=quantize,
                parameters=num_params,
                memory_usage=self._estimate_model_size(model_name),
                is_ready=True,
            )

            logger.info(
                f"Verifier model loaded successfully: {model_name} "
                f"({num_params / 1e6:.1f}M parameters, device={device}, quantized={quantize})"
            )

            return model, tokenizer

        except Exception as e:
            error_msg = (
                f"Failed to load verifier model '{model_name}': {str(e)}. "
                f"Please check that the model name is correct and you have sufficient memory."
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def load_judge_ensemble(
        self,
        model_names: Optional[List[str]] = None,
        quantize: Optional[bool] = None,
    ) -> Dict[str, tuple[Any, Any]]:
        """
        Load the judge model ensemble.

        This method loads 2-3 judge LLMs for ensemble evaluation with support
        for quantization and lazy loading.

        Args:
            model_names: List of model identifiers (uses config if not provided)
            quantize: Whether to quantize (uses config if not provided)

        Returns:
            Dictionary mapping model names to (model, tokenizer) tuples

        Raises:
            RuntimeError: If any model loading fails
            ValueError: If model names list is empty

        Example:
            >>> manager = ModelManager(config)
            >>> ensemble = manager.load_judge_ensemble()
            >>> print(f"Loaded {len(ensemble)} judge models")
        """
        # Use config values if not provided
        model_names = model_names or self.config.judge_models
        quantize = quantize if quantize is not None else self.config.quantize

        if not model_names:
            raise ValueError("No judge models specified in configuration")

        ensemble = {}
        failed_models = []

        for model_name in model_names:
            # Return cached model if already loaded
            if model_name in self._judge_models and model_name in self._judge_tokenizers:
                logger.info(f"Using cached judge model: {model_name}")
                ensemble[model_name] = (
                    self._judge_models[model_name],
                    self._judge_tokenizers[model_name],
                )
                continue

            try:
                logger.info(f"Loading judge model: {model_name}")
                
                # Download model if needed
                model_path = self.model_downloader.get_model_path(model_name)
                if model_path is None:
                    logger.info(f"Model not cached, downloading: {model_name}")
                    model_path = self.model_downloader.download_model(model_name)

                # Get device and quantization config
                device = self._get_device_string()
                quantization_config = self._get_quantization_config() if quantize else None

                # Load tokenizer
                logger.info(f"Loading tokenizer for {model_name}")
                tokenizer = AutoTokenizer.from_pretrained(
                    str(model_path),
                    trust_remote_code=True,
                )
                
                # Set padding token if not set
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token

                # Load model (judge models are typically causal LMs)
                logger.info(f"Loading model from {model_path}")
                model = AutoModelForCausalLM.from_pretrained(
                    str(model_path),
                    quantization_config=quantization_config,
                    device_map=device if quantization_config else None,
                    trust_remote_code=True,
                    torch_dtype=torch.float16 if device != "cpu" else torch.float32,
                )

                # Move to device if not using quantization
                if not quantization_config:
                    model = model.to(device)

                # Set to eval mode
                model.eval()

                # Cache the model
                self._judge_models[model_name] = model
                self._judge_tokenizers[model_name] = tokenizer
                ensemble[model_name] = (model, tokenizer)

                # Store model info
                num_params = sum(p.numel() for p in model.parameters())
                self._model_info[model_name] = ModelInfo(
                    name=model_name,
                    model_type="judge",
                    device=device,
                    quantized=quantize,
                    parameters=num_params,
                    memory_usage=self._estimate_model_size(model_name),
                    is_ready=True,
                )

                logger.info(
                    f"Judge model loaded successfully: {model_name} "
                    f"({num_params / 1e6:.1f}M parameters, device={device}, quantized={quantize})"
                )

            except Exception as e:
                error_msg = (
                    f"Failed to load judge model '{model_name}': {str(e)}. "
                    f"Please check that the model name is correct and you have sufficient memory."
                )
                logger.error(error_msg)
                failed_models.append((model_name, str(e)))

        # If all models failed, raise an error
        if not ensemble:
            error_details = "\n".join(f"  - {name}: {error}" for name, error in failed_models)
            raise RuntimeError(
                f"Failed to load any judge models. Errors:\n{error_details}"
            )

        # If some models failed, log a warning
        if failed_models:
            logger.warning(
                f"Successfully loaded {len(ensemble)}/{len(model_names)} judge models. "
                f"Failed models: {[name for name, _ in failed_models]}"
            )

        return ensemble

    def verify_models_ready(self) -> bool:
        """
        Verify that all configured models are loaded and ready for inference.

        Returns:
            True if all models are ready, False otherwise

        Example:
            >>> manager = ModelManager(config)
            >>> manager.load_verifier()
            >>> manager.load_judge_ensemble()
            >>> if manager.verify_models_ready():
            ...     print("All models ready!")
        """
        # Check verifier
        if self.config.verifier_model:
            if self._verifier_model is None or self._verifier_tokenizer is None:
                logger.warning("Verifier model not loaded")
                return False
            
            verifier_info = self._model_info.get(self.config.verifier_model)
            if verifier_info is None or not verifier_info.is_ready:
                logger.warning("Verifier model not ready")
                return False

        # Check judge ensemble
        if self.config.judge_models:
            for model_name in self.config.judge_models:
                if model_name not in self._judge_models or model_name not in self._judge_tokenizers:
                    logger.warning(f"Judge model not loaded: {model_name}")
                    return False
                
                judge_info = self._model_info.get(model_name)
                if judge_info is None or not judge_info.is_ready:
                    logger.warning(f"Judge model not ready: {model_name}")
                    return False

        logger.info("All configured models are loaded and ready")
        return True

    def get_model_info(self) -> Dict[str, ModelInfo]:
        """
        Get information about all loaded models.

        Returns:
            Dictionary mapping model names to ModelInfo objects

        Example:
            >>> manager = ModelManager(config)
            >>> manager.load_verifier()
            >>> info = manager.get_model_info()
            >>> for name, model_info in info.items():
            ...     print(f"{name}: {model_info.parameters / 1e6:.1f}M params")
        """
        return self._model_info.copy()

    def unload_models(self) -> None:
        """
        Unload all models to free memory.

        Example:
            >>> manager = ModelManager(config)
            >>> manager.load_verifier()
            >>> manager.unload_models()
            >>> print("Models unloaded")
        """
        # Clear model references
        self._verifier_model = None
        self._verifier_tokenizer = None
        self._judge_models.clear()
        self._judge_tokenizers.clear()
        
        # Update model info
        for model_info in self._model_info.values():
            model_info.is_ready = False

        # Clear GPU cache if using CUDA
        if self.config.device == DeviceType.CUDA:
            try:
                import torch
                torch.cuda.empty_cache()
                logger.info("Cleared CUDA cache")
            except Exception as e:
                logger.warning(f"Could not clear CUDA cache: {e}")

        logger.info("All models unloaded")

    def get_verifier(self) -> Optional[tuple[Any, Any]]:
        """
        Get the loaded verifier model and tokenizer.

        Returns:
            Tuple of (model, tokenizer) if loaded, None otherwise
        """
        if self._verifier_model is not None and self._verifier_tokenizer is not None:
            return self._verifier_model, self._verifier_tokenizer
        return None

    def get_judge(self, model_name: str) -> Optional[tuple[Any, Any]]:
        """
        Get a specific loaded judge model and tokenizer.

        Args:
            model_name: Name of the judge model

        Returns:
            Tuple of (model, tokenizer) if loaded, None otherwise
        """
        if model_name in self._judge_models and model_name in self._judge_tokenizers:
            return self._judge_models[model_name], self._judge_tokenizers[model_name]
        return None

    def get_all_judges(self) -> Dict[str, tuple[Any, Any]]:
        """
        Get all loaded judge models and tokenizers.

        Returns:
            Dictionary mapping model names to (model, tokenizer) tuples
        """
        return {
            name: (self._judge_models[name], self._judge_tokenizers[name])
            for name in self._judge_models.keys()
            if name in self._judge_tokenizers
        }
