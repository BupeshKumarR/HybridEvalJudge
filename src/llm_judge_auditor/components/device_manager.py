"""
Device Manager for hardware detection and optimization.

This module provides automatic detection of available compute devices (CUDA, MPS, CPU)
and auto-configuration for optimal device selection based on available hardware.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional

from llm_judge_auditor.config import DeviceType, ToolkitConfig

logger = logging.getLogger(__name__)


@dataclass
class Device:
    """
    Represents a compute device.

    Attributes:
        device_type: Type of device (cuda, mps, cpu)
        device_id: Device identifier (e.g., 0 for cuda:0)
        name: Human-readable device name
        total_memory: Total memory in bytes (None for CPU)
        available_memory: Available memory in bytes (None for CPU)
    """

    device_type: DeviceType
    device_id: int = 0
    name: str = ""
    total_memory: Optional[int] = None
    available_memory: Optional[int] = None

    def __str__(self) -> str:
        """String representation of the device."""
        if self.device_type == DeviceType.CPU:
            return f"{self.device_type.value}"
        elif self.total_memory:
            memory_gb = self.total_memory / (1024**3)
            return f"{self.device_type.value}:{self.device_id} ({self.name}, {memory_gb:.1f}GB)"
        else:
            return f"{self.device_type.value}:{self.device_id}"


class DeviceManager:
    """
    Manages hardware detection and device selection.

    This class automatically detects available compute devices and provides
    recommendations for optimal configuration based on hardware capabilities.
    """

    def __init__(self):
        """Initialize the DeviceManager."""
        self._detected_devices: Optional[List[Device]] = None

    def detect_devices(self) -> List[Device]:
        """
        Detect all available compute devices.

        Returns:
            List of available Device objects, ordered by preference (CUDA > MPS > CPU)

        Example:
            >>> manager = DeviceManager()
            >>> devices = manager.detect_devices()
            >>> print(f"Found {len(devices)} devices")
        """
        if self._detected_devices is not None:
            return self._detected_devices

        devices = []

        # Try to detect CUDA devices
        try:
            import torch

            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    device = Device(
                        device_type=DeviceType.CUDA,
                        device_id=i,
                        name=props.name,
                        total_memory=props.total_memory,
                        available_memory=props.total_memory
                        - torch.cuda.memory_allocated(i),
                    )
                    devices.append(device)
                    logger.info(f"Detected CUDA device: {device}")
        except (ImportError, RuntimeError) as e:
            logger.debug(f"CUDA not available: {e}")

        # Try to detect MPS (Apple Silicon)
        try:
            import torch

            if torch.backends.mps.is_available():
                device = Device(
                    device_type=DeviceType.MPS,
                    device_id=0,
                    name="Apple Silicon GPU",
                    total_memory=None,  # MPS doesn't expose memory info easily
                    available_memory=None,
                )
                devices.append(device)
                logger.info(f"Detected MPS device: {device}")
        except (ImportError, AttributeError, RuntimeError) as e:
            logger.debug(f"MPS not available: {e}")

        # CPU is always available
        cpu_device = Device(
            device_type=DeviceType.CPU,
            device_id=0,
            name="CPU",
            total_memory=None,
            available_memory=None,
        )
        devices.append(cpu_device)
        logger.info(f"CPU device available: {cpu_device}")

        self._detected_devices = devices
        return devices

    def select_optimal_device(self, model_size: Optional[int] = None) -> Device:
        """
        Select the optimal device for model execution.

        Args:
            model_size: Estimated model size in bytes (optional)

        Returns:
            The best available Device for the given model size

        Example:
            >>> manager = DeviceManager()
            >>> device = manager.select_optimal_device(model_size=1_000_000_000)
            >>> print(f"Selected device: {device}")
        """
        devices = self.detect_devices()

        # If model size is specified, filter devices with sufficient memory
        if model_size is not None:
            suitable_devices = []
            for device in devices:
                if device.device_type == DeviceType.CPU:
                    # CPU always suitable (uses system RAM)
                    suitable_devices.append(device)
                elif device.available_memory and device.available_memory >= model_size * 1.5:
                    # Require 1.5x model size for activations and overhead
                    suitable_devices.append(device)

            if suitable_devices:
                devices = suitable_devices
            else:
                logger.warning(
                    f"No device with sufficient memory for model size {model_size / (1024**3):.2f}GB. "
                    "Falling back to CPU."
                )
                return next(d for d in devices if d.device_type == DeviceType.CPU)

        # Return first device (already ordered by preference: CUDA > MPS > CPU)
        selected = devices[0]
        logger.info(f"Selected optimal device: {selected}")
        return selected

    def auto_configure(self, config: ToolkitConfig) -> ToolkitConfig:
        """
        Automatically configure device settings based on available hardware.

        This method updates the config with optimal device selection and
        quantization recommendations based on detected hardware.

        Args:
            config: The toolkit configuration to update

        Returns:
            Updated ToolkitConfig with optimal device settings

        Example:
            >>> manager = DeviceManager()
            >>> config = ToolkitConfig(device=DeviceType.AUTO)
            >>> config = manager.auto_configure(config)
            >>> print(f"Auto-configured device: {config.device}")
        """
        # If device is set to AUTO, detect and select optimal device
        if config.device == DeviceType.AUTO:
            optimal_device = self.select_optimal_device()
            config.device = optimal_device.device_type
            logger.info(f"Auto-configured device: {config.device}")

        # Recommend quantization based on device
        devices = self.detect_devices()
        selected_device = next(
            (d for d in devices if d.device_type == config.device), None
        )

        if selected_device:
            # Check if we should enable quantization
            if selected_device.device_type == DeviceType.CPU:
                # Always recommend quantization for CPU
                if not config.quantize:
                    logger.info(
                        "Enabling quantization for CPU execution (recommended)"
                    )
                    config.quantize = True
            elif (
                selected_device.device_type == DeviceType.CUDA
                and selected_device.total_memory
            ):
                # Recommend quantization for GPUs with < 16GB VRAM
                memory_gb = selected_device.total_memory / (1024**3)
                if memory_gb < 16 and not config.quantize:
                    logger.info(
                        f"Enabling quantization for GPU with {memory_gb:.1f}GB VRAM (recommended)"
                    )
                    config.quantize = True

        return config

    def get_device_info(self) -> dict:
        """
        Get information about all detected devices.

        Returns:
            Dictionary with device information

        Example:
            >>> manager = DeviceManager()
            >>> info = manager.get_device_info()
            >>> print(info)
        """
        devices = self.detect_devices()
        return {
            "num_devices": len(devices),
            "devices": [
                {
                    "type": d.device_type.value,
                    "id": d.device_id,
                    "name": d.name,
                    "total_memory_gb": (
                        d.total_memory / (1024**3) if d.total_memory else None
                    ),
                    "available_memory_gb": (
                        d.available_memory / (1024**3)
                        if d.available_memory
                        else None
                    ),
                }
                for d in devices
            ],
            "optimal_device": str(self.select_optimal_device()),
        }
