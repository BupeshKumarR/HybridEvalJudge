"""
Unit tests for DeviceManager component.

Tests hardware detection, device selection, and auto-configuration functionality.
"""

import pytest

from llm_judge_auditor.components import Device, DeviceManager
from llm_judge_auditor.config import DeviceType, ToolkitConfig


class TestDeviceManager:
    """Test suite for DeviceManager."""

    def test_detect_devices_returns_list(self):
        """Test that detect_devices returns a non-empty list."""
        manager = DeviceManager()
        devices = manager.detect_devices()

        assert isinstance(devices, list)
        assert len(devices) > 0, "Should detect at least CPU"

    def test_cpu_always_available(self):
        """Test that CPU is always detected as available."""
        manager = DeviceManager()
        devices = manager.detect_devices()

        cpu_devices = [d for d in devices if d.device_type == DeviceType.CPU]
        assert len(cpu_devices) > 0, "CPU should always be available"

    def test_detect_devices_caching(self):
        """Test that detect_devices caches results."""
        manager = DeviceManager()
        devices1 = manager.detect_devices()
        devices2 = manager.detect_devices()

        # Should return the same list object (cached)
        assert devices1 is devices2

    def test_select_optimal_device_returns_device(self):
        """Test that select_optimal_device returns a valid Device."""
        manager = DeviceManager()
        device = manager.select_optimal_device()

        assert isinstance(device, Device)
        assert device.device_type in [DeviceType.CUDA, DeviceType.MPS, DeviceType.CPU]

    def test_select_optimal_device_prefers_gpu(self):
        """Test that select_optimal_device prefers GPU over CPU when available."""
        manager = DeviceManager()
        devices = manager.detect_devices()

        # If GPU is available, optimal device should not be CPU
        has_gpu = any(
            d.device_type in [DeviceType.CUDA, DeviceType.MPS] for d in devices
        )
        optimal = manager.select_optimal_device()

        if has_gpu:
            assert optimal.device_type != DeviceType.CPU, "Should prefer GPU over CPU"

    def test_auto_configure_with_auto_device(self):
        """Test auto_configure selects a device when set to AUTO."""
        manager = DeviceManager()
        config = ToolkitConfig(device=DeviceType.AUTO)

        updated_config = manager.auto_configure(config)

        assert updated_config.device != DeviceType.AUTO
        assert updated_config.device in [DeviceType.CUDA, DeviceType.MPS, DeviceType.CPU]

    def test_auto_configure_preserves_explicit_device(self):
        """Test auto_configure preserves explicitly set device."""
        manager = DeviceManager()
        config = ToolkitConfig(device=DeviceType.CPU)

        updated_config = manager.auto_configure(config)

        assert updated_config.device == DeviceType.CPU

    def test_auto_configure_enables_quantization_for_cpu(self):
        """Test auto_configure enables quantization for CPU."""
        manager = DeviceManager()
        config = ToolkitConfig(device=DeviceType.CPU, quantize=False)

        updated_config = manager.auto_configure(config)

        assert updated_config.quantize is True, "Should enable quantization for CPU"

    def test_get_device_info_returns_dict(self):
        """Test that get_device_info returns a properly structured dict."""
        manager = DeviceManager()
        info = manager.get_device_info()

        assert isinstance(info, dict)
        assert "num_devices" in info
        assert "devices" in info
        assert "optimal_device" in info
        assert info["num_devices"] > 0
        assert isinstance(info["devices"], list)

    def test_device_str_representation(self):
        """Test Device string representation."""
        cpu_device = Device(device_type=DeviceType.CPU, name="CPU")
        assert "cpu" in str(cpu_device).lower()

        cuda_device = Device(
            device_type=DeviceType.CUDA,
            device_id=0,
            name="NVIDIA GPU",
            total_memory=8 * 1024**3,
        )
        device_str = str(cuda_device)
        assert "cuda" in device_str.lower()
        assert "0" in device_str

    def test_select_optimal_device_with_model_size(self):
        """Test device selection considers model size."""
        manager = DeviceManager()

        # Test with a very large model size (should fall back to CPU if no GPU)
        large_model_size = 100 * 1024**3  # 100GB
        device = manager.select_optimal_device(model_size=large_model_size)

        # Should return a device (likely CPU for such a large model)
        assert isinstance(device, Device)

    def test_select_optimal_device_with_small_model_size(self):
        """Test device selection with small model size."""
        manager = DeviceManager()

        # Test with a small model size (should work on any device)
        small_model_size = 100 * 1024**2  # 100MB
        device = manager.select_optimal_device(model_size=small_model_size)

        assert isinstance(device, Device)
