"""
Example demonstrating DeviceManager usage for hardware detection.

This script shows how to:
1. Detect available compute devices
2. Select optimal device for model execution
3. Auto-configure toolkit settings based on hardware
"""

from llm_judge_auditor.components import DeviceManager
from llm_judge_auditor.config import DeviceType, ToolkitConfig


def main():
    """Demonstrate DeviceManager functionality."""
    print("=" * 60)
    print("Device Detection Example")
    print("=" * 60)

    # Create DeviceManager instance
    manager = DeviceManager()

    # Detect all available devices
    print("\n1. Detecting available devices...")
    devices = manager.detect_devices()
    print(f"   Found {len(devices)} device(s):")
    for i, device in enumerate(devices, 1):
        print(f"   {i}. {device}")

    # Select optimal device
    print("\n2. Selecting optimal device...")
    optimal_device = manager.select_optimal_device()
    print(f"   Optimal device: {optimal_device}")

    # Select device for a specific model size
    print("\n3. Selecting device for 1GB model...")
    model_size = 1 * 1024**3  # 1GB
    device_for_model = manager.select_optimal_device(model_size=model_size)
    print(f"   Selected device: {device_for_model}")

    # Get detailed device information
    print("\n4. Getting detailed device information...")
    device_info = manager.get_device_info()
    print(f"   Number of devices: {device_info['num_devices']}")
    print(f"   Optimal device: {device_info['optimal_device']}")
    print("   Device details:")
    for device in device_info["devices"]:
        print(f"     - Type: {device['type']}, Name: {device['name']}")
        if device["total_memory_gb"]:
            print(f"       Memory: {device['total_memory_gb']:.2f}GB total, "
                  f"{device['available_memory_gb']:.2f}GB available")

    # Auto-configure toolkit
    print("\n5. Auto-configuring toolkit...")
    config = ToolkitConfig(device=DeviceType.AUTO, quantize=False)
    print(f"   Before: device={config.device}, quantize={config.quantize}")

    config = manager.auto_configure(config)
    print(f"   After:  device={config.device}, quantize={config.quantize}")

    print("\n" + "=" * 60)
    print("Device detection complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
