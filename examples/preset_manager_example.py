"""
Example demonstrating PresetManager usage.

This script shows how to:
1. Initialize the PresetManager
2. List available presets
3. Load preset configurations
4. Access preset details
"""

from pathlib import Path

from llm_judge_auditor.components import PresetManager


def main():
    """Demonstrate PresetManager functionality."""
    print("=" * 60)
    print("PresetManager Example")
    print("=" * 60)

    # Initialize PresetManager (without custom presets)
    print("\n1. Initializing PresetManager...")
    manager = PresetManager()
    print("   ✓ PresetManager initialized")

    # List all available presets
    print("\n2. Available Presets:")
    presets = manager.list_presets()
    for preset in presets:
        print(f"\n   • {preset.name}")
        print(f"     Description: {preset.description}")
        print(f"     Verifier: {preset.config.verifier_model}")
        print(f"     Judges: {len(preset.config.judge_models)} model(s)")
        print(f"     Retrieval: {'Enabled' if preset.config.enable_retrieval else 'Disabled'}")
        print(f"     Quantization: {'Enabled' if preset.config.quantize else 'Disabled'}")

    # Load specific presets
    print("\n3. Loading 'fast' preset...")
    fast_config = manager.load_preset("fast")
    print(f"   ✓ Loaded fast preset")
    print(f"     - Verifier: {fast_config.verifier_model}")
    print(f"     - Judges: {fast_config.judge_models}")
    print(f"     - Max length: {fast_config.max_length}")

    print("\n4. Loading 'balanced' preset...")
    balanced_config = manager.load_preset("balanced")
    print(f"   ✓ Loaded balanced preset")
    print(f"     - Verifier: {balanced_config.verifier_model}")
    print(f"     - Judges: {balanced_config.judge_models}")
    print(f"     - Retrieval: {balanced_config.enable_retrieval}")

    # Check if preset exists
    print("\n5. Checking preset existence...")
    print(f"   'fast' exists: {manager.preset_exists('fast')}")
    print(f"   'balanced' exists: {manager.preset_exists('balanced')}")
    print(f"   'custom' exists: {manager.preset_exists('custom')}")

    # Get preset names
    print("\n6. Available preset names:")
    names = manager.get_preset_names()
    print(f"   {', '.join(names)}")

    # Example with custom preset directory
    print("\n7. Loading with custom preset directory...")
    preset_dir = Path("config/presets")
    if preset_dir.exists():
        custom_manager = PresetManager(preset_dir=preset_dir)
        print(f"   ✓ Loaded PresetManager with custom directory: {preset_dir}")
        print(f"   Available presets: {', '.join(custom_manager.get_preset_names())}")
    else:
        print(f"   ⚠ Custom preset directory not found: {preset_dir}")

    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
