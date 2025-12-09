"""
Example demonstrating API key setup guide and validation.

This example shows how to:
1. Load API keys from environment
2. Validate API keys with test calls
3. Display formatted setup guides
4. Show validation status
5. Display troubleshooting information
"""

import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from llm_judge_auditor.components.api_key_manager import APIKeyManager


def example_basic_setup_guide():
    """Example 1: Display basic setup guide without validation."""
    print("=" * 70)
    print("Example 1: Basic Setup Guide (No Validation)")
    print("=" * 70)
    
    manager = APIKeyManager()
    manager.load_keys()
    
    # Display setup guide without validation
    print(manager.get_setup_instructions(show_validation=False))


def example_setup_guide_with_validation():
    """Example 2: Display setup guide with API key validation."""
    print("\n" + "=" * 70)
    print("Example 2: Setup Guide with Validation")
    print("=" * 70)
    
    manager = APIKeyManager()
    manager.load_keys()
    
    # Validate available keys
    if manager.has_any_keys():
        print("\n🔍 Validating API keys...")
        results = manager.validate_all_keys(verbose=True)
        
        print("\nValidation Results:")
        for service, valid in results.items():
            status = "✅ VALID" if valid else "❌ INVALID"
            print(f"  {service}: {status}")
            
            # Show error details if validation failed
            if not valid:
                error = manager.get_error_details(service)
                if error:
                    print(f"    Error: {error}")
    
    # Display setup guide with validation status
    print(manager.get_setup_instructions(show_validation=True))


def example_validation_summary():
    """Example 3: Display validation summary."""
    print("\n" + "=" * 70)
    print("Example 3: Validation Summary")
    print("=" * 70)
    
    manager = APIKeyManager()
    manager.load_keys()
    
    # Validate keys if available
    if manager.has_any_keys():
        manager.validate_all_keys(verbose=False)
    
    # Display validation summary
    print(manager.get_validation_summary())


def example_troubleshooting_guide():
    """Example 4: Display troubleshooting guide."""
    print("\n" + "=" * 70)
    print("Example 4: Troubleshooting Guide")
    print("=" * 70)
    
    manager = APIKeyManager()
    
    # Display troubleshooting guide
    print(manager.get_troubleshooting_guide())


def example_check_key_status():
    """Example 5: Check individual key status."""
    print("\n" + "=" * 70)
    print("Example 5: Check Individual Key Status")
    print("=" * 70)
    
    manager = APIKeyManager()
    manager.load_keys()
    
    # Check Groq key status
    groq_status = manager.get_key_status('groq')
    if groq_status:
        print(f"\nGroq API Key Status:")
        print(f"  Available: {groq_status.available}")
        print(f"  Validated: {groq_status.validated}")
        if groq_status.error_message:
            print(f"  Error: {groq_status.error_message}")
    
    # Check Gemini key status
    gemini_status = manager.get_key_status('gemini')
    if gemini_status:
        print(f"\nGemini API Key Status:")
        print(f"  Available: {gemini_status.available}")
        print(f"  Validated: {gemini_status.validated}")
        if gemini_status.error_message:
            print(f"  Error: {gemini_status.error_message}")
    
    # Show available services
    available = manager.get_available_services()
    print(f"\nAvailable Services: {', '.join(available) if available else 'None'}")


def example_complete_workflow():
    """Example 6: Complete workflow with validation and display."""
    print("\n" + "=" * 70)
    print("Example 6: Complete Workflow")
    print("=" * 70)
    
    manager = APIKeyManager()
    
    # Use the convenience method that does everything
    manager.display_setup_guide_with_validation(validate=True)


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("API Key Setup Guide Examples")
    print("=" * 70)
    
    # Check if any API keys are set
    has_groq = os.environ.get('GROQ_API_KEY')
    has_gemini = os.environ.get('GEMINI_API_KEY')
    
    print("\nCurrent Environment:")
    print(f"  GROQ_API_KEY: {'✅ Set' if has_groq else '❌ Not set'}")
    print(f"  GEMINI_API_KEY: {'✅ Set' if has_gemini else '❌ Not set'}")
    
    if not has_groq and not has_gemini:
        print("\n⚠️  No API keys found in environment.")
        print("Set at least one key to see validation in action:")
        print("  export GROQ_API_KEY='your-key'")
        print("  export GEMINI_API_KEY='your-key'")
    
    # Run examples
    try:
        example_basic_setup_guide()
        example_setup_guide_with_validation()
        example_validation_summary()
        example_troubleshooting_guide()
        example_check_key_status()
        example_complete_workflow()
        
        print("\n" + "=" * 70)
        print("All examples completed successfully!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
