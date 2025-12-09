"""
Comprehensive Error Handling Example for API Judge Integration.

This example demonstrates all error handling scenarios:
- Missing API keys
- Authentication errors
- Network errors
- Rate limit errors
- Malformed responses
- Partial failures

Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6
"""

import logging
import os
import sys
from typing import Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.llm_judge_auditor.components.api_key_manager import APIKeyManager
from src.llm_judge_auditor.components.api_judge_ensemble import APIJudgeEnsemble
from src.llm_judge_auditor.components.groq_judge_client import (
    GroqJudgeClient,
    GroqAPIError,
    GroqAuthenticationError,
    GroqNetworkError,
    GroqRateLimitError,
)
from src.llm_judge_auditor.components.gemini_judge_client import (
    GeminiJudgeClient,
    GeminiAPIError,
    GeminiAuthenticationError,
    GeminiNetworkError,
    GeminiRateLimitError,
)
from src.llm_judge_auditor.config import ToolkitConfig
from src.llm_judge_auditor.utils.error_handling import APIErrorHandler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def demonstrate_missing_keys_handling():
    """
    Demonstrate handling of missing API keys.
    
    Requirements: 6.1
    """
    print("\n" + "=" * 70)
    print("DEMO 1: Missing API Keys Handling")
    print("=" * 70)
    
    # Temporarily clear API keys
    original_groq = os.environ.get("GROQ_API_KEY")
    original_gemini = os.environ.get("GEMINI_API_KEY")
    
    try:
        # Clear keys
        if "GROQ_API_KEY" in os.environ:
            del os.environ["GROQ_API_KEY"]
        if "GEMINI_API_KEY" in os.environ:
            del os.environ["GEMINI_API_KEY"]
        
        # Try to load keys
        api_key_manager = APIKeyManager()
        available_keys = api_key_manager.load_keys()
        
        print(f"\nAvailable keys: {available_keys}")
        
        # Handle missing keys
        error_info = APIErrorHandler.handle_missing_keys(available_keys)
        
        if error_info["error"]:
            print(f"\n❌ Error: {error_info['message']}")
            print(f"\nAction: {error_info['action']}")
            print(f"\nHelp:\n{error_info['help']}")
        
        # Display setup guide
        print("\n" + api_key_manager.get_setup_instructions())
        
    finally:
        # Restore original keys
        if original_groq:
            os.environ["GROQ_API_KEY"] = original_groq
        if original_gemini:
            os.environ["GEMINI_API_KEY"] = original_gemini


def demonstrate_authentication_error_handling():
    """
    Demonstrate handling of authentication errors.
    
    Requirements: 6.2
    """
    print("\n" + "=" * 70)
    print("DEMO 2: Authentication Error Handling")
    print("=" * 70)
    
    # Try with invalid API key
    invalid_key = "invalid_key_12345"
    
    try:
        print("\nAttempting to use invalid Groq API key...")
        client = GroqJudgeClient(api_key=invalid_key)
        
        # This should fail with authentication error
        verdict = client.evaluate(
            source_text="The sky is blue.",
            candidate_output="The sky is green.",
            task="factual_accuracy"
        )
        
    except GroqAuthenticationError as e:
        print(f"\n✓ Caught authentication error: {e}")
        
        # Handle the error
        error_info = APIErrorHandler.handle_authentication_error("groq", e)
        
        print(f"\nError Type: {error_info['error_type']}")
        print(f"Service: {error_info['service']}")
        print(f"Action: {error_info['action']}")
        print(f"\nHelp:\n{error_info['help']}")
    
    except Exception as e:
        print(f"\n✓ Caught error: {type(e).__name__}: {e}")


def demonstrate_network_error_handling():
    """
    Demonstrate handling of network errors.
    
    Requirements: 6.3
    """
    print("\n" + "=" * 70)
    print("DEMO 3: Network Error Handling")
    print("=" * 70)
    
    print("\nNetwork errors are handled automatically with retry logic:")
    print("- Exponential backoff (1s, 2s, 4s)")
    print("- Maximum 2 retries")
    print("- Detailed error logging")
    
    # Simulate network error handling
    try:
        # Create a mock network error
        error = Exception("Connection timeout after 30 seconds")
        
        error_info = APIErrorHandler.handle_network_error(
            service="groq",
            error=error,
            retry_count=2
        )
        
        print(f"\nError Type: {error_info['error_type']}")
        print(f"Service: {error_info['service']}")
        print(f"Retry Count: {error_info['retry_count']}")
        print(f"Action: {error_info['action']}")
        print(f"\nHelp:\n{error_info['help']}")
        
    except Exception as e:
        print(f"\nError: {e}")


def demonstrate_rate_limit_handling():
    """
    Demonstrate handling of rate limit errors.
    
    Requirements: 6.3
    """
    print("\n" + "=" * 70)
    print("DEMO 4: Rate Limit Error Handling")
    print("=" * 70)
    
    print("\nRate limits for free tiers:")
    print("- Groq: 30 requests per minute")
    print("- Gemini: 15 requests per minute")
    
    # Simulate rate limit error
    try:
        error = Exception("Rate limit exceeded. Please try again in 60 seconds.")
        
        error_info = APIErrorHandler.handle_rate_limit_error(
            service="groq",
            error=error,
            retry_after=60.0
        )
        
        print(f"\nError Type: {error_info['error_type']}")
        print(f"Service: {error_info['service']}")
        print(f"Rate Limit: {error_info['rate_limit']}")
        print(f"Retry After: {error_info['retry_after']} seconds")
        print(f"Action: {error_info['action']}")
        print(f"\nHelp:\n{error_info['help']}")
        
    except Exception as e:
        print(f"\nError: {e}")


def demonstrate_malformed_response_handling():
    """
    Demonstrate handling of malformed API responses.
    
    Requirements: 6.4
    """
    print("\n" + "=" * 70)
    print("DEMO 5: Malformed Response Handling")
    print("=" * 70)
    
    # Simulate malformed responses
    test_cases = [
        {
            "name": "Invalid JSON",
            "response": "This is not valid JSON {score: 85",
            "expected": "Fallback parsing extracts score"
        },
        {
            "name": "Missing fields",
            "response": '{"score": 75}',
            "expected": "Uses default values for missing fields"
        },
        {
            "name": "Completely malformed",
            "response": "Random text with no structure",
            "expected": "Returns default score (50.0)"
        }
    ]
    
    for test_case in test_cases:
        print(f"\n--- Test Case: {test_case['name']} ---")
        print(f"Response: {test_case['response'][:50]}...")
        print(f"Expected: {test_case['expected']}")
        
        try:
            import json
            json.loads(test_case['response'])
            print("✓ Valid JSON")
        except json.JSONDecodeError as e:
            print(f"✗ JSON parse error: {e}")
            
            # Handle malformed response
            error_info = APIErrorHandler.handle_malformed_response(
                service="groq",
                response_text=test_case['response'],
                parse_error=e
            )
            
            print(f"\nError Type: {error_info['error_type']}")
            print(f"Partial Data: {error_info['partial_data']}")
            print(f"Action: {error_info['action']}")


def demonstrate_partial_failure_handling():
    """
    Demonstrate handling of partial failures in ensemble.
    
    Requirements: 6.4
    """
    print("\n" + "=" * 70)
    print("DEMO 6: Partial Failure Handling")
    print("=" * 70)
    
    # Scenario 1: Some judges succeed
    print("\n--- Scenario 1: Partial Success ---")
    error_info = APIErrorHandler.handle_partial_failure(
        total_judges=2,
        successful_judges=1,
        failed_judges=["gemini-flash"]
    )
    
    print(f"Status: {'Warning' if error_info.get('warning') else 'Error'}")
    print(f"Message: {error_info['message']}")
    print(f"Successful: {error_info['successful_judges']}/{error_info['total_judges']}")
    print(f"Failed: {error_info['failed_judges']}")
    print(f"Action: {error_info['action']}")
    
    # Scenario 2: All judges fail
    print("\n--- Scenario 2: Complete Failure ---")
    error_info = APIErrorHandler.handle_partial_failure(
        total_judges=2,
        successful_judges=0,
        failed_judges=["groq-llama-3.3-70b", "gemini-flash"]
    )
    
    print(f"Status: {'Error' if error_info.get('error') else 'OK'}")
    print(f"Message: {error_info['message']}")
    print(f"Successful: {error_info['successful_judges']}/{error_info['total_judges']}")
    print(f"Failed: {error_info['failed_judges']}")
    print(f"Action: {error_info['action']}")
    print(f"\nHelp:\n{error_info['help']}")


def demonstrate_comprehensive_troubleshooting():
    """
    Display comprehensive troubleshooting guide.
    
    Requirements: 6.5, 6.6
    """
    print("\n" + "=" * 70)
    print("DEMO 7: Comprehensive Troubleshooting Guide")
    print("=" * 70)
    
    guide = APIErrorHandler.get_comprehensive_troubleshooting_guide()
    print(guide)


def demonstrate_real_world_scenario():
    """
    Demonstrate a real-world scenario with proper error handling.
    
    Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6
    """
    print("\n" + "=" * 70)
    print("DEMO 8: Real-World Scenario with Error Handling")
    print("=" * 70)
    
    try:
        # Step 1: Load and validate API keys
        print("\n1. Loading API keys...")
        api_key_manager = APIKeyManager()
        available_keys = api_key_manager.load_keys()
        
        if not api_key_manager.has_any_keys():
            error_info = APIErrorHandler.handle_missing_keys(available_keys)
            print(f"\n❌ {error_info['message']}")
            print(api_key_manager.get_setup_instructions())
            return
        
        print(f"✓ Found keys for: {api_key_manager.get_available_services()}")
        
        # Step 2: Validate keys
        print("\n2. Validating API keys...")
        validation_results = api_key_manager.validate_all_keys(verbose=False)
        
        for service, valid in validation_results.items():
            if valid:
                print(f"✓ {service.title()}: Valid")
            else:
                print(f"✗ {service.title()}: Invalid")
                status = api_key_manager.get_key_status(service)
                if status and status.error_message:
                    print(f"  Error: {status.error_message}")
        
        # Step 3: Initialize ensemble
        print("\n3. Initializing judge ensemble...")
        config = ToolkitConfig()
        ensemble = APIJudgeEnsemble(
            config=config,
            api_key_manager=api_key_manager,
            parallel_execution=True
        )
        
        judge_count = ensemble.get_judge_count()
        print(f"✓ Initialized {judge_count} judges: {ensemble.get_judge_names()}")
        
        if judge_count == 0:
            print("\n❌ No judges available!")
            print(APIErrorHandler.get_comprehensive_troubleshooting_guide())
            return
        
        # Step 4: Perform evaluation
        print("\n4. Performing evaluation...")
        source_text = "The capital of France is Paris."
        candidate_output = "The capital of France is London."
        
        try:
            verdicts = ensemble.evaluate(
                source_text=source_text,
                candidate_output=candidate_output,
                task="factual_accuracy"
            )
            
            print(f"✓ Evaluation complete: {len(verdicts)} judges succeeded")
            
            # Display results
            for verdict in verdicts:
                print(f"\n  {verdict.judge_name}:")
                print(f"    Score: {verdict.score:.1f}/100")
                print(f"    Confidence: {verdict.confidence:.2f}")
                print(f"    Issues: {len(verdict.issues)}")
            
            # Check for partial failures
            if len(verdicts) < judge_count:
                failed_judges = [
                    name for name in ensemble.get_judge_names()
                    if name not in [v.judge_name for v in verdicts]
                ]
                
                error_info = APIErrorHandler.handle_partial_failure(
                    total_judges=judge_count,
                    successful_judges=len(verdicts),
                    failed_judges=failed_judges
                )
                
                print(f"\n⚠️  {error_info['message']}")
                print(f"Failed judges: {failed_judges}")
            
        except RuntimeError as e:
            print(f"\n❌ Evaluation failed: {e}")
            
            # Handle complete failure
            error_info = APIErrorHandler.handle_partial_failure(
                total_judges=judge_count,
                successful_judges=0,
                failed_judges=ensemble.get_judge_names()
            )
            
            print(f"\n{error_info['help']}")
        
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        logger.exception("Unexpected error in real-world scenario")


def main():
    """Run all error handling demonstrations."""
    print("\n" + "=" * 70)
    print("COMPREHENSIVE ERROR HANDLING DEMONSTRATION")
    print("API Judge Integration - Error Handling Examples")
    print("=" * 70)
    
    # Run all demonstrations
    demonstrate_missing_keys_handling()
    demonstrate_authentication_error_handling()
    demonstrate_network_error_handling()
    demonstrate_rate_limit_handling()
    demonstrate_malformed_response_handling()
    demonstrate_partial_failure_handling()
    demonstrate_comprehensive_troubleshooting()
    demonstrate_real_world_scenario()
    
    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)
    print("\nAll error handling scenarios have been demonstrated.")
    print("Review the output above to see how each error type is handled.")


if __name__ == "__main__":
    main()
