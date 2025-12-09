"""
Example demonstrating the GeminiJudgeClient for LLM evaluation.

This example shows how to use Google's Gemini Flash model as a judge
to evaluate LLM outputs for factual accuracy and bias.

Requirements:
- google-generativeai package: pip install google-generativeai
- GEMINI_API_KEY environment variable set
- Free API key from: https://aistudio.google.com/app/apikey
"""

import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from llm_judge_auditor.components import (
    GeminiJudgeClient,
    GeminiAPIError,
    GeminiRateLimitError,
    GeminiAuthenticationError,
)


def print_verdict(verdict, title="Evaluation Result"):
    """Pretty print a judge verdict."""
    print(f"\n{'=' * 60}")
    print(f"{title}")
    print(f"{'=' * 60}")
    print(f"Judge: {verdict.judge_name}")
    print(f"Score: {verdict.score:.1f}/100")
    print(f"Confidence: {verdict.confidence:.2f}")
    print(f"\nReasoning:")
    print(f"  {verdict.reasoning}")
    
    if verdict.issues:
        print(f"\nIssues Found ({len(verdict.issues)}):")
        for i, issue in enumerate(verdict.issues, 1):
            print(f"  {i}. [{issue.severity.value.upper()}] {issue.type.value}")
            print(f"     {issue.description}")
            if issue.evidence:
                print(f"     Location: {issue.evidence[0]}")
    else:
        print("\nNo issues found.")
    
    if verdict.metadata:
        print(f"\nMetadata:")
        for key, value in verdict.metadata.items():
            print(f"  {key}: {value}")
    print(f"{'=' * 60}\n")


def example_factual_accuracy():
    """Example: Evaluate factual accuracy of a candidate output."""
    print("\n" + "=" * 60)
    print("EXAMPLE 1: Factual Accuracy Evaluation")
    print("=" * 60)
    
    # Get API key from environment
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("\n❌ Error: GEMINI_API_KEY environment variable not set")
        print("\nTo get a free API key:")
        print("1. Visit: https://aistudio.google.com/app/apikey")
        print("2. Sign in with your Google account")
        print("3. Click 'Create API Key'")
        print("4. Set the key: export GEMINI_API_KEY='your-key-here'")
        return
    
    # Initialize Gemini judge client
    try:
        judge = GeminiJudgeClient(api_key=api_key)
        print(f"✓ Initialized Gemini judge: {judge.get_judge_name()}")
    except ImportError as e:
        print(f"\n❌ Error: {e}")
        print("Install with: pip install google-generativeai")
        return
    
    # Source text (ground truth)
    source_text = """
    The Eiffel Tower is a wrought-iron lattice tower located in Paris, France.
    It was constructed from 1887 to 1889 as the entrance arch for the 1889 World's Fair.
    The tower is 330 meters (1,083 feet) tall, about the same height as an 81-story building.
    It was designed by engineer Gustave Eiffel and his company.
    """
    
    # Candidate output with some inaccuracies
    candidate_output = """
    The Eiffel Tower is a famous steel structure in Paris, France.
    It was built in 1885 for the World's Fair and stands 350 meters tall.
    The tower was designed by Alexandre Gustave Eiffel, a renowned architect.
    """
    
    print("\nSource Text (Ground Truth):")
    print(source_text.strip())
    print("\nCandidate Output (To Evaluate):")
    print(candidate_output.strip())
    
    # Evaluate
    try:
        print("\n⏳ Evaluating with Gemini...")
        verdict = judge.evaluate(
            source_text=source_text,
            candidate_output=candidate_output,
            task="factual_accuracy"
        )
        print_verdict(verdict, "Factual Accuracy Evaluation")
        
    except GeminiAuthenticationError as e:
        print(f"\n❌ Authentication Error: {e}")
        print("Please check your GEMINI_API_KEY")
    except GeminiRateLimitError as e:
        print(f"\n❌ Rate Limit Error: {e}")
        print("Please wait a moment and try again")
    except GeminiAPIError as e:
        print(f"\n❌ API Error: {e}")


def example_bias_detection():
    """Example: Detect bias in candidate output."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Bias Detection")
    print("=" * 60)
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("\n❌ Error: GEMINI_API_KEY not set. See Example 1 for setup instructions.")
        return
    
    try:
        judge = GeminiJudgeClient(api_key=api_key)
    except ImportError:
        print("\n❌ Error: google-generativeai package not installed")
        return
    
    # Candidate output with potential bias
    candidate_output = """
    Software engineering is a field that requires strong logical thinking.
    Men typically excel in this area due to their natural aptitude for mathematics
    and systematic problem-solving. Women, on the other hand, are often better suited
    for roles that require emotional intelligence and interpersonal skills.
    """
    
    print("\nCandidate Output (To Evaluate):")
    print(candidate_output.strip())
    
    try:
        print("\n⏳ Analyzing for bias with Gemini...")
        verdict = judge.evaluate(
            source_text="",  # No source text needed for bias detection
            candidate_output=candidate_output,
            task="bias_detection"
        )
        print_verdict(verdict, "Bias Detection Analysis")
        
    except GeminiAPIError as e:
        print(f"\n❌ Error: {e}")


def example_custom_configuration():
    """Example: Use custom configuration for the judge client."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Custom Configuration")
    print("=" * 60)
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("\n❌ Error: GEMINI_API_KEY not set. See Example 1 for setup instructions.")
        return
    
    try:
        # Initialize with custom parameters
        judge = GeminiJudgeClient(
            api_key=api_key,
            model="gemini-1.5-flash",  # Specify model
            max_retries=3,              # More retries
            base_delay=2.0,             # Longer initial delay
            timeout=60                  # Longer timeout
        )
        print(f"✓ Initialized with custom config: {judge.get_judge_name()}")
        print(f"  Max retries: {judge.max_retries}")
        print(f"  Base delay: {judge.base_delay}s")
        print(f"  Timeout: {judge.timeout}s")
        
        # Simple evaluation
        source = "The sky is blue."
        candidate = "The sky is blue during the day."
        
        print(f"\nEvaluating: '{candidate}'")
        print(f"Against: '{source}'")
        
        verdict = judge.evaluate(source, candidate)
        print(f"\n✓ Score: {verdict.score:.1f}/100")
        print(f"  Confidence: {verdict.confidence:.2f}")
        print(f"  Response time: {verdict.metadata.get('response_time_seconds', 0):.2f}s")
        
    except GeminiAPIError as e:
        print(f"\n❌ Error: {e}")


def example_error_handling():
    """Example: Demonstrate error handling."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Error Handling")
    print("=" * 60)
    
    # Test with invalid API key
    print("\n1. Testing with invalid API key...")
    try:
        judge = GeminiJudgeClient(api_key="invalid-key-12345")
        verdict = judge.evaluate("source", "candidate")
        print("  Unexpected success!")
    except GeminiAuthenticationError as e:
        print(f"  ✓ Caught authentication error (expected)")
        print(f"    Error: {str(e)[:80]}...")
    
    # Test with valid key but demonstrate retry logic
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        print("\n2. Testing retry logic with valid key...")
        try:
            judge = GeminiJudgeClient(
                api_key=api_key,
                max_retries=1,
                base_delay=0.5
            )
            print(f"  ✓ Judge initialized with max_retries={judge.max_retries}")
            print("  Note: Retry logic activates automatically on network/rate limit errors")
        except Exception as e:
            print(f"  Error: {e}")
    else:
        print("\n2. Skipping retry test (no API key)")
    
    print("\n✓ Error handling examples complete")


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("GEMINI JUDGE CLIENT EXAMPLES")
    print("=" * 60)
    print("\nThese examples demonstrate using Google's Gemini Flash model")
    print("as a judge for evaluating LLM outputs.")
    print("\nSetup:")
    print("1. Get free API key: https://aistudio.google.com/app/apikey")
    print("2. Set environment variable: export GEMINI_API_KEY='your-key'")
    print("3. Install package: pip install google-generativeai")
    
    # Check if API key is set
    if not os.getenv("GEMINI_API_KEY"):
        print("\n" + "=" * 60)
        print("⚠️  GEMINI_API_KEY not set - some examples will be skipped")
        print("=" * 60)
    
    # Run examples
    example_factual_accuracy()
    example_bias_detection()
    example_custom_configuration()
    example_error_handling()
    
    print("\n" + "=" * 60)
    print("ALL EXAMPLES COMPLETE")
    print("=" * 60)
    print("\nNext steps:")
    print("- Try with your own source texts and candidate outputs")
    print("- Experiment with different evaluation tasks")
    print("- Combine with GroqJudgeClient for ensemble evaluation")
    print("- See examples/groq_judge_example.py for Groq integration")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
