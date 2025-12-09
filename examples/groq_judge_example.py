"""
Example demonstrating the Groq Judge Client for LLM evaluation.

This example shows how to:
1. Initialize the Groq judge client
2. Evaluate candidate outputs for factual accuracy
3. Handle API errors gracefully
4. Parse and display evaluation results

Requirements:
- Set GROQ_API_KEY environment variable
- Install groq package: pip install groq
"""

import os
from llm_judge_auditor.components.groq_judge_client import (
    GroqJudgeClient,
    GroqAPIError,
    GroqRateLimitError,
    GroqAuthenticationError,
)


def main():
    """Demonstrate Groq judge client usage."""
    
    # Check for API key
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("❌ GROQ_API_KEY environment variable not set")
        print("\nTo get a free API key:")
        print("1. Visit https://console.groq.com")
        print("2. Sign up for a free account")
        print("3. Get your API key from https://console.groq.com/keys")
        print("4. Set it: export GROQ_API_KEY='your-key-here'")
        return
    
    print("=" * 70)
    print("Groq Judge Client Example")
    print("=" * 70)
    
    # Initialize the client
    print("\n1. Initializing Groq judge client...")
    try:
        client = GroqJudgeClient(
            api_key=api_key,
            model="llama-3.3-70b-versatile",
            max_retries=2,
            timeout=30
        )
        print(f"✅ Client initialized: {client.get_judge_name()}")
    except Exception as e:
        print(f"❌ Failed to initialize client: {e}")
        return
    
    # Example 1: Factual accuracy evaluation
    print("\n2. Example 1: Factual Accuracy Evaluation")
    print("-" * 70)
    
    source_text = """
    Paris is the capital and largest city of France. It is located on the 
    Seine River in northern France. The city has a population of over 2 million 
    people within its administrative limits.
    """
    
    candidate_output = """
    Paris is the capital of France and is located on the Thames River. 
    It has a population of about 5 million people.
    """
    
    print(f"\nSource Text: {source_text.strip()}")
    print(f"\nCandidate Output: {candidate_output.strip()}")
    
    try:
        print("\n⏳ Evaluating...")
        verdict = client.evaluate(
            source_text=source_text,
            candidate_output=candidate_output,
            task="factual_accuracy"
        )
        
        print(f"\n✅ Evaluation Complete!")
        print(f"\nScore: {verdict.score:.1f}/100")
        print(f"Confidence: {verdict.confidence:.2f}")
        print(f"\nReasoning:\n{verdict.reasoning}")
        
        if verdict.issues:
            print(f"\n⚠️  Issues Found ({len(verdict.issues)}):")
            for i, issue in enumerate(verdict.issues, 1):
                print(f"\n  {i}. [{issue.severity.value.upper()}] {issue.type.value}")
                print(f"     {issue.description}")
                if issue.evidence:
                    print(f"     Location: {issue.evidence[0]}")
        
        print(f"\nMetadata:")
        print(f"  - Response time: {verdict.metadata.get('response_time_seconds', 0):.2f}s")
        print(f"  - Tokens used: {verdict.metadata.get('tokens_used', 0)}")
        
    except GroqAuthenticationError as e:
        print(f"\n❌ Authentication Error: {e}")
        print("Please check your GROQ_API_KEY")
    except GroqRateLimitError as e:
        print(f"\n❌ Rate Limit Error: {e}")
        print("Please wait a moment before trying again")
    except GroqAPIError as e:
        print(f"\n❌ API Error: {e}")
    except Exception as e:
        print(f"\n❌ Unexpected Error: {e}")
    
    # Example 2: Bias detection
    print("\n\n3. Example 2: Bias Detection")
    print("-" * 70)
    
    candidate_output_bias = """
    Women are naturally better at nursing and caregiving roles, while men 
    excel in leadership and technical positions. This is just how things are.
    """
    
    print(f"\nCandidate Output: {candidate_output_bias.strip()}")
    
    try:
        print("\n⏳ Evaluating for bias...")
        verdict = client.evaluate(
            source_text="",  # No source needed for bias detection
            candidate_output=candidate_output_bias,
            task="bias_detection"
        )
        
        print(f"\n✅ Evaluation Complete!")
        print(f"\nScore: {verdict.score:.1f}/100 (100 = no bias)")
        print(f"Confidence: {verdict.confidence:.2f}")
        print(f"\nReasoning:\n{verdict.reasoning}")
        
        if verdict.issues:
            print(f"\n⚠️  Bias Issues Found ({len(verdict.issues)}):")
            for i, issue in enumerate(verdict.issues, 1):
                print(f"\n  {i}. [{issue.severity.value.upper()}]")
                print(f"     {issue.description}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
    
    # Example 3: Accurate output
    print("\n\n4. Example 3: Accurate Output")
    print("-" * 70)
    
    accurate_output = """
    Paris is the capital and largest city of France. It is situated on the 
    Seine River in northern France. The city proper has a population of 
    approximately 2.1 million people.
    """
    
    print(f"\nSource Text: {source_text.strip()}")
    print(f"\nCandidate Output: {accurate_output.strip()}")
    
    try:
        print("\n⏳ Evaluating...")
        verdict = client.evaluate(
            source_text=source_text,
            candidate_output=accurate_output,
            task="factual_accuracy"
        )
        
        print(f"\n✅ Evaluation Complete!")
        print(f"\nScore: {verdict.score:.1f}/100")
        print(f"Confidence: {verdict.confidence:.2f}")
        print(f"\nReasoning:\n{verdict.reasoning}")
        
        if verdict.issues:
            print(f"\n⚠️  Issues Found: {len(verdict.issues)}")
        else:
            print(f"\n✅ No issues found!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
    
    print("\n" + "=" * 70)
    print("Example Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
