#!/usr/bin/env python3
"""
LLM Judge Auditor - Free API Judge Evaluation Demo
===================================================

Professional LLM evaluation using FREE API-based judges - no model downloads!

✅ Features:
   - Uses free API judges (Groq Llama 3.3 70B, Google Gemini Flash)
   - No model downloads required
   - Fast evaluation (2-5 seconds)
   - Professional quality scores
   - Completely free tier

✅ Requirements:
   - Free API keys (Groq and/or Gemini)
   - Internet connection
   - No GPU or high RAM needed

Quick Setup (2 minutes):
   1. Get free API keys:
      - Groq: https://console.groq.com/keys
      - Gemini: https://aistudio.google.com/app/apikey
   2. Set environment variables:
      export GROQ_API_KEY="your-key"
      export GEMINI_API_KEY="your-key"
   3. Run demo: python demo/demo.py
"""

import sys
import json
import os
import subprocess
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))



def print_header(title: str):
    print("\n" + "=" * 80)
    print(f"🔍 {title}")
    print("=" * 80 + "\n")

def print_section(title: str):
    print(f"\n{'─' * 80}")
    print(f"📋 {title}")
    print('─' * 80 + "\n")

def prompt_for_api_key(service: str, signup_url: str, key_url: str) -> Optional[str]:
    """
    Prompt user to enter an API key for a service.
    
    Args:
        service: Service name (e.g., "Groq", "Gemini")
        signup_url: URL to sign up for the service
        key_url: URL to get API key
    
    Returns:
        API key string or None if user skips
    """
    print(f"\n🔑 {service} API Key Setup")
    print(f"   Sign up: {signup_url}")
    print(f"   Get key: {key_url}")
    print(f"\n   Enter your {service} API key (or press Enter to skip):")
    
    api_key = input("   > ").strip()
    
    if api_key:
        return api_key
    else:
        print(f"   ⚠️  Skipped {service} setup")
        return None

def save_api_keys_to_env(groq_key: Optional[str] = None, gemini_key: Optional[str] = None) -> bool:
    """
    Save API keys to environment variables for current session.
    
    Args:
        groq_key: Groq API key
        gemini_key: Gemini API key
    
    Returns:
        True if at least one key was saved
    """
    saved_any = False
    
    if groq_key:
        os.environ["GROQ_API_KEY"] = groq_key
        print("   ✅ Groq API key set for this session")
        saved_any = True
    
    if gemini_key:
        os.environ["GEMINI_API_KEY"] = gemini_key
        print("   ✅ Gemini API key set for this session")
        saved_any = True
    
    if saved_any:
        print("\n   💡 To make these permanent, add to your shell profile:")
        if groq_key:
            print(f'      export GROQ_API_KEY="{groq_key}"')
        if gemini_key:
            print(f'      export GEMINI_API_KEY="{gemini_key}"')
    
    return saved_any

def interactive_api_key_setup() -> bool:
    """
    Interactive setup for API keys.
    
    Prompts user for API keys and validates them.
    
    Returns:
        True if at least one valid key was configured
    """
    print_header("Interactive API Key Setup")
    
    print("This demo requires at least one free API key.")
    print("Both APIs are completely FREE with generous limits!\n")
    
    # Prompt for Groq key
    groq_key = prompt_for_api_key(
        service="Groq",
        signup_url="https://console.groq.com",
        key_url="https://console.groq.com/keys"
    )
    
    # Prompt for Gemini key
    gemini_key = prompt_for_api_key(
        service="Gemini",
        signup_url="https://aistudio.google.com",
        key_url="https://aistudio.google.com/app/apikey"
    )
    
    if not groq_key and not gemini_key:
        print("\n❌ No API keys provided. Cannot proceed without at least one key.")
        return False
    
    # Save keys to environment
    print("\n📝 Saving API keys...")
    save_api_keys_to_env(groq_key, gemini_key)
    
    # Validate keys
    print("\n🔍 Validating API keys...")
    
    try:
        from llm_judge_auditor.components.api_key_manager import APIKeyManager
        
        api_key_manager = APIKeyManager()
        api_key_manager.load_keys()
        
        validation_results = api_key_manager.validate_all_keys(verbose=False)
        
        valid_count = sum(1 for valid in validation_results.values() if valid)
        
        if valid_count > 0:
            print(f"\n✅ {valid_count} API key(s) validated successfully!")
            print(api_key_manager.get_validation_summary())
            return True
        else:
            print("\n❌ API key validation failed.")
            print(api_key_manager.get_validation_summary())
            print(api_key_manager.get_troubleshooting_guide())
            return False
    
    except ImportError:
        print("\n⚠️  Could not validate keys (toolkit not installed)")
        print("   Keys have been set, but validation skipped")
        return True
    except Exception as e:
        print(f"\n⚠️  Validation error: {e}")
        print("   Keys have been set, but validation failed")
        return True



def demo_api_judge_evaluation():
    """Demonstrate API judge evaluation"""
    print_header("DEMO: API Judge Evaluation")
    
    # Ask user for question
    print("💬 Enter your question (or press Enter for default diabetes question):")
    user_question = input("   > ").strip()
    
    if user_question:
        question = user_question
        reference = "User-provided question (no reference available)"
    else:
        # Default medical question for testing
        question = "What are the early warning signs of Type 2 diabetes? List the main symptoms."
        
        # Reference information
        reference = """Type 2 diabetes symptoms often develop slowly. Common symptoms include:
- Increased thirst and frequent urination
- Increased hunger
- Unintended weight loss
- Fatigue and weakness
- Blurred vision
- Slow-healing sores or frequent infections
- Numbness or tingling in hands or feet
- Darkened skin areas, usually in armpits and neck"""
    
    print(f"\n📝 Test Question:")
    print(f"   {question}\n")
    
    print(f"📚 Reference (Medical Source):")
    print(f"   {reference[:150]}...\n")
    
    # Generate a sample response to evaluate
    print("🤖 Generating sample response to evaluate...")
    
    # For demo purposes, create a sample response
    # In a real scenario, this would come from an LLM
    sample_response = """Type 2 diabetes has several early warning signs:
1. Increased thirst and frequent urination - Your body tries to flush out excess sugar
2. Increased hunger - Cells aren't getting enough glucose for energy
3. Fatigue - Without proper glucose metabolism, you feel tired
4. Blurred vision - High blood sugar affects the lens of your eye
5. Slow healing - High blood sugar impairs circulation and immune function
6. Tingling in extremities - Nerve damage from prolonged high blood sugar

These symptoms develop gradually and may be mild at first."""
    
    print(f"   Generated response ({len(sample_response)} characters)\n")
    
    # Display the response
    print_section("Response to Evaluate")
    print(sample_response)
    
    # Evaluate using API judges
    print_section("Evaluating with API Judges")
    
    try:
        from llm_judge_auditor.components.api_key_manager import APIKeyManager
        from llm_judge_auditor.components.api_judge_ensemble import APIJudgeEnsemble
        from llm_judge_auditor.config import ToolkitConfig
        
        # Initialize API key manager
        api_key_manager = APIKeyManager()
        api_key_manager.load_keys()
        
        if not api_key_manager.has_any_keys():
            print("❌ No API keys found. Cannot evaluate.")
            return
        
        # Show which judges are available
        available_services = api_key_manager.get_available_services()
        print(f"✅ Available judges: {', '.join(available_services)}")
        print()
        
        # Initialize config and ensemble
        config = ToolkitConfig()
        ensemble = APIJudgeEnsemble(
            config=config,
            api_key_manager=api_key_manager,
            parallel_execution=True
        )
        
        judge_count = ensemble.get_judge_count()
        judge_names = ensemble.get_judge_names()
        
        print(f"🎯 Evaluating with {judge_count} judge(s): {', '.join(judge_names)}")
        print("   (This may take 5-10 seconds...)\n")
        
        # Evaluate
        start_time = time.time()
        verdicts = ensemble.evaluate(
            source_text=reference,
            candidate_output=sample_response,
            task="factual_accuracy"
        )
        elapsed = time.time() - start_time
        
        print(f"✅ Evaluation complete in {elapsed:.1f} seconds\n")
        
        # Display results
        print_section("EVALUATION RESULTS")
        
        # Show individual judge scores
        print("📊 Individual Judge Scores:\n")
        for verdict in verdicts:
            print(f"   🤖 {verdict.judge_name}")
            print(f"      Score: {verdict.score:.1f}/100")
            print(f"      Confidence: {verdict.confidence:.2f}")
            print(f"      Reasoning: {verdict.reasoning[:200]}...")
            if verdict.issues:
                print(f"      Issues found: {len(verdict.issues)}")
                for issue in verdict.issues[:3]:  # Show first 3 issues
                    print(f"         - [{issue.severity}] {issue.description}")
            print()
        
        # Aggregate scores
        consensus, individual_scores, disagreement = ensemble.aggregate_verdicts(verdicts)
        
        print("🎯 Consensus Score:")
        print(f"   {consensus:.1f}/100")
        print(f"   Disagreement level: {disagreement:.1f}")
        print()
        
        # Check for disagreements
        disagreement_analysis = ensemble.identify_disagreements(verdicts, threshold=20.0)
        
        if disagreement_analysis['has_disagreement']:
            print("⚠️  Judges disagree significantly!")
            print(f"   Score range: {disagreement_analysis['score_range'][0]:.1f} - {disagreement_analysis['score_range'][1]:.1f}")
            if disagreement_analysis['outliers']:
                print(f"   Outliers: {', '.join(disagreement_analysis['outliers'])}")
            print()
        
        # Final verdict
        verdict_emoji = "✅" if consensus >= 70 else "⚠️" if consensus >= 50 else "❌"
        verdict_text = "APPROVED" if consensus >= 70 else "NEEDS REVIEW" if consensus >= 50 else "REJECTED"
        
        print(f"🏆 FINAL VERDICT: {verdict_emoji} {verdict_text}")
        print(f"   Consensus Score: {consensus:.1f}/100")
        print(f"   Judges Used: {', '.join(judge_names)}")
        
        # Save results
        output_data = {
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "reference": reference,
            "response": sample_response,
            "judges_used": judge_names,
            "individual_scores": individual_scores,
            "consensus_score": consensus,
            "disagreement_level": disagreement,
            "verdict": verdict_text,
            "verdicts": [
                {
                    "judge_name": v.judge_name,
                    "score": v.score,
                    "confidence": v.confidence,
                    "reasoning": v.reasoning,
                    "issues": [
                        {
                            "type": str(i.type),
                            "severity": str(i.severity),
                            "description": i.description,
                            "evidence": i.evidence
                        }
                        for i in v.issues
                    ]
                }
                for v in verdicts
            ]
        }
        
        output_file = Path("demo/results.json")
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\n📁 Results saved to: {output_file}")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Install required packages:")
        print("   pip install groq google-generativeai")
        print("   pip install -e .")
    except Exception as e:
        print(f"❌ Evaluation error: {e}")
        import traceback
        traceback.print_exc()



def main():
    """Main demo function"""
    print_header("LLM Judge Auditor - API Judge Evaluation Demo")
    print()
    
    # Check for API keys
    print("🔍 Checking for API keys...")
    
    try:
        from llm_judge_auditor.components.api_key_manager import APIKeyManager
        
        api_key_manager = APIKeyManager()
        available_keys = api_key_manager.load_keys()
        
        has_groq = available_keys.get("groq", False)
        has_gemini = available_keys.get("gemini", False)
        
        if has_groq or has_gemini:
            print(f"✅ Found API keys:")
            if has_groq:
                print("   • Groq API key detected")
            if has_gemini:
                print("   • Gemini API key detected")
            print()
            
            # Validate keys
            print("🔍 Validating API keys...")
            validation_results = api_key_manager.validate_all_keys(verbose=False)
            
            valid_count = sum(1 for valid in validation_results.values() if valid)
            
            if valid_count > 0:
                print(f"✅ {valid_count} API key(s) validated successfully!\n")
                
                # Run the demo
                demo_api_judge_evaluation()
                
                # Summary
                print_header("DEMO COMPLETE")
                print("✅ Successfully evaluated response with API judges")
                print("📁 Results saved to: demo/results.json")
                print("\n💡 Try different questions by running the demo again")
                print("=" * 80 + "\n")
            else:
                print("❌ API key validation failed\n")
                print(api_key_manager.get_validation_summary())
                print(api_key_manager.get_troubleshooting_guide())
                
                # Offer interactive setup
                print("\n💡 Would you like to re-enter your API keys? (y/n)")
                response = input("   > ").strip().lower()
                
                if response == 'y':
                    if interactive_api_key_setup():
                        # Try demo again
                        demo_api_judge_evaluation()
                        
                        print_header("DEMO COMPLETE")
                        print("✅ Successfully evaluated response with API judges")
                        print("📁 Results saved to: demo/results.json")
                        print("=" * 80 + "\n")
        else:
            # No API keys found - show setup guide
            print("❌ No API keys found\n")
            print(api_key_manager.get_setup_instructions(show_validation=False))
            
            # Offer interactive setup
            print("\n💡 Would you like to set up API keys now? (y/n)")
            response = input("   > ").strip().lower()
            
            if response == 'y':
                if interactive_api_key_setup():
                    # Run the demo
                    demo_api_judge_evaluation()
                    
                    # Summary
                    print_header("DEMO COMPLETE")
                    print("✅ Successfully evaluated response with API judges")
                    print("📁 Results saved to: demo/results.json")
                    print("\n💡 Try different questions by running the demo again")
                    print("=" * 80 + "\n")
            else:
                print("\n📋 To set up API keys manually:")
                print("   1. Get free API keys from:")
                print("      • Groq: https://console.groq.com/keys")
                print("      • Gemini: https://aistudio.google.com/app/apikey")
                print("   2. Set environment variables:")
                print('      export GROQ_API_KEY="your-key"')
                print('      export GEMINI_API_KEY="your-key"')
                print("   3. Run this demo again: python demo/demo.py")
                print()
    
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("\n📦 Please install required packages:")
        print("   pip install groq google-generativeai")
        print("   pip install -e .")
        print()
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
