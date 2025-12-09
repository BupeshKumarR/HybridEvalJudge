#!/usr/bin/env python3
"""
Test script to verify API keys are working correctly.
"""

import os
import sys

def test_groq_api():
    """Test Groq API connection."""
    print("=" * 60)
    print("Testing Groq API")
    print("=" * 60)
    
    api_key = os.environ.get("GROQ_API_KEY")
    
    if not api_key:
        print("❌ GROQ_API_KEY not found in environment")
        print("   Set it with: export GROQ_API_KEY='your-key'")
        return False
    
    print(f"✅ API key found: {api_key[:10]}...{api_key[-4:]}")
    
    try:
        from groq import Groq
        
        print("📡 Making test API call to Groq...")
        client = Groq(api_key=api_key)
        
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "user", "content": "Say 'Hello from Groq!' in exactly 3 words."}
            ],
            max_tokens=10
        )
        
        result = response.choices[0].message.content
        print(f"✅ Groq API is working!")
        print(f"   Response: {result}")
        print(f"   Model: {response.model}")
        print(f"   Tokens used: {response.usage.total_tokens}")
        return True
        
    except ImportError:
        print("❌ groq package not installed")
        print("   Install with: pip install groq")
        return False
    except Exception as e:
        print(f"❌ Groq API call failed: {str(e)}")
        if "401" in str(e) or "authentication" in str(e).lower():
            print("   This looks like an authentication error.")
            print("   Please check your API key is correct.")
        return False


def test_gemini_api():
    """Test Gemini API connection."""
    print("\n" + "=" * 60)
    print("Testing Gemini API")
    print("=" * 60)
    
    api_key = os.environ.get("GEMINI_API_KEY")
    
    if not api_key:
        print("❌ GEMINI_API_KEY not found in environment")
        print("   Set it with: export GEMINI_API_KEY='your-key'")
        return False
    
    print(f"✅ API key found: {api_key[:10]}...{api_key[-4:]}")
    
    try:
        import google.generativeai as genai
        
        print("📡 Making test API call to Gemini...")
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.0-flash-exp")
        
        response = model.generate_content(
            "Say 'Hello from Gemini!' in exactly 3 words.",
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=10,
                temperature=0.1
            )
        )
        
        result = response.text
        print(f"✅ Gemini API is working!")
        print(f"   Response: {result}")
        print(f"   Model: gemini-2.0-flash-exp")
        return True
        
    except ImportError:
        print("❌ google-generativeai package not installed")
        print("   Install with: pip install google-generativeai")
        return False
    except Exception as e:
        print(f"❌ Gemini API call failed: {str(e)}")
        if "401" in str(e) or "403" in str(e) or "API key" in str(e):
            print("   This looks like an authentication error.")
            print("   Please check your API key is correct.")
        return False


def test_api_key_manager():
    """Test the APIKeyManager component."""
    print("\n" + "=" * 60)
    print("Testing APIKeyManager Component")
    print("=" * 60)
    
    try:
        from llm_judge_auditor.components import APIKeyManager
        
        manager = APIKeyManager()
        keys = manager.load_keys()
        
        print(f"Keys detected: {keys}")
        print(f"Available services: {manager.get_available_services()}")
        
        if keys['groq']:
            print("\n🔍 Validating Groq key through APIKeyManager...")
            if manager.validate_groq_key():
                print("✅ Groq key validated successfully")
            else:
                print("❌ Groq key validation failed")
                status = manager.get_key_status('groq')
                if status and status.error_message:
                    print(f"   Error: {status.error_message}")
        
        if keys['gemini']:
            print("\n🔍 Validating Gemini key through APIKeyManager...")
            if manager.validate_gemini_key():
                print("✅ Gemini key validated successfully")
            else:
                print("❌ Gemini key validation failed")
                status = manager.get_key_status('gemini')
                if status and status.error_message:
                    print(f"   Error: {status.error_message}")
        
        return True
        
    except Exception as e:
        print(f"❌ APIKeyManager test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all API tests."""
    print("\n" + "🔑" * 30)
    print("API Key Validation Test")
    print("🔑" * 30 + "\n")
    
    results = {
        'groq': test_groq_api(),
        'gemini': test_gemini_api(),
        'manager': test_api_key_manager()
    }
    
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    for service, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{service.capitalize():15} {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n🎉 All tests passed! Your API keys are working correctly.")
        print("\nYou're ready to continue with the next implementation tasks!")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
        print("\nTroubleshooting:")
        print("1. Make sure you've exported the API keys in your current shell")
        print("2. Verify the keys are correct (no extra spaces or quotes)")
        print("3. Check you have internet connectivity")
        return 1


if __name__ == "__main__":
    sys.exit(main())
