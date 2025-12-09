#!/usr/bin/env python3
"""
LLM Judge Auditor - Model Setup Script
======================================

Automatically sets up the best free models for your laptop.
Detects your system specs and installs optimal models.

Usage:
    python demo/setup.py
"""

import subprocess
import sys
import time
from pathlib import Path

try:
    import psutil
except ImportError:
    print("Installing psutil for system detection...")
    subprocess.run([sys.executable, "-m", "pip", "install", "psutil"])
    import psutil

def check_ollama():
    """Check if Ollama is installed"""
    try:
        result = subprocess.run(['ollama', '--version'], 
                              capture_output=True, 
                              timeout=5)
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False

def get_system_info():
    """Get system information"""
    import platform
    
    return {
        'os': platform.system(),
        'ram_gb': psutil.virtual_memory().total / (1024**3),
        'available_ram_gb': psutil.virtual_memory().available / (1024**3),
        'cpu_count': psutil.cpu_count()
    }

def recommend_models(ram_gb):
    """Recommend models based on available RAM"""
    if ram_gb >= 32:
        return {
            'tier': 'High-end (32GB+ RAM)',
            'models': [
                ('phi3', 'Phi-3 Mini (3.8B) - Excellent factual accuracy', '~2.3GB'),
                ('llama3.2:3b', 'Llama 3.2 3B - Strong reasoning', '~1.9GB'),
                ('qwen2.5:3b', 'Qwen2.5 3B - Multilingual support', '~1.9GB'),
                ('mistral', 'Mistral 7B - High quality general purpose', '~4.1GB')
            ],
            'note': 'You can run all models simultaneously!'
        }
    elif ram_gb >= 16:
        return {
            'tier': 'Mid-range (16GB RAM)',
            'models': [
                ('phi3', 'Phi-3 Mini (3.8B) - Best for factual Q&A', '~2.3GB'),
                ('llama3.2:3b', 'Llama 3.2 3B - Good reasoning', '~1.9GB'),
                ('qwen2.5:3b', 'Qwen2.5 3B - Multilingual', '~1.9GB')
            ],
            'note': 'Run 2-3 models, switch between them'
        }
    elif ram_gb >= 8:
        return {
            'tier': 'Standard (8GB RAM)',
            'models': [
                ('phi3', 'Phi-3 Mini (3.8B) - Best overall choice', '~2.3GB'),
                ('llama3.2:1b', 'Llama 3.2 1B - Fastest, still good', '~1GB')
            ],
            'note': 'Run one model at a time'
        }
    else:
        return {
            'tier': 'Limited (<8GB RAM)',
            'models': [
                ('llama3.2:1b', 'Llama 3.2 1B - Smallest, fastest', '~1GB')
            ],
            'note': 'Consider upgrading RAM for better performance'
        }

def pull_model(model_name, description, size):
    """Pull a model with progress indication"""
    print(f"\n📥 Installing {model_name}...")
    print(f"   {description}")
    print(f"   Size: {size}")
    print("   This may take 2-5 minutes...")
    
    try:
        # Run ollama pull with real-time output
        process = subprocess.Popen(
            ['ollama', 'pull', model_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Show progress
        for line in process.stdout:
            if 'pulling' in line.lower() or 'downloading' in line.lower():
                print(f"   {line.strip()}")
        
        process.wait()
        
        if process.returncode == 0:
            print(f"   ✅ {model_name} installed successfully!")
            return True
        else:
            print(f"   ❌ Failed to install {model_name}")
            return False
            
    except Exception as e:
        print(f"   ❌ Error installing {model_name}: {e}")
        return False

def main():
    print("=" * 80)
    print("🚀 FREE MODEL SETUP: Best LLMs for Your Laptop")
    print("=" * 80)
    print()
    
    # Check Ollama
    if not check_ollama():
        print("❌ Ollama not found. Please install it first:")
        print()
        print("1. Visit: https://ollama.ai")
        print("2. Download and install Ollama")
        print("3. Run this script again")
        print()
        return
    
    print("✅ Ollama detected!")
    
    # Get system info
    sys_info = get_system_info()
    
    print(f"\n💻 Your System:")
    print(f"   OS: {sys_info['os']}")
    print(f"   RAM: {sys_info['ram_gb']:.1f} GB")
    print(f"   Available: {sys_info['available_ram_gb']:.1f} GB")
    print(f"   CPU Cores: {sys_info['cpu_count']}")
    
    # Get recommendations
    recommendations = recommend_models(sys_info['ram_gb'])
    
    print(f"\n🎯 Recommended Setup: {recommendations['tier']}")
    print(f"   💡 {recommendations['note']}")
    print()
    
    print("📦 Recommended Models:")
    for model, description, size in recommendations['models']:
        print(f"   • {model}: {description} ({size})")
    
    # Ask for confirmation
    print(f"\n❓ Install {len(recommendations['models'])} recommended models?")
    print("   This will download ~2-8GB depending on your system.")
    print()
    
    try:
        response = input("Continue? (y/n): ").strip().lower()
    except KeyboardInterrupt:
        print("\n\n❌ Installation cancelled.")
        return
    
    if response != 'y':
        print("\n❌ Installation cancelled.")
        print("\n💡 To install manually:")
        for model, _, _ in recommendations['models']:
            print(f"   ollama pull {model}")
        return
    
    # Install models
    print("\n🚀 Starting installation...")
    
    installed = []
    failed = []
    
    for model, description, size in recommendations['models']:
        if pull_model(model, description, size):
            installed.append(model)
        else:
            failed.append(model)
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 INSTALLATION SUMMARY")
    print("=" * 80)
    
    if installed:
        print(f"\n✅ Successfully installed ({len(installed)}/{len(recommendations['models'])}):")
        for model in installed:
            print(f"   • {model}")
    
    if failed:
        print(f"\n❌ Failed to install ({len(failed)}/{len(recommendations['models'])}):")
        for model in failed:
            print(f"   • {model}")
        print("\n💡 Try installing manually:")
        for model in failed:
            print(f"   ollama pull {model}")
    
    if installed:
        print("\n🎯 Next Steps:")
        print("   1. Run the demo: python demo/demo.py")
        print("   2. Test individual models: ollama run <model>")
        print("   3. Try different questions and prompts")
        
        print("\n💡 Quick Test:")
        print(f"   ollama run {installed[0]} 'What is machine learning?'")
        
        print("\n🚀 Ready for professional LLM evaluation!")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
