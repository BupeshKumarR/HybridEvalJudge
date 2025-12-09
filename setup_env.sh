#!/bin/bash
# Setup script for creating a dedicated virtual environment for LLM Judge Auditor

set -e

echo "=========================================="
echo "LLM Judge Auditor - Environment Setup"
echo "=========================================="

# Check if Python 3.9+ is available
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 is not installed"
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "Found Python version: $PYTHON_VERSION"

# Create virtual environment
if [ -d ".venv" ]; then
    echo "Virtual environment already exists at ./.venv"
    read -p "Do you want to recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing virtual environment..."
        rm -rf .venv
    else
        echo "Using existing virtual environment"
        exit 0
    fi
fi

echo "Creating virtual environment..."
python3 -m venv .venv

echo "Activating virtual environment..."
source .venv/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Installing package in editable mode..."
pip install -e .

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "To activate the environment, run:"
echo "  source .venv/bin/activate"
echo ""
echo "To deactivate, run:"
echo "  deactivate"
echo ""
echo "To run tests:"
echo "  pytest"
echo ""
echo "=========================================="
echo "🔑 API Key Setup (Optional)"
echo "=========================================="
echo ""
echo "To use free API-based judges, set these environment variables:"
echo ""
echo "1. Groq (FREE - Llama 3.1 70B):"
echo "   Get key: https://console.groq.com/keys"
echo "   export GROQ_API_KEY=\"your-groq-key\""
echo ""
echo "2. Gemini (FREE - Gemini 1.5 Flash):"
echo "   Get key: https://aistudio.google.com/app/apikey"
echo "   export GEMINI_API_KEY=\"your-gemini-key\""
echo ""
echo "To make them permanent, add to your ~/.zshrc or ~/.bashrc:"
echo "  echo 'export GROQ_API_KEY=\"your-key\"' >> ~/.zshrc"
echo "  echo 'export GEMINI_API_KEY=\"your-key\"' >> ~/.zshrc"
echo ""
