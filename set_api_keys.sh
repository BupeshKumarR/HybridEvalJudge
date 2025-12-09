#!/bin/bash
# Helper script to set API keys for LLM Judge Auditor

echo "=========================================="
echo "🔑 API Key Setup for LLM Judge Auditor"
echo "=========================================="
echo ""
echo "This script will help you set up free API keys for judge models."
echo ""

# Function to get API key from user
get_api_key() {
    local service=$1
    local url=$2
    local var_name=$3
    
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Setting up $service API Key"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "1. Get your FREE API key from: $url"
    echo "2. Enter it below (or press Enter to skip)"
    echo ""
    read -p "Enter $service API key: " api_key
    
    if [ -n "$api_key" ]; then
        export $var_name="$api_key"
        echo "✅ $service API key set for this session"
        
        # Ask if user wants to make it permanent
        read -p "Make this permanent? (add to ~/.zshrc) [y/N]: " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            # Check if key already exists in .zshrc
            if grep -q "export $var_name=" ~/.zshrc 2>/dev/null; then
                echo "⚠️  $var_name already exists in ~/.zshrc"
                read -p "Replace it? [y/N]: " -n 1 -r
                echo
                if [[ $REPLY =~ ^[Yy]$ ]]; then
                    # Remove old line and add new one
                    sed -i.bak "/export $var_name=/d" ~/.zshrc
                    echo "export $var_name=\"$api_key\"" >> ~/.zshrc
                    echo "✅ Updated $var_name in ~/.zshrc"
                fi
            else
                echo "export $var_name=\"$api_key\"" >> ~/.zshrc
                echo "✅ Added $var_name to ~/.zshrc"
            fi
            echo "   Run 'source ~/.zshrc' to reload your shell"
        fi
    else
        echo "⏭️  Skipped $service"
    fi
    echo ""
}

# Get Groq API key
get_api_key "Groq" "https://console.groq.com/keys" "GROQ_API_KEY"

# Get Gemini API key
get_api_key "Gemini" "https://aistudio.google.com/app/apikey" "GEMINI_API_KEY"

echo "=========================================="
echo "✅ Setup Complete!"
echo "=========================================="
echo ""
echo "Current session environment variables:"
if [ -n "$GROQ_API_KEY" ]; then
    echo "  ✅ GROQ_API_KEY is set"
else
    echo "  ❌ GROQ_API_KEY is not set"
fi

if [ -n "$GEMINI_API_KEY" ]; then
    echo "  ✅ GEMINI_API_KEY is set"
else
    echo "  ❌ GEMINI_API_KEY is not set"
fi
echo ""
echo "You can now run the demo or use the evaluation toolkit!"
echo ""
