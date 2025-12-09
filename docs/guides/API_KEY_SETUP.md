# API Key Setup Guide

This guide explains how to set up free API keys for the LLM Judge Auditor's API-based judge models.

## Quick Start

### Option 1: Interactive Setup Script (Easiest)

```bash
./set_api_keys.sh
```

This script will:
- Guide you through getting API keys
- Set them for your current session
- Optionally add them permanently to your shell profile

### Option 2: Manual Setup

#### Get Your Free API Keys

1. **Groq (Llama 3.1 70B)** - FREE
   - Sign up: https://console.groq.com
   - Get API key: https://console.groq.com/keys
   - Free tier: 30 requests/minute

2. **Google Gemini Flash** - FREE
   - Sign up: https://aistudio.google.com
   - Get API key: https://aistudio.google.com/app/apikey
   - Free tier: 15 requests/minute

#### Set Environment Variables

**For current session only:**
```bash
export GROQ_API_KEY="your-groq-api-key-here"
export GEMINI_API_KEY="your-gemini-api-key-here"
```

**To make permanent (add to ~/.zshrc or ~/.bashrc):**
```bash
echo 'export GROQ_API_KEY="your-groq-api-key-here"' >> ~/.zshrc
echo 'export GEMINI_API_KEY="your-gemini-api-key-here"' >> ~/.zshrc
source ~/.zshrc
```

## Verify Setup

Test that your API keys are loaded:

```bash
python -c "
from llm_judge_auditor.components import APIKeyManager

manager = APIKeyManager()
keys = manager.load_keys()
print('Available services:', manager.get_available_services())
"
```

You should see:
```
Available services: ['groq', 'gemini']
```

## Usage

Once your API keys are set, the evaluation toolkit will automatically use them:

```python
from llm_judge_auditor import EvaluationToolkit

# The toolkit will automatically detect and use your API keys
toolkit = EvaluationToolkit()
```

## Troubleshooting

### Keys not detected

If your keys aren't being detected:

1. Check they're set in your current shell:
   ```bash
   echo $GROQ_API_KEY
   echo $GEMINI_API_KEY
   ```

2. Make sure you've sourced your shell profile:
   ```bash
   source ~/.zshrc  # or ~/.bashrc
   ```

3. Restart your terminal

### Invalid API key errors

If you get authentication errors:

1. Verify your key is correct (copy-paste from the API console)
2. Check for extra spaces or quotes
3. Make sure you're using the correct key for each service

### Rate limiting

Both APIs have free tier rate limits:
- Groq: 30 requests/minute
- Gemini: 15 requests/minute

The system will automatically retry with exponential backoff if you hit rate limits.

## Security Notes

- Never commit API keys to version control
- The `.gitignore` file already excludes common environment files
- Consider using a secrets manager for production deployments
- Rotate your keys periodically

## Need Help?

If you encounter issues:
1. Check the setup instructions are displayed when running without keys
2. Review the error messages - they include troubleshooting steps
3. Verify your API keys are valid by testing them directly in the API consoles
