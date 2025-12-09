# Quick Start: API Judge Demo

## What This Demo Does

Evaluates LLM responses using **free API-based judges** (Groq Llama 3.3 70B and Google Gemini Flash). No model downloads required!

## Prerequisites

- Python 3.8+
- Internet connection
- Free API keys (takes 2 minutes to get)

## Step 1: Get Free API Keys

### Groq (Llama 3.3 70B)
1. Visit: https://console.groq.com
2. Sign up (free)
3. Get API key: https://console.groq.com/keys
4. Copy your key

### Google Gemini Flash
1. Visit: https://aistudio.google.com
2. Sign up (free)
3. Get API key: https://aistudio.google.com/app/apikey
4. Copy your key

**Note**: You need at least ONE key. Having BOTH gives better accuracy.

## Step 2: Install Dependencies

```bash
# Install required packages
pip install groq google-generativeai

# Install the toolkit
pip install -e .
```

## Step 3: Run the Demo

### Option A: Set Keys First (Recommended)

```bash
# Set environment variables
export GROQ_API_KEY="your-groq-key-here"
export GEMINI_API_KEY="your-gemini-key-here"

# Run demo
python demo/demo.py
```

### Option B: Interactive Setup

```bash
# Run demo without keys
python demo/demo.py

# Follow the prompts to enter your keys
# Keys will be validated before proceeding
```

## What You'll See

1. **API Key Check**: Demo checks for available keys
2. **Validation**: Keys are validated with test calls
3. **Question Prompt**: Enter your question or use the default
4. **Evaluation**: Judges evaluate a sample response (2-5 seconds)
5. **Results**: See individual scores, consensus, and verdict

## Example Output

```
🔍 Checking for API keys...
✅ Found API keys:
   • Groq API key detected
   • Gemini API key detected

🔍 Validating API keys...
✅ 2 API key(s) validated successfully!

📝 Test Question:
   What are the early warning signs of Type 2 diabetes?

🎯 Evaluating with 2 judge(s): groq-llama-3.3-70b, gemini-flash
   (This may take 5-10 seconds...)

✅ Evaluation complete in 3.2 seconds

📊 Individual Judge Scores:

   🤖 groq-llama-3.3-70b
      Score: 85.0/100
      Confidence: 0.90
      Reasoning: Response accurately covers main symptoms...

   🤖 gemini-flash
      Score: 88.0/100
      Confidence: 0.92
      Reasoning: Comprehensive and well-structured response...

🎯 Consensus Score:
   86.5/100
   Disagreement level: 4.5

🏆 FINAL VERDICT: ✅ APPROVED
   Consensus Score: 86.5/100
   Judges Used: groq-llama-3.3-70b, gemini-flash

📁 Results saved to: demo/results.json
```

## Troubleshooting

### "No API keys found"
- Make sure you've set the environment variables
- Check spelling: `GROQ_API_KEY` and `GEMINI_API_KEY`
- Try the interactive setup option

### "API key validation failed"
- Verify your API key is correct (no extra spaces)
- Check your internet connection
- Make sure the API service is not experiencing issues
- Ensure packages are installed: `pip install groq google-generativeai`

### "Import error"
- Install required packages: `pip install groq google-generativeai`
- Install the toolkit: `pip install -e .`

### Rate Limits
- Groq free tier: 30 requests/minute
- Gemini free tier: 15 requests/minute
- If you hit limits, wait a minute and try again

## Making Keys Permanent

Add to your shell profile (`~/.bashrc`, `~/.zshrc`, etc.):

```bash
export GROQ_API_KEY="your-groq-key-here"
export GEMINI_API_KEY="your-gemini-key-here"
```

Then reload: `source ~/.bashrc` (or `~/.zshrc`)

## Features

✅ **No Model Downloads**: Uses API judges, no large downloads
✅ **Fast**: Evaluation completes in 2-5 seconds
✅ **Free**: Both APIs have generous free tiers
✅ **Professional**: Uses state-of-the-art models
✅ **Interactive Setup**: Easy to get started
✅ **Detailed Results**: Individual scores, consensus, and reasoning

## Next Steps

1. Try different questions
2. Compare results from different judges
3. Integrate into your own projects
4. Explore the full toolkit features

## Support

- Groq Docs: https://console.groq.com/docs
- Gemini Docs: https://ai.google.dev/docs
- Toolkit Docs: See README.md

## Cost

**Both APIs are completely FREE** with generous limits:
- Groq: 30 requests/minute (free tier)
- Gemini: 15 requests/minute (free tier)

Perfect for development, testing, and small-scale production use!
