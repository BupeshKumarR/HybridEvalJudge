# Quick Start Guide

## 1. Setup (You Already Have .venv!)

```bash
# Activate the virtual environment
source .venv/bin/activate          # macOS/Linux
# or
.venv\Scripts\activate.bat         # Windows
```

Your prompt will change from `(base)` to `(.venv)` when activated.

**Important:** Always activate the virtual environment before working on the project!

## 2. Get Free API Keys (Recommended)

The easiest way to use the toolkit is with free API judges (no model downloads needed):

### Groq (FREE)
1. Sign up: https://console.groq.com
2. Get API key: https://console.groq.com/keys
3. Set environment variable:
   ```bash
   export GROQ_API_KEY="your-groq-key-here"
   ```

### Google Gemini (FREE)
1. Sign up: https://aistudio.google.com
2. Get API key: https://aistudio.google.com/app/apikey
3. Set environment variable:
   ```bash
   export GEMINI_API_KEY="your-gemini-key-here"
   ```

**Benefits:**
- ✅ No model downloads (start immediately)
- ✅ Completely free
- ✅ Fast inference
- ✅ Works on any machine (no GPU needed)

**Alternative:** You can also use local models (see docs/USAGE_GUIDE.md)

## 3. Install API Dependencies

```bash
# Install API judge dependencies
pip install groq google-generativeai
```

## 4. Verify Installation

```bash
# Run tests to verify everything works
pytest

# Should see all tests passing
```

## 5. Basic Usage

```python
from llm_judge_auditor import EvaluationToolkit

# Initialize toolkit (uses API judges if keys are set)
toolkit = EvaluationToolkit.from_preset("fast")

# Evaluate an output
result = toolkit.evaluate(
    source_text="The Eiffel Tower is in Paris, France.",
    candidate_output="The Eiffel Tower is in London.",
)

# View results
print(f"Score: {result.consensus_score:.2f}/100")
print(f"Judges used: {[v.judge_name for v in result.judge_verdicts]}")
```

## 6. Run Examples

```bash
# Run the demo (best starting point)
python demo/demo.py

# Simple evaluation example
python examples/simple_evaluation.py

# API judge examples
python examples/api_judge_ensemble_example.py
python examples/groq_judge_example.py
python examples/gemini_judge_example.py

# Batch processing
python examples/batch_processing_example.py
```

## 7. Development Commands

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/unit/test_config.py

# Run with coverage
pytest --cov=llm_judge_auditor

# Format code
make format

# Check linting
make lint
```

## 8. Daily Workflow

**Start working:**
```bash
cd llm-judge-auditor
source .venv/bin/activate  # Don't forget this!
```

**Make changes, run tests:**
```bash
pytest
```

**Stop working:**
```bash
deactivate
```

## Troubleshooting

### "No API keys configured"
```bash
# Set your API keys
export GROQ_API_KEY="your-key"
export GEMINI_API_KEY="your-key"
```

### "Invalid API key"
- Verify key is correct (no spaces)
- Check key hasn't expired
- Try generating a new key

### "Rate limit exceeded"
- Free tier limits: Groq (30/min), Gemini (15/min)
- Wait a minute and retry
- System automatically retries with backoff

See [docs/API_TROUBLESHOOTING.md](docs/API_TROUBLESHOOTING.md) for more help.

## Need Help?

- **API Setup**: `demo/FREE_SETUP_GUIDE.md`
- **Troubleshooting**: `docs/API_TROUBLESHOOTING.md`
- **Full Setup**: `docs/ENVIRONMENT_SETUP.md`
- **Usage Guide**: `docs/USAGE_GUIDE.md`
- **Contributing**: `CONTRIBUTING.md`

## Next Steps

- Try the demo: `python demo/demo.py`
- Explore examples in `examples/`
- Read the full documentation in `docs/`
- Check out API judge examples for advanced usage
