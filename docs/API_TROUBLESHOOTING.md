# API Judge Troubleshooting Guide

Quick reference guide for troubleshooting common issues with API-based judges (Groq and Gemini).

## Quick Diagnostics

Run this to check your API setup:

```python
from llm_judge_auditor.components.api_key_manager import APIKeyManager

# Check API key status
api_key_manager = APIKeyManager()
api_key_manager.load_keys()

# Validate all keys
validation_results = api_key_manager.validate_all_keys(verbose=True)

# Display summary
print(api_key_manager.get_validation_summary())
```

## Common Issues

### 1. "No API keys configured"

**Symptoms:**
- Error message: "No API keys configured"
- Demo won't start
- Empty judge ensemble

**Solutions:**

**Option A: Set Environment Variables (Recommended)**
```bash
# macOS/Linux
export GROQ_API_KEY="your-groq-key-here"
export GEMINI_API_KEY="your-gemini-key-here"

# Windows
set GROQ_API_KEY=your-groq-key-here
set GEMINI_API_KEY=your-gemini-key-here
```

**Option B: Add to Shell Profile (Persistent)**
```bash
# macOS/Linux - Add to ~/.bashrc or ~/.zshrc
echo 'export GROQ_API_KEY="your-groq-key-here"' >> ~/.bashrc
echo 'export GEMINI_API_KEY="your-gemini-key-here"' >> ~/.bashrc
source ~/.bashrc

# Windows - Use setx for persistence
setx GROQ_API_KEY "your-groq-key-here"
setx GEMINI_API_KEY "your-gemini-key-here"
```

**Option C: Pass Directly in Code**
```python
from llm_judge_auditor import ToolkitConfig, APIConfig

config = ToolkitConfig(
    api_config=APIConfig(
        groq_api_key="your-groq-key-here",
        gemini_api_key="your-gemini-key-here",
    )
)
```

**Get Free API Keys:**
- **Groq**: https://console.groq.com/keys
- **Gemini**: https://aistudio.google.com/app/apikey

---

### 2. "Invalid API key"

**Symptoms:**
- Error: "Authentication failed"
- 401 or 403 HTTP errors
- "Invalid API key" message

**Solutions:**

1. **Verify Key Format**
   ```bash
   # Check your key (should be long alphanumeric string)
   echo $GROQ_API_KEY
   echo $GEMINI_API_KEY
   ```

2. **Check for Common Mistakes**
   - Extra spaces before/after key
   - Quotes included in the key value
   - Wrong environment variable name
   - Key copied incorrectly

3. **Regenerate API Key**
   - Groq: https://console.groq.com/keys → Create new key
   - Gemini: https://aistudio.google.com/app/apikey → Create new key

4. **Verify Key is Active**
   - Check your account dashboard
   - Ensure key hasn't been revoked
   - Confirm account is in good standing

---

### 3. "Rate limit exceeded"

**Symptoms:**
- Error: "Rate limit exceeded"
- 429 HTTP status code
- "Too many requests" message

**Free Tier Limits:**
- **Groq**: 30 requests per minute
- **Gemini**: 15 requests per minute

**Solutions:**

1. **Wait and Retry**
   ```python
   # The system automatically retries with backoff
   # Just wait a minute and try again
   import time
   time.sleep(60)
   ```

2. **Reduce Request Frequency**
   ```python
   # Add delays between requests
   import time
   
   for request in requests:
       result = toolkit.evaluate(source, candidate)
       time.sleep(2)  # 2 second delay
   ```

3. **Use Batch Processing**
   ```python
   # More efficient than individual requests
   batch_result = toolkit.batch_evaluate(requests)
   ```

4. **Upgrade to Paid Tier** (if needed)
   - Groq: Higher limits available
   - Gemini: Google Cloud pricing

---

### 4. "Network error" or "Connection timeout"

**Symptoms:**
- "Connection timeout" error
- "Network error" message
- "Failed to connect" error

**Solutions:**

1. **Check Internet Connection**
   ```bash
   # Test connectivity
   ping google.com
   curl https://api.groq.com/openai/v1/models
   ```

2. **Check Firewall/Proxy**
   - Ensure firewall allows HTTPS connections
   - Configure proxy if needed:
     ```bash
     export HTTPS_PROXY=http://proxy.example.com:8080
     ```

3. **Verify API Status**
   - Groq status: https://status.groq.com
   - Google status: https://status.cloud.google.com

4. **Increase Timeout**
   ```python
   config = ToolkitConfig(
       api_config=APIConfig(
           timeout=60,  # Increase from default 30s
       )
   )
   ```

---

### 5. "No module named 'groq'" or "'google.generativeai'"

**Symptoms:**
- ImportError when running code
- "No module named" errors
- Missing dependencies

**Solutions:**

1. **Install API Dependencies**
   ```bash
   pip install groq google-generativeai
   ```

2. **Verify Installation**
   ```bash
   pip list | grep groq
   pip list | grep google-generativeai
   ```

3. **Reinstall if Needed**
   ```bash
   pip uninstall groq google-generativeai
   pip install groq google-generativeai
   ```

4. **Check Virtual Environment**
   ```bash
   # Ensure you're in the right environment
   which python
   source .venv/bin/activate  # Activate if needed
   ```

---

### 6. "Malformed response" or "Parse error"

**Symptoms:**
- JSON parse errors
- "Unexpected response format"
- Missing fields in response

**Solutions:**

1. **System Handles Automatically**
   - Fallback parsing extracts what it can
   - Uses default values for missing fields
   - Continues with low confidence score

2. **Check Logs**
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   # Run your code to see detailed logs
   ```

3. **Verify Model Version**
   ```python
   # Ensure using correct model versions
   config = ToolkitConfig(
       api_config=APIConfig(
           groq_model="llama-3.1-70b-versatile",
           gemini_model="gemini-1.5-flash",
       )
   )
   ```

4. **Report Issue**
   - If persistent, report to GitHub Issues
   - Include response preview from logs

---

### 7. "Some judges failed" (Partial Failure)

**Symptoms:**
- Warning: "Some judges failed"
- Fewer verdicts than expected
- One API works, another doesn't

**Solutions:**

1. **Check Individual Keys**
   ```python
   api_key_manager = APIKeyManager()
   api_key_manager.load_keys()
   
   # Validate each key separately
   groq_valid = api_key_manager.validate_key("groq", api_key_manager.groq_key)
   gemini_valid = api_key_manager.validate_key("gemini", api_key_manager.gemini_key)
   
   print(f"Groq valid: {groq_valid}")
   print(f"Gemini valid: {gemini_valid}")
   ```

2. **Continue with Available Judges**
   ```python
   # System automatically uses available judges
   # No action needed - evaluation continues
   ```

3. **Fix Failed Judge**
   - Check specific error message
   - Follow troubleshooting for that API
   - Verify that API key is valid

---

### 8. "All judges failed"

**Symptoms:**
- Error: "All judges failed"
- No verdicts returned
- Complete evaluation failure

**Solutions:**

1. **Run Full Diagnostics**
   ```python
   from llm_judge_auditor.components.api_key_manager import APIKeyManager
   
   api_key_manager = APIKeyManager()
   api_key_manager.load_keys()
   
   if not api_key_manager.has_any_keys():
       print("No API keys found!")
       print(api_key_manager.get_setup_instructions())
   else:
       validation = api_key_manager.validate_all_keys(verbose=True)
       print(api_key_manager.get_validation_summary())
   ```

2. **Check All Requirements**
   - ✅ API keys set correctly
   - ✅ Dependencies installed
   - ✅ Internet connection working
   - ✅ No firewall blocking
   - ✅ API services operational

3. **Try One Judge at a Time**
   ```python
   # Test Groq only
   from llm_judge_auditor.components.groq_judge_client import GroqJudgeClient
   
   client = GroqJudgeClient(api_key="your-groq-key")
   verdict = client.evaluate(source, candidate, task="factual_accuracy")
   print(f"Groq works: {verdict.score}")
   ```

4. **Get Help**
   - Review logs for specific errors
   - Check [API_ERROR_HANDLING.md](API_ERROR_HANDLING.md)
   - Report issue on GitHub

---

## Testing Your Setup

### Quick Test Script

```python
from llm_judge_auditor import EvaluationToolkit

# Simple test
toolkit = EvaluationToolkit.from_preset("fast")

source = "The capital of France is Paris."
candidate = "The capital of France is Paris."

try:
    result = toolkit.evaluate(source, candidate)
    print(f"✅ Success! Score: {result.consensus_score:.2f}")
    print(f"Judges used: {[v.judge_name for v in result.judge_verdicts]}")
except Exception as e:
    print(f"❌ Error: {e}")
```

### Comprehensive Test

```bash
# Run the demo
python demo/demo.py

# Should show:
# - API key validation
# - Judge initialization
# - Evaluation results
# - No errors
```

---

## Environment Variable Debugging

### Check Current Values

```bash
# macOS/Linux
echo "Groq: $GROQ_API_KEY"
echo "Gemini: $GEMINI_API_KEY"

# Windows
echo %GROQ_API_KEY%
echo %GEMINI_API_KEY%
```

### Check in Python

```python
import os

groq_key = os.getenv("GROQ_API_KEY")
gemini_key = os.getenv("GEMINI_API_KEY")

print(f"Groq key set: {bool(groq_key)}")
print(f"Gemini key set: {bool(gemini_key)}")

if groq_key:
    print(f"Groq key length: {len(groq_key)}")
if gemini_key:
    print(f"Gemini key length: {len(gemini_key)}")
```

### Verify After Setting

```bash
# Set key
export GROQ_API_KEY="your-key"

# Verify immediately
echo $GROQ_API_KEY

# Verify in new shell
bash -c 'echo $GROQ_API_KEY'
```

---

## Performance Issues

### Slow Evaluation

**Symptoms:**
- Evaluation takes >30 seconds
- Timeouts occurring
- Slow response times

**Solutions:**

1. **Enable Parallel Execution**
   ```python
   config = ToolkitConfig(
       api_config=APIConfig(
           parallel_calls=True,  # Faster
       )
   )
   ```

2. **Reduce Timeout**
   ```python
   config = ToolkitConfig(
       api_config=APIConfig(
           timeout=15,  # Fail faster
       )
   )
   ```

3. **Use Faster Models**
   ```python
   # Already using fastest free models
   # Groq Llama 3.1 70B - very fast
   # Gemini Flash - optimized for speed
   ```

---

## Getting More Help

### Documentation
- **[API Error Handling](API_ERROR_HANDLING.md)** - Detailed error handling
- **[Free Setup Guide](../demo/FREE_SETUP_GUIDE.md)** - Complete setup instructions
- **[Usage Guide](USAGE_GUIDE.md)** - General usage information

### Examples
- **[API Judge Ensemble Example](../examples/api_judge_ensemble_example.py)**
- **[Groq Judge Example](../examples/groq_judge_example.py)**
- **[Gemini Judge Example](../examples/gemini_judge_example.py)**
- **[Error Handling Example](../examples/comprehensive_error_handling_example.py)**

### Support
- **GitHub Issues**: https://github.com/yourusername/llm-judge-auditor/issues
- **Discussions**: https://github.com/yourusername/llm-judge-auditor/discussions

---

## Checklist for New Users

Before reporting an issue, verify:

- [ ] API keys obtained from correct sources
- [ ] Environment variables set correctly
- [ ] Dependencies installed (`pip install groq google-generativeai`)
- [ ] Internet connection working
- [ ] No firewall blocking HTTPS
- [ ] Keys validated with test script
- [ ] Using latest version of toolkit
- [ ] Checked logs for specific errors

---

## Requirements Coverage

This troubleshooting guide satisfies:

- **7.1** - Setup guide display
- **7.2** - Links to API signup
- **7.3** - Environment variable setup
- **7.4** - Free API indication
- **6.5** - Comprehensive troubleshooting
- **6.6** - Error-specific solutions
