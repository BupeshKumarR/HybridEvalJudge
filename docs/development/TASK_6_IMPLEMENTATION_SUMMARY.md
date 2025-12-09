# Task 6 Implementation Summary: Setup Guide Display

## Overview
Successfully implemented task 6 and subtask 6.1 from the ollama-judge-integration spec, adding comprehensive API key setup guide display and validation functionality.

## What Was Implemented

### 1. API Key Validation on Startup (Subtask 6.1)
Added methods to the `APIKeyManager` class to validate API keys with lightweight test calls:

- `validate_all_keys(verbose: bool = True)` - Validates all available API keys
- `get_validation_summary()` - Returns formatted validation status summary
- `get_error_details(service: str)` - Gets detailed error messages for specific services

### 2. Setup Guide Display (Task 6)
Enhanced the `APIKeyManager` class with comprehensive setup guide functionality:

- `get_setup_instructions(show_validation: bool = False)` - Generates formatted setup guide with ASCII art
- `display_setup_guide_with_validation(validate: bool = True)` - Convenience method that loads, validates, and displays
- `get_troubleshooting_guide()` - Provides detailed troubleshooting information

## Key Features

### Formatted ASCII Art Display
The setup guide uses beautiful ASCII box drawing characters to create a professional-looking interface:

```
╔══════════════════════════════════════════════════════════════╗
║  🔑 API Key Setup Required                                   ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  This system uses FREE API-based judges for evaluation:     ║
║                                                              ║
║  1️⃣  Groq Llama 3.3 70B (FREE)                              ║
║     • Sign up: https://console.groq.com                     ║
║     • Get API key: https://console.groq.com/keys            ║
║     • Set: export GROQ_API_KEY="your-key"                   ║
║     ✅ Groq key: VALID                                       ║
║                                                              ║
║  2️⃣  Google Gemini Flash (FREE)                             ║
║     • Sign up: https://aistudio.google.com                  ║
║     • Get API key: https://aistudio.google.com/app/apikey   ║
║     • Set: export GEMINI_API_KEY="your-key"                 ║
║     ✅ Gemini key: VALID                                     ║
║                                                              ║
║  💡 Both APIs are completely FREE!                          ║
║  💡 You need at least ONE key to use API judges             ║
║  💡 Using BOTH keys gives better evaluation accuracy        ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
```

### API Key Validation
- Tests API keys with minimal API calls (1 token generation)
- Displays validation status with clear visual indicators (✅/❌/⚠️)
- Shows specific error messages for each API
- Handles missing packages, authentication errors, and network issues gracefully

### Comprehensive Troubleshooting
Provides detailed troubleshooting guide covering:
1. Invalid API Key / Authentication Failed
2. Rate Limit Exceeded
3. Package Not Installed
4. Network Error / Connection Timeout
5. Environment Variables Not Set

### Links to Documentation
- Groq signup: https://console.groq.com
- Groq API keys: https://console.groq.com/keys
- Groq docs: https://console.groq.com/docs
- Gemini signup: https://aistudio.google.com
- Gemini API keys: https://aistudio.google.com/app/apikey
- Gemini docs: https://ai.google.dev/docs

## Testing

### Unit Tests
Created comprehensive test suite in `tests/unit/test_api_key_manager.py`:
- 27 tests covering all functionality
- 100% pass rate
- Tests for loading, validation, utilities, setup guide, and validation workflows

### Test Coverage
- API key loading from environment variables
- Validation with success and failure scenarios
- Setup guide generation with various key states
- Validation summary display
- Troubleshooting guide generation
- Error detail retrieval
- Service availability checking

## Example Usage

Created `examples/api_key_setup_guide_example.py` demonstrating:
1. Basic setup guide without validation
2. Setup guide with validation
3. Validation summary display
4. Troubleshooting guide display
5. Individual key status checking
6. Complete workflow with convenience method

## Files Modified

### Core Implementation
- `src/llm_judge_auditor/components/api_key_manager.py` - Enhanced with validation and display methods

### Tests
- `tests/unit/test_api_key_manager.py` - New comprehensive test suite (27 tests)

### Examples
- `examples/api_key_setup_guide_example.py` - New example demonstrating all features

### Documentation
- `TASK_6_IMPLEMENTATION_SUMMARY.md` - This summary document

## Requirements Validated

### Requirement 7.1 ✅
"WHEN the system detects missing API keys THEN the system SHALL display a setup guide"
- Implemented in `get_setup_instructions()` method

### Requirement 7.2 ✅
"WHEN displaying the setup guide THEN the system SHALL include links to Groq API signup"
- Links included: https://console.groq.com and https://console.groq.com/keys

### Requirement 7.3 ✅
"WHEN displaying the setup guide THEN the system SHALL include links to Google AI Studio"
- Links included: https://aistudio.google.com and https://aistudio.google.com/app/apikey

### Requirement 7.4 ✅
"WHEN displaying the setup guide THEN the system SHALL show how to set environment variables"
- Shows: `export GROQ_API_KEY="your-key"` and `export GEMINI_API_KEY="your-key"`

### Requirement 7.5 ✅
"WHEN displaying the setup guide THEN the system SHALL indicate that both APIs are free"
- Displays: "💡 Both APIs are completely FREE!"

### Requirement 6.1 ✅
"WHEN API keys are missing THEN the system SHALL provide links to obtain free API keys"
- All signup and API key links provided in setup guide

### Requirement 3.6 ✅
"WHEN API keys are invalid THEN the system SHALL provide clear error messages with troubleshooting steps"
- Validation shows specific errors and comprehensive troubleshooting guide

### Requirement 6.2 ✅
"WHEN API authentication fails THEN the system SHALL indicate which API key is invalid"
- Validation summary shows per-service status with error details

### Requirement 7.6 ✅
"WHEN API keys are set THEN the system SHALL verify they work before proceeding"
- `validate_all_keys()` method performs lightweight test calls

## Status Indicators

The implementation uses clear visual indicators:
- ✅ - Valid/Available
- ❌ - Invalid/Not found
- ⚠️ - Detected but not validated
- 🔑 - API Key Setup
- 🔍 - Validation Status
- 🔧 - Troubleshooting
- 💡 - Tips and information
- 1️⃣/2️⃣ - Step numbers

## Next Steps

This implementation completes task 6 and subtask 6.1. The next tasks in the spec are:
- Task 7: Update demo to use API judges
- Task 7.1: Add interactive API key setup to demo
- Task 8: Add comprehensive error handling
- Task 8.1: Create troubleshooting guide (partially complete)

## Conclusion

Task 6 has been successfully implemented with:
- ✅ All subtasks completed
- ✅ All requirements validated
- ✅ Comprehensive test coverage (27 tests, 100% pass rate)
- ✅ Example code demonstrating usage
- ✅ Beautiful ASCII art formatting
- ✅ Clear error messages and troubleshooting
- ✅ Links to all necessary resources
