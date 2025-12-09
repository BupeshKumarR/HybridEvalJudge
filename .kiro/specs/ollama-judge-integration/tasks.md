# Implementation Plan: Free API Judge Integration

- [x] 1. Set up API client infrastructure
  - Create base classes and interfaces for API judges
  - Install required packages (groq, google-generativeai)
  - _Requirements: 1.1, 1.2, 5.1, 5.2_

- [x] 1.1 Create APIKeyManager component
  - Implement key loading from environment variables
  - Add validation methods for each API
  - Create setup instructions generator
  - _Requirements: 3.1, 3.2, 3.3, 7.1, 7.2, 7.3_

- [x] 1.2 Create base BaseJudgeClient interface
  - Define common interface for all judge clients
  - Add abstract methods for evaluate, format_prompt, parse_response
  - _Requirements: 1.3, 1.4, 5.3, 5.4_

- [x] 2. Implement Groq judge client
  - Create GroqJudgeClient class
  - Implement Groq API integration using groq SDK
  - Add prompt formatting for Groq
  - Add response parsing logic
  - _Requirements: 1.1, 1.3, 5.1, 5.3_

- [x] 2.1 Add error handling for Groq client
  - Implement retry logic with exponential backoff
  - Handle rate limiting (429 errors)
  - Handle authentication errors
  - _Requirements: 5.5, 5.6, 6.2, 6.3_

- [x] 3. Implement Gemini judge client
  - Create GeminiJudgeClient class
  - Implement Gemini API integration using google-generativeai SDK
  - Add prompt formatting for Gemini
  - Add response parsing logic
  - _Requirements: 1.2, 1.4, 5.2, 5.4_

- [x] 3.1 Add error handling for Gemini client
  - Implement retry logic with exponential backoff
  - Handle rate limiting
  - Handle authentication errors
  - _Requirements: 5.5, 5.6, 6.2, 6.3_

- [x] 4. Create API Judge Ensemble
  - Implement APIJudgeEnsemble class
  - Add logic to initialize judges based on available API keys
  - Implement parallel judge execution using asyncio or concurrent.futures
  - Handle partial failures gracefully
  - _Requirements: 1.5, 1.6, 5.7, 6.4_

- [x] 4.1 Add ensemble aggregation logic
  - Combine scores from multiple judges
  - Calculate consensus score
  - Identify disagreements between judges
  - _Requirements: 1.5_

- [x] 5. Update EvaluationToolkit integration
  - Modify EvaluationToolkit.__init__ to use APIKeyManager
  - Initialize APIJudgeEnsemble when API keys are available
  - Make verifier model optional (don't fail if unavailable)
  - Add fallback logic when no judges are available
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 6.1_

- [x] 5.1 Update configuration system
  - Add APIConfig dataclass
  - Add API-related settings to ToolkitConfig
  - Support environment variable configuration
  - _Requirements: 3.1, 3.2, 3.4, 3.5_

- [x] 6. Create setup guide display
  - Implement formatted setup guide with ASCII art
  - Add links to API key signup pages
  - Show environment variable setup commands
  - Display when API keys are missing
  - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 6.1_

- [x] 6.1 Add API key validation on startup
  - Test API keys with lightweight calls
  - Display which keys are valid/invalid
  - Show specific error messages for each API
  - _Requirements: 3.6, 6.2, 7.6_

- [x] 7. Update demo to use API judges
  - Modify demo.py to check for API keys
  - Display setup guide if keys are missing
  - Use API judge ensemble for evaluation
  - Show which judges were used in results
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5_

- [x] 7.1 Add interactive API key setup to demo
  - Prompt user for API keys if not set
  - Validate keys before proceeding
  - Save keys to environment or config file
  - _Requirements: 4.2, 4.6_

- [x] 8. Add comprehensive error handling
  - Implement error handling for missing keys
  - Add network error handling
  - Add rate limit handling
  - Add malformed response handling
  - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6_

- [x] 8.1 Create troubleshooting guide
  - Document common errors and solutions
  - Add to demo output when errors occur
  - Include in documentation
  - _Requirements: 6.5, 4.6_

- [x] 9. Add unit tests
  - Test APIKeyManager key loading and validation
  - Test GroqJudgeClient prompt formatting and parsing
  - Test GeminiJudgeClient prompt formatting and parsing
  - Test APIJudgeEnsemble initialization and execution
  - Mock API responses for testing
  - _Requirements: All_

- [x] 10. Add integration tests
  - Test real Groq API integration (with test key)
  - Test real Gemini API integration (with test key)
  - Test end-to-end evaluation flow
  - Test demo with API judges
  - _Requirements: All_

- [x] 11. Update documentation
  - Add API setup guide to README
  - Document new configuration options
  - Add troubleshooting section
  - Update examples to use API judges
  - _Requirements: 7.1, 7.2, 7.3, 7.4_

- [x] 12. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.
