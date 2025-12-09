# Requirements Document: Free API Judge Integration

## Introduction

The LLM Judge Auditor currently requires downloading models from HuggingFace, which may fail due to authentication issues, private repositories, or network problems. This feature will enable the system to use **free API-based judge models** (Groq Llama 3.1 70B and Google Gemini Flash) as the judge ensemble, making the evaluation toolkit work without downloading any models.

## Glossary

- **Groq**: A fast inference platform offering free API access to Llama models
- **Gemini Flash**: Google's fast, free-tier LLM API
- **Judge Model**: An LLM used to evaluate the quality of another LLM's output
- **Judge Ensemble**: Multiple judge models working together to evaluate outputs
- **Verifier Model**: A specialized model for fact-checking claims (optional)
- **Evaluation Toolkit**: The main orchestrator that coordinates all evaluation components
- **API Key**: Authentication credential for accessing external LLM APIs

## Requirements

### Requirement 1: Free API Judge Ensemble

**User Story:** As a developer, I want to use free API-based judge models (Groq and Gemini), so that I can evaluate LLM outputs without downloading models or paying for API access.

#### Acceptance Criteria

1. WHEN the system initializes the evaluation toolkit THEN the system SHALL support Groq Llama 3.1 70B as a judge
2. WHEN the system initializes the evaluation toolkit THEN the system SHALL support Google Gemini Flash as a judge
3. WHEN calling a Groq judge THEN the system SHALL use the Groq API with proper authentication
4. WHEN calling a Gemini judge THEN the system SHALL use the Google Gemini API with proper authentication
5. WHEN both judges are available THEN the system SHALL use them as an ensemble for evaluation
6. WHEN a judge API call fails THEN the system SHALL continue with remaining judges

### Requirement 2: Verifier Model Fallback

**User Story:** As a developer, I want the system to work without a specialized verifier model, so that I can run evaluations even when the verifier model is unavailable.

#### Acceptance Criteria

1. WHEN the verifier model fails to load THEN the system SHALL continue with judge-only evaluation
2. WHEN running without a verifier THEN the system SHALL log a warning but not fail
3. WHEN the verifier is unavailable THEN the system SHALL rely on judge ensemble for all evaluations
4. WHEN generating a report without verifier THEN the system SHALL indicate which components were used

### Requirement 3: API Key Configuration

**User Story:** As a developer, I want to configure API keys for Groq and Gemini, so that the system can authenticate with these services.

#### Acceptance Criteria

1. WHEN configuring the toolkit THEN the system SHALL accept a Groq API key via environment variable or config
2. WHEN configuring the toolkit THEN the system SHALL accept a Gemini API key via environment variable or config
3. WHERE no API keys are provided THEN the system SHALL provide clear instructions on obtaining free API keys
4. WHERE only one API key is provided THEN the system SHALL use that single judge for evaluation
5. WHERE both API keys are provided THEN the system SHALL use both judges in ensemble mode
6. WHEN API keys are invalid THEN the system SHALL provide clear error messages with troubleshooting steps

### Requirement 4: Demo Integration

**User Story:** As a user running the demo, I want it to use free API judges for evaluation, so that I can see real evaluation scores without downloading models.

#### Acceptance Criteria

1. WHEN running the demo THEN the system SHALL check for Groq and Gemini API keys
2. WHERE API keys are not set THEN the system SHALL display instructions for obtaining free API keys
3. WHERE API keys are set THEN the system SHALL use the judge ensemble for evaluation automatically
4. WHEN the demo completes THEN the system SHALL show which judges were used and their individual scores
5. WHEN evaluation succeeds THEN the system SHALL display consensus scores and rankings
6. WHEN evaluation fails THEN the system SHALL provide clear troubleshooting steps

### Requirement 5: API Integration

**User Story:** As a developer, I want the system to call Groq and Gemini APIs efficiently, so that evaluations complete quickly with minimal latency.

#### Acceptance Criteria

1. WHEN calling Groq API THEN the system SHALL use the official Groq Python SDK or REST API
2. WHEN calling Gemini API THEN the system SHALL use the official Google Generative AI SDK
3. WHEN formatting prompts THEN the system SHALL use appropriate templates for each API
4. WHEN receiving API responses THEN the system SHALL parse them into structured evaluation data
5. WHEN an API call fails THEN the system SHALL retry once with exponential backoff
6. WHEN rate limits are hit THEN the system SHALL wait and retry according to API guidelines
7. WHEN multiple judges are used THEN the system SHALL call them in parallel for faster evaluation

### Requirement 6: Error Handling and Fallbacks

**User Story:** As a developer, I want clear error messages when APIs are unavailable, so that I can quickly fix configuration issues.

#### Acceptance Criteria

1. WHEN API keys are missing THEN the system SHALL provide links to obtain free API keys
2. WHEN API authentication fails THEN the system SHALL indicate which API key is invalid
3. WHEN network errors occur THEN the system SHALL suggest checking internet connectivity
4. WHEN one judge fails THEN the system SHALL continue with remaining judges
5. WHEN all judges fail THEN the system SHALL provide a comprehensive troubleshooting guide
6. WHEN running in demo mode THEN the system SHALL check API connectivity before starting evaluation

### Requirement 7: Free API Key Setup Guide

**User Story:** As a new user, I want clear instructions on getting free API keys, so that I can start using the evaluation toolkit immediately.

#### Acceptance Criteria

1. WHEN the system detects missing API keys THEN the system SHALL display a setup guide
2. WHEN displaying the setup guide THEN the system SHALL include links to Groq API signup
3. WHEN displaying the setup guide THEN the system SHALL include links to Google AI Studio
4. WHEN displaying the setup guide THEN the system SHALL show how to set environment variables
5. WHEN displaying the setup guide THEN the system SHALL indicate that both APIs are free
6. WHEN API keys are set THEN the system SHALL verify they work before proceeding
