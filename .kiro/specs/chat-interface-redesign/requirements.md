# Requirements Document

## Introduction

This feature redesigns the LLM Judge Auditor web application to provide a chat-style interface where users can ask questions, receive responses from local LLMs (Ollama), and see real-time evaluation metrics from API judges (Groq/Gemini). The interface will visualize how judges assess the LLM's responses for hallucinations and accuracy, with support for multi-model comparison and detailed claim-level analysis.

## Glossary

- **Chat_Interface**: A conversational UI where users type questions and receive responses
- **Ollama**: A local LLM runtime that generates responses to user questions
- **API_Judges**: External LLM services (Groq Llama 3.3 70B, Gemini 2.0 Flash) that evaluate response quality
- **Hallucination_Score**: A metric (0-100) indicating the likelihood of factual errors in a response
- **Consensus_Score**: The aggregated accuracy score from multiple judges
- **Judge_Verdict**: The evaluation result from a single judge including score, confidence, and reasoning
- **Claim**: An individual factual statement extracted from an LLM response for verification
- **Evaluation_Pipeline**: The multi-stage process: Generation → Claim Extraction → Verification → Scoring → Aggregation

## Requirements

### Requirement 1

**User Story:** As a user, I want to ask questions in a simple chat input, so that I can interact with the LLM naturally without complex forms.

#### Acceptance Criteria

1. WHEN the user visits the chat page THEN the Chat_Interface SHALL display a single text input field at the bottom of the screen
2. WHEN the user types a question and presses Enter or clicks Send THEN the Chat_Interface SHALL submit the question for processing
3. WHEN a question is submitted THEN the Chat_Interface SHALL display the user's question as a chat message bubble
4. WHEN the input field is empty THEN the Chat_Interface SHALL disable the Send button

### Requirement 2

**User Story:** As a user, I want the system to automatically generate a response using a local LLM with real-time streaming, so that I can see the response as it's being generated.

#### Acceptance Criteria

1. WHEN a user submits a question THEN the Backend SHALL send the question to Ollama for response generation
2. WHILE Ollama is generating THEN the Chat_Interface SHALL stream the response token-by-token in real-time
3. WHEN Ollama returns a complete response THEN the Chat_Interface SHALL display the full response as an assistant message bubble
4. IF Ollama is unavailable THEN the Backend SHALL return an error message indicating the LLM service is offline
5. WHEN streaming THEN the Chat_Interface SHALL display a typing indicator alongside the streaming text

### Requirement 3

**User Story:** As a user, I want the LLM response to be automatically evaluated by API judges with visible progress, so that I can see the evaluation happening in real-time.

#### Acceptance Criteria

1. WHEN Ollama generates a response THEN the Backend SHALL automatically send the question and response to API judges for evaluation
2. WHILE judges are evaluating THEN the Chat_Interface SHALL display an evaluation progress indicator showing current stage
3. WHEN a judge completes evaluation THEN the Chat_Interface SHALL display the judge's verdict incrementally
4. WHEN all judges complete THEN the Chat_Interface SHALL display the aggregated consensus score
5. WHEN evaluation is in progress THEN the Chat_Interface SHALL show which judges are still processing

### Requirement 4

**User Story:** As a user, I want to see visual metrics showing how judges evaluated the response, so that I can understand the assessment at a glance.

#### Acceptance Criteria

1. WHEN evaluation completes THEN the Chat_Interface SHALL display a hallucination score thermometer (0-100)
2. WHEN evaluation completes THEN the Chat_Interface SHALL display individual judge scores with confidence levels
3. WHEN judges disagree significantly THEN the Chat_Interface SHALL highlight the disagreement visually with a warning indicator
4. WHEN a judge flags specific issues THEN the Chat_Interface SHALL display the flagged issues with severity badges (Minor, Moderate, Severe)
5. WHEN displaying scores THEN the Chat_Interface SHALL use color coding (green for high accuracy, red for low)

### Requirement 5

**User Story:** As a user, I want to see the reasoning behind judge decisions with claim-level breakdown, so that I can understand exactly why a response was scored a certain way.

#### Acceptance Criteria

1. WHEN evaluation completes THEN the Chat_Interface SHALL display expandable reasoning sections for each judge
2. WHEN the user clicks on a judge's verdict THEN the Chat_Interface SHALL expand to show detailed reasoning
3. WHEN issues are flagged THEN the Chat_Interface SHALL display the issue type, severity, and description
4. WHEN claims are extracted THEN the Chat_Interface SHALL display a claim-by-claim breakdown with verdict badges (Supported, Refuted, Not Enough Info)
5. WHEN a claim is problematic THEN the Chat_Interface SHALL highlight the corresponding text in the LLM response

### Requirement 6

**User Story:** As a user, I want to continue the conversation with follow-up questions, so that I can have a natural dialogue.

#### Acceptance Criteria

1. WHEN the user submits a follow-up question THEN the Backend SHALL include conversation history in the Ollama prompt
2. WHEN displaying the conversation THEN the Chat_Interface SHALL show all messages in chronological order
3. WHEN the user scrolls up THEN the Chat_Interface SHALL load previous messages from the session
4. WHEN clicking on any previous message THEN the Chat_Interface SHALL allow viewing its evaluation summary

### Requirement 7

**User Story:** As a user, I want to select which Ollama model to use and configure judge settings, so that I can customize the evaluation.

#### Acceptance Criteria

1. WHEN the user opens settings THEN the Chat_Interface SHALL display available Ollama models
2. WHEN the user selects a different model THEN the Backend SHALL use that model for subsequent responses
3. IF no Ollama models are available THEN the Chat_Interface SHALL display a message to install Ollama
4. WHEN the user opens settings THEN the Chat_Interface SHALL allow enabling/disabling specific judge APIs
5. WHEN the user changes judge settings THEN the Backend SHALL use the selected judges for evaluation

### Requirement 8

**User Story:** As a user, I want to see the evaluation pipeline stages, so I understand what the system is doing at each step.

#### Acceptance Criteria

1. WHEN evaluation starts THEN the Chat_Interface SHALL display a step indicator showing pipeline stages
2. WHEN a stage completes THEN the Chat_Interface SHALL visually update the indicator to show progress
3. WHEN all stages complete THEN the Chat_Interface SHALL display the final aggregated results
4. THE pipeline stages SHALL include: Response Generation, Claim Extraction, Fact Checking, Judge Scoring, Aggregation

### Requirement 9

**User Story:** As a user, I want graceful error handling when judges or the generator fails, so the system remains usable.

#### Acceptance Criteria

1. IF a judge API fails or times out THEN the Chat_Interface SHALL annotate its verdict as "Unavailable" and continue with other judges
2. IF all judges fail THEN the Chat_Interface SHALL show a warning and display only the LLM response without evaluation
3. IF Ollama fails THEN the Chat_Interface SHALL show a clear error message with troubleshooting suggestions
4. WHEN partial results are available THEN the Chat_Interface SHALL display them with appropriate annotations

### Requirement 10

**User Story:** As a user, I want to see metadata about which models and parameters were used, so I can reproduce and audit results.

#### Acceptance Criteria

1. WHEN evaluation completes THEN the Chat_Interface SHALL display the Ollama model name and version used
2. WHEN evaluation completes THEN the Chat_Interface SHALL display which judge models evaluated the response
3. WHEN displaying metadata THEN the Chat_Interface SHALL show timestamps for generation and evaluation
4. WHEN the user expands metadata THEN the Chat_Interface SHALL show generation parameters (temperature, top-p if applicable)

### Requirement 11

**User Story:** As a user, I want to export evaluation results, so I can save and share my findings.

#### Acceptance Criteria

1. WHEN viewing an evaluation THEN the Chat_Interface SHALL provide an export button
2. WHEN the user clicks export THEN the Chat_Interface SHALL offer JSON and CSV format options
3. WHEN exporting THEN the system SHALL include the question, response, all judge verdicts, and metrics
4. WHEN viewing conversation history THEN the Chat_Interface SHALL allow exporting the entire session


### Requirement 12

**User Story:** As a user, I want the interface to be responsive and accessible, so I can use it on any device.

#### Acceptance Criteria

1. THE Chat_Interface SHALL remain usable at all screen sizes (desktop, tablet, mobile)
2. THE Chat_Interface SHALL be operable with keyboard navigation
3. THE Chat_Interface SHALL support screen readers for accessibility
4. IF the browser disconnects or reloads THEN the Chat_Interface SHALL restore conversation and evaluation state

### Requirement 13

**User Story:** As a user, I want to see which judge evaluated which type of claim, so I understand judge specialization.

#### Acceptance Criteria

1. WHEN a claim is routed to a specialized judge THEN the Chat_Interface SHALL display which judge handled it
2. WHEN viewing claim-level results THEN the Chat_Interface SHALL show routing metadata (claim type: numerical, temporal, definitional, etc.)
3. IF routing fails THEN the system SHALL fall back to default ensemble scoring and indicate this in the UI
