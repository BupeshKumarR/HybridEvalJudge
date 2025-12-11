# Implementation Plan

## Phase 1: Backend - Ollama Integration

- [x] 1. Create Ollama service for LLM communication
  - [x] 1.1 Create OllamaService class with streaming support
    - Implement `generate_stream()` method using httpx async streaming
    - Implement `list_models()` to fetch available models
    - Implement `check_health()` for connection status
    - _Requirements: 2.1, 2.2, 2.4_
  - [ ]* 1.2 Write property test for Ollama streaming
    - **Property 4: Streaming tokens appear incrementally**
    - **Validates: Requirements 2.2, 2.5**
  - [x] 1.3 Create Ollama router endpoints
    - POST `/api/v1/ollama/generate` for streaming generation
    - GET `/api/v1/ollama/models` for model list
    - GET `/api/v1/ollama/health` for status check
    - _Requirements: 7.1, 7.3_

## Phase 2: Backend - Chat Session Management

- [x] 2. Implement chat session and message models
  - [x] 2.1 Create ChatSession and ChatMessage database models
    - Add session_id, user_id, ollama_model fields
    - Add message role, content, evaluation_id fields
    - _Requirements: 6.1, 6.2_
  - [x] 2.2 Create chat router for session management
    - POST `/api/v1/chat/sessions` to create session
    - GET `/api/v1/chat/sessions/{id}/messages` to get history
    - POST `/api/v1/chat/sessions/{id}/messages` to add message
    - _Requirements: 6.1, 6.2, 6.3_
  - [ ]* 2.3 Write property test for conversation history
    - **Property 14: Conversation history inclusion**
    - **Validates: Requirements 6.1**
  - [ ]* 2.4 Write property test for message ordering
    - **Property 15: Chronological message order**
    - **Validates: Requirements 6.2**

## Phase 3: Backend - WebSocket Chat Handler

- [x] 3. Implement WebSocket handler for chat streaming
  - [x] 3.1 Create chat WebSocket endpoint
    - Handle `chat_message` events from client
    - Stream tokens back as `stream_token` events
    - Emit `evaluation_progress` events during evaluation
    - _Requirements: 2.2, 3.2, 3.3_
  - [x] 3.2 Integrate Ollama streaming with WebSocket
    - Forward Ollama tokens to WebSocket client
    - Handle generation completion
    - Trigger evaluation after response complete
    - _Requirements: 2.2, 2.3, 3.1_
  - [ ]* 3.3 Write property test for automatic evaluation trigger
    - **Property 6: Automatic judge evaluation**
    - **Validates: Requirements 3.1**

- [x] 4. Checkpoint - Ensure all backend tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Phase 4: Backend - Claim Extraction and Evaluation

- [x] 5. Implement claim extraction service
  - [x] 5.1 Create ClaimExtractionService
    - Extract factual claims from LLM response
    - Classify claim types (numerical, temporal, definitional, general)
    - Return claim text with span positions
    - _Requirements: 5.4, 8.4_
  - [x] 5.2 Create ClaimVerdict database model
    - Store claim_text, claim_type, verdict, confidence
    - Store text_span_start, text_span_end for highlighting
    - _Requirements: 5.4, 5.5_
  - [ ]* 5.3 Write property test for claim verdict display
    - **Property 13: Claim verdict display**
    - **Validates: Requirements 5.4**

- [x] 6. Update evaluation service for chat flow
  - [x] 6.1 Modify EvaluationService to emit pipeline stages
    - Emit progress for each stage (generation, claim_extraction, verification, scoring, aggregation)
    - Stream judge verdicts as they complete
    - _Requirements: 8.1, 8.2, 3.3_
  - [ ]* 6.2 Write property test for pipeline progression
    - **Property 17: Pipeline stage progression**
    - **Validates: Requirements 8.1, 8.2, 8.4**
  - [ ]* 6.3 Write property test for judge verdict streaming
    - **Property 7: Judge verdicts appear incrementally**
    - **Validates: Requirements 3.3**

## Phase 5: Frontend - Chat Input Component

- [x] 7. Create simplified chat input component
  - [x] 7.1 Replace ChatInputForm with single-input ChatInput
    - Single text input field at bottom of screen
    - Send button with Enter key support
    - Disabled state when processing
    - _Requirements: 1.1, 1.2, 1.4_
  - [ ]* 7.2 Write property test for empty input handling
    - **Property 2: Empty input disables send**
    - **Validates: Requirements 1.4**
  - [ ]* 7.3 Write property test for question submission
    - **Property 1: Question submission creates message**
    - **Validates: Requirements 1.2, 1.3**

## Phase 6: Frontend - Message Display

- [x] 8. Update message list for chat interface
  - [x] 8.1 Create ChatMessageBubble component
    - User message styling (right-aligned, blue)
    - Assistant message styling (left-aligned, gray)
    - Streaming text display with cursor
    - _Requirements: 1.3, 2.3_
  - [x] 8.2 Create StreamingText component
    - Display tokens as they arrive
    - Show typing indicator during streaming
    - Smooth text accumulation
    - _Requirements: 2.2, 2.5_
  - [ ]* 8.3 Write property test for complete response display
    - **Property 5: Complete responses become messages**
    - **Validates: Requirements 2.3**

## Phase 7: Frontend - Evaluation Visualization

- [x] 9. Create pipeline indicator component
  - [x] 9.1 Create PipelineIndicator component
    - Display 5 stages with icons
    - Highlight current stage
    - Show completion checkmarks
    - _Requirements: 8.1, 8.2, 8.4_

- [x] 10. Update evaluation panel for chat context
  - [x] 10.1 Create inline EvaluationSummary component
    - Compact view showing consensus score and hallucination score
    - Expandable to show full details
    - Attach to assistant message bubbles
    - _Requirements: 4.1, 4.2_
  - [x] 10.2 Create JudgeVerdictCard component
    - Show judge name, score, confidence
    - Color coding based on score
    - Expandable reasoning section
    - _Requirements: 4.2, 4.5, 5.1, 5.2_
  - [ ]* 10.3 Write property test for score color coding
    - **Property 12: Score color coding**
    - **Validates: Requirements 4.5**
  - [x] 10.4 Create DisagreementWarning component
    - Display when judge variance exceeds threshold
    - Show which judges disagree
    - _Requirements: 4.3_
  - [ ]* 10.5 Write property test for disagreement detection
    - **Property 10: Disagreement detection**
    - **Validates: Requirements 4.3**

- [x] 11. Checkpoint - Ensure all frontend tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Phase 8: Frontend - Claim Breakdown

- [x] 12. Create claim breakdown component
  - [x] 12.1 Create ClaimBreakdown component
    - List extracted claims with verdict badges
    - Show claim type labels
    - Display routing metadata (which judge handled)
    - _Requirements: 5.4, 13.1, 13.2_
  - [x] 12.2 Create ClaimHighlighter component
    - Highlight problematic claims in response text
    - Color code by verdict (green=supported, red=refuted, yellow=unknown)
    - _Requirements: 5.5_
  - [ ]* 12.3 Write property test for claim routing display
    - **Property 23: Claim routing display**
    - **Validates: Requirements 13.1, 13.2**

## Phase 9: Frontend - Settings and Configuration

- [x] 13. Update settings for chat interface
  - [x] 13.1 Add Ollama model selector to settings
    - Fetch available models from backend
    - Display model selection dropdown
    - Show "Install Ollama" message if unavailable
    - _Requirements: 7.1, 7.3_
  - [x] 13.2 Add judge configuration to settings
    - Toggle switches for each judge API
    - Show API key status
    - _Requirements: 7.4, 7.5_
  - [ ]* 13.3 Write property test for model selection persistence
    - **Property 16: Model selection persistence**
    - **Validates: Requirements 7.2**

## Phase 10: Error Handling and State Persistence

- [x] 14. Implement error handling
  - [x] 14.1 Add Ollama error handling
    - Display connection error messages
    - Show troubleshooting suggestions
    - _Requirements: 2.4, 9.3_
  - [x] 14.2 Add judge failure handling
    - Annotate failed judges as "Unavailable"
    - Continue with available judges
    - _Requirements: 9.1, 9.2_
  - [ ]* 14.3 Write property test for judge failure annotation
    - **Property 18: Judge failure annotation**
    - **Validates: Requirements 9.1**
  - [ ]* 14.4 Write property test for partial results display
    - **Property 19: Partial results display**
    - **Validates: Requirements 9.4**

- [x] 15. Implement state persistence
  - [x] 15.1 Add session state to localStorage
    - Save conversation history
    - Save evaluation results
    - Restore on page reload
    - _Requirements: 12.4_
  - [ ]* 15.2 Write property test for state restoration
    - **Property 22: State restoration**
    - **Validates: Requirements 12.4**

## Phase 11: Export and Metadata

- [x] 16. Add export functionality
  - [x] 16.1 Create export service
    - Generate JSON export with all data
    - Generate CSV export for tabular data
    - _Requirements: 11.1, 11.2, 11.3_
  - [x] 16.2 Add export buttons to UI
    - Export button on evaluation summary
    - Export session button in history
    - _Requirements: 11.1, 11.4_
  - [ ]* 16.3 Write property test for export completeness
    - **Property 21: Export data completeness**
    - **Validates: Requirements 11.3**

- [x] 17. Add metadata display
  - [x] 17.1 Create MetadataPanel component
    - Show Ollama model name and version
    - Show judge models used
    - Show timestamps
    - _Requirements: 10.1, 10.2, 10.3_
  - [ ]* 17.2 Write property test for metadata completeness
    - **Property 20: Metadata completeness**
    - **Validates: Requirements 10.1, 10.2, 10.3**

## Phase 12: Final Integration

- [x] 18. Update ChatPage to use new components
  - [x] 18.1 Replace existing ChatPage layout
    - Remove source text and candidate output fields
    - Add single chat input at bottom
    - Add message list with evaluation summaries
    - _Requirements: 1.1, 1.3, 4.1_
  - [x] 18.2 Connect WebSocket for streaming
    - Handle stream_token events
    - Handle evaluation_progress events
    - Handle judge_verdict events
    - _Requirements: 2.2, 3.2, 3.3_

- [x] 19. Final Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.
