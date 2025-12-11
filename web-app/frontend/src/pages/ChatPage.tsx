import React, { useEffect, useCallback, useState, useRef } from 'react';
import MainLayout from '../components/layout/MainLayout';
import {
  ChatInput,
  ChatMessageBubble,
  PipelineIndicator,
  EvaluationSummary,
  OllamaErrorDisplay,
  JudgeFailureDisplay,
  StreamingJudgeResult,
} from '../components/chat';
import { HistorySidebar } from '../components/history';
import { useEvaluationStore } from '../store/evaluationStore';
import { useChatStream } from '../hooks/useChatStream';
import { useWebSocket } from '../hooks/useWebSocket';
// Note: useSessionRestoration is for evaluation sessions, not chat sessions
import { useToastStore } from '../store/toastStore';
import { chatApi, evaluationsApi } from '../api';
import { v4 as uuidv4 } from 'uuid';
import type { PipelineStage } from '../components/chat/PipelineIndicator';
import type { EvaluationResults } from '../api/types';
import type { WebSocketEvents } from '../services/websocket';

/**
 * Chat message type for the new chat interface
 */
interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  evaluationResults?: EvaluationResults;
  isStreaming?: boolean;
}

/**
 * Streaming judge result for incremental display
 */
interface StreamingJudge {
  judgeName: string;
  score: number;
  confidence: number;
  reasoning: string;
  flaggedIssues: any[];
  isNew: boolean;
}

/**
 * ChatPage - Redesigned chat interface for LLM Judge Auditor
 * 
 * Features:
 * - Single chat input at bottom (Requirements: 1.1, 1.2, 1.4)
 * - Message list with user/assistant bubbles (Requirements: 1.3, 2.3)
 * - Real-time streaming of LLM responses (Requirements: 2.2, 2.5)
 * - Evaluation summaries attached to messages (Requirements: 4.1)
 * - Pipeline progress indicator (Requirements: 8.1, 8.2)
 * - Error handling for Ollama and judges (Requirements: 2.4, 9.1, 9.3)
 */
const ChatPage: React.FC = () => {
  const {
    currentSessionId,
    setSessionId,
    restoreState,
  } = useEvaluationStore();

  const [chatSessionId, setChatSessionId] = useState<string | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [completedStages, setCompletedStages] = useState<PipelineStage[]>([]);
  const [selectedModel] = useState<string>('llama3.2');
  const [streamingJudges, setStreamingJudges] = useState<StreamingJudge[]>([]);
  
  const messagesEndRef = useRef<HTMLDivElement>(null);
  
  // Get WebSocket event handlers for direct event subscription
  const { on, off } = useWebSocket();

  const {
    isConnected,
    isGenerating,
    streamedText,
    ollamaError,
    judgeStatuses,
    evaluationProgress,
    evaluationComplete,
    evaluationError,
    sendMessage,
    joinSession,
    leaveSession,
    clearError,
    reset,
  } = useChatStream(chatSessionId || undefined);

  const { success, error: showError } = useToastStore();

  // Restore state from localStorage on page load (Requirements: 12.4)
  useEffect(() => {
    restoreState();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  /**
   * WebSocket event handlers for evaluation events (Requirements: 3.2, 3.3)
   * Note: stream_token is handled by useChatStream hook to avoid duplicate processing
   */
  useEffect(() => {
    if (!chatSessionId) return;

    // Handle evaluation_progress events (Requirements: 3.2)
    const handleEvaluationProgress = (data: WebSocketEvents['evaluation_progress']) => {
      const stage = data.stage as PipelineStage;
      const stageOrder: PipelineStage[] = ['generation', 'claim_extraction', 'verification', 'scoring', 'aggregation'];
      const currentIndex = stageOrder.indexOf(stage);
      
      // Mark all previous stages as complete
      if (currentIndex > 0) {
        setCompletedStages(stageOrder.slice(0, currentIndex));
      }
    };

    // Handle judge_verdict events (Requirements: 3.3)
    const handleJudgeVerdict = (data: WebSocketEvents['judge_verdict']) => {
      // Add judge result to streaming judges for incremental display
      const newJudge: StreamingJudge = {
        judgeName: data.judge_name,
        score: data.score,
        confidence: data.confidence,
        reasoning: data.reasoning,
        flaggedIssues: data.issues || [],
        isNew: true,
      };

      setStreamingJudges(prev => {
        // Deduplicate by judge name - update existing or add new
        const existingIndex = prev.findIndex(j => j.judgeName === newJudge.judgeName);
        
        if (existingIndex >= 0) {
          // Update existing judge
          const updated = [...prev];
          updated[existingIndex] = newJudge;
          return updated;
        }
        
        // Mark previous judges as not new and add new one
        const updated = prev.map(j => ({ ...j, isNew: false }));
        return [...updated, newJudge];
      });
    };

    // Handle chat_evaluation_complete events
    const handleChatEvaluationComplete = (data: WebSocketEvents['chat_evaluation_complete']) => {
      if (data.chat_session_id === chatSessionId) {
        // Mark all stages as complete
        setCompletedStages(['generation', 'claim_extraction', 'verification', 'scoring', 'aggregation']);
        
        // Clear streaming judges after evaluation is complete
        setTimeout(() => {
          setStreamingJudges([]);
        }, 2000);
      }
    };

    // Subscribe to events (stream_token handled by useChatStream)
    on('evaluation_progress', handleEvaluationProgress);
    on('judge_verdict', handleJudgeVerdict);
    on('chat_evaluation_complete', handleChatEvaluationComplete);

    return () => {
      off('evaluation_progress', handleEvaluationProgress);
      off('judge_verdict', handleJudgeVerdict);
      off('chat_evaluation_complete', handleChatEvaluationComplete);
    };
  }, [chatSessionId, on, off]);

  // Auto-scroll to bottom when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, streamedText, evaluationProgress]);

  // Update streaming message content
  useEffect(() => {
    if (isGenerating && streamedText) {
      setMessages(prev => {
        const lastMessage = prev[prev.length - 1];
        if (lastMessage && lastMessage.role === 'assistant' && lastMessage.isStreaming) {
          return [
            ...prev.slice(0, -1),
            { ...lastMessage, content: streamedText },
          ];
        }
        return prev;
      });
    }
  }, [streamedText, isGenerating]);

  // Handle generation complete - mark message as not streaming
  useEffect(() => {
    if (!isGenerating && streamedText && messages.length > 0) {
      setMessages(prev => {
        const lastMessage = prev[prev.length - 1];
        if (lastMessage && lastMessage.role === 'assistant' && lastMessage.isStreaming) {
          return [
            ...prev.slice(0, -1),
            { ...lastMessage, content: streamedText, isStreaming: false },
          ];
        }
        return prev;
      });
      
      // Mark generation stage as complete
      if (!completedStages.includes('generation')) {
        setCompletedStages(prev => [...prev, 'generation']);
      }
    }
  }, [isGenerating, streamedText, messages.length, completedStages]);

  // Track pipeline stage progression (Requirements: 8.1, 8.2)
  useEffect(() => {
    if (evaluationProgress) {
      const stage = evaluationProgress.stage as PipelineStage;
      const stageOrder: PipelineStage[] = ['generation', 'claim_extraction', 'verification', 'scoring', 'aggregation'];
      const currentIndex = stageOrder.indexOf(stage);
      
      // Mark all previous stages as complete
      const newCompleted = stageOrder.slice(0, currentIndex);
      setCompletedStages(newCompleted);
    }
  }, [evaluationProgress]);

  // Handle evaluation complete
  useEffect(() => {
    if (evaluationComplete) {
      // Mark all stages as complete
      setCompletedStages(['generation', 'claim_extraction', 'verification', 'scoring', 'aggregation']);
      
      success(
        'Evaluation Complete',
        'The response has been evaluated by all available judges.'
      );
    }
  }, [evaluationComplete, success]);

  // Handle evaluation error
  useEffect(() => {
    if (evaluationError) {
      showError(
        'Evaluation Failed',
        evaluationError.message,
        10000
      );
    }
  }, [evaluationError, showError]);

  // Create or join chat session
  const initializeChatSession = useCallback(async () => {
    try {
      const response = await chatApi.createSession({ ollama_model: selectedModel });
      setChatSessionId(response.id);
      setSessionId(response.id);
      return response.id;
    } catch (err: any) {
      console.error('Failed to create chat session:', err);
      showError('Session Error', 'Failed to create chat session. Please try again.');
      return null;
    }
  }, [selectedModel, setSessionId, showError]);

  // Join session when chatSessionId changes
  // Note: We use refs to avoid cleanup running when callbacks change
  const joinSessionRef = useRef(joinSession);
  const leaveSessionRef = useRef(leaveSession);
  
  useEffect(() => {
    joinSessionRef.current = joinSession;
    leaveSessionRef.current = leaveSession;
  }, [joinSession, leaveSession]);
  
  useEffect(() => {
    if (chatSessionId && isConnected) {
      joinSessionRef.current();
    }
    
    const currentSessionId = chatSessionId;
    return () => {
      if (currentSessionId) {
        leaveSessionRef.current();
      }
    };
  }, [chatSessionId, isConnected]);

  /**
   * Handle submitting a new question
   * Requirements: 1.2, 2.1
   */
  const handleSubmitQuestion = useCallback(
    async (question: string) => {
      try {
        // Ensure we have a chat session
        let sessionId = chatSessionId;
        if (!sessionId) {
          sessionId = await initializeChatSession();
          if (!sessionId) return;
        }

        // Add user message to the list (Requirements: 1.3)
        const userMessage: ChatMessage = {
          id: uuidv4(),
          role: 'user',
          content: question,
          timestamp: new Date(),
        };
        setMessages(prev => [...prev, userMessage]);

        // Add placeholder for assistant response
        const assistantMessage: ChatMessage = {
          id: uuidv4(),
          role: 'assistant',
          content: '',
          timestamp: new Date(),
          isStreaming: true,
        };
        setMessages(prev => [...prev, assistantMessage]);

        // Reset pipeline state
        setCompletedStages([]);
        setStreamingJudges([]);
        reset();

        // Send message via WebSocket (Requirements: 2.1, 2.2)
        // Pass sessionId explicitly for newly created sessions (state may not be updated yet)
        sendMessage(question, selectedModel, sessionId);

      } catch (err: any) {
        console.error('Failed to submit question:', err);
        showError('Error', 'Failed to send message. Please try again.');
      }
    },
    [chatSessionId, initializeChatSession, selectedModel, sendMessage, reset, showError]
  );

  /**
   * Handle starting a new chat session
   */
  const handleNewChat = useCallback(() => {
    // Clear current state
    setMessages([]);
    setChatSessionId(null);
    setSessionId(null);
    setCompletedStages([]);
    setStreamingJudges([]);
    reset();
  }, [setSessionId, reset]);

  /**
   * Handle selecting a previous session from history
   * Supports both chat sessions and evaluation sessions
   */
  const handleSelectSession = useCallback(
    async (sessionId: string, isEvaluation: boolean = false) => {
      try {
        if (isEvaluation) {
          // Load evaluation session with full results
          const evaluation = await evaluationsApi.getEvaluation(sessionId);
          
          // Create messages from evaluation data
          const loadedMessages: ChatMessage[] = [];
          const evalTimestamp = new Date(evaluation.created_at);
          
          // Add user message (the source text that was evaluated)
          if (evaluation.source_text) {
            loadedMessages.push({
              id: `${sessionId}-user`,
              role: 'user',
              content: evaluation.source_text,
              timestamp: evalTimestamp,
              isStreaming: false,
            });
          }
          
          // Add assistant message with evaluation results
          if (evaluation.candidate_output) {
            // Build evaluation results from the session data
            // Use defaults for fields not available in EvaluationSession
            const evaluationResults: EvaluationResults = {
              session_id: sessionId,
              consensus_score: evaluation.consensus_score || 0,
              judge_results: evaluation.judge_results || [],
              verifier_verdicts: evaluation.verifier_verdicts || [],
              confidence_metrics: {
                mean_confidence: 0.8,
                confidence_interval: [0.6, 1.0],
                confidence_level: 0.95,
                is_low_confidence: false,
              },
              inter_judge_agreement: {
                fleiss_kappa: evaluation.inter_judge_agreement || 0,
                interpretation:
                  evaluation.inter_judge_agreement && evaluation.inter_judge_agreement > 0.6
                    ? 'substantial'
                    : 'moderate',
                pairwise_correlations: {},
              },
              hallucination_metrics: {
                overall_score: evaluation.hallucination_score || 0,
                breakdown_by_type: {},
                affected_text_spans: [],
                severity_distribution: {},
              },
              variance: 0,
              standard_deviation: 0,
              processing_time_ms: 0,
              timestamp: evaluation.created_at,
            };
            
            loadedMessages.push({
              id: `${sessionId}-assistant`,
              role: 'assistant',
              content: evaluation.candidate_output,
              timestamp: evalTimestamp,
              isStreaming: false,
              evaluationResults,
            });
          }
          
          setMessages(loadedMessages);
          setChatSessionId(null); // Not a chat session
          setSessionId(sessionId);
          setCompletedStages(['generation', 'claim_extraction', 'verification', 'scoring', 'aggregation']);
          setStreamingJudges([]);
          reset();
          
          success('Evaluation Loaded', 'Previous evaluation report loaded successfully');
        } else {
          // Fetch chat messages for the session
          const response = await chatApi.getMessages(sessionId, 100, 0);
          
          // Convert API messages to ChatMessage format
          const loadedMessages: ChatMessage[] = response.messages.map(msg => ({
            id: msg.id,
            role: msg.role,
            content: msg.content,
            timestamp: new Date(msg.created_at),
            isStreaming: false,
          }));
          
          // Update state
          setMessages(loadedMessages);
          setChatSessionId(sessionId);
          setSessionId(sessionId);
          setCompletedStages([]);
          setStreamingJudges([]);
          reset();
          
          success('Session Restored', 'Previous conversation loaded successfully');
        }
      } catch (err: any) {
        console.error('Failed to load session:', err);
        showError('Session Error', 'Failed to load session. Please try again.');
      }
    },
    [setSessionId, reset, showError, success]
  );

  /**
   * Handle retry after Ollama error
   */
  const handleRetry = useCallback(() => {
    clearError();
    // Remove the failed assistant message
    setMessages(prev => {
      const lastMessage = prev[prev.length - 1];
      if (lastMessage && lastMessage.role === 'assistant' && !lastMessage.content) {
        return prev.slice(0, -1);
      }
      return prev;
    });
  }, [clearError]);

  // Determine if we're processing (generating or evaluating)
  const isProcessing = isGenerating || (evaluationProgress !== null && !evaluationComplete);

  // Get current pipeline stage
  const currentStage = evaluationProgress?.stage as PipelineStage | null;

  return (
    <MainLayout showSidebar={false}>
      <div className="flex h-full">
        {/* History Sidebar */}
        <HistorySidebar
          currentSessionId={currentSessionId}
          onSelectSession={handleSelectSession}
          onNewChat={handleNewChat}
          className="w-80 flex-shrink-0"
        />

        {/* Main Chat Area */}
        <div className="flex flex-col flex-1 min-w-0">
          {/* Connection Status */}
          {!isConnected && (
            <div className="bg-yellow-50 border-b border-yellow-200 px-4 py-2">
              <div className="flex items-center gap-2 text-sm text-yellow-800">
                <svg
                  className="w-4 h-4 animate-pulse"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
                  />
                </svg>
                <span>Connecting to server...</span>
              </div>
            </div>
          )}

          {/* Message List Area */}
          <div className="flex-1 overflow-y-auto px-4 py-6">
            {/* Empty State */}
            {messages.length === 0 && !isProcessing && (
              <div className="flex flex-col items-center justify-center h-full text-center py-12">
                <svg
                  className="w-16 h-16 text-gray-300 mb-4"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z"
                  />
                </svg>
                <h3 className="text-lg font-medium text-gray-900 mb-2">
                  Start a conversation
                </h3>
                <p className="text-sm text-gray-500 max-w-sm">
                  Ask a question below. The response will be automatically evaluated
                  by AI judges for accuracy and potential hallucinations.
                </p>
              </div>
            )}

            {/* Chat Messages */}
            {messages.map((message) => (
              <ChatMessageBubble
                key={message.id}
                role={message.role}
                content={message.content}
                timestamp={message.timestamp}
                isStreaming={message.isStreaming}
                evaluationSummary={
                  message.role === 'assistant' && 
                  message.evaluationResults && 
                  !message.isStreaming ? (
                    <EvaluationSummary
                      results={message.evaluationResults}
                      sessionId={chatSessionId || undefined}
                    />
                  ) : undefined
                }
              />
            ))}

            {/* Ollama Error Display (Requirements: 2.4, 9.3) */}
            {ollamaError && (
              <OllamaErrorDisplay
                errorType={ollamaError.type}
                message={ollamaError.message}
                suggestions={ollamaError.suggestions}
                onRetry={handleRetry}
                onDismiss={clearError}
              />
            )}

            {/* Pipeline Progress Indicator (Requirements: 8.1, 8.2) */}
            {(isProcessing || evaluationProgress) && !ollamaError && (
              <div className="my-4">
                <PipelineIndicator
                  currentStage={currentStage}
                  completedStages={completedStages}
                  progress={evaluationProgress?.progress}
                  message={evaluationProgress?.message}
                />
              </div>
            )}

            {/* Streaming Judge Results (Requirements: 3.3) */}
            {streamingJudges.length > 0 && !evaluationComplete && (
              <div className="my-4 space-y-2">
                <h4 className="text-sm font-medium text-gray-700 mb-2">Judge Verdicts</h4>
                {streamingJudges.map((judge, idx) => (
                  <StreamingJudgeResult
                    key={`${judge.judgeName}-${idx}`}
                    judgeName={judge.judgeName}
                    score={judge.score}
                    confidence={judge.confidence}
                    reasoning={judge.reasoning}
                    flaggedIssues={judge.flaggedIssues}
                    isNew={judge.isNew}
                  />
                ))}
              </div>
            )}

            {/* Judge Status Display (Requirements: 9.1, 9.2) */}
            {judgeStatuses.length > 0 && evaluationComplete && (
              <div className="my-4">
                <JudgeFailureDisplay
                  judges={judgeStatuses}
                  showAvailable={true}
                />
              </div>
            )}

            {/* Scroll anchor */}
            <div ref={messagesEndRef} />
          </div>

          {/* Chat Input (Requirements: 1.1, 1.2, 1.4) */}
          <ChatInput
            onSubmit={handleSubmitQuestion}
            isProcessing={isProcessing}
            placeholder="Ask a question..."
          />
        </div>
      </div>
    </MainLayout>
  );
};

export default ChatPage;
