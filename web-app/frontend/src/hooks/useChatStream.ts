import { useEffect, useCallback, useState } from 'react';
import { useWebSocket } from './useWebSocket';
import { WebSocketEvents } from '../services/websocket';

export interface OllamaError {
  type: 'connection_error' | 'model_not_found' | 'timeout' | 'service_error';
  message: string;
  suggestions: string[];
}

export interface JudgeStatus {
  judgeName: string;
  status: 'available' | 'unavailable' | 'failed' | 'timeout' | 'rate_limited';
  errorMessage?: string;
  score?: number;
  confidence?: number;
  reasoning?: string;
  flaggedIssues?: any[];
}

export interface ChatStreamState {
  isGenerating: boolean;
  streamedText: string;
  ollamaError: OllamaError | null;
  judgeStatuses: JudgeStatus[];
  evaluationProgress: {
    stage: string;
    progress: number;
    message: string;
  } | null;
  evaluationComplete: boolean;
  evaluationError: {
    error: string;
    message: string;
  } | null;
}

/**
 * Hook for managing chat streaming with Ollama error handling
 * and judge failure tracking.
 * 
 * Requirements: 2.4, 9.1, 9.2, 9.3
 */
export const useChatStream = (chatSessionId?: string) => {
  const { emit, on, off, isConnected } = useWebSocket();
  
  const [state, setState] = useState<ChatStreamState>({
    isGenerating: false,
    streamedText: '',
    ollamaError: null,
    judgeStatuses: [],
    evaluationProgress: null,
    evaluationComplete: false,
    evaluationError: null,
  });

  useEffect(() => {
    if (!chatSessionId) return;

    // Handle generation started
    const handleGenerationStarted = (data: WebSocketEvents['generation_started']) => {
      if (data.chat_session_id === chatSessionId) {
        setState(prev => ({
          ...prev,
          isGenerating: true,
          streamedText: '',
          ollamaError: null,
        }));
      }
    };

    // Handle streaming tokens
    const handleStreamToken = (data: WebSocketEvents['stream_token']) => {
      setState(prev => ({
        ...prev,
        streamedText: prev.streamedText + data.token,
        isGenerating: !data.done,
      }));
    };

    // Handle generation complete
    const handleGenerationComplete = (data: WebSocketEvents['generation_complete']) => {
      if (data.chat_session_id === chatSessionId) {
        setState(prev => ({
          ...prev,
          isGenerating: false,
          streamedText: data.response,
        }));
      }
    };

    // Handle Ollama errors (Requirements 2.4, 9.3)
    const handleGenerationError = (data: WebSocketEvents['generation_error']) => {
      setState(prev => ({
        ...prev,
        isGenerating: false,
        ollamaError: {
          type: data.error,
          message: data.message,
          suggestions: data.suggestions,
        },
      }));
    };

    // Handle evaluation progress
    const handleEvaluationProgress = (data: WebSocketEvents['evaluation_progress']) => {
      setState(prev => ({
        ...prev,
        evaluationProgress: {
          stage: data.stage,
          progress: data.progress,
          message: data.message,
        },
      }));
    };

    // Handle judge verdicts with status (Requirements 9.1)
    const handleJudgeVerdict = (data: WebSocketEvents['judge_verdict']) => {
      const judgeStatus: JudgeStatus = {
        judgeName: data.judge_name,
        status: data.status || 'available',
        errorMessage: data.error_message,
        score: data.score,
        confidence: data.confidence,
        reasoning: data.reasoning,
        flaggedIssues: data.issues || [],
      };

      setState(prev => {
        // Deduplicate judges by name - update existing or add new
        const existingIndex = prev.judgeStatuses.findIndex(
          js => js.judgeName === judgeStatus.judgeName
        );
        
        if (existingIndex >= 0) {
          // Update existing judge status
          const updated = [...prev.judgeStatuses];
          updated[existingIndex] = judgeStatus;
          return { ...prev, judgeStatuses: updated };
        }
        
        // Add new judge
        return {
          ...prev,
          judgeStatuses: [...prev.judgeStatuses, judgeStatus],
        };
      });
    };

    // Handle chat evaluation complete
    const handleChatEvaluationComplete = (data: WebSocketEvents['chat_evaluation_complete']) => {
      if (data.chat_session_id === chatSessionId) {
        // Update judge statuses if provided, but preserve existing reasoning
        const newJudgeStatuses = data.judge_statuses;
        if (newJudgeStatuses && newJudgeStatuses.length > 0) {
          setState(prev => {
            // Merge new status info with existing reasoning data
            const mergedStatuses = newJudgeStatuses.map(js => {
              const existing = prev.judgeStatuses.find(
                existing => existing.judgeName === js.judge_name
              );
              return {
                judgeName: js.judge_name,
                status: js.status,
                errorMessage: js.error_message,
                // Preserve reasoning and other data from streaming
                score: existing?.score,
                confidence: existing?.confidence,
                reasoning: existing?.reasoning,
                flaggedIssues: existing?.flaggedIssues,
              };
            });
            return {
              ...prev,
              evaluationComplete: true,
              evaluationProgress: null,
              judgeStatuses: mergedStatuses,
            };
          });
        } else {
          setState(prev => ({
            ...prev,
            evaluationComplete: true,
            evaluationProgress: null,
          }));
        }
      }
    };

    // Handle chat evaluation error
    const handleChatEvaluationError = (data: WebSocketEvents['chat_evaluation_error']) => {
      if (data.chat_session_id === chatSessionId) {
        setState(prev => ({
          ...prev,
          evaluationError: {
            error: data.error,
            message: data.message,
          },
          evaluationProgress: null,
        }));
      }
    };

    // Subscribe to events
    on('generation_started', handleGenerationStarted);
    on('stream_token', handleStreamToken);
    on('generation_complete', handleGenerationComplete);
    on('generation_error', handleGenerationError);
    on('evaluation_progress', handleEvaluationProgress);
    on('judge_verdict', handleJudgeVerdict);
    on('chat_evaluation_complete', handleChatEvaluationComplete);
    on('chat_evaluation_error', handleChatEvaluationError);

    return () => {
      off('generation_started', handleGenerationStarted);
      off('stream_token', handleStreamToken);
      off('generation_complete', handleGenerationComplete);
      off('generation_error', handleGenerationError);
      off('evaluation_progress', handleEvaluationProgress);
      off('judge_verdict', handleJudgeVerdict);
      off('chat_evaluation_complete', handleChatEvaluationComplete);
      off('chat_evaluation_error', handleChatEvaluationError);
    };
  }, [chatSessionId, on, off]);

  const sendMessage = useCallback(
    (question: string, model?: string, overrideSessionId?: string) => {
      // Use override session ID if provided (for newly created sessions)
      const sessionId = overrideSessionId || chatSessionId;
      
      if (!sessionId) {
        console.error('sendMessage: No session ID available');
        return;
      }

      // Reset state for new message
      setState(prev => ({
        ...prev,
        streamedText: '',
        ollamaError: null,
        judgeStatuses: [],
        evaluationProgress: null,
        evaluationComplete: false,
        evaluationError: null,
      }));

      // Ensure we're in the chat session room before sending
      // This handles the case where session was just created
      emit('join_chat_session', { chat_session_id: sessionId });

      // Small delay to ensure room join is processed before message
      setTimeout(() => {
        emit('chat_message', {
          chat_session_id: sessionId,
          question,
          model,
        });
      }, 100);
    },
    [chatSessionId, emit]
  );

  const joinSession = useCallback(() => {
    if (!chatSessionId) return;
    emit('join_chat_session', { chat_session_id: chatSessionId });
  }, [chatSessionId, emit]);

  const leaveSession = useCallback(() => {
    if (!chatSessionId) return;
    emit('leave_chat_session', { chat_session_id: chatSessionId });
  }, [chatSessionId, emit]);

  const clearError = useCallback(() => {
    setState(prev => ({
      ...prev,
      ollamaError: null,
      evaluationError: null,
    }));
  }, []);

  const reset = useCallback(() => {
    setState({
      isGenerating: false,
      streamedText: '',
      ollamaError: null,
      judgeStatuses: [],
      evaluationProgress: null,
      evaluationComplete: false,
      evaluationError: null,
    });
  }, []);

  // Computed properties
  const hasAvailableJudges = state.judgeStatuses.some(js => js.status === 'available');
  const allJudgesFailed = state.judgeStatuses.length > 0 && !hasAvailableJudges;
  const hasPartialResults = state.judgeStatuses.some(js => js.status === 'available') &&
    state.judgeStatuses.some(js => js.status !== 'available');

  return {
    ...state,
    isConnected,
    hasAvailableJudges,
    allJudgesFailed,
    hasPartialResults,
    sendMessage,
    joinSession,
    leaveSession,
    clearError,
    reset,
  };
};

export default useChatStream;
