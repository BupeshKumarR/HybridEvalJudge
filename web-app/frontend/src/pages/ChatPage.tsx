import React, { useEffect, useCallback } from 'react';
import MainLayout from '../components/layout/MainLayout';
import { MessageList, ChatInputForm } from '../components/chat';
import { HistorySidebar } from '../components/history';
import { useEvaluationStore } from '../store/evaluationStore';
import { useEvaluationStream } from '../hooks/useWebSocket';
import { useSessionRestoration } from '../hooks/useSessionRestoration';
import { evaluationsApi } from '../api/evaluations';
import { v4 as uuidv4 } from 'uuid';

const ChatPage: React.FC = () => {
  const {
    messages,
    config,
    isEvaluating,
    currentSessionId,
    addMessage,
    setEvaluating,
    setSessionId,
    updateConfig,
  } = useEvaluationStore();

  const {
    isConnected,
    progress,
    judgeResults,
    finalResults,
    error,
    startEvaluation,
    reset,
  } = useEvaluationStream();

  const { restoreSession } = useSessionRestoration();

  // Handle evaluation completion
  useEffect(() => {
    if (finalResults) {
      // Add evaluation result message
      addMessage({
        id: uuidv4(),
        type: 'evaluation',
        timestamp: new Date(),
        content: {
          results: finalResults.full_results,
        },
      });

      // Reset streaming state
      setEvaluating(false);
      reset();
    }
  }, [finalResults, addMessage, setEvaluating, reset]);

  // Handle evaluation error
  useEffect(() => {
    if (error) {
      // Add error message
      addMessage({
        id: uuidv4(),
        type: 'error',
        timestamp: new Date(),
        content: {
          message: `${error.error_type}: ${error.message}`,
        },
      });

      // Reset streaming state
      setEvaluating(false);
      reset();
    }
  }, [error, addMessage, setEvaluating, reset]);

  const handleSubmitEvaluation = useCallback(
    async (sourceText: string, candidateOutput: string) => {
      try {
        // Add user message
        addMessage({
          id: uuidv4(),
          type: 'user',
          timestamp: new Date(),
          content: {
            sourceText,
            candidateOutput,
          },
        });

        // Set evaluating state
        setEvaluating(true);

        // Create evaluation via API
        const response = await evaluationsApi.createEvaluation({
          source_text: sourceText,
          candidate_output: candidateOutput,
          config: {
            judge_models: config.judgeModels,
            enable_retrieval: config.enableRetrieval,
            aggregation_strategy: config.aggregationStrategy,
          },
        });

        // Store session ID
        setSessionId(response.session_id);

        // Start WebSocket streaming
        if (isConnected) {
          startEvaluation({
            session_id: response.session_id,
            source_text: sourceText,
            candidate_output: candidateOutput,
            config: {
              judge_models: config.judgeModels,
              enable_retrieval: config.enableRetrieval,
              aggregation_strategy: config.aggregationStrategy,
            },
          });
        } else {
          // WebSocket not connected, show error
          addMessage({
            id: uuidv4(),
            type: 'error',
            timestamp: new Date(),
            content: {
              message:
                'WebSocket connection not available. Please refresh the page.',
            },
          });
          setEvaluating(false);
        }
      } catch (err: any) {
        console.error('Failed to create evaluation:', err);
        addMessage({
          id: uuidv4(),
          type: 'error',
          timestamp: new Date(),
          content: {
            message:
              err.response?.data?.message ||
              'Failed to start evaluation. Please try again.',
          },
        });
        setEvaluating(false);
      }
    },
    [
      addMessage,
      setEvaluating,
      setSessionId,
      config,
      isConnected,
      startEvaluation,
    ]
  );

  const handleSelectSession = useCallback(
    async (sessionId: string) => {
      const success = await restoreSession(sessionId);
      if (!success) {
        addMessage({
          id: uuidv4(),
          type: 'error',
          timestamp: new Date(),
          content: {
            message: 'Failed to load session. Please try again.',
          },
        });
      }
    },
    [restoreSession, addMessage]
  );

  return (
    <MainLayout showSidebar={true}>
      <div className="flex h-full">
        {/* History Sidebar */}
        <HistorySidebar
          currentSessionId={currentSessionId}
          onSelectSession={handleSelectSession}
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

          {/* Message List */}
          <MessageList
            messages={messages}
            streamingProgress={progress}
            streamingJudgeResults={judgeResults}
            streamingError={error}
          />

          {/* Input Form */}
          <ChatInputForm
            onSubmit={handleSubmitEvaluation}
            isEvaluating={isEvaluating}
            config={config}
            onConfigChange={updateConfig}
          />
        </div>
      </div>
    </MainLayout>
  );
};

export default ChatPage;
