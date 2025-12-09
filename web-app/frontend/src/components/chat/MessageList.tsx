import React, { useEffect, useRef, useState } from 'react';
import { EvaluationMessage } from '../../store/evaluationStore';
import UserMessage from './UserMessage';
import SystemMessage from './SystemMessage';
import EvaluationResultMessage from './EvaluationResultMessage';
import StreamingProgress from './StreamingProgress';
import StreamingJudgeResult from './StreamingJudgeResult';
import StreamingError from './StreamingError';

interface MessageListProps {
  messages: EvaluationMessage[];
  streamingProgress?: {
    stage: string;
    progress: number;
    message: string;
  } | null;
  streamingJudgeResults?: any[];
  streamingError?: {
    error_type: string;
    message: string;
    recovery_suggestions: string[];
  } | null;
  onLoadMore?: () => void;
  hasMore?: boolean;
  isLoadingMore?: boolean;
}

const MessageList: React.FC<MessageListProps> = ({
  messages,
  streamingProgress,
  streamingJudgeResults = [],
  streamingError,
  onLoadMore,
  hasMore = false,
  isLoadingMore = false,
}) => {
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const messagesContainerRef = useRef<HTMLDivElement>(null);
  const [shouldAutoScroll, setShouldAutoScroll] = useState(true);
  const [showScrollButton, setShowScrollButton] = useState(false);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    if (shouldAutoScroll) {
      scrollToBottom();
    }
  }, [messages, streamingProgress, streamingJudgeResults, streamingError, shouldAutoScroll]);

  // Check if user has scrolled up
  const handleScroll = () => {
    if (!messagesContainerRef.current) return;

    const { scrollTop, scrollHeight, clientHeight } = messagesContainerRef.current;
    const isNearBottom = scrollHeight - scrollTop - clientHeight < 100;

    setShouldAutoScroll(isNearBottom);
    setShowScrollButton(!isNearBottom);

    // Load more when scrolled to top
    if (scrollTop === 0 && hasMore && !isLoadingMore && onLoadMore) {
      onLoadMore();
    }
  };

  const scrollToBottom = (smooth = true) => {
    messagesEndRef.current?.scrollIntoView({
      behavior: smooth ? 'smooth' : 'auto',
    });
  };

  const handleScrollToBottom = () => {
    setShouldAutoScroll(true);
    scrollToBottom();
  };

  const renderMessage = (message: EvaluationMessage) => {
    switch (message.type) {
      case 'user':
        return (
          <UserMessage
            key={message.id}
            sourceText={message.content.sourceText || ''}
            candidateOutput={message.content.candidateOutput || ''}
            timestamp={message.timestamp}
          />
        );
      case 'system':
        return (
          <SystemMessage
            key={message.id}
            message={message.content.message || ''}
            timestamp={message.timestamp}
          />
        );
      case 'evaluation':
        return (
          <EvaluationResultMessage
            key={message.id}
            results={message.content.results}
            timestamp={message.timestamp}
          />
        );
      case 'error':
        return (
          <SystemMessage
            key={message.id}
            message={message.content.message || 'An error occurred'}
            timestamp={message.timestamp}
            type="error"
          />
        );
      default:
        return null;
    }
  };

  return (
    <div className="relative flex-1 flex flex-col">
      {/* Messages Container */}
      <div
        ref={messagesContainerRef}
        onScroll={handleScroll}
        className="flex-1 overflow-y-auto px-4 py-6 space-y-4"
      >
        {/* Load More Indicator */}
        {hasMore && (
          <div className="flex justify-center mb-4">
            {isLoadingMore ? (
              <div className="flex items-center gap-2 text-sm text-gray-500">
                <svg
                  className="animate-spin h-4 w-4"
                  fill="none"
                  viewBox="0 0 24 24"
                >
                  <circle
                    className="opacity-25"
                    cx="12"
                    cy="12"
                    r="10"
                    stroke="currentColor"
                    strokeWidth="4"
                  />
                  <path
                    className="opacity-75"
                    fill="currentColor"
                    d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                  />
                </svg>
                Loading more messages...
              </div>
            ) : (
              <button
                onClick={onLoadMore}
                className="text-sm text-blue-600 hover:text-blue-700 font-medium"
              >
                Load more messages
              </button>
            )}
          </div>
        )}

        {/* Empty State */}
        {messages.length === 0 && !streamingProgress && (
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
              No messages yet
            </h3>
            <p className="text-sm text-gray-500 max-w-sm">
              Start by entering source text and candidate output below to begin
              an evaluation.
            </p>
          </div>
        )}

        {/* Messages */}
        {messages.map(renderMessage)}

        {/* Streaming Progress */}
        {streamingProgress && (
          <StreamingProgress
            stage={streamingProgress.stage}
            progress={streamingProgress.progress}
            message={streamingProgress.message}
          />
        )}

        {/* Streaming Judge Results */}
        {streamingJudgeResults.map((result, idx) => (
          <StreamingJudgeResult
            key={`judge-${idx}`}
            judgeName={result.judge_name}
            score={result.score}
            confidence={result.confidence}
            reasoning={result.reasoning}
            flaggedIssues={result.flagged_issues}
            isNew={idx === streamingJudgeResults.length - 1}
          />
        ))}

        {/* Streaming Error */}
        {streamingError && (
          <StreamingError
            errorType={streamingError.error_type}
            message={streamingError.message}
            recoverySuggestions={streamingError.recovery_suggestions}
          />
        )}

        {/* Scroll anchor */}
        <div ref={messagesEndRef} />
      </div>

      {/* Scroll to Bottom Button */}
      {showScrollButton && (
        <button
          onClick={handleScrollToBottom}
          className="absolute bottom-4 right-4 bg-blue-600 text-white rounded-full p-3 shadow-lg hover:bg-blue-700 transition-colors"
          aria-label="Scroll to bottom"
        >
          <svg
            className="w-5 h-5"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M19 14l-7 7m0 0l-7-7m7 7V3"
            />
          </svg>
        </button>
      )}
    </div>
  );
};

export default MessageList;
