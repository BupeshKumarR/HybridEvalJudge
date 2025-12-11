import React, { useMemo } from 'react';
import { formatDistanceToNow, isValid } from 'date-fns';
import StreamingText from './StreamingText';

export type MessageRole = 'user' | 'assistant';

interface ChatMessageBubbleProps {
  /** The role of the message sender */
  role: MessageRole;
  /** The message content */
  content: string;
  /** Timestamp of the message (Date object or ISO string) */
  timestamp: Date | string;
  /** Whether this message is currently streaming (for assistant messages) */
  isStreaming?: boolean;
  /** Optional evaluation summary to display with assistant messages */
  evaluationSummary?: React.ReactNode;
}

/**
 * Safely format a timestamp for display
 * Handles both Date objects and ISO strings, with fallback for invalid dates
 */
const formatTimestamp = (timestamp: Date | string): string => {
  try {
    let date: Date;
    if (timestamp instanceof Date) {
      date = timestamp;
    } else if (typeof timestamp === 'string') {
      // Ensure UTC timestamps are parsed correctly
      const isoString = timestamp.includes('Z') || timestamp.includes('+') || timestamp.includes('-', 10)
        ? timestamp : timestamp + 'Z';
      date = new Date(isoString);
    } else {
      return 'Just now';
    }
    if (!isValid(date) || isNaN(date.getTime())) return 'Just now';
    if (date > new Date()) return 'Just now'; // Handle future dates
    return formatDistanceToNow(date, { addSuffix: true });
  } catch {
    return 'Just now';
  }
};

/**
 * ChatMessageBubble - Displays a chat message with role-based styling
 * 
 * Features:
 * - User message styling (right-aligned, blue)
 * - Assistant message styling (left-aligned, gray)
 * - Streaming text display with cursor
 * 
 * Requirements: 1.3, 2.3
 */
const ChatMessageBubble: React.FC<ChatMessageBubbleProps> = ({
  role,
  content,
  timestamp,
  isStreaming = false,
  evaluationSummary,
}) => {
  const isUser = role === 'user';
  const formattedTime = useMemo(() => formatTimestamp(timestamp), [timestamp]);

  return (
    <div 
      className={`flex mb-4 ${isUser ? 'justify-end' : 'justify-start'}`}
      role="article"
      aria-label={`${isUser ? 'Your' : 'Assistant'} message`}
    >
      <div className={`max-w-3xl ${isUser ? 'w-auto' : 'w-full'}`}>
        {/* Message bubble */}
        <div
          className={`
            rounded-2xl px-4 py-3 shadow-sm
            ${isUser 
              ? 'bg-blue-600 text-white rounded-br-md' 
              : 'bg-gray-100 text-gray-900 rounded-bl-md border border-gray-200'
            }
          `}
        >
          {isUser ? (
            // User message - simple text display
            <p className="text-sm leading-relaxed whitespace-pre-wrap break-words">
              {content}
            </p>
          ) : (
            // Assistant message - supports streaming
            <StreamingText
              text={content}
              isStreaming={isStreaming}
              className={`text-sm leading-relaxed ${isStreaming ? 'text-gray-800' : 'text-gray-900'}`}
            />
          )}
        </div>

        {/* Evaluation summary (for assistant messages) */}
        {!isUser && evaluationSummary && !isStreaming && (
          <div className="mt-2">
            {evaluationSummary}
          </div>
        )}

        {/* Timestamp */}
        <div 
          className={`text-xs text-gray-500 mt-1 ${isUser ? 'text-right' : 'text-left'}`}
          aria-label={`Sent ${formattedTime}`}
        >
          {formattedTime}
        </div>
      </div>
    </div>
  );
};

export default ChatMessageBubble;
