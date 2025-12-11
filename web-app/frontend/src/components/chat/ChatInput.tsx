import React, { useState, useRef, useEffect, KeyboardEvent, ChangeEvent } from 'react';

interface ChatInputProps {
  onSubmit: (question: string) => void;
  isProcessing: boolean;
  placeholder?: string;
}

/**
 * ChatInput - A simplified single-input chat component
 * 
 * Features:
 * - Single text input field at bottom of screen
 * - Send button with Enter key support
 * - Disabled state when processing
 * 
 * Requirements: 1.1, 1.2, 1.4
 */
const ChatInput: React.FC<ChatInputProps> = ({
  onSubmit,
  isProcessing,
  placeholder = 'Ask a question...',
}) => {
  const [inputValue, setInputValue] = useState('');
  const inputRef = useRef<HTMLTextAreaElement>(null);

  // Focus input on mount
  useEffect(() => {
    if (inputRef.current && !isProcessing) {
      inputRef.current.focus();
    }
  }, [isProcessing]);

  // Auto-resize textarea based on content
  useEffect(() => {
    if (inputRef.current) {
      inputRef.current.style.height = 'auto';
      inputRef.current.style.height = `${Math.min(inputRef.current.scrollHeight, 200)}px`;
    }
  }, [inputValue]);

  const isInputEmpty = !inputValue.trim();

  const handleSubmit = () => {
    if (isInputEmpty || isProcessing) {
      return;
    }

    onSubmit(inputValue.trim());
    setInputValue('');
    
    // Reset textarea height
    if (inputRef.current) {
      inputRef.current.style.height = 'auto';
    }
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    // Submit on Enter (without Shift)
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const handleChange = (e: ChangeEvent<HTMLTextAreaElement>) => {
    setInputValue(e.target.value);
  };

  return (
    <div className="bg-white border-t border-gray-200 p-4">
      <div className="max-w-4xl mx-auto">
        <div className="flex items-end gap-3">
          {/* Text Input */}
          <div className="flex-1 relative">
            <textarea
              ref={inputRef}
              value={inputValue}
              onChange={handleChange}
              onKeyDown={handleKeyDown}
              placeholder={placeholder}
              disabled={isProcessing}
              rows={1}
              aria-label="Chat input"
              className={`
                w-full px-4 py-3 pr-12
                border border-gray-300 rounded-xl
                resize-none overflow-hidden
                focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent
                transition-colors
                ${isProcessing ? 'bg-gray-100 cursor-not-allowed text-gray-500' : 'bg-white'}
              `}
              style={{ minHeight: '48px', maxHeight: '200px' }}
            />
          </div>

          {/* Send Button */}
          <button
            type="button"
            onClick={handleSubmit}
            disabled={isInputEmpty || isProcessing}
            aria-label="Send message"
            className={`
              flex-shrink-0
              w-12 h-12
              flex items-center justify-center
              rounded-xl
              transition-all duration-200
              ${
                isInputEmpty || isProcessing
                  ? 'bg-gray-200 text-gray-400 cursor-not-allowed'
                  : 'bg-blue-600 text-white hover:bg-blue-700 active:scale-95'
              }
            `}
          >
            {isProcessing ? (
              <svg
                className="w-5 h-5 animate-spin"
                fill="none"
                viewBox="0 0 24 24"
                aria-hidden="true"
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
            ) : (
              <svg
                className="w-5 h-5"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
                aria-hidden="true"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8"
                />
              </svg>
            )}
          </button>
        </div>

        {/* Helper text */}
        <div className="mt-2 text-xs text-gray-500 text-center">
          Press <kbd className="px-1.5 py-0.5 bg-gray-100 rounded text-gray-600">Enter</kbd> to send, <kbd className="px-1.5 py-0.5 bg-gray-100 rounded text-gray-600">Shift + Enter</kbd> for new line
        </div>
      </div>
    </div>
  );
};

export default ChatInput;
