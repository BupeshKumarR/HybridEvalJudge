import React from 'react';
import { ExclamationTriangleIcon, ArrowPathIcon } from '@heroicons/react/24/outline';

export interface ErrorDisplayProps {
  title?: string;
  message: string;
  details?: string;
  onRetry?: () => void;
  retryText?: string;
  fullScreen?: boolean;
  className?: string;
}

const ErrorDisplay: React.FC<ErrorDisplayProps> = ({
  title = 'Something went wrong',
  message,
  details,
  onRetry,
  retryText = 'Try again',
  fullScreen = false,
  className = '',
}) => {
  const content = (
    <div className={`text-center ${className}`}>
      <div className="inline-flex items-center justify-center w-16 h-16 bg-red-100 rounded-full mb-4">
        <ExclamationTriangleIcon className="w-8 h-8 text-red-600" />
      </div>
      
      <h3 className="text-lg font-semibold text-gray-900 mb-2">{title}</h3>
      
      <p className="text-sm text-gray-600 mb-4 max-w-md mx-auto">{message}</p>
      
      {details && (
        <details className="text-left bg-gray-50 rounded-lg p-3 mb-4 max-w-md mx-auto">
          <summary className="text-sm font-medium text-gray-700 cursor-pointer hover:text-gray-900">
            Technical details
          </summary>
          <pre className="mt-2 text-xs text-gray-600 overflow-auto whitespace-pre-wrap">
            {details}
          </pre>
        </details>
      )}
      
      {onRetry && (
        <button
          onClick={onRetry}
          className="inline-flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
        >
          <ArrowPathIcon className="w-4 h-4" />
          {retryText}
        </button>
      )}
    </div>
  );

  if (fullScreen) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-50 p-4">
        {content}
      </div>
    );
  }

  return content;
};

export default ErrorDisplay;

// Inline error component for forms
export const InlineError: React.FC<{ message: string }> = ({ message }) => (
  <div className="flex items-center gap-2 text-sm text-red-600 mt-1">
    <ExclamationTriangleIcon className="w-4 h-4 flex-shrink-0" />
    <span>{message}</span>
  </div>
);

// Banner error component
export const ErrorBanner: React.FC<{
  message: string;
  onDismiss?: () => void;
}> = ({ message, onDismiss }) => (
  <div className="bg-red-50 border-l-4 border-red-500 p-4 mb-4">
    <div className="flex items-start">
      <ExclamationTriangleIcon className="w-5 h-5 text-red-500 mt-0.5 mr-3 flex-shrink-0" />
      <div className="flex-1">
        <p className="text-sm text-red-800">{message}</p>
      </div>
      {onDismiss && (
        <button
          onClick={onDismiss}
          className="ml-3 text-red-500 hover:text-red-700"
          aria-label="Dismiss"
        >
          <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
            <path
              fillRule="evenodd"
              d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z"
              clipRule="evenodd"
            />
          </svg>
        </button>
      )}
    </div>
  </div>
);
