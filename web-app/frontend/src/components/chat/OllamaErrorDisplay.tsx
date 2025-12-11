import React from 'react';
import { ExclamationTriangleIcon, ArrowPathIcon, CommandLineIcon } from '@heroicons/react/24/outline';

export interface OllamaErrorDisplayProps {
  errorType: 'connection_error' | 'model_not_found' | 'timeout' | 'service_error';
  message: string;
  suggestions?: string[];
  onRetry?: () => void;
  onDismiss?: () => void;
}

/**
 * OllamaErrorDisplay component for displaying Ollama-specific errors
 * with troubleshooting suggestions.
 * 
 * Requirements: 2.4, 9.3
 */
const OllamaErrorDisplay: React.FC<OllamaErrorDisplayProps> = ({
  errorType,
  message,
  suggestions = [],
  onRetry,
  onDismiss,
}) => {
  const getErrorTitle = () => {
    switch (errorType) {
      case 'connection_error':
        return 'Ollama Connection Error';
      case 'model_not_found':
        return 'Model Not Found';
      case 'timeout':
        return 'Request Timeout';
      case 'service_error':
        return 'Ollama Service Error';
      default:
        return 'Error';
    }
  };

  const getDefaultSuggestions = (): string[] => {
    switch (errorType) {
      case 'connection_error':
        return [
          'Ensure Ollama is installed: https://ollama.ai',
          'Start Ollama with: ollama serve',
          'Check if Ollama is running on the configured host',
          'Verify network connectivity',
        ];
      case 'model_not_found':
        return [
          'Pull the required model: ollama pull <model-name>',
          'Check available models: ollama list',
          'Select a different model in settings',
        ];
      case 'timeout':
        return [
          'The request took too long to complete',
          'Try a shorter prompt or simpler question',
          'Check if Ollama is under heavy load',
          'Consider using a smaller model',
        ];
      case 'service_error':
        return [
          'Restart Ollama: ollama serve',
          'Check Ollama logs for more details',
          'Ensure sufficient system resources',
        ];
      default:
        return [];
    }
  };

  const displaySuggestions = suggestions.length > 0 ? suggestions : getDefaultSuggestions();

  const getIconColor = () => {
    switch (errorType) {
      case 'connection_error':
        return 'text-red-600';
      case 'model_not_found':
        return 'text-orange-600';
      case 'timeout':
        return 'text-yellow-600';
      default:
        return 'text-red-600';
    }
  };

  const getBgColor = () => {
    switch (errorType) {
      case 'connection_error':
        return 'bg-red-50 border-red-200';
      case 'model_not_found':
        return 'bg-orange-50 border-orange-200';
      case 'timeout':
        return 'bg-yellow-50 border-yellow-200';
      default:
        return 'bg-red-50 border-red-200';
    }
  };

  return (
    <div className="flex justify-start mb-4">
      <div className="max-w-2xl w-full">
        <div className={`${getBgColor()} border rounded-lg shadow-md p-4`}>
          <div className="flex items-start gap-3">
            <div className="flex-shrink-0 mt-0.5">
              <ExclamationTriangleIcon className={`w-6 h-6 ${getIconColor()}`} />
            </div>
            <div className="flex-1">
              <div className="flex items-center justify-between">
                <h4 className="font-semibold text-gray-900 mb-1">
                  {getErrorTitle()}
                </h4>
                {onDismiss && (
                  <button
                    onClick={onDismiss}
                    className="text-gray-400 hover:text-gray-600"
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
              
              <p className="text-sm text-gray-700 mb-3">{message}</p>
              
              {displaySuggestions.length > 0 && (
                <div className="mt-3 bg-white bg-opacity-50 rounded-md p-3">
                  <div className="flex items-center gap-2 text-xs font-medium text-gray-700 mb-2">
                    <CommandLineIcon className="w-4 h-4" />
                    <span>Troubleshooting Steps:</span>
                  </div>
                  <ul className="space-y-1.5">
                    {displaySuggestions.map((suggestion, idx) => (
                      <li
                        key={idx}
                        className="text-xs text-gray-600 flex items-start gap-2"
                      >
                        <span className="text-gray-400 mt-0.5 font-mono">{idx + 1}.</span>
                        <span className="font-mono">{suggestion}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}
              
              {onRetry && (
                <div className="mt-4">
                  <button
                    onClick={onRetry}
                    className="inline-flex items-center gap-2 px-3 py-1.5 bg-blue-600 text-white text-sm rounded-md hover:bg-blue-700 transition-colors"
                  >
                    <ArrowPathIcon className="w-4 h-4" />
                    Try Again
                  </button>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default OllamaErrorDisplay;
