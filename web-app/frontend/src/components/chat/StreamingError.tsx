import React from 'react';

interface StreamingErrorProps {
  errorType: string;
  message: string;
  recoverySuggestions: string[];
}

const StreamingError: React.FC<StreamingErrorProps> = ({
  errorType,
  message,
  recoverySuggestions,
}) => {
  return (
    <div className="flex justify-start mb-4">
      <div className="max-w-2xl w-full">
        <div className="bg-red-50 border border-red-200 rounded-lg shadow-md p-4">
          <div className="flex items-start gap-3">
            <div className="flex-shrink-0 mt-0.5">
              <svg
                className="w-6 h-6 text-red-600"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                />
              </svg>
            </div>
            <div className="flex-1">
              <h4 className="font-semibold text-red-900 mb-1">
                Evaluation Error
              </h4>
              <div className="text-sm text-red-800 mb-2">
                <span className="font-medium">{errorType}:</span> {message}
              </div>
              {recoverySuggestions.length > 0 && (
                <div className="mt-3">
                  <div className="text-xs font-medium text-red-900 mb-2">
                    Recovery Suggestions:
                  </div>
                  <ul className="space-y-1">
                    {recoverySuggestions.map((suggestion, idx) => (
                      <li
                        key={idx}
                        className="text-xs text-red-700 flex items-start gap-2"
                      >
                        <span className="text-red-500 mt-0.5">•</span>
                        <span>{suggestion}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default StreamingError;
