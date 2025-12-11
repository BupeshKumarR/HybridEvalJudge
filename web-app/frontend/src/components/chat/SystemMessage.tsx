import React from 'react';
import { formatDistanceToNow, isValid } from 'date-fns';

interface SystemMessageProps {
  message: string;
  timestamp: Date | string;
  type?: 'info' | 'success' | 'warning' | 'error';
}

/**
 * Safely format a timestamp for display
 * Handles UTC timestamps from backend properly
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

const SystemMessage: React.FC<SystemMessageProps> = ({
  message,
  timestamp,
  type = 'info',
}) => {
  const getBackgroundColor = () => {
    switch (type) {
      case 'success':
        return 'bg-green-50 border-green-200';
      case 'warning':
        return 'bg-yellow-50 border-yellow-200';
      case 'error':
        return 'bg-red-50 border-red-200';
      default:
        return 'bg-gray-50 border-gray-200';
    }
  };

  const getTextColor = () => {
    switch (type) {
      case 'success':
        return 'text-green-800';
      case 'warning':
        return 'text-yellow-800';
      case 'error':
        return 'text-red-800';
      default:
        return 'text-gray-800';
    }
  };

  const getIcon = () => {
    switch (type) {
      case 'success':
        return (
          <svg
            className="w-5 h-5 text-green-600"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"
            />
          </svg>
        );
      case 'warning':
        return (
          <svg
            className="w-5 h-5 text-yellow-600"
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
        );
      case 'error':
        return (
          <svg
            className="w-5 h-5 text-red-600"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M10 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2m7-2a9 9 0 11-18 0 9 9 0 0118 0z"
            />
          </svg>
        );
      default:
        return (
          <svg
            className="w-5 h-5 text-gray-600"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
            />
          </svg>
        );
    }
  };

  return (
    <div className="flex justify-center mb-4">
      <div className="max-w-2xl w-full">
        <div
          className={`border rounded-lg p-3 shadow-sm ${getBackgroundColor()}`}
        >
          <div className="flex items-start gap-3">
            <div className="flex-shrink-0 mt-0.5">{getIcon()}</div>
            <div className="flex-1">
              <p className={`text-sm leading-relaxed ${getTextColor()}`}>
                {message}
              </p>
            </div>
          </div>
        </div>
        <div className="text-xs text-gray-500 mt-1 text-center">
          {formatTimestamp(timestamp)}
        </div>
      </div>
    </div>
  );
};

export default SystemMessage;
