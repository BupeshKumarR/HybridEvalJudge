import React from 'react';
import { formatDistanceToNow, isValid } from 'date-fns';

interface UserMessageProps {
  sourceText: string;
  candidateOutput: string;
  timestamp: Date | string;
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

const UserMessage: React.FC<UserMessageProps> = ({
  sourceText,
  candidateOutput,
  timestamp,
}) => {
  return (
    <div className="flex justify-end mb-4">
      <div className="max-w-3xl w-full">
        <div className="bg-blue-600 text-white rounded-lg p-4 shadow-md">
          <div className="mb-3">
            <h4 className="text-sm font-semibold mb-2 opacity-90">Source Text:</h4>
            <p className="text-sm leading-relaxed whitespace-pre-wrap">
              {sourceText}
            </p>
          </div>
          <div className="border-t border-blue-500 pt-3">
            <h4 className="text-sm font-semibold mb-2 opacity-90">
              Candidate Output:
            </h4>
            <p className="text-sm leading-relaxed whitespace-pre-wrap">
              {candidateOutput}
            </p>
          </div>
        </div>
        <div className="text-xs text-gray-500 mt-1 text-right">
          {formatTimestamp(timestamp)}
        </div>
      </div>
    </div>
  );
};

export default UserMessage;
