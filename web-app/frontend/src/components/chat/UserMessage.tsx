import React from 'react';
import { formatDistanceToNow } from 'date-fns';

interface UserMessageProps {
  sourceText: string;
  candidateOutput: string;
  timestamp: Date;
}

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
          {formatDistanceToNow(timestamp, { addSuffix: true })}
        </div>
      </div>
    </div>
  );
};

export default UserMessage;
