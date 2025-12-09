import React, { useEffect, useState } from 'react';

interface StreamingJudgeResultProps {
  judgeName: string;
  score: number;
  confidence: number;
  reasoning: string;
  flaggedIssues: any[];
  isNew?: boolean;
}

const StreamingJudgeResult: React.FC<StreamingJudgeResultProps> = ({
  judgeName,
  score,
  confidence,
  reasoning,
  flaggedIssues,
  isNew = false,
}) => {
  const [animate, setAnimate] = useState(isNew);

  useEffect(() => {
    if (isNew) {
      const timer = setTimeout(() => setAnimate(false), 500);
      return () => clearTimeout(timer);
    }
    return undefined;
  }, [isNew]);

  const getScoreColor = (score: number) => {
    if (score >= 80) return 'text-green-600';
    if (score >= 50) return 'text-yellow-600';
    return 'text-red-600';
  };

  const getScoreBgColor = (score: number) => {
    if (score >= 80) return 'bg-green-100';
    if (score >= 50) return 'bg-yellow-100';
    return 'bg-red-100';
  };

  return (
    <div className="flex justify-start mb-4">
      <div className="max-w-2xl w-full">
        <div
          className={`bg-white border border-gray-200 rounded-lg shadow-md p-4 transition-all duration-500 ${
            animate ? 'scale-105 ring-2 ring-blue-400' : 'scale-100'
          }`}
        >
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center gap-2">
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
              <span className="font-medium text-gray-900">{judgeName}</span>
            </div>
            <span
              className={`px-3 py-1 rounded-full text-sm font-semibold ${getScoreBgColor(
                score
              )} ${getScoreColor(score)}`}
            >
              {score.toFixed(1)}
            </span>
          </div>

          <div className="mb-2">
            <div className="flex items-center justify-between text-sm mb-1">
              <span className="text-gray-600">Confidence:</span>
              <span className="font-medium text-gray-900">
                {(confidence * 100).toFixed(1)}%
              </span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-1.5">
              <div
                className="bg-blue-600 h-1.5 rounded-full transition-all duration-300"
                style={{ width: `${confidence * 100}%` }}
              />
            </div>
          </div>

          <div className="text-sm text-gray-700 bg-gray-50 rounded p-3 mb-2">
            {reasoning}
          </div>

          {flaggedIssues.length > 0 && (
            <div>
              <div className="text-xs font-medium text-gray-700 mb-2">
                Flagged Issues ({flaggedIssues.length}):
              </div>
              <div className="space-y-1">
                {flaggedIssues.slice(0, 3).map((issue, idx) => (
                  <div
                    key={idx}
                    className="text-xs bg-red-50 border border-red-200 rounded p-2"
                  >
                    <span className="font-medium text-red-800">
                      {issue.issue_type}
                    </span>
                    {' - '}
                    <span className="text-red-700">{issue.description}</span>
                  </div>
                ))}
                {flaggedIssues.length > 3 && (
                  <div className="text-xs text-gray-500 text-center">
                    +{flaggedIssues.length - 3} more issues
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default StreamingJudgeResult;
