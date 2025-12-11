import React from 'react';
import { ExclamationCircleIcon, CheckCircleIcon, XCircleIcon } from '@heroicons/react/24/outline';

export interface JudgeStatus {
  judgeName: string;
  status: 'available' | 'unavailable' | 'failed' | 'timeout' | 'rate_limited';
  errorMessage?: string;
  score?: number;
  confidence?: number;
  reasoning?: string;
  flaggedIssues?: any[];
}

export interface JudgeFailureDisplayProps {
  judges: JudgeStatus[];
  showAvailable?: boolean;
  onRetryJudge?: (judgeName: string) => void;
}

/**
 * JudgeFailureDisplay component for showing judge availability status
 * and annotating failed judges as "Unavailable".
 * 
 * Requirements: 9.1, 9.2
 */
const JudgeFailureDisplay: React.FC<JudgeFailureDisplayProps> = ({
  judges,
  showAvailable = true,
  onRetryJudge,
}) => {
  const availableJudges = judges.filter(j => j.status === 'available');
  const unavailableJudges = judges.filter(j => j.status !== 'available');
  
  const allJudgesFailed = availableJudges.length === 0 && judges.length > 0;

  const getStatusIcon = (status: JudgeStatus['status']) => {
    switch (status) {
      case 'available':
        return <CheckCircleIcon className="w-5 h-5 text-green-500" />;
      case 'unavailable':
      case 'failed':
        return <XCircleIcon className="w-5 h-5 text-red-500" />;
      case 'timeout':
        return <ExclamationCircleIcon className="w-5 h-5 text-yellow-500" />;
      case 'rate_limited':
        return <ExclamationCircleIcon className="w-5 h-5 text-orange-500" />;
      default:
        return <ExclamationCircleIcon className="w-5 h-5 text-gray-500" />;
    }
  };

  const getStatusLabel = (status: JudgeStatus['status']) => {
    switch (status) {
      case 'available':
        return 'Available';
      case 'unavailable':
        return 'Unavailable';
      case 'failed':
        return 'Failed';
      case 'timeout':
        return 'Timed Out';
      case 'rate_limited':
        return 'Rate Limited';
      default:
        return 'Unknown';
    }
  };

  const getStatusBadgeColor = (status: JudgeStatus['status']) => {
    switch (status) {
      case 'available':
        return 'bg-green-100 text-green-800';
      case 'unavailable':
      case 'failed':
        return 'bg-red-100 text-red-800';
      case 'timeout':
        return 'bg-yellow-100 text-yellow-800';
      case 'rate_limited':
        return 'bg-orange-100 text-orange-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };

  if (judges.length === 0) {
    return null;
  }

  return (
    <div className="space-y-3">
      {/* All judges failed warning - Requirements 9.2 */}
      {allJudgesFailed && (
        <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
          <div className="flex items-start gap-3">
            <ExclamationCircleIcon className="w-6 h-6 text-yellow-600 flex-shrink-0" />
            <div>
              <h4 className="font-semibold text-yellow-900 mb-1">
                All Judges Unavailable
              </h4>
              <p className="text-sm text-yellow-800">
                No judges were able to evaluate this response. The LLM response is displayed without evaluation metrics.
              </p>
              <p className="text-xs text-yellow-700 mt-2">
                Check your API keys and network connectivity, then try again.
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Judge status list */}
      <div className="bg-white border border-gray-200 rounded-lg overflow-hidden">
        <div className="px-4 py-2 bg-gray-50 border-b border-gray-200">
          <h5 className="text-sm font-medium text-gray-700">Judge Status</h5>
        </div>
        <ul className="divide-y divide-gray-100">
          {/* Show unavailable judges first */}
          {unavailableJudges.map((judge) => (
            <li key={judge.judgeName} className="px-4 py-3">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  {getStatusIcon(judge.status)}
                  <div>
                    <span className="text-sm font-medium text-gray-900">
                      {judge.judgeName}
                    </span>
                    {judge.errorMessage && (
                      <p className="text-xs text-gray-500 mt-0.5">
                        {judge.errorMessage}
                      </p>
                    )}
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <span className={`px-2 py-0.5 text-xs font-medium rounded-full ${getStatusBadgeColor(judge.status)}`}>
                    {getStatusLabel(judge.status)}
                  </span>
                  {onRetryJudge && judge.status !== 'rate_limited' && (
                    <button
                      onClick={() => onRetryJudge(judge.judgeName)}
                      className="text-xs text-blue-600 hover:text-blue-800"
                    >
                      Retry
                    </button>
                  )}
                </div>
              </div>
            </li>
          ))}
          
          {/* Show available judges if requested */}
          {showAvailable && availableJudges.map((judge) => (
            <li key={judge.judgeName} className="px-4 py-3">
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-3">
                  {getStatusIcon(judge.status)}
                  <div>
                    <span className="text-sm font-medium text-gray-900">
                      {judge.judgeName}
                    </span>
                    {judge.score !== undefined && (
                      <p className="text-xs text-gray-500 mt-0.5">
                        Score: {judge.score.toFixed(1)} | Confidence: {((judge.confidence || 0) * 100).toFixed(0)}%
                      </p>
                    )}
                  </div>
                </div>
                <span className={`px-2 py-0.5 text-xs font-medium rounded-full ${getStatusBadgeColor(judge.status)}`}>
                  {getStatusLabel(judge.status)}
                </span>
              </div>
              {/* Show reasoning if available */}
              {judge.reasoning && (
                <div className="ml-8 mt-2 text-sm text-gray-700 bg-gray-50 rounded p-2">
                  {judge.reasoning}
                </div>
              )}
              {/* Show flagged issues if available */}
              {judge.flaggedIssues && judge.flaggedIssues.length > 0 && (
                <div className="ml-8 mt-2">
                  <div className="text-xs font-medium text-gray-700 mb-1">
                    Flagged Issues:
                  </div>
                  <div className="space-y-1">
                    {judge.flaggedIssues.slice(0, 3).map((issue, issueIdx) => (
                      <div
                        key={issueIdx}
                        className="text-xs bg-red-50 border border-red-200 rounded p-2"
                      >
                        <span className="font-medium text-red-800">
                          {issue.issue_type}
                        </span>
                        {' - '}
                        <span className="text-red-700">{issue.description}</span>
                      </div>
                    ))}
                    {judge.flaggedIssues.length > 3 && (
                      <div className="text-xs text-gray-500">
                        +{judge.flaggedIssues.length - 3} more issues
                      </div>
                    )}
                  </div>
                </div>
              )}
            </li>
          ))}
        </ul>
      </div>

      {/* Partial results notice */}
      {unavailableJudges.length > 0 && availableJudges.length > 0 && (
        <p className="text-xs text-gray-500 italic">
          * Results shown are based on {availableJudges.length} of {judges.length} judges. 
          Some judges were unavailable.
        </p>
      )}
    </div>
  );
};

export default JudgeFailureDisplay;

/**
 * Compact inline badge for showing judge unavailability
 */
export const JudgeUnavailableBadge: React.FC<{ judgeName: string; reason?: string }> = ({
  judgeName,
  reason,
}) => (
  <div className="inline-flex items-center gap-1.5 px-2 py-1 bg-gray-100 rounded-md">
    <XCircleIcon className="w-4 h-4 text-red-500" />
    <span className="text-xs text-gray-700">{judgeName}</span>
    <span className="text-xs font-medium text-red-600">Unavailable</span>
    {reason && (
      <span className="text-xs text-gray-500">({reason})</span>
    )}
  </div>
);
