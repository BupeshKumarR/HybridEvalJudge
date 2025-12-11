import React, { useState } from 'react';
import { JudgeResult } from '../../api/types';

interface JudgeVerdictCardProps {
  /** The judge result to display */
  judge: JudgeResult;
  /** Whether to start with reasoning expanded */
  defaultExpanded?: boolean;
}

/**
 * JudgeVerdictCard - Displays a single judge's verdict with expandable reasoning
 * 
 * Features:
 * - Show judge name, score, confidence
 * - Color coding based on score (green >= 80, yellow 50-79, red < 50)
 * - Expandable reasoning section
 * - Display flagged issues with severity badges
 * 
 * Requirements: 4.2, 4.5, 5.1, 5.2
 */
const JudgeVerdictCard: React.FC<JudgeVerdictCardProps> = ({
  judge,
  defaultExpanded = false,
}) => {
  const [isExpanded, setIsExpanded] = useState(defaultExpanded);

  /**
   * Get color class based on score
   * Property 12: Score color coding
   * - scores >= 80 SHALL be green
   * - scores 50-79 SHALL be yellow
   * - scores < 50 SHALL be red
   */
  const getScoreColor = (score: number): string => {
    if (score >= 80) return 'text-green-600';
    if (score >= 50) return 'text-yellow-600';
    return 'text-red-600';
  };

  const getScoreBgColor = (score: number): string => {
    if (score >= 80) return 'bg-green-100';
    if (score >= 50) return 'bg-yellow-100';
    return 'bg-red-100';
  };

  const getScoreBorderColor = (score: number): string => {
    if (score >= 80) return 'border-green-200';
    if (score >= 50) return 'border-yellow-200';
    return 'border-red-200';
  };

  const getSeverityBadgeStyle = (severity: string): string => {
    switch (severity.toLowerCase()) {
      case 'severe':
      case 'critical':
        return 'bg-red-100 text-red-800 border-red-200';
      case 'moderate':
      case 'medium':
        return 'bg-orange-100 text-orange-800 border-orange-200';
      case 'minor':
      case 'low':
        return 'bg-yellow-100 text-yellow-800 border-yellow-200';
      default:
        return 'bg-gray-100 text-gray-800 border-gray-200';
    }
  };

  const getConfidenceLabel = (confidence: number): string => {
    if (confidence >= 0.9) return 'Very High';
    if (confidence >= 0.7) return 'High';
    if (confidence >= 0.5) return 'Medium';
    return 'Low';
  };

  return (
    <div 
      className={`border rounded-lg overflow-hidden ${getScoreBorderColor(judge.score)}`}
      role="article"
      aria-label={`${judge.judge_name} verdict`}
    >
      {/* Header - Always visible */}
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full px-3 py-2 flex items-center justify-between hover:bg-gray-50 transition-colors"
        aria-expanded={isExpanded}
        aria-controls={`judge-details-${judge.judge_name}`}
      >
        <div className="flex items-center gap-3">
          {/* Judge Icon */}
          <div className="w-8 h-8 rounded-full bg-gray-200 flex items-center justify-center">
            <svg className="w-4 h-4 text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          </div>

          {/* Judge Name */}
          <div className="text-left">
            <div className="text-sm font-medium text-gray-900">{judge.judge_name}</div>
            <div className="text-xs text-gray-500">
              Confidence: {getConfidenceLabel(judge.confidence)} ({(judge.confidence * 100).toFixed(0)}%)
            </div>
          </div>
        </div>

        <div className="flex items-center gap-2">
          {/* Score Badge */}
          <div 
            className={`px-3 py-1 rounded-full text-sm font-bold ${getScoreBgColor(judge.score)} ${getScoreColor(judge.score)}`}
          >
            {judge.score.toFixed(1)}
          </div>

          {/* Flagged Issues Count */}
          {judge.flagged_issues.length > 0 && (
            <div className="flex items-center gap-1 px-2 py-1 bg-red-50 rounded-full">
              <svg className="w-3 h-3 text-red-500" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
              </svg>
              <span className="text-xs text-red-600">{judge.flagged_issues.length}</span>
            </div>
          )}

          {/* Expand Icon */}
          <svg
            className={`w-4 h-4 text-gray-400 transition-transform ${isExpanded ? 'rotate-180' : ''}`}
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
          </svg>
        </div>
      </button>


      {/* Expanded Details */}
      {isExpanded && (
        <div 
          id={`judge-details-${judge.judge_name}`}
          className="px-3 py-3 border-t border-gray-100 bg-gray-50"
        >
          {/* Reasoning Section */}
          {judge.reasoning && (
            <div className="mb-3">
              <h5 className="text-xs font-medium text-gray-700 mb-1">Reasoning</h5>
              <p className="text-sm text-gray-600 bg-white rounded p-2 border border-gray-200">
                {judge.reasoning}
              </p>
            </div>
          )}

          {/* Flagged Issues */}
          {judge.flagged_issues.length > 0 && (
            <div>
              <h5 className="text-xs font-medium text-gray-700 mb-2">
                Flagged Issues ({judge.flagged_issues.length})
              </h5>
              <div className="space-y-2">
                {judge.flagged_issues.map((issue, idx) => (
                  <div 
                    key={idx}
                    className={`p-2 rounded border ${getSeverityBadgeStyle(issue.severity)}`}
                  >
                    <div className="flex items-center gap-2 mb-1">
                      <span className={`text-xs font-semibold px-2 py-0.5 rounded ${getSeverityBadgeStyle(issue.severity)}`}>
                        {issue.severity}
                      </span>
                      <span className="text-xs font-medium text-gray-700">
                        {issue.issue_type}
                      </span>
                    </div>
                    <p className="text-xs text-gray-600">{issue.description}</p>
                    {issue.text_span_start !== undefined && issue.text_span_end !== undefined && (
                      <div className="mt-1 text-xs text-gray-500">
                        Text span: [{issue.text_span_start}-{issue.text_span_end}]
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Response Time */}
          {judge.response_time_ms && (
            <div className="mt-2 text-xs text-gray-500">
              Response time: {(judge.response_time_ms / 1000).toFixed(2)}s
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default JudgeVerdictCard;
