import React, { useState } from 'react';
import { formatDistanceToNow, isValid } from 'date-fns';
import { EvaluationResults } from '../../api/types';
import ExportMenu from '../export/ExportMenu';
import ShareButton from '../export/ShareButton';

interface EvaluationResultMessageProps {
  results: EvaluationResults;
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

const EvaluationResultMessage: React.FC<EvaluationResultMessageProps> = ({
  results,
  timestamp,
}) => {
  const [expandedSections, setExpandedSections] = useState<{
    judges: boolean;
    confidence: boolean;
    hallucination: boolean;
    statistics: boolean;
  }>({
    judges: false,
    confidence: false,
    hallucination: false,
    statistics: false,
  });

  const toggleSection = (section: keyof typeof expandedSections) => {
    setExpandedSections((prev) => ({
      ...prev,
      [section]: !prev[section],
    }));
  };

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
      <div className="max-w-4xl w-full">
        <div className="bg-white border border-gray-200 rounded-lg shadow-md overflow-hidden">
          {/* Header with Consensus Score */}
          <div className="bg-gradient-to-r from-purple-600 to-blue-600 text-white p-4">
            <div className="flex items-center justify-between mb-3">
              <div>
                <h3 className="text-lg font-semibold mb-1">
                  Evaluation Complete
                </h3>
                <p className="text-sm opacity-90">
                  Session: {results.session_id.substring(0, 8)}...
                </p>
              </div>
              <div className="text-right">
                <div className="text-3xl font-bold">
                  {results.consensus_score.toFixed(1)}
                </div>
                <div className="text-xs opacity-90">Consensus Score</div>
              </div>
            </div>
            {/* Export and Share Actions */}
            <div className="flex justify-end gap-2">
              <ShareButton sessionId={results.session_id} />
              <ExportMenu sessionId={results.session_id} />
            </div>
          </div>

          {/* Key Metrics Summary */}
          <div className="grid grid-cols-3 gap-4 p-4 bg-gray-50 border-b border-gray-200">
            <div className="text-center">
              <div
                className={`text-2xl font-bold ${getScoreColor(
                  100 - results.hallucination_metrics.overall_score
                )}`}
              >
                {results.hallucination_metrics.overall_score.toFixed(1)}
              </div>
              <div className="text-xs text-gray-600 mt-1">
                Hallucination Score
              </div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-blue-600">
                {(results.confidence_metrics.mean_confidence * 100).toFixed(1)}%
              </div>
              <div className="text-xs text-gray-600 mt-1">
                Mean Confidence
              </div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-purple-600">
                {results.judge_results.length}
              </div>
              <div className="text-xs text-gray-600 mt-1">Judges</div>
            </div>
          </div>

          {/* Expandable Sections */}
          <div className="divide-y divide-gray-200">
            {/* Judge Results */}
            <div>
              <button
                onClick={() => toggleSection('judges')}
                className="w-full px-4 py-3 flex items-center justify-between hover:bg-gray-50 transition-colors"
              >
                <div className="flex items-center gap-2">
                  <svg
                    className={`w-5 h-5 text-gray-500 transition-transform ${
                      expandedSections.judges ? 'rotate-90' : ''
                    }`}
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M9 5l7 7-7 7"
                    />
                  </svg>
                  <span className="font-medium text-gray-900">
                    Judge Results ({results.judge_results.length})
                  </span>
                </div>
              </button>
              {expandedSections.judges && (
                <div className="px-4 pb-4 space-y-3">
                  {results.judge_results.map((judge, idx) => (
                    <div
                      key={idx}
                      className="border border-gray-200 rounded-lg p-3"
                    >
                      <div className="flex items-center justify-between mb-2">
                        <span className="font-medium text-gray-900">
                          {judge.judge_name}
                        </span>
                        <span
                          className={`px-3 py-1 rounded-full text-sm font-semibold ${getScoreBgColor(
                            judge.score
                          )} ${getScoreColor(judge.score)}`}
                        >
                          {judge.score.toFixed(1)}
                        </span>
                      </div>
                      <div className="text-sm text-gray-600 mb-2">
                        <span className="font-medium">Confidence:</span>{' '}
                        {(judge.confidence * 100).toFixed(1)}%
                      </div>
                      <div className="text-sm text-gray-700 bg-gray-50 rounded p-2">
                        {judge.reasoning}
                      </div>
                      {judge.flagged_issues.length > 0 && (
                        <div className="mt-2">
                          <div className="text-xs font-medium text-gray-700 mb-1">
                            Flagged Issues:
                          </div>
                          <div className="space-y-1">
                            {judge.flagged_issues.map((issue, issueIdx) => (
                              <div
                                key={issueIdx}
                                className="text-xs bg-red-50 border border-red-200 rounded p-2"
                              >
                                <span className="font-medium text-red-800">
                                  {issue.issue_type}
                                </span>
                                {' - '}
                                <span className="text-red-700">
                                  {issue.description}
                                </span>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Confidence Metrics */}
            <div>
              <button
                onClick={() => toggleSection('confidence')}
                className="w-full px-4 py-3 flex items-center justify-between hover:bg-gray-50 transition-colors"
              >
                <div className="flex items-center gap-2">
                  <svg
                    className={`w-5 h-5 text-gray-500 transition-transform ${
                      expandedSections.confidence ? 'rotate-90' : ''
                    }`}
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M9 5l7 7-7 7"
                    />
                  </svg>
                  <span className="font-medium text-gray-900">
                    Confidence Metrics
                  </span>
                </div>
                {results.confidence_metrics.is_low_confidence && (
                  <span className="px-2 py-1 bg-yellow-100 text-yellow-800 text-xs rounded-full">
                    Low Confidence
                  </span>
                )}
              </button>
              {expandedSections.confidence && (
                <div className="px-4 pb-4">
                  <div className="space-y-2 text-sm">
                    <div className="flex justify-between">
                      <span className="text-gray-600">Mean Confidence:</span>
                      <span className="font-medium">
                        {(
                          results.confidence_metrics.mean_confidence * 100
                        ).toFixed(1)}
                        %
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600">
                        Confidence Interval:
                      </span>
                      <span className="font-medium">
                        [{results.confidence_metrics.confidence_interval[0].toFixed(1)},{' '}
                        {results.confidence_metrics.confidence_interval[1].toFixed(1)}]
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600">
                        Inter-Judge Agreement:
                      </span>
                      <span className="font-medium capitalize">
                        {results.inter_judge_agreement.interpretation}
                      </span>
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* Hallucination Details */}
            <div>
              <button
                onClick={() => toggleSection('hallucination')}
                className="w-full px-4 py-3 flex items-center justify-between hover:bg-gray-50 transition-colors"
              >
                <div className="flex items-center gap-2">
                  <svg
                    className={`w-5 h-5 text-gray-500 transition-transform ${
                      expandedSections.hallucination ? 'rotate-90' : ''
                    }`}
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M9 5l7 7-7 7"
                    />
                  </svg>
                  <span className="font-medium text-gray-900">
                    Hallucination Analysis
                  </span>
                </div>
              </button>
              {expandedSections.hallucination && (
                <div className="px-4 pb-4">
                  <div className="space-y-3">
                    <div>
                      <div className="text-xs font-medium text-gray-700 mb-2">
                        Breakdown by Type:
                      </div>
                      <div className="space-y-1">
                        {Object.entries(
                          results.hallucination_metrics.breakdown_by_type
                        ).map(([type, score]) => (
                          <div
                            key={type}
                            className="flex items-center justify-between text-sm"
                          >
                            <span className="text-gray-600 capitalize">
                              {type.replace(/_/g, ' ')}:
                            </span>
                            <span
                              className={`font-medium ${getScoreColor(
                                100 - (score as number)
                              )}`}
                            >
                              {(score as number).toFixed(1)}
                            </span>
                          </div>
                        ))}
                      </div>
                    </div>
                    {results.hallucination_metrics.affected_text_spans.length >
                      0 && (
                      <div>
                        <div className="text-xs font-medium text-gray-700 mb-2">
                          Affected Text Spans:
                        </div>
                        <div className="text-xs text-gray-600">
                          {results.hallucination_metrics.affected_text_spans.length}{' '}
                          span(s) identified
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>

            {/* Statistics */}
            <div>
              <button
                onClick={() => toggleSection('statistics')}
                className="w-full px-4 py-3 flex items-center justify-between hover:bg-gray-50 transition-colors"
              >
                <div className="flex items-center gap-2">
                  <svg
                    className={`w-5 h-5 text-gray-500 transition-transform ${
                      expandedSections.statistics ? 'rotate-90' : ''
                    }`}
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M9 5l7 7-7 7"
                    />
                  </svg>
                  <span className="font-medium text-gray-900">Statistics</span>
                </div>
              </button>
              {expandedSections.statistics && (
                <div className="px-4 pb-4">
                  <div className="space-y-2 text-sm">
                    <div className="flex justify-between">
                      <span className="text-gray-600">Variance:</span>
                      <span className="font-medium">
                        {results.variance.toFixed(2)}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600">Std Deviation:</span>
                      <span className="font-medium">
                        {results.standard_deviation.toFixed(2)}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600">Processing Time:</span>
                      <span className="font-medium">
                        {(results.processing_time_ms / 1000).toFixed(2)}s
                      </span>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
        <div className="text-xs text-gray-500 mt-1">
          {formatTimestamp(timestamp)}
        </div>
      </div>
    </div>
  );
};

export default EvaluationResultMessage;
