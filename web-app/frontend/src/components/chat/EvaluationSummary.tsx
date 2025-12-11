import React, { useState } from 'react';
import { EvaluationResults, JudgeResult } from '../../api/types';
import JudgeVerdictCard from './JudgeVerdictCard';
import DisagreementWarning from './DisagreementWarning';
import { ExportMenu } from '../export';

interface EvaluationSummaryProps {
  /** The evaluation results to display */
  results: EvaluationResults;
  /** Whether to start in expanded state */
  defaultExpanded?: boolean;
  /** Optional session ID for export functionality */
  sessionId?: string;
}

/**
 * EvaluationSummary - Compact inline evaluation summary for chat messages
 * 
 * Features:
 * - Compact view showing consensus score and hallucination score
 * - Expandable to show full details including judge verdicts
 * - Attaches to assistant message bubbles
 * 
 * Requirements: 4.1, 4.2
 */
const EvaluationSummary: React.FC<EvaluationSummaryProps> = ({
  results,
  defaultExpanded = false,
  sessionId,
}) => {
  const [isExpanded, setIsExpanded] = useState(defaultExpanded);

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

  // Calculate judge variance for disagreement detection
  const calculateVariance = (judges: JudgeResult[]): number => {
    if (judges.length < 2) return 0;
    const scores = judges.map(j => j.score);
    const mean = scores.reduce((a, b) => a + b, 0) / scores.length;
    return scores.reduce((sum, score) => sum + Math.pow(score - mean, 2), 0) / scores.length;
  };

  const variance = calculateVariance(results.judge_results);
  const DISAGREEMENT_THRESHOLD = 200; // Variance threshold for showing warning
  const hasDisagreement = variance > DISAGREEMENT_THRESHOLD;

  return (
    <div 
      className="bg-white border border-gray-200 rounded-lg shadow-sm overflow-hidden"
      role="region"
      aria-label="Evaluation summary"
    >
      {/* Compact Summary Header */}
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full px-4 py-3 flex items-center justify-between hover:bg-gray-50 transition-colors"
        aria-expanded={isExpanded}
        aria-controls="evaluation-details"
      >
        <div className="flex items-center gap-4">
          {/* Consensus Score */}
          <div className="flex items-center gap-2">
            <div 
              className={`w-10 h-10 rounded-full flex items-center justify-center ${getScoreBgColor(results.consensus_score)}`}
            >
              <span className={`text-sm font-bold ${getScoreColor(results.consensus_score)}`}>
                {results.consensus_score.toFixed(0)}
              </span>
            </div>
            <div className="text-left">
              <div className="text-xs text-gray-500">Consensus</div>
              <div className={`text-sm font-medium ${getScoreColor(results.consensus_score)}`}>
                {results.consensus_score >= 80 ? 'High' : results.consensus_score >= 50 ? 'Medium' : 'Low'}
              </div>
            </div>
          </div>

          {/* Hallucination Score */}
          <div className="flex items-center gap-2">
            <div 
              className={`w-10 h-10 rounded-full flex items-center justify-center ${getScoreBgColor(100 - results.hallucination_metrics.overall_score)}`}
            >
              <span className={`text-sm font-bold ${getScoreColor(100 - results.hallucination_metrics.overall_score)}`}>
                {results.hallucination_metrics.overall_score.toFixed(0)}
              </span>
            </div>
            <div className="text-left">
              <div className="text-xs text-gray-500">Hallucination</div>
              <div className={`text-sm font-medium ${getScoreColor(100 - results.hallucination_metrics.overall_score)}`}>
                {results.hallucination_metrics.overall_score < 25 ? 'Low' : results.hallucination_metrics.overall_score < 50 ? 'Medium' : 'High'}
              </div>
            </div>
          </div>

          {/* Judge Count Badge */}
          <div className="flex items-center gap-1 px-2 py-1 bg-gray-100 rounded-full">
            <svg className="w-4 h-4 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z" />
            </svg>
            <span className="text-xs text-gray-600">{results.judge_results.length} judges</span>
          </div>

          {/* Disagreement Warning Indicator */}
          {hasDisagreement && (
            <div className="flex items-center gap-1 px-2 py-1 bg-amber-100 rounded-full">
              <svg className="w-4 h-4 text-amber-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
              </svg>
              <span className="text-xs text-amber-700">Disagreement</span>
            </div>
          )}
        </div>

        {/* Expand/Collapse Icon */}
        <svg
          className={`w-5 h-5 text-gray-400 transition-transform ${isExpanded ? 'rotate-180' : ''}`}
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>
      </button>


      {/* Expanded Details */}
      {isExpanded && (
        <div id="evaluation-details" className="border-t border-gray-200">
          {/* Disagreement Warning */}
          {hasDisagreement && (
            <div className="px-4 py-3 border-b border-gray-200">
              <DisagreementWarning 
                judges={results.judge_results}
                variance={variance}
              />
            </div>
          )}

          {/* Judge Verdicts */}
          <div className="px-4 py-3">
            <h4 className="text-sm font-medium text-gray-700 mb-3">Judge Verdicts</h4>
            <div className="space-y-3">
              {results.judge_results.map((judge, idx) => (
                <JudgeVerdictCard key={idx} judge={judge} />
              ))}
            </div>
          </div>

          {/* Additional Metrics */}
          <div className="px-4 py-3 bg-gray-50 border-t border-gray-200">
            <div className="grid grid-cols-3 gap-4 text-center">
              <div>
                <div className="text-xs text-gray-500">Confidence</div>
                <div className="text-sm font-medium text-gray-900">
                  {(results.confidence_metrics.mean_confidence * 100).toFixed(0)}%
                </div>
              </div>
              <div>
                <div className="text-xs text-gray-500">Agreement</div>
                <div className="text-sm font-medium text-gray-900 capitalize">
                  {results.inter_judge_agreement.interpretation}
                </div>
              </div>
              <div>
                <div className="text-xs text-gray-500">Processing</div>
                <div className="text-sm font-medium text-gray-900">
                  {(results.processing_time_ms / 1000).toFixed(1)}s
                </div>
              </div>
            </div>
          </div>

          {/* Export Button - Requirements: 11.1 */}
          {sessionId && (
            <div className="px-4 py-3 border-t border-gray-200 flex justify-end">
              <ExportMenu 
                sessionId={sessionId}
                onExportError={(error) => console.error('Export failed:', error)}
              />
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default EvaluationSummary;
