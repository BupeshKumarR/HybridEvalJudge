import React from 'react';
import { JudgeResult } from '../../api/types';

interface DisagreementWarningProps {
  /** The judge results to analyze for disagreement */
  judges: JudgeResult[];
  /** Pre-calculated variance (optional - will calculate if not provided) */
  variance?: number;
  /** Threshold for showing warning (default: 200) */
  threshold?: number;
}

/**
 * DisagreementWarning - Displays a warning when judges significantly disagree
 * 
 * Features:
 * - Display when judge variance exceeds threshold
 * - Show which judges disagree (highest and lowest scores)
 * - Visual warning indicator
 * 
 * Requirements: 4.3
 * Property 10: Disagreement detection - For any evaluation where judge scores 
 * have variance above threshold, a disagreement warning SHALL be displayed.
 */
const DisagreementWarning: React.FC<DisagreementWarningProps> = ({
  judges,
  variance: providedVariance,
  threshold = 200,
}) => {
  // Calculate variance if not provided
  const calculateVariance = (judgeResults: JudgeResult[]): number => {
    if (judgeResults.length < 2) return 0;
    const scores = judgeResults.map(j => j.score);
    const mean = scores.reduce((a, b) => a + b, 0) / scores.length;
    return scores.reduce((sum, score) => sum + Math.pow(score - mean, 2), 0) / scores.length;
  };

  const variance = providedVariance ?? calculateVariance(judges);

  // Don't render if variance is below threshold
  if (variance <= threshold || judges.length < 2) {
    return null;
  }

  // Find judges with highest and lowest scores
  const sortedJudges = [...judges].sort((a, b) => b.score - a.score);
  const highestJudge = sortedJudges[0];
  const lowestJudge = sortedJudges[sortedJudges.length - 1];
  const scoreDifference = highestJudge.score - lowestJudge.score;

  // Find all judges that significantly deviate from mean
  const scores = judges.map(j => j.score);
  const mean = scores.reduce((a, b) => a + b, 0) / scores.length;
  const stdDev = Math.sqrt(variance);
  
  const disagreeingJudges = judges.filter(j => 
    Math.abs(j.score - mean) > stdDev
  );

  return (
    <div 
      className="bg-amber-50 border border-amber-200 rounded-lg p-3"
      role="alert"
      aria-label="Judge disagreement warning"
    >
      <div className="flex items-start gap-3">
        {/* Warning Icon */}
        <div className="flex-shrink-0">
          <svg 
            className="w-5 h-5 text-amber-600" 
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
        </div>

        <div className="flex-1">
          {/* Warning Title */}
          <h4 className="text-sm font-semibold text-amber-800 mb-1">
            Judges Disagree
          </h4>

          {/* Warning Description */}
          <p className="text-sm text-amber-700 mb-2">
            There is significant disagreement between judges on this evaluation. 
            The score difference is {scoreDifference.toFixed(1)} points.
          </p>

          {/* Disagreeing Judges */}
          <div className="space-y-2">
            {/* Highest Score */}
            <div className="flex items-center justify-between text-sm">
              <span className="text-amber-700">
                <span className="font-medium">{highestJudge.judge_name}</span>
                <span className="text-amber-600"> (highest)</span>
              </span>
              <span className="font-bold text-green-600">
                {highestJudge.score.toFixed(1)}
              </span>
            </div>

            {/* Lowest Score */}
            <div className="flex items-center justify-between text-sm">
              <span className="text-amber-700">
                <span className="font-medium">{lowestJudge.judge_name}</span>
                <span className="text-amber-600"> (lowest)</span>
              </span>
              <span className="font-bold text-red-600">
                {lowestJudge.score.toFixed(1)}
              </span>
            </div>
          </div>

          {/* Additional Stats */}
          <div className="mt-2 pt-2 border-t border-amber-200 text-xs text-amber-600">
            <span>Variance: {variance.toFixed(1)}</span>
            <span className="mx-2">•</span>
            <span>Std Dev: {stdDev.toFixed(1)}</span>
            {disagreeingJudges.length > 0 && (
              <>
                <span className="mx-2">•</span>
                <span>{disagreeingJudges.length} judge(s) deviate significantly</span>
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default DisagreementWarning;
