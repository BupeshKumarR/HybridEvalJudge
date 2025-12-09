import React, { useState } from 'react';
import { EvaluationResults } from '../../api/types';
import { ChevronDownIcon, ChevronUpIcon } from '@heroicons/react/24/outline';

interface StatisticsPanelProps {
  evaluationResults: EvaluationResults;
  defaultExpanded?: boolean;
}

interface StatisticItem {
  label: string;
  value: string | number;
  tooltip: string;
  formula?: string;
}

/**
 * Expandable statistics panel that displays detailed statistical metrics
 * including variance, standard deviation, and score distribution.
 * 
 * Requirements: 6.1, 6.2, 6.5
 */
export const StatisticsPanel: React.FC<StatisticsPanelProps> = ({
  evaluationResults,
  defaultExpanded = false,
}) => {
  const [isExpanded, setIsExpanded] = useState(defaultExpanded);

  // Calculate additional statistics
  const scores = evaluationResults.judge_results.map((jr) => jr.score);
  const mean = evaluationResults.consensus_score;
  const variance = evaluationResults.variance;
  const stdDev = evaluationResults.standard_deviation;

  // Calculate min, max, median, and quartiles
  const sortedScores = [...scores].sort((a, b) => a - b);
  const min = sortedScores[0] || 0;
  const max = sortedScores[sortedScores.length - 1] || 0;
  
  const getMedian = (arr: number[]): number => {
    const mid = Math.floor(arr.length / 2);
    return arr.length % 2 === 0 ? (arr[mid - 1] + arr[mid]) / 2 : arr[mid];
  };
  
  const median = getMedian(sortedScores);
  
  const getQuartile = (arr: number[], q: number): number => {
    const pos = (arr.length - 1) * q;
    const base = Math.floor(pos);
    const rest = pos - base;
    if (arr[base + 1] !== undefined) {
      return arr[base] + rest * (arr[base + 1] - arr[base]);
    }
    return arr[base];
  };
  
  const q1 = getQuartile(sortedScores, 0.25);
  const q3 = getQuartile(sortedScores, 0.75);
  const iqr = q3 - q1;

  // Calculate coefficient of variation
  const coefficientOfVariation = mean !== 0 ? (stdDev / mean) * 100 : 0;

  // Calculate range
  const range = max - min;

  // Calculate confidence metrics
  const meanConfidence = evaluationResults.confidence_metrics.mean_confidence;
  const [ciLower, ciUpper] = evaluationResults.confidence_metrics.confidence_interval;
  const ciWidth = ciUpper - ciLower;

  const statistics: StatisticItem[] = [
    {
      label: 'Mean Score',
      value: mean.toFixed(2),
      tooltip: 'Average score across all judges (consensus score)',
      formula: 'Σ(scores) / n',
    },
    {
      label: 'Variance',
      value: variance.toFixed(2),
      tooltip: 'Measure of score dispersion. Higher values indicate more disagreement between judges.',
      formula: 'Σ(score - mean)² / n',
    },
    {
      label: 'Standard Deviation',
      value: stdDev.toFixed(2),
      tooltip: 'Square root of variance. Indicates typical deviation from the mean score.',
      formula: '√variance',
    },
    {
      label: 'Coefficient of Variation',
      value: `${coefficientOfVariation.toFixed(2)}%`,
      tooltip: 'Relative variability. Lower values indicate more consistent judge scores.',
      formula: '(std_dev / mean) × 100',
    },
    {
      label: 'Minimum Score',
      value: min.toFixed(2),
      tooltip: 'Lowest score given by any judge',
    },
    {
      label: 'Maximum Score',
      value: max.toFixed(2),
      tooltip: 'Highest score given by any judge',
    },
    {
      label: 'Range',
      value: range.toFixed(2),
      tooltip: 'Difference between maximum and minimum scores',
      formula: 'max - min',
    },
    {
      label: 'Median Score',
      value: median.toFixed(2),
      tooltip: 'Middle value when scores are sorted. Less affected by outliers than mean.',
    },
    {
      label: 'Q1 (25th Percentile)',
      value: q1.toFixed(2),
      tooltip: '25% of scores fall below this value',
    },
    {
      label: 'Q3 (75th Percentile)',
      value: q3.toFixed(2),
      tooltip: '75% of scores fall below this value',
    },
    {
      label: 'Interquartile Range (IQR)',
      value: iqr.toFixed(2),
      tooltip: 'Range of the middle 50% of scores. Robust measure of spread.',
      formula: 'Q3 - Q1',
    },
    {
      label: 'Mean Confidence',
      value: `${(meanConfidence * 100).toFixed(1)}%`,
      tooltip: 'Average confidence level across all judges',
    },
    {
      label: 'Confidence Interval',
      value: `[${ciLower.toFixed(2)}, ${ciUpper.toFixed(2)}]`,
      tooltip: `${(evaluationResults.confidence_metrics.confidence_level * 100).toFixed(0)}% confidence interval for the consensus score`,
    },
    {
      label: 'CI Width',
      value: ciWidth.toFixed(2),
      tooltip: 'Width of confidence interval. Narrower intervals indicate more reliable estimates.',
    },
  ];

  return (
    <div className="bg-white rounded-lg shadow">
      {/* Header */}
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full px-6 py-4 flex items-center justify-between hover:bg-gray-50 transition-colors"
        aria-expanded={isExpanded}
        aria-controls="statistics-panel-content"
      >
        <div className="flex items-center space-x-3">
          <h3 className="text-lg font-semibold text-gray-900">
            Statistical Metrics
          </h3>
          <span className="text-sm text-gray-500">
            ({evaluationResults.judge_results.length} judges)
          </span>
        </div>
        {isExpanded ? (
          <ChevronUpIcon className="h-5 w-5 text-gray-500" />
        ) : (
          <ChevronDownIcon className="h-5 w-5 text-gray-500" />
        )}
      </button>

      {/* Content */}
      {isExpanded && (
        <div
          id="statistics-panel-content"
          className="px-6 pb-6 border-t border-gray-200"
        >
          <div className="mt-4 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {statistics.map((stat, index) => (
              <StatisticCard key={index} statistic={stat} />
            ))}
          </div>

          {/* Score Distribution Summary */}
          <div className="mt-6 p-4 bg-gray-50 rounded-lg">
            <h4 className="text-sm font-semibold text-gray-900 mb-2">
              Score Distribution Summary
            </h4>
            <div className="text-sm text-gray-700 space-y-1">
              <p>
                <span className="font-medium">Distribution:</span> The scores
                range from {min.toFixed(2)} to {max.toFixed(2)}, with a median
                of {median.toFixed(2)}.
              </p>
              <p>
                <span className="font-medium">Spread:</span> The standard
                deviation of {stdDev.toFixed(2)} indicates{' '}
                {stdDev < 10
                  ? 'high agreement'
                  : stdDev < 20
                  ? 'moderate agreement'
                  : 'significant disagreement'}{' '}
                among judges.
              </p>
              <p>
                <span className="font-medium">Confidence:</span> The mean
                confidence of {(meanConfidence * 100).toFixed(1)}% suggests{' '}
                {meanConfidence > 0.8
                  ? 'high certainty'
                  : meanConfidence > 0.6
                  ? 'moderate certainty'
                  : 'low certainty'}{' '}
                in the evaluations.
              </p>
              {evaluationResults.confidence_metrics.is_low_confidence && (
                <p className="text-amber-700 font-medium">
                  ⚠️ Low confidence detected. Consider additional evaluation or
                  review.
                </p>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

interface StatisticCardProps {
  statistic: StatisticItem;
}

const StatisticCard: React.FC<StatisticCardProps> = ({ statistic }) => {
  const [showTooltip, setShowTooltip] = useState(false);

  return (
    <div className="relative">
      <div
        className="p-4 border border-gray-200 rounded-lg hover:border-blue-300 transition-colors cursor-help"
        onMouseEnter={() => setShowTooltip(true)}
        onMouseLeave={() => setShowTooltip(false)}
      >
        <div className="text-sm font-medium text-gray-600 mb-1">
          {statistic.label}
        </div>
        <div className="text-2xl font-bold text-gray-900">
          {statistic.value}
        </div>
        {statistic.formula && (
          <div className="text-xs text-gray-500 mt-1 font-mono">
            {statistic.formula}
          </div>
        )}
      </div>

      {/* Tooltip */}
      {showTooltip && (
        <div className="absolute z-10 bottom-full left-1/2 transform -translate-x-1/2 mb-2 px-3 py-2 bg-gray-900 text-white text-xs rounded shadow-lg max-w-xs">
          {statistic.tooltip}
          <div className="absolute top-full left-1/2 transform -translate-x-1/2 -mt-1">
            <div className="border-4 border-transparent border-t-gray-900"></div>
          </div>
        </div>
      )}
    </div>
  );
};
