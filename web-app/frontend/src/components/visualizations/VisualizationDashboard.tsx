import React, { lazy, Suspense } from 'react';
import { EvaluationResults } from '../../api/types';
import { JudgeComparisonChart } from './JudgeComparisonChart';
import { ConfidenceGauge } from './ConfidenceGauge';
import { HallucinationThermometer } from './HallucinationThermometer';

// Lazy load heavy visualization components
const ScoreDistributionChart = lazy(() => import('./ScoreDistributionChart').then(m => ({ default: m.ScoreDistributionChart })));
const InterJudgeAgreementHeatmap = lazy(() => import('./InterJudgeAgreementHeatmap').then(m => ({ default: m.InterJudgeAgreementHeatmap })));
const HallucinationBreakdownChart = lazy(() => import('./HallucinationBreakdownChart').then(m => ({ default: m.HallucinationBreakdownChart })));
const StatisticsPanel = lazy(() => import('./StatisticsPanel').then(m => ({ default: m.StatisticsPanel })));

// Loading fallback component
const ChartLoadingFallback: React.FC = () => (
  <div className="bg-white p-6 rounded-lg shadow flex items-center justify-center h-64">
    <div className="text-center">
      <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mb-2"></div>
      <p className="text-sm text-gray-500">Loading chart...</p>
    </div>
  </div>
);

interface VisualizationDashboardProps {
  evaluationResults: EvaluationResults;
  showAdvancedMetrics?: boolean;
  historicalScores?: number[];
}

/**
 * Comprehensive visualization dashboard that displays all evaluation metrics
 * and charts in an organized layout.
 */
export const VisualizationDashboard: React.FC<
  VisualizationDashboardProps
> = ({ evaluationResults, showAdvancedMetrics = true, historicalScores }) => {
  // Extract all flagged issues from judge results for drill-down
  const allFlaggedIssues = evaluationResults.judge_results.flatMap(
    (jr) => jr.flagged_issues
  );

  return (
    <div className="space-y-6">
      {/* Top Row: Key Metrics */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Consensus Score Card */}
        <div className="bg-white p-6 rounded-lg shadow">
          <h3 className="text-sm font-medium text-gray-600 mb-2">
            Consensus Score
          </h3>
          <p className="text-4xl font-bold text-gray-900">
            {evaluationResults.consensus_score.toFixed(1)}
          </p>
          <p className="text-sm text-gray-500 mt-1">
            Based on {evaluationResults.judge_results.length} judges
          </p>
        </div>

        {/* Processing Time Card */}
        <div className="bg-white p-6 rounded-lg shadow">
          <h3 className="text-sm font-medium text-gray-600 mb-2">
            Processing Time
          </h3>
          <p className="text-4xl font-bold text-gray-900">
            {(evaluationResults.processing_time_ms / 1000).toFixed(2)}s
          </p>
          <p className="text-sm text-gray-500 mt-1">
            {evaluationResults.verifier_verdicts.length} claims verified
          </p>
        </div>

        {/* Variance Card */}
        <div className="bg-white p-6 rounded-lg shadow">
          <h3 className="text-sm font-medium text-gray-600 mb-2">
            Score Variance
          </h3>
          <p className="text-4xl font-bold text-gray-900">
            {evaluationResults.variance.toFixed(2)}
          </p>
          <p className="text-sm text-gray-500 mt-1">
            Std Dev: {evaluationResults.standard_deviation.toFixed(2)}
          </p>
        </div>
      </div>

      {/* Second Row: Judge Comparison and Confidence */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2">
          <JudgeComparisonChart
            judgeResults={evaluationResults.judge_results}
          />
        </div>
        <div>
          <ConfidenceGauge
            confidenceMetrics={evaluationResults.confidence_metrics}
          />
        </div>
      </div>

      {/* Third Row: Hallucination Metrics */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div>
          <HallucinationThermometer
            hallucinationMetrics={evaluationResults.hallucination_metrics}
          />
        </div>
        <div className="lg:col-span-2">
          <Suspense fallback={<ChartLoadingFallback />}>
            <HallucinationBreakdownChart
              hallucinationMetrics={evaluationResults.hallucination_metrics}
              flaggedIssues={allFlaggedIssues}
            />
          </Suspense>
        </div>
      </div>

      {/* Advanced Metrics Section */}
      {showAdvancedMetrics && (
        <>
          {/* Statistics Panel */}
          <Suspense fallback={<ChartLoadingFallback />}>
            <StatisticsPanel evaluationResults={evaluationResults} />
          </Suspense>

          {/* Fourth Row: Score Distribution */}
          <Suspense fallback={<ChartLoadingFallback />}>
            <ScoreDistributionChart
              judgeResults={evaluationResults.judge_results}
              historicalScores={historicalScores}
            />
          </Suspense>

          {/* Fifth Row: Inter-Judge Agreement */}
          {evaluationResults.judge_results.length >= 2 && (
            <Suspense fallback={<ChartLoadingFallback />}>
              <InterJudgeAgreementHeatmap
                interJudgeAgreement={evaluationResults.inter_judge_agreement}
              />
            </Suspense>
          )}

          {/* Verifier Verdicts Section */}
          {evaluationResults.verifier_verdicts.length > 0 && (
            <div className="bg-white p-6 rounded-lg shadow">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">
                Verifier Verdicts ({evaluationResults.verifier_verdicts.length})
              </h3>
              <div className="space-y-3 max-h-96 overflow-y-auto">
                {evaluationResults.verifier_verdicts.map((verdict, idx) => (
                  <div
                    key={idx}
                    className="border border-gray-200 rounded p-4"
                  >
                    <div className="flex items-start justify-between mb-2">
                      <p className="text-sm font-medium text-gray-900 flex-1">
                        {verdict.claim_text}
                      </p>
                      <span
                        className={`ml-2 px-2 py-1 rounded text-xs font-semibold ${
                          verdict.label === 'SUPPORTED'
                            ? 'bg-green-100 text-green-800'
                            : verdict.label === 'REFUTED'
                            ? 'bg-red-100 text-red-800'
                            : 'bg-yellow-100 text-yellow-800'
                        }`}
                      >
                        {verdict.label}
                      </span>
                    </div>
                    <p className="text-xs text-gray-600 mb-2">
                      Confidence: {(verdict.confidence * 100).toFixed(1)}%
                    </p>
                    {verdict.reasoning && (
                      <p className="text-xs text-gray-700 mt-2">
                        {verdict.reasoning}
                      </p>
                    )}
                    {verdict.evidence.length > 0 && (
                      <div className="mt-2">
                        <p className="text-xs font-medium text-gray-700">
                          Evidence:
                        </p>
                        <ul className="text-xs text-gray-600 list-disc list-inside">
                          {verdict.evidence.slice(0, 2).map((ev, evIdx) => (
                            <li key={evIdx}>{ev}</li>
                          ))}
                          {verdict.evidence.length > 2 && (
                            <li className="text-gray-500">
                              +{verdict.evidence.length - 2} more...
                            </li>
                          )}
                        </ul>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
};
