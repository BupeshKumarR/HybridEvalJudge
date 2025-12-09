import React, { useState } from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
  ErrorBar,
} from 'recharts';
import { JudgeResult } from '../../api/types';

interface JudgeComparisonChartProps {
  judgeResults: JudgeResult[];
}

interface ChartData {
  judgeName: string;
  score: number;
  confidence: number;
  confidenceLower: number;
  confidenceUpper: number;
  color: string;
  reasoning?: string;
  flaggedIssuesCount: number;
}

const getScoreColor = (score: number): string => {
  if (score >= 80) return '#10b981'; // green
  if (score >= 50) return '#f59e0b'; // yellow
  return '#ef4444'; // red
};

const CustomTooltip = ({ active, payload }: any) => {
  if (active && payload && payload.length) {
    const data = payload[0].payload as ChartData;
    return (
      <div className="bg-white p-4 rounded-lg shadow-lg border border-gray-200 max-w-md">
        <p className="font-semibold text-gray-900 mb-2">{data.judgeName}</p>
        <div className="space-y-1 text-sm">
          <p className="text-gray-700">
            <span className="font-medium">Score:</span> {data.score.toFixed(1)}
          </p>
          <p className="text-gray-700">
            <span className="font-medium">Confidence:</span>{' '}
            {(data.confidence * 100).toFixed(1)}%
          </p>
          <p className="text-gray-700">
            <span className="font-medium">Confidence Range:</span>{' '}
            {data.confidenceLower.toFixed(1)} - {data.confidenceUpper.toFixed(1)}
          </p>
          <p className="text-gray-700">
            <span className="font-medium">Flagged Issues:</span>{' '}
            {data.flaggedIssuesCount}
          </p>
          {data.reasoning && (
            <div className="mt-2 pt-2 border-t border-gray-200">
              <p className="font-medium text-gray-900 mb-1">Reasoning:</p>
              <p className="text-gray-600 text-xs leading-relaxed">
                {data.reasoning.length > 200
                  ? `${data.reasoning.substring(0, 200)}...`
                  : data.reasoning}
              </p>
            </div>
          )}
        </div>
      </div>
    );
  }
  return null;
};

export const JudgeComparisonChart: React.FC<JudgeComparisonChartProps> = React.memo(({
  judgeResults,
}) => {
  const [expandedJudge, setExpandedJudge] = useState<string | null>(null);

  // Transform judge results into chart data
  const chartData: ChartData[] = judgeResults.map((result) => {
    const errorMargin = (1 - result.confidence) * result.score * 0.5;
    return {
      judgeName: result.judge_name,
      score: result.score,
      confidence: result.confidence,
      confidenceLower: Math.max(0, result.score - errorMargin),
      confidenceUpper: Math.min(100, result.score + errorMargin),
      color: getScoreColor(result.score),
      reasoning: result.reasoning,
      flaggedIssuesCount: result.flagged_issues.length,
    };
  });

  const handleBarClick = (data: ChartData) => {
    setExpandedJudge(
      expandedJudge === data.judgeName ? null : data.judgeName
    );
  };

  const expandedJudgeData = judgeResults.find(
    (j) => j.judge_name === expandedJudge
  );

  return (
    <div className="space-y-4">
      <div className="bg-white p-4 rounded-lg shadow">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">
          Judge Comparison
        </h3>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart
            data={chartData}
            layout="vertical"
            margin={{ top: 5, right: 30, left: 100, bottom: 5 }}
          >
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis type="number" domain={[0, 100]} />
            <YAxis dataKey="judgeName" type="category" width={90} />
            <Tooltip content={<CustomTooltip />} />
            <Bar
              dataKey="score"
              onClick={(data) => handleBarClick(data)}
              cursor="pointer"
            >
              {chartData.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={entry.color} />
              ))}
              <ErrorBar
                dataKey="confidenceLower"
                width={4}
                strokeWidth={2}
                stroke="#666"
                direction="x"
              />
            </Bar>
          </BarChart>
        </ResponsiveContainer>
        <div className="mt-4 flex items-center justify-center space-x-6 text-sm">
          <div className="flex items-center">
            <div className="w-4 h-4 bg-green-500 rounded mr-2"></div>
            <span className="text-gray-600">High (80-100)</span>
          </div>
          <div className="flex items-center">
            <div className="w-4 h-4 bg-yellow-500 rounded mr-2"></div>
            <span className="text-gray-600">Medium (50-79)</span>
          </div>
          <div className="flex items-center">
            <div className="w-4 h-4 bg-red-500 rounded mr-2"></div>
            <span className="text-gray-600">Low (0-49)</span>
          </div>
        </div>
      </div>

      {/* Expanded Judge Details */}
      {expandedJudge && expandedJudgeData && (
        <div className="bg-white p-6 rounded-lg shadow">
          <div className="flex items-center justify-between mb-4">
            <h4 className="text-lg font-semibold text-gray-900">
              {expandedJudge} - Detailed Analysis
            </h4>
            <button
              onClick={() => setExpandedJudge(null)}
              className="text-gray-500 hover:text-gray-700"
            >
              <svg
                className="w-5 h-5"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M6 18L18 6M6 6l12 12"
                />
              </svg>
            </button>
          </div>

          <div className="space-y-4">
            <div>
              <h5 className="font-medium text-gray-900 mb-2">Reasoning</h5>
              <p className="text-gray-700 text-sm leading-relaxed">
                {expandedJudgeData.reasoning}
              </p>
            </div>

            {expandedJudgeData.flagged_issues.length > 0 && (
              <div>
                <h5 className="font-medium text-gray-900 mb-2">
                  Flagged Issues ({expandedJudgeData.flagged_issues.length})
                </h5>
                <div className="space-y-2">
                  {expandedJudgeData.flagged_issues.map((issue, idx) => (
                    <div
                      key={idx}
                      className="border border-gray-200 rounded p-3"
                    >
                      <div className="flex items-center justify-between mb-1">
                        <span className="text-sm font-medium text-gray-900">
                          {issue.issue_type.replace(/_/g, ' ').toUpperCase()}
                        </span>
                        <span
                          className={`text-xs px-2 py-1 rounded ${
                            issue.severity === 'critical'
                              ? 'bg-red-100 text-red-800'
                              : issue.severity === 'high'
                              ? 'bg-orange-100 text-orange-800'
                              : issue.severity === 'medium'
                              ? 'bg-yellow-100 text-yellow-800'
                              : 'bg-blue-100 text-blue-800'
                          }`}
                        >
                          {issue.severity}
                        </span>
                      </div>
                      <p className="text-sm text-gray-600">
                        {issue.description}
                      </p>
                      {issue.evidence && Object.keys(issue.evidence).length > 0 && (
                        <div className="mt-2">
                          <p className="text-xs font-medium text-gray-700">
                            Evidence:
                          </p>
                          <ul className="text-xs text-gray-600 list-disc list-inside">
                            {Object.values(issue.evidence).map((ev: any, evIdx) => (
                              <li key={evIdx}>{String(ev)}</li>
                            ))}
                          </ul>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}

            <div className="grid grid-cols-2 gap-4 pt-4 border-t border-gray-200">
              <div>
                <p className="text-sm text-gray-600">Response Time</p>
                <p className="text-lg font-semibold text-gray-900">
                  {expandedJudgeData.response_time_ms}ms
                </p>
              </div>
              <div>
                <p className="text-sm text-gray-600">Confidence</p>
                <p className="text-lg font-semibold text-gray-900">
                  {(expandedJudgeData.confidence * 100).toFixed(1)}%
                </p>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
});
