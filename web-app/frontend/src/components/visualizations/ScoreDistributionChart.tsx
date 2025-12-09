import React, { useMemo } from 'react';
import {
  ComposedChart,
  Bar,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts';
import { JudgeResult } from '../../api/types';
import { sampleData, needsOptimization } from '../../utils/dataOptimization';

interface ScoreDistributionChartProps {
  judgeResults: JudgeResult[];
  historicalScores?: number[]; // Optional historical data for comparison
}

interface DistributionData {
  range: string;
  current: number;
  historical?: number;
}

interface StatisticsData {
  min: number;
  q1: number;
  median: number;
  q3: number;
  max: number;
  mean: number;
  outliers: number[];
}

const calculateStatistics = (scores: number[]): StatisticsData => {
  const sorted = [...scores].sort((a, b) => a - b);
  const n = sorted.length;

  const median =
    n % 2 === 0
      ? (sorted[n / 2 - 1] + sorted[n / 2]) / 2
      : sorted[Math.floor(n / 2)];

  const q1 =
    n % 4 === 0
      ? (sorted[Math.floor(n / 4) - 1] + sorted[Math.floor(n / 4)]) / 2
      : sorted[Math.floor(n / 4)];

  const q3 =
    n % 4 === 0
      ? (sorted[Math.floor((3 * n) / 4) - 1] + sorted[Math.floor((3 * n) / 4)]) / 2
      : sorted[Math.floor((3 * n) / 4)];

  const mean = scores.reduce((sum, score) => sum + score, 0) / n;

  // Detect outliers using IQR method
  const iqr = q3 - q1;
  const lowerBound = q1 - 1.5 * iqr;
  const upperBound = q3 + 1.5 * iqr;
  const outliers = scores.filter(
    (score) => score < lowerBound || score > upperBound
  );

  return {
    min: sorted[0],
    q1,
    median,
    q3,
    max: sorted[n - 1],
    mean,
    outliers,
  };
};

const createDistribution = (
  scores: number[],
  binSize: number = 10
): DistributionData[] => {
  const bins: { [key: string]: number } = {};

  // Initialize bins
  for (let i = 0; i <= 100; i += binSize) {
    const range = `${i}-${i + binSize - 1}`;
    bins[range] = 0;
  }

  // Count scores in each bin
  scores.forEach((score) => {
    const binIndex = Math.floor(score / binSize) * binSize;
    const range = `${binIndex}-${binIndex + binSize - 1}`;
    if (bins[range] !== undefined) {
      bins[range]++;
    }
  });

  return Object.entries(bins).map(([range, count]) => ({
    range,
    current: count,
  }));
};

export const ScoreDistributionChart: React.FC<
  ScoreDistributionChartProps
> = ({ judgeResults, historicalScores }) => {
  // Memoize expensive calculations
  const currentScores = useMemo(() => judgeResults.map((jr) => jr.score), [judgeResults]);
  
  // Sample historical data if it's too large
  const sampledHistoricalScores = useMemo(() => {
    if (!historicalScores) return undefined;
    if (needsOptimization(historicalScores.length, 1000)) {
      return sampleData(historicalScores, 1000);
    }
    return historicalScores;
  }, [historicalScores]);
  
  const stats = useMemo(() => calculateStatistics(currentScores), [currentScores]);
  const distribution = useMemo(() => createDistribution(currentScores), [currentScores]);

  // Add historical data if provided
  const distributionWithHistory = useMemo(() => {
    if (sampledHistoricalScores && sampledHistoricalScores.length > 0) {
      const historicalDist = createDistribution(sampledHistoricalScores);
      return distribution.map((item, idx) => ({
        ...item,
        historical: historicalDist[idx]?.current || 0,
      }));
    }
    return distribution;
  }, [distribution, sampledHistoricalScores]);

  return (
    <div className="bg-white p-4 rounded-lg shadow space-y-4">
      <h3 className="text-lg font-semibold text-gray-900">
        Score Distribution
      </h3>

      {/* Box Plot Visualization */}
      <div className="bg-gray-50 p-4 rounded">
        <h4 className="text-sm font-semibold text-gray-700 mb-3">
          Box Plot Summary
        </h4>
        <div className="relative h-24">
          {/* Box plot */}
          <svg width="100%" height="100" viewBox="0 0 400 100">
            {/* Scale */}
            <line
              x1="50"
              y1="50"
              x2="350"
              y2="50"
              stroke="#9ca3af"
              strokeWidth="2"
            />

            {/* Min to Q1 whisker */}
            <line
              x1={50 + (stats.min / 100) * 300}
              y1="50"
              x2={50 + (stats.q1 / 100) * 300}
              y2="50"
              stroke="#6b7280"
              strokeWidth="2"
            />

            {/* Q3 to Max whisker */}
            <line
              x1={50 + (stats.q3 / 100) * 300}
              y1="50"
              x2={50 + (stats.max / 100) * 300}
              y2="50"
              stroke="#6b7280"
              strokeWidth="2"
            />

            {/* Box (Q1 to Q3) */}
            <rect
              x={50 + (stats.q1 / 100) * 300}
              y="30"
              width={((stats.q3 - stats.q1) / 100) * 300}
              height="40"
              fill="#3b82f6"
              fillOpacity="0.3"
              stroke="#3b82f6"
              strokeWidth="2"
            />

            {/* Median line */}
            <line
              x1={50 + (stats.median / 100) * 300}
              y1="30"
              x2={50 + (stats.median / 100) * 300}
              y2="70"
              stroke="#1f2937"
              strokeWidth="3"
            />

            {/* Mean marker */}
            <circle
              cx={50 + (stats.mean / 100) * 300}
              cy="50"
              r="4"
              fill="#ef4444"
            />

            {/* Outliers */}
            {stats.outliers.map((outlier, idx) => (
              <circle
                key={idx}
                cx={50 + (outlier / 100) * 300}
                cy="50"
                r="3"
                fill="#f59e0b"
                stroke="#d97706"
                strokeWidth="1"
              />
            ))}

            {/* Scale labels */}
            {[0, 25, 50, 75, 100].map((val) => (
              <text
                key={val}
                x={50 + (val / 100) * 300}
                y="85"
                textAnchor="middle"
                fontSize="10"
                fill="#6b7280"
              >
                {val}
              </text>
            ))}
          </svg>
        </div>

        {/* Statistics */}
        <div className="grid grid-cols-3 gap-4 mt-4">
          <div>
            <p className="text-xs text-gray-600">Min</p>
            <p className="text-sm font-semibold text-gray-900">
              {stats.min.toFixed(1)}
            </p>
          </div>
          <div>
            <p className="text-xs text-gray-600">Q1</p>
            <p className="text-sm font-semibold text-gray-900">
              {stats.q1.toFixed(1)}
            </p>
          </div>
          <div>
            <p className="text-xs text-gray-600">Median</p>
            <p className="text-sm font-semibold text-gray-900">
              {stats.median.toFixed(1)}
            </p>
          </div>
          <div>
            <p className="text-xs text-gray-600">Q3</p>
            <p className="text-sm font-semibold text-gray-900">
              {stats.q3.toFixed(1)}
            </p>
          </div>
          <div>
            <p className="text-xs text-gray-600">Max</p>
            <p className="text-sm font-semibold text-gray-900">
              {stats.max.toFixed(1)}
            </p>
          </div>
          <div>
            <p className="text-xs text-gray-600">Mean</p>
            <p className="text-sm font-semibold text-red-600">
              {stats.mean.toFixed(1)}
            </p>
          </div>
        </div>

        {stats.outliers.length > 0 && (
          <div className="mt-3 p-2 bg-yellow-50 border border-yellow-200 rounded">
            <p className="text-xs font-medium text-yellow-800">
              {stats.outliers.length} Outlier(s) Detected
            </p>
            <p className="text-xs text-yellow-700">
              Scores: {stats.outliers.map((o) => o.toFixed(1)).join(', ')}
            </p>
          </div>
        )}
      </div>

      {/* Histogram */}
      <div>
        <h4 className="text-sm font-semibold text-gray-700 mb-3">
          Distribution Histogram
        </h4>
        <ResponsiveContainer width="100%" height={250}>
          <ComposedChart data={distributionWithHistory}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis
              dataKey="range"
              angle={-45}
              textAnchor="end"
              height={80}
              tick={{ fontSize: 10 }}
            />
            <YAxis label={{ value: 'Count', angle: -90, position: 'insideLeft' }} />
            <Tooltip />
            <Legend />
            <Bar dataKey="current" fill="#3b82f6" name="Current Session" />
            {sampledHistoricalScores && (
              <Bar
                dataKey="historical"
                fill="#9ca3af"
                name="Historical Average"
              />
            )}
            <Line
              type="monotone"
              dataKey="current"
              stroke="#1f2937"
              strokeWidth={2}
              dot={false}
              name="Trend"
            />
            <ReferenceLine
              y={stats.mean}
              stroke="#ef4444"
              strokeDasharray="3 3"
              label={{ value: 'Mean', position: 'right', fill: '#ef4444' }}
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      {/* Legend */}
      <div className="flex items-center justify-center space-x-6 text-xs">
        <div className="flex items-center">
          <div className="w-3 h-3 bg-blue-500 rounded mr-1"></div>
          <span className="text-gray-600">Current</span>
        </div>
        {sampledHistoricalScores && (
          <div className="flex items-center">
            <div className="w-3 h-3 bg-gray-400 rounded mr-1"></div>
            <span className="text-gray-600">Historical</span>
          </div>
        )}
        <div className="flex items-center">
          <div className="w-3 h-3 bg-red-500 rounded-full mr-1"></div>
          <span className="text-gray-600">Mean</span>
        </div>
        <div className="flex items-center">
          <div className="w-3 h-3 bg-yellow-500 rounded-full mr-1"></div>
          <span className="text-gray-600">Outlier</span>
        </div>
      </div>
    </div>
  );
};
