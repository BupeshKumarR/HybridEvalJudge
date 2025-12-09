import React, { useState, useEffect } from 'react';
import { AggregateStatistics } from '../../api/types';
import { evaluationsApi } from '../../api/evaluations';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';

interface AggregateStatisticsPanelProps {
  days?: number;
}

/**
 * Component that displays aggregate statistics across multiple evaluation sessions,
 * including historical trends and judge performance over time.
 * 
 * Requirements: 6.3
 */
export const AggregateStatisticsPanel: React.FC<
  AggregateStatisticsPanelProps
> = ({ days = 30 }) => {
  const [statistics, setStatistics] = useState<AggregateStatistics | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedDays, setSelectedDays] = useState(days);

  useEffect(() => {
    loadStatistics();
  }, [selectedDays]);

  const loadStatistics = async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await evaluationsApi.getAggregateStatistics(selectedDays);
      setStatistics(data);
    } catch (err) {
      setError('Failed to load aggregate statistics');
      console.error('Error loading aggregate statistics:', err);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="bg-white p-6 rounded-lg shadow">
        <div className="animate-pulse">
          <div className="h-6 bg-gray-200 rounded w-1/4 mb-4"></div>
          <div className="space-y-3">
            <div className="h-4 bg-gray-200 rounded"></div>
            <div className="h-4 bg-gray-200 rounded w-5/6"></div>
          </div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-white p-6 rounded-lg shadow">
        <div className="text-red-600">
          <p className="font-semibold">Error</p>
          <p className="text-sm">{error}</p>
          <button
            onClick={loadStatistics}
            className="mt-2 text-sm text-blue-600 hover:text-blue-800"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  if (!statistics || statistics.total_evaluations === 0) {
    return (
      <div className="bg-white p-6 rounded-lg shadow">
        <h3 className="text-lg font-semibold text-gray-900 mb-2">
          Aggregate Statistics
        </h3>
        <p className="text-gray-600">
          {statistics?.message || 'No evaluation data available for the selected time range.'}
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header with time range selector */}
      <div className="bg-white p-6 rounded-lg shadow">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-gray-900">
            Aggregate Statistics
          </h3>
          <div className="flex items-center space-x-2">
            <label className="text-sm text-gray-600">Time Range:</label>
            <select
              value={selectedDays}
              onChange={(e) => setSelectedDays(Number(e.target.value))}
              className="border border-gray-300 rounded px-3 py-1 text-sm"
            >
              <option value={7}>Last 7 days</option>
              <option value={30}>Last 30 days</option>
              <option value={90}>Last 90 days</option>
              <option value={365}>Last year</option>
            </select>
          </div>
        </div>

        {/* Summary Cards */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <SummaryCard
            label="Total Evaluations"
            value={statistics.total_evaluations}
            icon="📊"
          />
          {statistics.consensus_score_stats && (
            <SummaryCard
              label="Avg Consensus Score"
              value={statistics.consensus_score_stats.mean.toFixed(2)}
              subtitle={`Range: ${statistics.consensus_score_stats.min.toFixed(1)} - ${statistics.consensus_score_stats.max.toFixed(1)}`}
              icon="⭐"
            />
          )}
          {statistics.hallucination_score_stats && (
            <SummaryCard
              label="Avg Hallucination Score"
              value={statistics.hallucination_score_stats.mean.toFixed(2)}
              subtitle={`Std Dev: ${statistics.hallucination_score_stats.std_dev.toFixed(2)}`}
              icon="🔍"
            />
          )}
          {statistics.processing_time_stats && (
            <SummaryCard
              label="Avg Processing Time"
              value={`${(statistics.processing_time_stats.mean / 1000).toFixed(2)}s`}
              subtitle={`Min: ${(statistics.processing_time_stats.min / 1000).toFixed(2)}s`}
              icon="⏱️"
            />
          )}
        </div>

        {/* Trend Analysis */}
        {statistics.consensus_score_trend && (
          <div className="mt-6 p-4 bg-gray-50 rounded-lg">
            <h4 className="text-sm font-semibold text-gray-900 mb-2">
              Consensus Score Trend
            </h4>
            <div className="flex items-center space-x-4">
              <div>
                <p className="text-xs text-gray-600">First Half Average</p>
                <p className="text-lg font-bold text-gray-900">
                  {statistics.consensus_score_trend.first_half_average.toFixed(2)}
                </p>
              </div>
              <div className="text-2xl">→</div>
              <div>
                <p className="text-xs text-gray-600">Second Half Average</p>
                <p className="text-lg font-bold text-gray-900">
                  {statistics.consensus_score_trend.second_half_average.toFixed(2)}
                </p>
              </div>
              <div className="flex-1">
                <div
                  className={`inline-flex items-center px-3 py-1 rounded-full text-sm font-semibold ${
                    statistics.consensus_score_trend.direction === 'improving'
                      ? 'bg-green-100 text-green-800'
                      : statistics.consensus_score_trend.direction === 'declining'
                      ? 'bg-red-100 text-red-800'
                      : 'bg-gray-100 text-gray-800'
                  }`}
                >
                  {statistics.consensus_score_trend.direction === 'improving' && '↗️'}
                  {statistics.consensus_score_trend.direction === 'declining' && '↘️'}
                  {statistics.consensus_score_trend.direction === 'stable' && '→'}
                  <span className="ml-1">
                    {Math.abs(statistics.consensus_score_trend.percent_change).toFixed(1)}%{' '}
                    {statistics.consensus_score_trend.direction}
                  </span>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Daily Trends Chart */}
      {statistics.daily_trends.length > 0 && (
        <div className="bg-white p-6 rounded-lg shadow">
          <h4 className="text-lg font-semibold text-gray-900 mb-4">
            Daily Trends
          </h4>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={statistics.daily_trends}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis
                dataKey="date"
                tick={{ fontSize: 12 }}
                tickFormatter={(value) => {
                  const date = new Date(value);
                  return `${date.getMonth() + 1}/${date.getDate()}`;
                }}
              />
              <YAxis yAxisId="left" domain={[0, 100]} />
              <YAxis yAxisId="right" orientation="right" domain={[0, 100]} />
              <Tooltip
                labelFormatter={(value) => new Date(value).toLocaleDateString()}
              />
              <Legend />
              <Line
                yAxisId="left"
                type="monotone"
                dataKey="avg_consensus_score"
                stroke="#3b82f6"
                name="Avg Consensus Score"
                strokeWidth={2}
                dot={{ r: 3 }}
              />
              <Line
                yAxisId="right"
                type="monotone"
                dataKey="avg_hallucination_score"
                stroke="#ef4444"
                name="Avg Hallucination Score"
                strokeWidth={2}
                dot={{ r: 3 }}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Judge Performance */}
      {Object.keys(statistics.judge_performance).length > 0 && (
        <div className="bg-white p-6 rounded-lg shadow">
          <h4 className="text-lg font-semibold text-gray-900 mb-4">
            Judge Performance Over Time
          </h4>
          <div className="space-y-4">
            {Object.entries(statistics.judge_performance).map(([judgeName, perf]) => (
              <JudgePerformanceCard
                key={judgeName}
                judgeName={judgeName}
                performance={perf}
              />
            ))}
          </div>
        </div>
      )}

      {/* Detailed Statistics */}
      {statistics.consensus_score_stats && (
        <div className="bg-white p-6 rounded-lg shadow">
          <h4 className="text-lg font-semibold text-gray-900 mb-4">
            Detailed Statistics
          </h4>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {statistics.consensus_score_stats && (
              <StatisticsTable
                title="Consensus Score Distribution"
                stats={statistics.consensus_score_stats}
              />
            )}
            {statistics.hallucination_score_stats && (
              <StatisticsTable
                title="Hallucination Score Distribution"
                stats={statistics.hallucination_score_stats}
              />
            )}
            {statistics.variance_stats && (
              <StatisticsTable
                title="Variance Distribution"
                stats={statistics.variance_stats}
              />
            )}
            {statistics.std_dev_stats && (
              <StatisticsTable
                title="Standard Deviation Distribution"
                stats={statistics.std_dev_stats}
              />
            )}
          </div>
        </div>
      )}
    </div>
  );
};

interface SummaryCardProps {
  label: string;
  value: string | number;
  subtitle?: string;
  icon?: string;
}

const SummaryCard: React.FC<SummaryCardProps> = ({ label, value, subtitle, icon }) => (
  <div className="p-4 border border-gray-200 rounded-lg">
    <div className="flex items-center justify-between mb-2">
      <p className="text-sm font-medium text-gray-600">{label}</p>
      {icon && <span className="text-2xl">{icon}</span>}
    </div>
    <p className="text-2xl font-bold text-gray-900">{value}</p>
    {subtitle && <p className="text-xs text-gray-500 mt-1">{subtitle}</p>}
  </div>
);

interface JudgePerformanceCardProps {
  judgeName: string;
  performance: {
    total_evaluations: number;
    score_stats: any;
    confidence_stats: any;
    response_time_stats?: any;
  };
}

const JudgePerformanceCard: React.FC<JudgePerformanceCardProps> = ({
  judgeName,
  performance,
}) => (
  <div className="border border-gray-200 rounded-lg p-4">
    <div className="flex items-center justify-between mb-3">
      <h5 className="font-semibold text-gray-900">{judgeName}</h5>
      <span className="text-sm text-gray-600">
        {performance.total_evaluations} evaluations
      </span>
    </div>
    <div className="grid grid-cols-3 gap-4 text-sm">
      <div>
        <p className="text-gray-600">Avg Score</p>
        <p className="font-semibold text-gray-900">
          {performance.score_stats.mean.toFixed(2)}
        </p>
        <p className="text-xs text-gray-500">
          ±{performance.score_stats.std_dev.toFixed(2)}
        </p>
      </div>
      <div>
        <p className="text-gray-600">Avg Confidence</p>
        <p className="font-semibold text-gray-900">
          {(performance.confidence_stats.mean * 100).toFixed(1)}%
        </p>
        <p className="text-xs text-gray-500">
          Range: {(performance.confidence_stats.min * 100).toFixed(0)}%-
          {(performance.confidence_stats.max * 100).toFixed(0)}%
        </p>
      </div>
      {performance.response_time_stats && (
        <div>
          <p className="text-gray-600">Avg Response Time</p>
          <p className="font-semibold text-gray-900">
            {(performance.response_time_stats.mean / 1000).toFixed(2)}s
          </p>
          <p className="text-xs text-gray-500">
            Min: {(performance.response_time_stats.min / 1000).toFixed(2)}s
          </p>
        </div>
      )}
    </div>
  </div>
);

interface StatisticsTableProps {
  title: string;
  stats: any;
}

const StatisticsTable: React.FC<StatisticsTableProps> = ({ title, stats }) => (
  <div>
    <h5 className="font-semibold text-gray-900 mb-3">{title}</h5>
    <table className="w-full text-sm">
      <tbody className="divide-y divide-gray-200">
        <tr>
          <td className="py-2 text-gray-600">Count</td>
          <td className="py-2 text-right font-medium text-gray-900">{stats.count}</td>
        </tr>
        <tr>
          <td className="py-2 text-gray-600">Mean</td>
          <td className="py-2 text-right font-medium text-gray-900">
            {stats.mean.toFixed(2)}
          </td>
        </tr>
        <tr>
          <td className="py-2 text-gray-600">Median</td>
          <td className="py-2 text-right font-medium text-gray-900">
            {stats.median.toFixed(2)}
          </td>
        </tr>
        <tr>
          <td className="py-2 text-gray-600">Std Dev</td>
          <td className="py-2 text-right font-medium text-gray-900">
            {stats.std_dev.toFixed(2)}
          </td>
        </tr>
        <tr>
          <td className="py-2 text-gray-600">Min / Max</td>
          <td className="py-2 text-right font-medium text-gray-900">
            {stats.min.toFixed(2)} / {stats.max.toFixed(2)}
          </td>
        </tr>
        <tr>
          <td className="py-2 text-gray-600">Q1 / Q3</td>
          <td className="py-2 text-right font-medium text-gray-900">
            {stats.q1.toFixed(2)} / {stats.q3.toFixed(2)}
          </td>
        </tr>
      </tbody>
    </table>
  </div>
);
