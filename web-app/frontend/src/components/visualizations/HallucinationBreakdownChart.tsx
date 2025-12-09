import React, { useState } from 'react';
import {
  PieChart,
  Pie,
  Cell,
  ResponsiveContainer,
  Tooltip,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
} from 'recharts';
import { HallucinationMetrics, FlaggedIssue } from '../../api/types';

interface HallucinationBreakdownChartProps {
  hallucinationMetrics: HallucinationMetrics;
  flaggedIssues?: FlaggedIssue[]; // Optional detailed issues for drill-down
}

type ViewMode = 'pie' | 'bar';

interface BreakdownData {
  name: string;
  value: number;
  color: string;
}

interface SeverityData {
  severity: string;
  count: number;
  color: string;
}

const ISSUE_TYPE_COLORS: { [key: string]: string } = {
  factual_error: '#ef4444',
  hallucination: '#dc2626',
  unsupported_claim: '#f97316',
  temporal_inconsistency: '#f59e0b',
  numerical_error: '#eab308',
  bias: '#8b5cf6',
};

const SEVERITY_COLORS: { [key: string]: string } = {
  critical: '#dc2626',
  high: '#f97316',
  medium: '#f59e0b',
  low: '#3b82f6',
};

export const HallucinationBreakdownChart: React.FC<
  HallucinationBreakdownChartProps
> = ({ hallucinationMetrics, flaggedIssues }) => {
  const [viewMode, setViewMode] = useState<ViewMode>('pie');
  const [selectedType, setSelectedType] = useState<string | null>(null);

  // Prepare breakdown data
  const breakdownData: BreakdownData[] = Object.entries(
    hallucinationMetrics.breakdown_by_type
  )
    .filter(([_, value]) => value > 0)
    .map(([type, value]) => ({
      name: type.replace(/_/g, ' ').toUpperCase(),
      value: value,
      color: ISSUE_TYPE_COLORS[type] || '#6b7280',
    }))
    .sort((a, b) => b.value - a.value);

  // Prepare severity data
  const severityData: SeverityData[] = Object.entries(
    hallucinationMetrics.severity_distribution
  )
    .filter(([_, count]) => count > 0)
    .map(([severity, count]) => ({
      severity: severity.charAt(0).toUpperCase() + severity.slice(1),
      count: count,
      color: SEVERITY_COLORS[severity] || '#6b7280',
    }))
    .sort((a, b) => {
      const order = ['Critical', 'High', 'Medium', 'Low'];
      return order.indexOf(a.severity) - order.indexOf(b.severity);
    });

  // Get issues for selected type (drill-down)
  const selectedIssues = selectedType
    ? flaggedIssues?.filter(
        (issue) =>
          issue.issue_type.replace(/_/g, ' ').toUpperCase() === selectedType
      ) || []
    : [];

  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      const data = payload[0];
      return (
        <div className="bg-white p-3 rounded-lg shadow-lg border border-gray-200">
          <p className="font-semibold text-gray-900">{data.name}</p>
          <p className="text-sm text-gray-600">
            Score: {data.value.toFixed(1)}
          </p>
        </div>
      );
    }
    return null;
  };

  const handlePieClick = (data: BreakdownData) => {
    setSelectedType(selectedType === data.name ? null : data.name);
  };

  return (
    <div className="bg-white p-4 rounded-lg shadow space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold text-gray-900">
          Hallucination Breakdown
        </h3>
        <div className="flex space-x-2">
          <button
            onClick={() => setViewMode('pie')}
            className={`px-3 py-1 rounded text-sm font-medium ${
              viewMode === 'pie'
                ? 'bg-blue-500 text-white'
                : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
            }`}
          >
            Pie Chart
          </button>
          <button
            onClick={() => setViewMode('bar')}
            className={`px-3 py-1 rounded text-sm font-medium ${
              viewMode === 'bar'
                ? 'bg-blue-500 text-white'
                : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
            }`}
          >
            Bar Chart
          </button>
        </div>
      </div>

      {breakdownData.length === 0 ? (
        <div className="text-center py-8 text-gray-500">
          No hallucination issues detected
        </div>
      ) : (
        <>
          {/* Main Chart */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {/* Type Breakdown */}
            <div>
              <h4 className="text-sm font-semibold text-gray-700 mb-2">
                By Issue Type
              </h4>
              {viewMode === 'pie' ? (
                <ResponsiveContainer width="100%" height={250}>
                  <PieChart>
                    <Pie
                      data={breakdownData}
                      cx="50%"
                      cy="50%"
                      labelLine={false}
                      label={({ name, percent }) =>
                        `${name}: ${(percent * 100).toFixed(0)}%`
                      }
                      outerRadius={80}
                      fill="#8884d8"
                      dataKey="value"
                      onClick={handlePieClick}
                      style={{ cursor: 'pointer' }}
                    >
                      {breakdownData.map((entry, index) => (
                        <Cell
                          key={`cell-${index}`}
                          fill={entry.color}
                          stroke={
                            selectedType === entry.name ? '#000' : '#fff'
                          }
                          strokeWidth={selectedType === entry.name ? 3 : 1}
                        />
                      ))}
                    </Pie>
                    <Tooltip content={<CustomTooltip />} />
                  </PieChart>
                </ResponsiveContainer>
              ) : (
                <ResponsiveContainer width="100%" height={250}>
                  <BarChart data={breakdownData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis
                      dataKey="name"
                      angle={-45}
                      textAnchor="end"
                      height={100}
                      tick={{ fontSize: 10 }}
                    />
                    <YAxis />
                    <Tooltip content={<CustomTooltip />} />
                    <Bar
                      dataKey="value"
                      onClick={(data) => handlePieClick(data)}
                      style={{ cursor: 'pointer' }}
                    >
                      {breakdownData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.color} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              )}
            </div>

            {/* Severity Breakdown */}
            <div>
              <h4 className="text-sm font-semibold text-gray-700 mb-2">
                By Severity
              </h4>
              <ResponsiveContainer width="100%" height={250}>
                <BarChart data={severityData} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis type="number" />
                  <YAxis dataKey="severity" type="category" width={80} />
                  <Tooltip />
                  <Bar dataKey="count">
                    {severityData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Interactive Legend */}
          <div className="border-t border-gray-200 pt-4">
            <h4 className="text-sm font-semibold text-gray-700 mb-2">
              Issue Types
            </h4>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-2">
              {breakdownData.map((item) => (
                <button
                  key={item.name}
                  onClick={() =>
                    setSelectedType(
                      selectedType === item.name ? null : item.name
                    )
                  }
                  className={`flex items-center p-2 rounded border transition-all ${
                    selectedType === item.name
                      ? 'border-gray-900 bg-gray-50'
                      : 'border-gray-200 hover:border-gray-400'
                  }`}
                >
                  <div
                    className="w-4 h-4 rounded mr-2 flex-shrink-0"
                    style={{ backgroundColor: item.color }}
                  ></div>
                  <div className="text-left flex-1 min-w-0">
                    <p className="text-xs font-medium text-gray-900 truncate">
                      {item.name}
                    </p>
                    <p className="text-xs text-gray-600">
                      {item.value.toFixed(1)}
                    </p>
                  </div>
                </button>
              ))}
            </div>
          </div>

          {/* Drill-down Details */}
          {selectedType && selectedIssues.length > 0 && (
            <div className="border-t border-gray-200 pt-4">
              <div className="flex items-center justify-between mb-3">
                <h4 className="text-sm font-semibold text-gray-700">
                  {selectedType} - Detailed Issues ({selectedIssues.length})
                </h4>
                <button
                  onClick={() => setSelectedType(null)}
                  className="text-sm text-gray-500 hover:text-gray-700"
                >
                  Close
                </button>
              </div>
              <div className="space-y-2 max-h-64 overflow-y-auto">
                {selectedIssues.map((issue, idx) => (
                  <div
                    key={idx}
                    className="border border-gray-200 rounded p-3 bg-gray-50"
                  >
                    <div className="flex items-center justify-between mb-1">
                      <span className="text-sm font-medium text-gray-900">
                        Issue #{idx + 1}
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
                    <p className="text-sm text-gray-700 mb-2">
                      {issue.description}
                    </p>
                    {issue.text_span_start !== undefined && issue.text_span_end !== undefined && (
                      <p className="text-xs text-gray-500">
                        Text span: [{issue.text_span_start}-{issue.text_span_end}]
                      </p>
                    )}
                    {issue.evidence && Object.keys(issue.evidence).length > 0 && (
                      <div className="mt-2">
                        <p className="text-xs font-medium text-gray-700">
                          Evidence:
                        </p>
                        <ul className="text-xs text-gray-600 list-disc list-inside">
                          {Object.values(issue.evidence).slice(0, 2).map((ev: any, evIdx) => (
                            <li key={evIdx}>{String(ev)}</li>
                          ))}
                          {Object.keys(issue.evidence).length > 2 && (
                            <li className="text-gray-500">
                              +{Object.keys(issue.evidence).length - 2} more...
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

          {/* Summary Statistics */}
          <div className="border-t border-gray-200 pt-4">
            <div className="grid grid-cols-3 gap-4">
              <div className="text-center">
                <p className="text-2xl font-bold text-gray-900">
                  {breakdownData.length}
                </p>
                <p className="text-xs text-gray-600">Issue Types</p>
              </div>
              <div className="text-center">
                <p className="text-2xl font-bold text-gray-900">
                  {Object.values(hallucinationMetrics.severity_distribution).reduce(
                    (sum, count) => sum + count,
                    0
                  )}
                </p>
                <p className="text-xs text-gray-600">Total Issues</p>
              </div>
              <div className="text-center">
                <p className="text-2xl font-bold text-gray-900">
                  {hallucinationMetrics.affected_text_spans.length}
                </p>
                <p className="text-xs text-gray-600">Affected Spans</p>
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  );
};
