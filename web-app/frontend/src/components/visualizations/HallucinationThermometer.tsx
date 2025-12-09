import React, { useState, useEffect } from 'react';
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip } from 'recharts';
import { HallucinationMetrics } from '../../api/types';

interface HallucinationThermometerProps {
  hallucinationMetrics: HallucinationMetrics;
}

export const HallucinationThermometer: React.FC<
  HallucinationThermometerProps
> = React.memo(({ hallucinationMetrics }) => {
  const [fillHeight, setFillHeight] = useState(0);
  const [showBreakdown, setShowBreakdown] = useState(false);

  useEffect(() => {
    // Animate fill on mount
    const timer = setTimeout(() => {
      setFillHeight(hallucinationMetrics.overall_score);
    }, 100);
    return () => clearTimeout(timer);
  }, [hallucinationMetrics.overall_score]);

  const getColor = (score: number): string => {
    if (score >= 75) return '#ef4444'; // red
    if (score >= 50) return '#f97316'; // orange
    if (score >= 25) return '#f59e0b'; // yellow
    return '#10b981'; // green
  };

  const getGradientId = 'hallucination-gradient';

  const getSeverityLabel = (score: number): string => {
    if (score >= 75) return 'Critical';
    if (score >= 50) return 'High';
    if (score >= 25) return 'Moderate';
    return 'Low';
  };

  // Prepare breakdown data for pie chart
  const breakdownData = Object.entries(
    hallucinationMetrics.breakdown_by_type
  )
    .filter(([_, value]) => value > 0)
    .map(([type, value]) => ({
      name: type.replace(/_/g, ' ').toUpperCase(),
      value: value,
    }));

  const COLORS = [
    '#ef4444',
    '#f97316',
    '#f59e0b',
    '#84cc16',
    '#06b6d4',
    '#8b5cf6',
  ];

  return (
    <div className="bg-white p-4 rounded-lg shadow">
      <h3 className="text-lg font-semibold text-gray-900 mb-4">
        Hallucination Score
      </h3>
      <div className="flex items-center justify-center space-x-8">
        {/* Thermometer */}
        <div
          className="relative"
          onMouseEnter={() => setShowBreakdown(true)}
          onMouseLeave={() => setShowBreakdown(false)}
        >
          <svg width="80" height="250" viewBox="0 0 80 250">
            {/* Define gradient */}
            <defs>
              <linearGradient
                id={getGradientId}
                x1="0%"
                y1="100%"
                x2="0%"
                y2="0%"
              >
                <stop offset="0%" stopColor="#10b981" />
                <stop offset="25%" stopColor="#f59e0b" />
                <stop offset="50%" stopColor="#f97316" />
                <stop offset="100%" stopColor="#ef4444" />
              </linearGradient>
            </defs>

            {/* Thermometer tube background */}
            <rect
              x="25"
              y="20"
              width="30"
              height="180"
              rx="15"
              fill="#e5e7eb"
              stroke="#9ca3af"
              strokeWidth="2"
            />

            {/* Thermometer bulb background */}
            <circle cx="40" cy="220" r="25" fill="#e5e7eb" stroke="#9ca3af" strokeWidth="2" />

            {/* Animated fill */}
            <rect
              x="25"
              y={20 + 180 * (1 - fillHeight / 100)}
              width="30"
              height={180 * (fillHeight / 100)}
              rx="15"
              fill={`url(#${getGradientId})`}
              style={{
                transition: 'all 1.5s ease-out',
              }}
            />

            {/* Bulb fill */}
            <circle
              cx="40"
              cy="220"
              r="25"
              fill={getColor(hallucinationMetrics.overall_score)}
              style={{
                transition: 'fill 1.5s ease-out',
              }}
            />

            {/* Scale marks */}
            {[0, 25, 50, 75, 100].map((mark) => {
              const y = 20 + 180 * (1 - mark / 100);
              return (
                <g key={mark}>
                  <line
                    x1="55"
                    y1={y}
                    x2="65"
                    y2={y}
                    stroke="#6b7280"
                    strokeWidth="2"
                  />
                  <text
                    x="70"
                    y={y + 4}
                    fontSize="10"
                    fill="#6b7280"
                    textAnchor="start"
                  >
                    {mark}
                  </text>
                </g>
              );
            })}
          </svg>

          {/* Score display */}
          <div className="text-center mt-2">
            <p className="text-2xl font-bold" style={{ color: getColor(hallucinationMetrics.overall_score) }}>
              {hallucinationMetrics.overall_score.toFixed(1)}
            </p>
            <p className="text-sm text-gray-600">
              {getSeverityLabel(hallucinationMetrics.overall_score)}
            </p>
          </div>
        </div>

        {/* Breakdown Pie Chart (shown on hover) */}
        {showBreakdown && breakdownData.length > 0 && (
          <div className="absolute left-full ml-4 bg-white p-4 rounded-lg shadow-lg border border-gray-200 z-10">
            <h4 className="text-sm font-semibold text-gray-900 mb-2">
              Breakdown by Type
            </h4>
            <ResponsiveContainer width={200} height={200}>
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
                >
                  {breakdownData.map((_, index) => (
                    <Cell
                      key={`cell-${index}`}
                      fill={COLORS[index % COLORS.length]}
                    />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>

      {/* Severity Distribution */}
      {Object.keys(hallucinationMetrics.severity_distribution).length > 0 && (
        <div className="mt-4 pt-4 border-t border-gray-200">
          <h4 className="text-sm font-semibold text-gray-900 mb-2">
            Severity Distribution
          </h4>
          <div className="grid grid-cols-2 gap-2">
            {Object.entries(hallucinationMetrics.severity_distribution)
              .filter(([_, count]) => count > 0)
              .map(([severity, count]) => (
                <div
                  key={severity}
                  className="flex items-center justify-between p-2 bg-gray-50 rounded"
                >
                  <span className="text-sm text-gray-700 capitalize">
                    {severity}
                  </span>
                  <span
                    className={`text-sm font-semibold px-2 py-1 rounded ${
                      severity === 'critical'
                        ? 'bg-red-100 text-red-800'
                        : severity === 'high'
                        ? 'bg-orange-100 text-orange-800'
                        : severity === 'medium'
                        ? 'bg-yellow-100 text-yellow-800'
                        : 'bg-blue-100 text-blue-800'
                    }`}
                  >
                    {count}
                  </span>
                </div>
              ))}
          </div>
        </div>
      )}

      {/* Affected Text Spans */}
      {hallucinationMetrics.affected_text_spans.length > 0 && (
        <div className="mt-4 pt-4 border-t border-gray-200">
          <h4 className="text-sm font-semibold text-gray-900 mb-2">
            Affected Text Spans ({hallucinationMetrics.affected_text_spans.length})
          </h4>
          <div className="space-y-1 max-h-32 overflow-y-auto">
            {hallucinationMetrics.affected_text_spans.map((span, idx) => (
              <div
                key={idx}
                className="text-xs text-gray-600 flex items-center"
              >
                <span className="font-mono bg-gray-100 px-2 py-1 rounded mr-2">
                  [{span[0]}-{span[1]}]
                </span>
                <span className="text-gray-500">
                  {span[2].replace(/_/g, ' ')}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Hover hint */}
      <p className="text-xs text-gray-500 text-center mt-4">
        Hover over thermometer to see breakdown by type
      </p>
    </div>
  );
});
