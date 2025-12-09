# Visualization Components Integration Guide

## Overview

This guide shows how to integrate the new visualization components into the existing `EvaluationResultMessage` component to provide rich, interactive visualizations of evaluation results.

## Quick Integration

### Option 1: Replace Expandable Sections with Visualizations

Update `EvaluationResultMessage.tsx` to use the `VisualizationDashboard`:

```tsx
import React, { useState } from 'react';
import { formatDistanceToNow } from 'date-fns';
import { EvaluationResults } from '../../api/types';
import { VisualizationDashboard } from '../visualizations';

interface EvaluationResultMessageProps {
  results: EvaluationResults;
  timestamp: Date;
}

const EvaluationResultMessage: React.FC<EvaluationResultMessageProps> = ({
  results,
  timestamp,
}) => {
  const [showVisualizations, setShowVisualizations] = useState(true);

  return (
    <div className="flex justify-start mb-4">
      <div className="max-w-6xl w-full">
        <div className="bg-white border border-gray-200 rounded-lg shadow-md overflow-hidden">
          {/* Header with Consensus Score */}
          <div className="bg-gradient-to-r from-purple-600 to-blue-600 text-white p-4">
            <div className="flex items-center justify-between">
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
          </div>

          {/* Toggle Button */}
          <div className="px-4 py-2 bg-gray-50 border-b border-gray-200">
            <button
              onClick={() => setShowVisualizations(!showVisualizations)}
              className="text-sm text-blue-600 hover:text-blue-800 font-medium"
            >
              {showVisualizations ? 'Hide' : 'Show'} Detailed Visualizations
            </button>
          </div>

          {/* Visualization Dashboard */}
          {showVisualizations && (
            <div className="p-4">
              <VisualizationDashboard
                evaluationResults={results}
                showAdvancedMetrics={true}
              />
            </div>
          )}
        </div>
        <div className="text-xs text-gray-500 mt-1">
          {formatDistanceToNow(timestamp, { addSuffix: true })}
        </div>
      </div>
    </div>
  );
};

export default EvaluationResultMessage;
```

### Option 2: Add Visualizations as Additional Tab

Create a tabbed interface with "Summary" and "Visualizations" tabs:

```tsx
import React, { useState } from 'react';
import { formatDistanceToNow } from 'date-fns';
import { EvaluationResults } from '../../api/types';
import { VisualizationDashboard } from '../visualizations';

type TabType = 'summary' | 'visualizations';

const EvaluationResultMessage: React.FC<EvaluationResultMessageProps> = ({
  results,
  timestamp,
}) => {
  const [activeTab, setActiveTab] = useState<TabType>('summary');

  return (
    <div className="flex justify-start mb-4">
      <div className="max-w-6xl w-full">
        <div className="bg-white border border-gray-200 rounded-lg shadow-md overflow-hidden">
          {/* Header */}
          <div className="bg-gradient-to-r from-purple-600 to-blue-600 text-white p-4">
            {/* ... header content ... */}
          </div>

          {/* Tabs */}
          <div className="flex border-b border-gray-200">
            <button
              onClick={() => setActiveTab('summary')}
              className={`px-6 py-3 font-medium text-sm ${
                activeTab === 'summary'
                  ? 'border-b-2 border-blue-600 text-blue-600'
                  : 'text-gray-600 hover:text-gray-900'
              }`}
            >
              Summary
            </button>
            <button
              onClick={() => setActiveTab('visualizations')}
              className={`px-6 py-3 font-medium text-sm ${
                activeTab === 'visualizations'
                  ? 'border-b-2 border-blue-600 text-blue-600'
                  : 'text-gray-600 hover:text-gray-900'
              }`}
            >
              Visualizations
            </button>
          </div>

          {/* Content */}
          <div className="p-4">
            {activeTab === 'summary' ? (
              <div>
                {/* Existing summary content */}
              </div>
            ) : (
              <VisualizationDashboard
                evaluationResults={results}
                showAdvancedMetrics={true}
              />
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
```

### Option 3: Use Individual Components in Expandable Sections

Replace the text-based expandable sections with visualization components:

```tsx
import React, { useState } from 'react';
import { formatDistanceToNow } from 'date-fns';
import { EvaluationResults } from '../../api/types';
import {
  JudgeComparisonChart,
  ConfidenceGauge,
  HallucinationThermometer,
  HallucinationBreakdownChart,
  ScoreDistributionChart,
  InterJudgeAgreementHeatmap,
} from '../visualizations';

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

  return (
    <div className="flex justify-start mb-4">
      <div className="max-w-6xl w-full">
        <div className="bg-white border border-gray-200 rounded-lg shadow-md overflow-hidden">
          {/* Header */}
          {/* ... existing header ... */}

          {/* Expandable Sections with Visualizations */}
          <div className="divide-y divide-gray-200">
            {/* Judge Results with Chart */}
            <div>
              <button
                onClick={() => toggleSection('judges')}
                className="w-full px-4 py-3 flex items-center justify-between hover:bg-gray-50 transition-colors"
              >
                <div className="flex items-center gap-2">
                  <svg className={/* ... */}>
                    {/* ... chevron icon ... */}
                  </svg>
                  <span className="font-medium text-gray-900">
                    Judge Comparison
                  </span>
                </div>
              </button>
              {expandedSections.judges && (
                <div className="px-4 pb-4">
                  <JudgeComparisonChart
                    judgeResults={results.judge_results}
                  />
                </div>
              )}
            </div>

            {/* Confidence with Gauge */}
            <div>
              <button
                onClick={() => toggleSection('confidence')}
                className="w-full px-4 py-3 flex items-center justify-between hover:bg-gray-50 transition-colors"
              >
                <div className="flex items-center gap-2">
                  <svg className={/* ... */}>
                    {/* ... chevron icon ... */}
                  </svg>
                  <span className="font-medium text-gray-900">
                    Confidence Metrics
                  </span>
                </div>
              </button>
              {expandedSections.confidence && (
                <div className="px-4 pb-4">
                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                    <ConfidenceGauge
                      confidenceMetrics={results.confidence_metrics}
                    />
                    {results.judge_results.length >= 2 && (
                      <InterJudgeAgreementHeatmap
                        interJudgeAgreement={results.inter_judge_agreement}
                      />
                    )}
                  </div>
                </div>
              )}
            </div>

            {/* Hallucination with Visualizations */}
            <div>
              <button
                onClick={() => toggleSection('hallucination')}
                className="w-full px-4 py-3 flex items-center justify-between hover:bg-gray-50 transition-colors"
              >
                <div className="flex items-center gap-2">
                  <svg className={/* ... */}>
                    {/* ... chevron icon ... */}
                  </svg>
                  <span className="font-medium text-gray-900">
                    Hallucination Analysis
                  </span>
                </div>
              </button>
              {expandedSections.hallucination && (
                <div className="px-4 pb-4">
                  <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
                    <HallucinationThermometer
                      hallucinationMetrics={results.hallucination_metrics}
                    />
                    <div className="lg:col-span-2">
                      <HallucinationBreakdownChart
                        hallucinationMetrics={results.hallucination_metrics}
                        flaggedIssues={results.judge_results.flatMap(
                          (jr) => jr.flagged_issues
                        )}
                      />
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* Statistics with Distribution */}
            <div>
              <button
                onClick={() => toggleSection('statistics')}
                className="w-full px-4 py-3 flex items-center justify-between hover:bg-gray-50 transition-colors"
              >
                <div className="flex items-center gap-2">
                  <svg className={/* ... */}>
                    {/* ... chevron icon ... */}
                  </svg>
                  <span className="font-medium text-gray-900">
                    Statistical Analysis
                  </span>
                </div>
              </button>
              {expandedSections.statistics && (
                <div className="px-4 pb-4">
                  <ScoreDistributionChart
                    judgeResults={results.judge_results}
                  />
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
```

## Responsive Considerations

The visualization components are designed to be responsive, but you may want to adjust the layout for mobile devices:

```tsx
// Mobile-friendly layout
<div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-4">
  <div className="lg:col-span-2">
    <JudgeComparisonChart judgeResults={results.judge_results} />
  </div>
  <div>
    <ConfidenceGauge confidenceMetrics={results.confidence_metrics} />
  </div>
</div>
```

## Performance Tips

1. **Lazy Loading**: Load visualizations only when expanded
2. **Memoization**: Use React.memo for expensive components
3. **Debouncing**: Debounce interactive features like hover tooltips

```tsx
import React, { memo } from 'react';

const MemoizedVisualizationDashboard = memo(VisualizationDashboard);

// Use in component
<MemoizedVisualizationDashboard
  evaluationResults={results}
  showAdvancedMetrics={true}
/>
```

## Export Functionality

Add export buttons to allow users to save visualizations:

```tsx
const handleExportPNG = async () => {
  // Use html2canvas or similar library
  const element = document.getElementById('visualization-dashboard');
  const canvas = await html2canvas(element);
  const link = document.createElement('a');
  link.download = `evaluation-${results.session_id}.png`;
  link.href = canvas.toDataURL();
  link.click();
};

// Add export button
<button
  onClick={handleExportPNG}
  className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
>
  Export as PNG
</button>
```

## Next Steps

1. Choose an integration approach (Option 1, 2, or 3)
2. Update `EvaluationResultMessage.tsx`
3. Test with real evaluation data
4. Add export functionality if needed
5. Optimize for performance
6. Add unit tests for the integrated components

## Example with Historical Data

If you have historical evaluation data, you can pass it to the dashboard:

```tsx
import { useHistoricalScores } from '../../hooks/useHistoricalScores';

const EvaluationResultMessage: React.FC<EvaluationResultMessageProps> = ({
  results,
  timestamp,
}) => {
  const { data: historicalScores } = useHistoricalScores();

  return (
    <VisualizationDashboard
      evaluationResults={results}
      showAdvancedMetrics={true}
      historicalScores={historicalScores}
    />
  );
};
```

## Conclusion

The visualization components are ready to be integrated into the existing chat interface. Choose the approach that best fits your UX requirements and user workflow.
