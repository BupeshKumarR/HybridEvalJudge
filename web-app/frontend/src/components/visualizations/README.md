# Visualization Components

This directory contains all visualization components for the LLM Judge Auditor web application. These components provide rich, interactive visualizations of evaluation results, judge decisions, confidence metrics, and hallucination detection.

## Components

### 1. JudgeComparisonChart

**Purpose:** Displays individual judge scores in a horizontal bar chart with confidence error bars.

**Features:**
- Color-coded bars (green/yellow/red) based on score ranges
- Confidence error bars showing uncertainty
- Interactive tooltips with detailed reasoning
- Click to expand full judge details
- Displays flagged issues for each judge

**Usage:**
```tsx
import { JudgeComparisonChart } from './components/visualizations';

<JudgeComparisonChart judgeResults={evaluationResults.judge_results} />
```

**Requirements Validated:** 3.1, 3.2, 3.3, 3.4, 3.5

---

### 2. ConfidenceGauge

**Purpose:** Displays confidence level as a radial gauge with animated needle.

**Features:**
- D3.js-powered radial gauge visualization
- Color gradient from red (low) to green (high)
- Animated needle with elastic easing
- Threshold markers at 70% and 90%
- Confidence interval display
- Low confidence warning indicator

**Usage:**
```tsx
import { ConfidenceGauge } from './components/visualizations';

<ConfidenceGauge confidenceMetrics={evaluationResults.confidence_metrics} />
```

**Requirements Validated:** 4.1, 4.2, 4.5

---

### 3. HallucinationThermometer

**Purpose:** Displays hallucination score as a vertical thermometer with color gradient.

**Features:**
- Animated fill with smooth transitions
- Color gradient from green (low) to red (high)
- Severity distribution breakdown
- Affected text spans listing
- Hover to show breakdown pie chart
- Scale markers at 0, 25, 50, 75, 100

**Usage:**
```tsx
import { HallucinationThermometer } from './components/visualizations';

<HallucinationThermometer 
  hallucinationMetrics={evaluationResults.hallucination_metrics} 
/>
```

**Requirements Validated:** 5.1, 5.3, 5.5

---

### 4. ScoreDistributionChart

**Purpose:** Displays score distribution using box plot and histogram.

**Features:**
- Box plot with min, Q1, median, Q3, max
- Outlier detection using IQR method
- Histogram with configurable bin sizes
- Historical comparison (optional)
- Mean reference line
- Statistical summary (variance, std dev)

**Usage:**
```tsx
import { ScoreDistributionChart } from './components/visualizations';

<ScoreDistributionChart 
  judgeResults={evaluationResults.judge_results}
  historicalScores={[85, 90, 88, 92]} // Optional
/>
```

**Requirements Validated:** 6.2, 11.4

---

### 5. InterJudgeAgreementHeatmap

**Purpose:** Displays inter-judge agreement as a correlation matrix heatmap.

**Features:**
- D3.js-powered heatmap visualization
- Color intensity mapping (red to green)
- Hover tooltips with exact correlation values
- Cohen's Kappa and Fleiss' Kappa metrics
- Interpretation labels (poor, fair, moderate, substantial, perfect)
- Color legend with gradient scale

**Usage:**
```tsx
import { InterJudgeAgreementHeatmap } from './components/visualizations';

<InterJudgeAgreementHeatmap 
  interJudgeAgreement={evaluationResults.inter_judge_agreement} 
/>
```

**Requirements Validated:** 4.3, 6.4, 11.1, 11.5

---

### 6. HallucinationBreakdownChart

**Purpose:** Displays hallucination breakdown by type and severity.

**Features:**
- Toggle between pie chart and bar chart views
- Color-coded by issue type
- Severity distribution (critical, high, medium, low)
- Interactive legend with filtering
- Drill-down to detailed issues
- Summary statistics

**Usage:**
```tsx
import { HallucinationBreakdownChart } from './components/visualizations';

<HallucinationBreakdownChart 
  hallucinationMetrics={evaluationResults.hallucination_metrics}
  flaggedIssues={allFlaggedIssues} // Optional for drill-down
/>
```

**Requirements Validated:** 5.5, 11.3

---

### 7. VisualizationDashboard

**Purpose:** Comprehensive dashboard that combines all visualization components.

**Features:**
- Organized layout with key metrics cards
- All visualization components in logical grouping
- Optional advanced metrics section
- Verifier verdicts display
- Responsive grid layout

**Usage:**
```tsx
import { VisualizationDashboard } from './components/visualizations';

<VisualizationDashboard 
  evaluationResults={evaluationResults}
  showAdvancedMetrics={true}
  historicalScores={[85, 90, 88, 92]} // Optional
/>
```

---

## Dependencies

- **recharts**: Bar charts, pie charts, line charts, and composed charts
- **d3**: Custom visualizations (gauge, heatmap)
- **React**: Component framework
- **TypeScript**: Type safety
- **TailwindCSS**: Styling

## Color Schemes

### Score Colors
- **Green (#10b981)**: High scores (80-100)
- **Yellow (#f59e0b)**: Medium scores (50-79)
- **Red (#ef4444)**: Low scores (0-49)

### Severity Colors
- **Critical**: #dc2626 (dark red)
- **High**: #f97316 (orange)
- **Medium**: #f59e0b (yellow)
- **Low**: #3b82f6 (blue)

### Issue Type Colors
- **Factual Error**: #ef4444
- **Hallucination**: #dc2626
- **Unsupported Claim**: #f97316
- **Temporal Inconsistency**: #f59e0b
- **Numerical Error**: #eab308
- **Bias**: #8b5cf6

## Accessibility

All components follow accessibility best practices:
- Semantic HTML structure
- ARIA labels where appropriate
- Keyboard navigation support
- Color contrast ratios meet WCAG AA standards
- Tooltips provide additional context

## Performance

- Components use React.memo for optimization (when needed)
- D3 visualizations clean up on unmount
- Animations use CSS transitions and D3 easing
- Large datasets are handled with pagination/virtualization

## Testing

Each component should be tested for:
- Correct rendering with valid data
- Handling of edge cases (empty data, single judge, etc.)
- Interactive features (clicks, hovers)
- Responsive behavior
- Accessibility compliance

## Future Enhancements

- Export individual charts as images
- Customizable color schemes
- Animation speed controls
- More chart types (violin plots, radar charts)
- Real-time updates for streaming data
