import React, { useEffect, useRef } from 'react';
import * as d3 from 'd3';
import { ConfidenceMetrics } from '../../api/types';

interface ConfidenceGaugeProps {
  confidenceMetrics: ConfidenceMetrics;
}

export const ConfidenceGauge: React.FC<ConfidenceGaugeProps> = ({
  confidenceMetrics,
}) => {
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!svgRef.current) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const width = 300;
    const height = 200;
    const radius = Math.min(width, height) / 2 - 20;

    const g = svg
      .attr('width', width)
      .attr('height', height)
      .append('g')
      .attr('transform', `translate(${width / 2}, ${height - 20})`);

    // Create arc generator
    const arc = d3
      .arc()
      .innerRadius(radius - 30)
      .outerRadius(radius)
      .startAngle(-Math.PI / 2)
      .cornerRadius(5);

    // Background arc
    g.append('path')
      .datum({ endAngle: Math.PI / 2 })
      .style('fill', '#e5e7eb')
      .attr('d', arc as any);

    // Color scale for gradient
    const colorScale = d3
      .scaleLinear<string>()
      .domain([0, 0.5, 1])
      .range(['#ef4444', '#f59e0b', '#10b981']);

    // Create gradient
    const defs = svg.append('defs');
    const gradient = defs
      .append('linearGradient')
      .attr('id', 'confidence-gradient')
      .attr('x1', '0%')
      .attr('y1', '0%')
      .attr('x2', '100%')
      .attr('y2', '0%');

    gradient
      .append('stop')
      .attr('offset', '0%')
      .attr('stop-color', '#ef4444');

    gradient
      .append('stop')
      .attr('offset', '50%')
      .attr('stop-color', '#f59e0b');

    gradient
      .append('stop')
      .attr('offset', '100%')
      .attr('stop-color', '#10b981');

    // Animated confidence arc
    const confidenceArc = g
      .append('path')
      .datum({ endAngle: -Math.PI / 2 })
      .style('fill', 'url(#confidence-gradient)')
      .attr('d', arc as any);

    // Animate the arc
    confidenceArc
      .transition()
      .duration(1500)
      .ease(d3.easeElasticOut)
      .attrTween('d', function (d: any) {
        const interpolate = d3.interpolate(
          d.endAngle,
          -Math.PI / 2 + Math.PI * confidenceMetrics.mean_confidence
        );
        return function (t) {
          d.endAngle = interpolate(t);
          return arc(d as any) || '';
        };
      });

    // Threshold markers
    const thresholds = [
      { value: 0.7, label: '70%' },
      { value: 0.9, label: '90%' },
    ];

    thresholds.forEach((threshold) => {
      const angle = -Math.PI / 2 + Math.PI * threshold.value;
      const x1 = Math.cos(angle) * (radius - 30);
      const y1 = Math.sin(angle) * (radius - 30);
      const x2 = Math.cos(angle) * (radius + 5);
      const y2 = Math.sin(angle) * (radius + 5);

      g.append('line')
        .attr('x1', x1)
        .attr('y1', y1)
        .attr('x2', x2)
        .attr('y2', y2)
        .attr('stroke', '#6b7280')
        .attr('stroke-width', 2);

      g.append('text')
        .attr('x', Math.cos(angle) * (radius + 15))
        .attr('y', Math.sin(angle) * (radius + 15))
        .attr('text-anchor', 'middle')
        .attr('dominant-baseline', 'middle')
        .attr('font-size', '10px')
        .attr('fill', '#6b7280')
        .text(threshold.label);
    });

    // Needle
    const needleLength = radius - 15;
    const needleAngle =
      -Math.PI / 2 + Math.PI * confidenceMetrics.mean_confidence;

    const needleGroup = g.append('g').attr('class', 'needle');

    // Needle line
    needleGroup
      .append('line')
      .attr('x1', 0)
      .attr('y1', 0)
      .attr('x2', 0)
      .attr('y2', -needleLength)
      .attr('stroke', '#1f2937')
      .attr('stroke-width', 3)
      .attr('stroke-linecap', 'round');

    // Needle circle
    needleGroup
      .append('circle')
      .attr('cx', 0)
      .attr('cy', 0)
      .attr('r', 8)
      .attr('fill', '#1f2937');

    // Animate needle
    needleGroup
      .transition()
      .duration(1500)
      .ease(d3.easeElasticOut)
      .attr('transform', `rotate(${(needleAngle * 180) / Math.PI})`);

    // Center text
    g.append('text')
      .attr('text-anchor', 'middle')
      .attr('y', -10)
      .attr('font-size', '24px')
      .attr('font-weight', 'bold')
      .attr('fill', colorScale(confidenceMetrics.mean_confidence))
      .text(`${(confidenceMetrics.mean_confidence * 100).toFixed(1)}%`);

    g.append('text')
      .attr('text-anchor', 'middle')
      .attr('y', 10)
      .attr('font-size', '12px')
      .attr('fill', '#6b7280')
      .text('Confidence');
  }, [confidenceMetrics]);

  const getConfidenceLabel = (confidence: number): string => {
    if (confidence >= 0.9) return 'Very High';
    if (confidence >= 0.7) return 'High';
    if (confidence >= 0.5) return 'Moderate';
    if (confidence >= 0.3) return 'Low';
    return 'Very Low';
  };

  const getConfidenceColor = (confidence: number): string => {
    if (confidence >= 0.9) return 'text-green-600';
    if (confidence >= 0.7) return 'text-green-500';
    if (confidence >= 0.5) return 'text-yellow-500';
    if (confidence >= 0.3) return 'text-orange-500';
    return 'text-red-500';
  };

  return (
    <div className="bg-white p-4 rounded-lg shadow">
      <h3 className="text-lg font-semibold text-gray-900 mb-2">
        Confidence Level
      </h3>
      <div className="flex flex-col items-center">
        <svg ref={svgRef}></svg>
        <div className="mt-4 w-full space-y-2">
          <div className="flex items-center justify-between text-sm">
            <span className="text-gray-600">Confidence Level:</span>
            <span
              className={`font-semibold ${getConfidenceColor(
                confidenceMetrics.mean_confidence
              )}`}
            >
              {getConfidenceLabel(confidenceMetrics.mean_confidence)}
            </span>
          </div>
          <div className="flex items-center justify-between text-sm">
            <span className="text-gray-600">Confidence Interval:</span>
            <span className="font-medium text-gray-900">
              [{confidenceMetrics.confidence_interval[0].toFixed(1)},{' '}
              {confidenceMetrics.confidence_interval[1].toFixed(1)}]
            </span>
          </div>
          {confidenceMetrics.is_low_confidence && (
            <div className="mt-2 p-2 bg-yellow-50 border border-yellow-200 rounded">
              <div className="flex items-start">
                <svg
                  className="w-5 h-5 text-yellow-600 mt-0.5 mr-2 flex-shrink-0"
                  fill="currentColor"
                  viewBox="0 0 20 20"
                >
                  <path
                    fillRule="evenodd"
                    d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z"
                    clipRule="evenodd"
                  />
                </svg>
                <div>
                  <p className="text-sm font-medium text-yellow-800">
                    Low Confidence Warning
                  </p>
                  <p className="text-xs text-yellow-700 mt-1">
                    The evaluation has low confidence. Consider using additional
                    judges or retrieval to improve reliability.
                  </p>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
