import React, { useEffect, useRef, useState } from 'react';
import * as d3 from 'd3';
import { InterJudgeAgreement } from '../../api/types';

interface InterJudgeAgreementHeatmapProps {
  interJudgeAgreement: InterJudgeAgreement;
}

interface TooltipData {
  judge1: string;
  judge2: string;
  correlation: number;
  x: number;
  y: number;
}

export const InterJudgeAgreementHeatmap: React.FC<
  InterJudgeAgreementHeatmapProps
> = ({ interJudgeAgreement }) => {
  const svgRef = useRef<SVGSVGElement>(null);
  const [tooltip, setTooltip] = useState<TooltipData | null>(null);

  useEffect(() => {
    if (!svgRef.current) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const judges = Object.keys(interJudgeAgreement.pairwise_correlations);
    if (judges.length === 0) return;

    const margin = { top: 80, right: 20, bottom: 20, left: 80 };
    const cellSize = 60;
    const width = judges.length * cellSize + margin.left + margin.right;
    const height = judges.length * cellSize + margin.top + margin.bottom;

    svg.attr('width', width).attr('height', height);

    const g = svg
      .append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    // Color scale
    const colorScale = d3
      .scaleSequential(d3.interpolateRdYlGn)
      .domain([0, 1]);

    // Create cells
    judges.forEach((judge1, i) => {
      judges.forEach((judge2, j) => {
        const correlation =
          i === j
            ? 1.0
            : interJudgeAgreement.pairwise_correlations[judge1]?.[judge2] || 0;

        g
          .append('rect')
          .attr('x', j * cellSize)
          .attr('y', i * cellSize)
          .attr('width', cellSize)
          .attr('height', cellSize)
          .attr('fill', colorScale(correlation))
          .attr('stroke', '#fff')
          .attr('stroke-width', 2)
          .style('cursor', 'pointer')
          .on('mouseenter', function (event) {
            d3.select(this).attr('stroke', '#000').attr('stroke-width', 3);
            
            const rect = (event.target as SVGRectElement).getBoundingClientRect();
            setTooltip({
              judge1,
              judge2,
              correlation,
              x: rect.left + rect.width / 2,
              y: rect.top,
            });
          })
          .on('mouseleave', function () {
            d3.select(this).attr('stroke', '#fff').attr('stroke-width', 2);
            setTooltip(null);
          });

        // Add correlation value text
        g.append('text')
          .attr('x', j * cellSize + cellSize / 2)
          .attr('y', i * cellSize + cellSize / 2)
          .attr('text-anchor', 'middle')
          .attr('dominant-baseline', 'middle')
          .attr('font-size', '12px')
          .attr('font-weight', 'bold')
          .attr('fill', correlation > 0.5 ? '#000' : '#fff')
          .attr('pointer-events', 'none')
          .text(correlation.toFixed(2));
      });
    });

    // Add row labels
    judges.forEach((judge, i) => {
      g.append('text')
        .attr('x', -10)
        .attr('y', i * cellSize + cellSize / 2)
        .attr('text-anchor', 'end')
        .attr('dominant-baseline', 'middle')
        .attr('font-size', '12px')
        .attr('fill', '#374151')
        .text(judge);
    });

    // Add column labels
    judges.forEach((judge, j) => {
      g.append('text')
        .attr('x', j * cellSize + cellSize / 2)
        .attr('y', -10)
        .attr('text-anchor', 'middle')
        .attr('dominant-baseline', 'middle')
        .attr('font-size', '12px')
        .attr('fill', '#374151')
        .attr('transform', `rotate(-45, ${j * cellSize + cellSize / 2}, -10)`)
        .text(judge);
    });

    // Add color legend
    const legendWidth = 200;
    const legendHeight = 20;
    const legendX = (judges.length * cellSize - legendWidth) / 2;
    const legendY = judges.length * cellSize + 40;

    const legendScale = d3
      .scaleLinear()
      .domain([0, 1])
      .range([0, legendWidth]);

    const legendAxis = d3
      .axisBottom(legendScale)
      .ticks(5)
      .tickFormat((d) => (d as number).toFixed(1));

    // Create gradient for legend
    const defs = svg.append('defs');
    const gradient = defs
      .append('linearGradient')
      .attr('id', 'legend-gradient')
      .attr('x1', '0%')
      .attr('x2', '100%')
      .attr('y1', '0%')
      .attr('y2', '0%');

    const numStops = 10;
    for (let i = 0; i <= numStops; i++) {
      gradient
        .append('stop')
        .attr('offset', `${(i / numStops) * 100}%`)
        .attr('stop-color', colorScale(i / numStops));
    }

    g.append('rect')
      .attr('x', legendX)
      .attr('y', legendY)
      .attr('width', legendWidth)
      .attr('height', legendHeight)
      .style('fill', 'url(#legend-gradient)');

    g.append('g')
      .attr('transform', `translate(${legendX}, ${legendY + legendHeight})`)
      .call(legendAxis);

    g.append('text')
      .attr('x', legendX + legendWidth / 2)
      .attr('y', legendY - 10)
      .attr('text-anchor', 'middle')
      .attr('font-size', '12px')
      .attr('fill', '#374151')
      .text('Correlation Coefficient');
  }, [interJudgeAgreement]);

  const getInterpretationColor = (interpretation: string): string => {
    const colors: { [key: string]: string } = {
      almost_perfect: 'bg-green-100 text-green-800',
      substantial: 'bg-green-100 text-green-700',
      moderate: 'bg-yellow-100 text-yellow-800',
      fair: 'bg-orange-100 text-orange-800',
      slight: 'bg-red-100 text-red-700',
      poor: 'bg-red-100 text-red-800',
      insufficient_judges: 'bg-gray-100 text-gray-800',
      insufficient_data: 'bg-gray-100 text-gray-800',
    };
    return colors[interpretation] || 'bg-gray-100 text-gray-800';
  };

  const getInterpretationLabel = (interpretation: string): string => {
    return interpretation.replace(/_/g, ' ').toUpperCase();
  };

  return (
    <div className="bg-white p-4 rounded-lg shadow">
      <h3 className="text-lg font-semibold text-gray-900 mb-4">
        Inter-Judge Agreement
      </h3>

      {/* Agreement Metrics */}
      <div className="mb-4 grid grid-cols-2 gap-4">
        {interJudgeAgreement.cohens_kappa !== null &&
          interJudgeAgreement.cohens_kappa !== undefined && (
            <div className="bg-gray-50 p-3 rounded">
              <p className="text-sm text-gray-600">Cohen's Kappa</p>
              <p className="text-xl font-semibold text-gray-900">
                {interJudgeAgreement.cohens_kappa.toFixed(3)}
              </p>
            </div>
          )}
        {interJudgeAgreement.fleiss_kappa !== null &&
          interJudgeAgreement.fleiss_kappa !== undefined && (
            <div className="bg-gray-50 p-3 rounded">
              <p className="text-sm text-gray-600">Fleiss' Kappa</p>
              <p className="text-xl font-semibold text-gray-900">
                {interJudgeAgreement.fleiss_kappa.toFixed(3)}
              </p>
            </div>
          )}
        {interJudgeAgreement.krippendorff_alpha !== null &&
          interJudgeAgreement.krippendorff_alpha !== undefined && (
            <div className="bg-gray-50 p-3 rounded">
              <p className="text-sm text-gray-600">Krippendorff's Alpha</p>
              <p className="text-xl font-semibold text-gray-900">
                {interJudgeAgreement.krippendorff_alpha.toFixed(3)}
              </p>
            </div>
          )}
      </div>

      {/* Interpretation Badge */}
      <div className="mb-4">
        <span
          className={`inline-block px-3 py-1 rounded-full text-sm font-semibold ${getInterpretationColor(
            interJudgeAgreement.interpretation
          )}`}
        >
          {getInterpretationLabel(interJudgeAgreement.interpretation)} Agreement
        </span>
      </div>

      {/* Heatmap */}
      <div className="flex justify-center overflow-x-auto">
        <svg ref={svgRef}></svg>
      </div>

      {/* Tooltip */}
      {tooltip && (
        <div
          className="fixed bg-white p-3 rounded-lg shadow-lg border border-gray-200 z-50 pointer-events-none"
          style={{
            left: `${tooltip.x}px`,
            top: `${tooltip.y - 80}px`,
            transform: 'translateX(-50%)',
          }}
        >
          <p className="text-sm font-semibold text-gray-900">
            {tooltip.judge1} ↔ {tooltip.judge2}
          </p>
          <p className="text-sm text-gray-600">
            Correlation: {tooltip.correlation.toFixed(3)}
          </p>
        </div>
      )}

      {/* Interpretation Guide */}
      <div className="mt-4 pt-4 border-t border-gray-200">
        <h4 className="text-sm font-semibold text-gray-700 mb-2">
          Interpretation Guide
        </h4>
        <div className="grid grid-cols-2 gap-2 text-xs">
          <div className="flex items-center">
            <div className="w-3 h-3 bg-green-500 rounded mr-2"></div>
            <span className="text-gray-600">0.81-1.00: Almost Perfect</span>
          </div>
          <div className="flex items-center">
            <div className="w-3 h-3 bg-green-400 rounded mr-2"></div>
            <span className="text-gray-600">0.61-0.80: Substantial</span>
          </div>
          <div className="flex items-center">
            <div className="w-3 h-3 bg-yellow-400 rounded mr-2"></div>
            <span className="text-gray-600">0.41-0.60: Moderate</span>
          </div>
          <div className="flex items-center">
            <div className="w-3 h-3 bg-orange-400 rounded mr-2"></div>
            <span className="text-gray-600">0.21-0.40: Fair</span>
          </div>
          <div className="flex items-center">
            <div className="w-3 h-3 bg-red-400 rounded mr-2"></div>
            <span className="text-gray-600">0.00-0.20: Slight</span>
          </div>
          <div className="flex items-center">
            <div className="w-3 h-3 bg-red-600 rounded mr-2"></div>
            <span className="text-gray-600">&lt;0.00: Poor</span>
          </div>
        </div>
      </div>
    </div>
  );
};
