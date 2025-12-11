import React, { useMemo } from 'react';
import { ClaimVerdict, ClaimVerdictType } from '../../api/types';

interface ClaimHighlighterProps {
  /** The original response text to highlight */
  text: string;
  /** List of claim verdicts with text spans */
  claims: ClaimVerdict[];
  /** Currently selected claim ID for emphasis */
  selectedClaimId?: string;
  /** Callback when a highlighted claim is clicked */
  onClaimClick?: (claim: ClaimVerdict) => void;
}

interface TextSegment {
  text: string;
  claim?: ClaimVerdict;
  isHighlighted: boolean;
}

/**
 * ClaimHighlighter - Highlights problematic claims in response text
 * 
 * Features:
 * - Highlight claims in the response text based on their verdict
 * - Color code by verdict (green=supported, red=refuted, yellow=unknown)
 * - Interactive highlighting with click support
 * 
 * Requirements: 5.5
 */
const ClaimHighlighter: React.FC<ClaimHighlighterProps> = ({
  text,
  claims,
  selectedClaimId,
  onClaimClick,
}) => {
  /**
   * Get highlight color based on verdict type
   * Color coding: green=supported, red=refuted, yellow=unknown
   */
  const getHighlightStyle = (verdict: ClaimVerdictType, isSelected: boolean): string => {
    const baseStyles = 'rounded px-0.5 cursor-pointer transition-all';
    const selectedStyles = isSelected ? 'ring-2 ring-offset-1' : '';
    
    switch (verdict) {
      case 'SUPPORTED':
        return `${baseStyles} bg-green-100 hover:bg-green-200 ${isSelected ? 'ring-green-400' : ''} ${selectedStyles}`;
      case 'REFUTED':
        return `${baseStyles} bg-red-100 hover:bg-red-200 ${isSelected ? 'ring-red-400' : ''} ${selectedStyles}`;
      case 'NOT_ENOUGH_INFO':
        return `${baseStyles} bg-yellow-100 hover:bg-yellow-200 ${isSelected ? 'ring-yellow-400' : ''} ${selectedStyles}`;
      default:
        return `${baseStyles} bg-gray-100 hover:bg-gray-200 ${selectedStyles}`;
    }
  };

  /**
   * Get tooltip text for a claim
   */
  const getTooltipText = (claim: ClaimVerdict): string => {
    const verdictLabels: Record<ClaimVerdictType, string> = {
      'SUPPORTED': 'Supported',
      'REFUTED': 'Refuted',
      'NOT_ENOUGH_INFO': 'Not Enough Info',
    };
    return `${verdictLabels[claim.verdict]} (${(claim.confidence * 100).toFixed(0)}% confidence) - ${claim.judge_name}`;
  };


  /**
   * Build text segments with highlighting information
   * Handles overlapping spans by prioritizing more severe verdicts
   */
  const segments = useMemo((): TextSegment[] => {
    if (!text || !claims || claims.length === 0) {
      return [{ text, isHighlighted: false }];
    }

    // Sort claims by start position, then by severity (REFUTED > NOT_ENOUGH_INFO > SUPPORTED)
    const sortedClaims = [...claims].sort((a, b) => {
      if (a.text_span_start !== b.text_span_start) {
        return a.text_span_start - b.text_span_start;
      }
      // Prioritize more severe verdicts
      const severityOrder: Record<ClaimVerdictType, number> = {
        'REFUTED': 0,
        'NOT_ENOUGH_INFO': 1,
        'SUPPORTED': 2,
      };
      return severityOrder[a.verdict] - severityOrder[b.verdict];
    });

    // Filter out claims with invalid spans
    const validClaims = sortedClaims.filter(
      claim => 
        claim.text_span_start >= 0 && 
        claim.text_span_end <= text.length &&
        claim.text_span_start < claim.text_span_end
    );

    if (validClaims.length === 0) {
      return [{ text, isHighlighted: false }];
    }

    const result: TextSegment[] = [];
    let currentPos = 0;

    // Build non-overlapping segments
    const processedRanges: Array<{ start: number; end: number; claim: ClaimVerdict }> = [];
    
    for (const claim of validClaims) {
      // Check if this claim overlaps with any processed range
      const overlaps = processedRanges.some(
        range => 
          (claim.text_span_start < range.end && claim.text_span_end > range.start)
      );
      
      if (!overlaps) {
        processedRanges.push({
          start: claim.text_span_start,
          end: claim.text_span_end,
          claim,
        });
      }
    }

    // Sort processed ranges by start position
    processedRanges.sort((a, b) => a.start - b.start);

    // Build segments
    for (const range of processedRanges) {
      // Add non-highlighted text before this range
      if (range.start > currentPos) {
        result.push({
          text: text.slice(currentPos, range.start),
          isHighlighted: false,
        });
      }

      // Add highlighted text
      result.push({
        text: text.slice(range.start, range.end),
        claim: range.claim,
        isHighlighted: true,
      });

      currentPos = range.end;
    }

    // Add remaining text after last highlight
    if (currentPos < text.length) {
      result.push({
        text: text.slice(currentPos),
        isHighlighted: false,
      });
    }

    return result;
  }, [text, claims]);

  const getClaimId = (claim: ClaimVerdict): string => {
    return claim.id || `claim-${claim.text_span_start}-${claim.text_span_end}`;
  };

  return (
    <div className="text-sm text-gray-900 leading-relaxed whitespace-pre-wrap">
      {segments.map((segment, index) => {
        if (!segment.isHighlighted || !segment.claim) {
          return <span key={index}>{segment.text}</span>;
        }

        const claimId = getClaimId(segment.claim);
        const isSelected = selectedClaimId === claimId;

        return (
          <span
            key={index}
            className={getHighlightStyle(segment.claim.verdict, isSelected)}
            onClick={() => onClaimClick?.(segment.claim!)}
            title={getTooltipText(segment.claim)}
            role="button"
            tabIndex={0}
            onKeyDown={(e) => {
              if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                onClaimClick?.(segment.claim!);
              }
            }}
            aria-label={`Claim: ${segment.text}. ${getTooltipText(segment.claim)}`}
          >
            {segment.text}
          </span>
        );
      })}
    </div>
  );
};

export default ClaimHighlighter;
