import React, { useState } from 'react';
import { ClaimVerdict, ClaimType, ClaimVerdictType } from '../../api/types';

interface ClaimBreakdownProps {
  /** List of claim verdicts to display */
  claims: ClaimVerdict[];
  /** Callback when a claim is clicked for highlighting */
  onClaimClick?: (claim: ClaimVerdict) => void;
  /** Currently selected claim for highlighting */
  selectedClaimId?: string;
}

/**
 * ClaimBreakdown - Displays extracted claims with verdict badges and routing metadata
 * 
 * Features:
 * - List extracted claims with verdict badges (Supported/Refuted/Not Enough Info)
 * - Show claim type labels (numerical, temporal, definitional, general)
 * - Display routing metadata (which judge handled each claim)
 * 
 * Requirements: 5.4, 13.1, 13.2
 */
const ClaimBreakdown: React.FC<ClaimBreakdownProps> = ({
  claims,
  onClaimClick,
  selectedClaimId,
}) => {
  const [expandedClaims, setExpandedClaims] = useState<Set<string>>(new Set());

  /**
   * Get verdict badge styling based on verdict type
   * Property 13: Claim verdict display
   */
  const getVerdictBadgeStyle = (verdict: ClaimVerdictType): string => {
    switch (verdict) {
      case 'SUPPORTED':
        return 'bg-green-100 text-green-800 border-green-200';
      case 'REFUTED':
        return 'bg-red-100 text-red-800 border-red-200';
      case 'NOT_ENOUGH_INFO':
        return 'bg-yellow-100 text-yellow-800 border-yellow-200';
      default:
        return 'bg-gray-100 text-gray-800 border-gray-200';
    }
  };

  /**
   * Get human-readable verdict label
   */
  const getVerdictLabel = (verdict: ClaimVerdictType): string => {
    switch (verdict) {
      case 'SUPPORTED':
        return 'Supported';
      case 'REFUTED':
        return 'Refuted';
      case 'NOT_ENOUGH_INFO':
        return 'Not Enough Info';
      default:
        return verdict;
    }
  };


  /**
   * Get claim type badge styling
   */
  const getClaimTypeBadgeStyle = (claimType: ClaimType): string => {
    switch (claimType) {
      case 'numerical':
        return 'bg-blue-50 text-blue-700 border-blue-200';
      case 'temporal':
        return 'bg-purple-50 text-purple-700 border-purple-200';
      case 'definitional':
        return 'bg-indigo-50 text-indigo-700 border-indigo-200';
      case 'general':
      default:
        return 'bg-gray-50 text-gray-700 border-gray-200';
    }
  };

  /**
   * Get human-readable claim type label
   */
  const getClaimTypeLabel = (claimType: ClaimType): string => {
    switch (claimType) {
      case 'numerical':
        return 'Numerical';
      case 'temporal':
        return 'Temporal';
      case 'definitional':
        return 'Definitional';
      case 'general':
      default:
        return 'General';
    }
  };

  /**
   * Get verdict icon
   */
  const getVerdictIcon = (verdict: ClaimVerdictType): React.ReactNode => {
    switch (verdict) {
      case 'SUPPORTED':
        return (
          <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
            <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
          </svg>
        );
      case 'REFUTED':
        return (
          <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
            <path fillRule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clipRule="evenodd" />
          </svg>
        );
      case 'NOT_ENOUGH_INFO':
        return (
          <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
            <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-8-3a1 1 0 00-.867.5 1 1 0 11-1.731-1A3 3 0 0113 8a3.001 3.001 0 01-2 2.83V11a1 1 0 11-2 0v-1a1 1 0 011-1 1 1 0 100-2zm0 8a1 1 0 100-2 1 1 0 000 2z" clipRule="evenodd" />
          </svg>
        );
      default:
        return null;
    }
  };

  const toggleClaimExpanded = (claimId: string) => {
    setExpandedClaims(prev => {
      const newSet = new Set(prev);
      if (newSet.has(claimId)) {
        newSet.delete(claimId);
      } else {
        newSet.add(claimId);
      }
      return newSet;
    });
  };

  const getClaimId = (claim: ClaimVerdict, index: number): string => {
    return claim.id || `claim-${index}`;
  };

  if (!claims || claims.length === 0) {
    return (
      <div className="text-sm text-gray-500 italic p-3 bg-gray-50 rounded-lg">
        No claims extracted from this response.
      </div>
    );
  }

  // Group claims by verdict for summary
  const verdictCounts = claims.reduce((acc, claim) => {
    acc[claim.verdict] = (acc[claim.verdict] || 0) + 1;
    return acc;
  }, {} as Record<string, number>);

  return (
    <div 
      className="bg-white border border-gray-200 rounded-lg overflow-hidden"
      role="region"
      aria-label="Claim breakdown"
    >
      {/* Header with summary */}
      <div className="px-4 py-3 bg-gray-50 border-b border-gray-200">
        <div className="flex items-center justify-between">
          <h4 className="text-sm font-medium text-gray-900">
            Claim Analysis ({claims.length} claims)
          </h4>
          <div className="flex items-center gap-2">
            {verdictCounts['SUPPORTED'] && (
              <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs bg-green-100 text-green-800">
                {getVerdictIcon('SUPPORTED')}
                {verdictCounts['SUPPORTED']}
              </span>
            )}
            {verdictCounts['REFUTED'] && (
              <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs bg-red-100 text-red-800">
                {getVerdictIcon('REFUTED')}
                {verdictCounts['REFUTED']}
              </span>
            )}
            {verdictCounts['NOT_ENOUGH_INFO'] && (
              <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs bg-yellow-100 text-yellow-800">
                {getVerdictIcon('NOT_ENOUGH_INFO')}
                {verdictCounts['NOT_ENOUGH_INFO']}
              </span>
            )}
          </div>
        </div>
      </div>


      {/* Claims list */}
      <div className="divide-y divide-gray-100">
        {claims.map((claim, index) => {
          const claimId = getClaimId(claim, index);
          const isExpanded = expandedClaims.has(claimId);
          const isSelected = selectedClaimId === claimId;

          return (
            <div
              key={claimId}
              className={`transition-colors ${isSelected ? 'bg-blue-50' : 'hover:bg-gray-50'}`}
            >
              {/* Claim header */}
              <button
                onClick={() => {
                  toggleClaimExpanded(claimId);
                  onClaimClick?.(claim);
                }}
                className="w-full px-4 py-3 text-left"
                aria-expanded={isExpanded}
                aria-controls={`claim-details-${claimId}`}
              >
                <div className="flex items-start gap-3">
                  {/* Verdict icon */}
                  <div className={`flex-shrink-0 mt-0.5 p-1 rounded-full ${getVerdictBadgeStyle(claim.verdict)}`}>
                    {getVerdictIcon(claim.verdict)}
                  </div>

                  {/* Claim content */}
                  <div className="flex-1 min-w-0">
                    <p className="text-sm text-gray-900 line-clamp-2">
                      {claim.claim_text}
                    </p>
                    
                    {/* Badges row */}
                    <div className="flex flex-wrap items-center gap-2 mt-2">
                      {/* Verdict badge */}
                      <span className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-medium border ${getVerdictBadgeStyle(claim.verdict)}`}>
                        {getVerdictLabel(claim.verdict)}
                      </span>

                      {/* Claim type badge */}
                      <span className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-medium border ${getClaimTypeBadgeStyle(claim.claim_type)}`}>
                        {getClaimTypeLabel(claim.claim_type)}
                      </span>

                      {/* Judge routing metadata - Requirements 13.1, 13.2 */}
                      <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs bg-gray-100 text-gray-600 border border-gray-200">
                        <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                        </svg>
                        {claim.judge_name}
                      </span>

                      {/* Confidence */}
                      <span className="text-xs text-gray-500">
                        {(claim.confidence * 100).toFixed(0)}% confidence
                      </span>
                    </div>
                  </div>

                  {/* Expand icon */}
                  <svg
                    className={`flex-shrink-0 w-4 h-4 text-gray-400 transition-transform ${isExpanded ? 'rotate-180' : ''}`}
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                  </svg>
                </div>
              </button>

              {/* Expanded details */}
              {isExpanded && (
                <div
                  id={`claim-details-${claimId}`}
                  className="px-4 pb-3 ml-10"
                >
                  {/* Reasoning */}
                  {claim.reasoning && (
                    <div className="mb-2">
                      <h5 className="text-xs font-medium text-gray-700 mb-1">Reasoning</h5>
                      <p className="text-sm text-gray-600 bg-white rounded p-2 border border-gray-200">
                        {claim.reasoning}
                      </p>
                    </div>
                  )}

                  {/* Text span info */}
                  <div className="text-xs text-gray-500">
                    Text position: characters {claim.text_span_start} - {claim.text_span_end}
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default ClaimBreakdown;
