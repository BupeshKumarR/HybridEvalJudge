import React, { useState } from 'react';
import { formatDistanceToNow, isValid } from 'date-fns';
import { useVirtualScroll } from '../../hooks/useVirtualScroll';
import { exportAsJSON, exportAsCSV } from '../../utils/exportUtils';

interface HistorySession {
  id: string;
  timestamp: Date | string;
  sourcePreview: string;
  consensusScore: number;
  hallucinationScore: number;
  status: string;
}

/**
 * Safely format a timestamp for display
 * Handles both Date objects and ISO strings, with fallback for invalid dates
 */
const formatTimestamp = (timestamp: Date | string): string => {
  try {
    let date: Date;
    if (timestamp instanceof Date) {
      date = timestamp;
    } else if (typeof timestamp === 'string') {
      // Ensure UTC timestamps are parsed correctly
      const isoString = timestamp.includes('Z') || timestamp.includes('+') || timestamp.includes('-', 10)
        ? timestamp : timestamp + 'Z';
      date = new Date(isoString);
    } else {
      return 'Just now';
    }
    if (!isValid(date) || isNaN(date.getTime())) return 'Just now';
    if (date > new Date()) return 'Just now'; // Handle future dates
    return formatDistanceToNow(date, { addSuffix: true });
  } catch {
    return 'Just now';
  }
};

interface VirtualizedHistoryListProps {
  sessions: HistorySession[];
  currentSessionId: string | null;
  onSelectSession: (sessionId: string) => void;
  containerHeight: number;
}

const ITEM_HEIGHT = 120; // Approximate height of each session item

/**
 * Compact export button for session items
 * Requirements: 11.1, 11.4
 */
const SessionExportButton: React.FC<{ sessionId: string }> = ({ sessionId }) => {
  const [isOpen, setIsOpen] = useState(false);
  const [isExporting, setIsExporting] = useState(false);

  const handleExport = async (format: 'json' | 'csv', e: React.MouseEvent) => {
    e.stopPropagation();
    setIsExporting(true);
    setIsOpen(false);
    
    try {
      if (format === 'json') {
        await exportAsJSON(sessionId);
      } else {
        await exportAsCSV(sessionId);
      }
    } catch (error) {
      console.error(`Failed to export as ${format}:`, error);
    } finally {
      setIsExporting(false);
    }
  };

  return (
    <div className="relative">
      <button
        onClick={(e) => {
          e.stopPropagation();
          setIsOpen(!isOpen);
        }}
        disabled={isExporting}
        className="p-1 text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded transition-colors"
        title="Export session"
        aria-label="Export session"
      >
        {isExporting ? (
          <svg className="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
          </svg>
        ) : (
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
          </svg>
        )}
      </button>
      
      {isOpen && (
        <>
          <div className="fixed inset-0 z-10" onClick={(e) => { e.stopPropagation(); setIsOpen(false); }} />
          <div className="absolute right-0 z-20 w-32 mt-1 bg-white border border-gray-200 rounded shadow-lg">
            <button
              onClick={(e) => handleExport('json', e)}
              className="w-full px-3 py-2 text-xs text-left text-gray-700 hover:bg-gray-100"
            >
              Export JSON
            </button>
            <button
              onClick={(e) => handleExport('csv', e)}
              className="w-full px-3 py-2 text-xs text-left text-gray-700 hover:bg-gray-100"
            >
              Export CSV
            </button>
          </div>
        </>
      )}
    </div>
  );
};

const SessionItem = React.memo<{
  session: HistorySession;
  isSelected: boolean;
  onSelect: (id: string) => void;
}>(({ session, isSelected, onSelect }) => {
  const getScoreBadgeColor = (score: number): string => {
    if (score >= 80) return 'bg-green-100 text-green-800';
    if (score >= 50) return 'bg-yellow-100 text-yellow-800';
    return 'bg-red-100 text-red-800';
  };

  const getStatusBadgeColor = (status: string): string => {
    switch (status) {
      case 'completed':
        return 'bg-green-100 text-green-800';
      case 'pending':
      case 'in_progress':
        return 'bg-blue-100 text-blue-800';
      case 'failed':
        return 'bg-red-100 text-red-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };

  return (
    <div
      onClick={() => onSelect(session.id)}
      className={`w-full px-4 py-3 text-left hover:bg-gray-50 transition-colors border-b border-gray-100 cursor-pointer ${
        isSelected ? 'bg-blue-50 border-l-4 border-blue-500' : ''
      }`}
      style={{ height: `${ITEM_HEIGHT}px` }}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => e.key === 'Enter' && onSelect(session.id)}
    >
      {/* Timestamp and Export */}
      <div className="flex items-center justify-between mb-2">
        <span className="text-xs text-gray-500">
          {formatTimestamp(session.timestamp)}
        </span>
        <div className="flex items-center gap-2">
          {/* Export button - Requirements: 11.1, 11.4 */}
          <SessionExportButton sessionId={session.id} />
          <span
            className={`text-xs px-2 py-0.5 rounded-full font-medium ${getStatusBadgeColor(
              session.status
            )}`}
          >
            {session.status}
          </span>
        </div>
      </div>

      {/* Source Preview */}
      <p className="text-sm text-gray-900 line-clamp-2 mb-2">
        {session.sourcePreview}
      </p>

      {/* Score Badges */}
      <div className="flex items-center gap-2">
        {session.consensusScore !== null && session.consensusScore !== undefined && (
          <div
            className={`flex items-center gap-1 text-xs px-2 py-1 rounded ${getScoreBadgeColor(
              session.consensusScore
            )}`}
          >
            <svg
              className="w-3 h-3"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"
              />
            </svg>
            <span className="font-medium">{session.consensusScore.toFixed(1)}</span>
          </div>
        )}

        {session.hallucinationScore !== null &&
          session.hallucinationScore !== undefined && (
            <div
              className={`flex items-center gap-1 text-xs px-2 py-1 rounded ${
                session.hallucinationScore < 30
                  ? 'bg-green-100 text-green-800'
                  : session.hallucinationScore < 60
                  ? 'bg-yellow-100 text-yellow-800'
                  : 'bg-red-100 text-red-800'
              }`}
            >
              <svg
                className="w-3 h-3"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
                />
              </svg>
              <span className="font-medium">
                {session.hallucinationScore.toFixed(1)}
              </span>
            </div>
          )}
      </div>
    </div>
  );
});

SessionItem.displayName = 'SessionItem';

export const VirtualizedHistoryList: React.FC<VirtualizedHistoryListProps> = React.memo(({
  sessions,
  currentSessionId,
  onSelectSession,
  containerHeight,
}) => {
  const { virtualItems, totalHeight, containerRef } = useVirtualScroll(sessions, {
    itemHeight: ITEM_HEIGHT,
    containerHeight,
    overscan: 5,
  });

  if (sessions.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center h-full px-4 text-center">
        <svg
          className="w-12 h-12 text-gray-400 mb-4"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
          />
        </svg>
        <p className="text-gray-600 font-medium">No evaluations yet</p>
        <p className="text-gray-500 text-sm mt-1">
          Start by submitting your first evaluation
        </p>
      </div>
    );
  }

  return (
    <div
      ref={containerRef}
      className="overflow-y-auto"
      style={{ height: `${containerHeight}px` }}
    >
      <div style={{ height: `${totalHeight}px`, position: 'relative' }}>
        {virtualItems.map((virtualItem) => {
          const session = sessions[virtualItem.index];
          return (
            <div
              key={session.id}
              style={{
                position: 'absolute',
                top: 0,
                left: 0,
                width: '100%',
                transform: `translateY(${virtualItem.start}px)`,
              }}
            >
              <SessionItem
                session={session}
                isSelected={currentSessionId === session.id}
                onSelect={onSelectSession}
              />
            </div>
          );
        })}
      </div>
    </div>
  );
});

VirtualizedHistoryList.displayName = 'VirtualizedHistoryList';
