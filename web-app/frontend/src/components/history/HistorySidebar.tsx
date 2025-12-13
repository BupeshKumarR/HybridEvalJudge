import React, { useEffect, useCallback, useRef, useState } from 'react';
import { formatDistanceToNow, isValid } from 'date-fns';
import { useHistoryStore } from '../../store/historyStore';
import { chatApi } from '../../api/chat';
import { evaluationsApi } from '../../api/evaluations';
import SearchAndFilter, { FilterOptions } from './SearchAndFilter';

type HistoryMode = 'chat' | 'evaluations';

interface HistorySession {
  id: string;
  timestamp: Date | string;
  sourcePreview: string;
  consensusScore: number;
  hallucinationScore: number;
  status: string;
}

const formatTimestamp = (timestamp: Date | string): string => {
  try {
    let date: Date;
    
    if (timestamp instanceof Date) {
      date = timestamp;
    } else if (typeof timestamp === 'string') {
      // Handle ISO strings - ensure proper parsing
      // If the string doesn't have timezone info, assume UTC
      const isoString = timestamp.includes('Z') || timestamp.includes('+') || timestamp.includes('-', 10)
        ? timestamp
        : timestamp + 'Z';
      date = new Date(isoString);
    } else {
      return 'Just now';
    }
    
    if (!isValid(date) || isNaN(date.getTime())) {
      return 'Just now';
    }
    
    // Check for future dates (clock skew) - treat as "Just now"
    const now = new Date();
    if (date > now) {
      return 'Just now';
    }
    
    return formatDistanceToNow(date, { addSuffix: true });
  } catch {
    return 'Just now';
  }
};

const SessionItem: React.FC<{
  session: HistorySession;
  isSelected: boolean;
  onSelect: (id: string) => void;
  onDelete: (id: string) => void;
}> = ({ session, isSelected, onSelect, onDelete }) => {
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed': return 'bg-green-100 text-green-800';
      case 'pending':
      case 'in_progress': return 'bg-blue-100 text-blue-800';
      case 'failed': return 'bg-red-100 text-red-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  const handleDeleteClick = (e: React.MouseEvent) => {
    e.stopPropagation();
    setShowDeleteConfirm(true);
  };

  const handleConfirmDelete = (e: React.MouseEvent) => {
    e.stopPropagation();
    onDelete(session.id);
    setShowDeleteConfirm(false);
  };

  const handleCancelDelete = (e: React.MouseEvent) => {
    e.stopPropagation();
    setShowDeleteConfirm(false);
  };

  return (
    <div
      onClick={() => onSelect(session.id)}
      className={`px-4 py-3 cursor-pointer hover:bg-gray-50 transition-colors group relative ${
        isSelected ? 'bg-blue-50 border-l-4 border-l-blue-500' : ''
      }`}
    >
      <div className="flex items-center justify-between mb-1">
        <span className="text-xs text-gray-500">{formatTimestamp(session.timestamp)}</span>
        <div className="flex items-center gap-2">
          <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${getStatusColor(session.status)}`}>
            {session.status}
          </span>
          {/* Delete button - visible on hover */}
          <button
            onClick={handleDeleteClick}
            className="opacity-0 group-hover:opacity-100 p-1 text-gray-400 hover:text-red-500 transition-all"
            title="Delete session"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
            </svg>
          </button>
        </div>
      </div>
      
      {/* Delete confirmation overlay */}
      {showDeleteConfirm && (
        <div className="absolute inset-0 bg-white bg-opacity-95 flex items-center justify-center z-10 rounded">
          <div className="text-center p-2">
            <p className="text-sm text-gray-700 mb-2">Delete this chat?</p>
            <div className="flex gap-2 justify-center">
              <button
                onClick={handleConfirmDelete}
                className="px-3 py-1 text-xs font-medium text-white bg-red-500 rounded hover:bg-red-600"
              >
                Delete
              </button>
              <button
                onClick={handleCancelDelete}
                className="px-3 py-1 text-xs font-medium text-gray-700 bg-gray-200 rounded hover:bg-gray-300"
              >
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}
      
      <p className="text-sm text-gray-900 line-clamp-2 mb-2">{session.sourcePreview}</p>
      {(session.consensusScore > 0 || session.hallucinationScore > 0) && (
        <div className="flex items-center gap-2 text-xs">
          {session.consensusScore > 0 && (
            <span className={`px-2 py-0.5 rounded ${
              session.consensusScore >= 80 ? 'bg-green-100 text-green-800' :
              session.consensusScore >= 50 ? 'bg-yellow-100 text-yellow-800' :
              'bg-red-100 text-red-800'
            }`}>
              Score: {session.consensusScore.toFixed(0)}
            </span>
          )}
          {session.hallucinationScore > 0 && (
            <span className={`px-2 py-0.5 rounded ${
              session.hallucinationScore < 30 ? 'bg-green-100 text-green-800' :
              session.hallucinationScore < 60 ? 'bg-yellow-100 text-yellow-800' :
              'bg-red-100 text-red-800'
            }`}>
              Halluc: {session.hallucinationScore.toFixed(0)}
            </span>
          )}
        </div>
      )}
    </div>
  );
};

interface HistorySidebarProps {
  currentSessionId: string | null;
  onSelectSession: (sessionId: string, isEvaluation?: boolean) => void;
  onNewChat?: () => void;
  className?: string;
}

const HistorySidebar: React.FC<HistorySidebarProps> = ({
  currentSessionId,
  onSelectSession,
  onNewChat,
  className = '',
}) => {
  const {
    sessions,
    currentPage,
    hasMore,
    isLoading,
    setSessions,
    addSessions,
    setCurrentPage,
    setHasMore,
    setLoading,
  } = useHistoryStore();

  const observerTarget = useRef<HTMLDivElement>(null);
  const [filters, setFilters] = useState<FilterOptions>({
    sortBy: 'created_at',
    sortOrder: 'desc',
  });
  const containerRef = useRef<HTMLDivElement>(null);
  const [historyMode, setHistoryMode] = useState<HistoryMode>('chat');

  // Load initial sessions
  useEffect(() => {
    loadSessions(1, filters);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [filters, historyMode]);

  // Infinite scroll observer
  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        if (entries[0].isIntersecting && hasMore && !isLoading) {
          loadMoreSessions();
        }
      },
      { threshold: 0.1 }
    );

    const currentTarget = observerTarget.current;
    if (currentTarget) {
      observer.observe(currentTarget);
    }

    return () => {
      if (currentTarget) {
        observer.unobserve(currentTarget);
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [hasMore, isLoading, currentPage]);

  const loadSessions = async (page: number, currentFilters: FilterOptions) => {
    try {
      setLoading(true);

      let formattedSessions: any[] = [];
      let responseHasMore = false;

      if (historyMode === 'chat') {
        // Fetch chat sessions
        const limit = 20;
        const response = await chatApi.listSessions(limit, page);
        responseHasMore = response.has_more;

        // Convert API response to store format
        formattedSessions = response.sessions.map((session) => ({
          id: session.id,
          timestamp: new Date(session.updated_at || session.created_at),
          sourcePreview: session.last_message_preview || `New chat (${session.ollama_model})`,
          consensusScore: 0,
          hallucinationScore: 0,
          status: 'completed' as const,
        }));
      } else {
        // Fetch evaluation sessions
        const params = {
          page,
          limit: 20,
          sort_by: currentFilters.sortBy || 'created_at',
          order: currentFilters.sortOrder || 'desc',
        };
        const response = await evaluationsApi.getHistory(params);
        responseHasMore = response.has_more;

        formattedSessions = response.sessions.map((session) => ({
          id: session.id,
          timestamp: session.created_at,
          sourcePreview: session.source_preview,
          consensusScore: session.consensus_score || 0,
          hallucinationScore: session.hallucination_score || 0,
          status: session.status,
        }));
      }

      // Client-side filtering for search query and date range
      if (currentFilters.searchQuery) {
        const query = currentFilters.searchQuery.toLowerCase();
        formattedSessions = formattedSessions.filter((session) =>
          session.sourcePreview.toLowerCase().includes(query)
        );
      }

      if (currentFilters.dateRange) {
        formattedSessions = formattedSessions.filter((session) => {
          const sessionDate = session.timestamp;
          const { start, end } = currentFilters.dateRange!;

          if (start && sessionDate < start) return false;
          if (end && sessionDate > end) return false;
          return true;
        });
      }

      if (page === 1) {
        setSessions(formattedSessions);
      } else {
        addSessions(formattedSessions);
      }

      setCurrentPage(page);
      setHasMore(responseHasMore);
    } catch (error) {
      console.error('Failed to load sessions:', error);
    } finally {
      setLoading(false);
    }
  };

  const loadMoreSessions = useCallback(() => {
    if (!isLoading && hasMore) {
      loadSessions(currentPage + 1, filters);
    }
  }, [currentPage, hasMore, isLoading, filters]);

  const handleFilterChange = useCallback((newFilters: FilterOptions) => {
    setFilters(newFilters);
    // Reset to page 1 when filters change
    loadSessions(1, newFilters);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleDeleteSession = useCallback(async (sessionId: string) => {
    try {
      if (historyMode === 'chat') {
        await chatApi.deleteSession(sessionId);
      } else {
        await evaluationsApi.deleteEvaluation(sessionId);
      }
      // Remove from local state
      setSessions(sessions.filter(s => s.id !== sessionId));
      // If deleted session was selected, clear selection
      if (currentSessionId === sessionId && onNewChat) {
        onNewChat();
      }
    } catch (error) {
      console.error('Failed to delete session:', error);
    }
  }, [historyMode, sessions, setSessions, currentSessionId, onNewChat]);

  return (
    <div className={`flex flex-col h-full bg-white border-r border-gray-200 ${className}`}>
      {/* Header with New Chat button */}
      <div className="px-4 py-3 border-b border-gray-200">
        <div className="flex items-center justify-between mb-2">
          <h2 className="text-lg font-semibold text-gray-900">History</h2>
          {onNewChat && (
            <button
              onClick={onNewChat}
              className="flex items-center gap-1 px-3 py-1.5 text-sm font-medium text-white bg-blue-600 rounded-md hover:bg-blue-700 transition-colors"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
              </svg>
              New Chat
            </button>
          )}
        </div>
        
        {/* Mode Toggle */}
        <div className="flex rounded-md bg-gray-100 p-1 mb-2">
          <button
            onClick={() => setHistoryMode('chat')}
            className={`flex-1 px-3 py-1 text-xs font-medium rounded transition-colors ${
              historyMode === 'chat'
                ? 'bg-white text-gray-900 shadow-sm'
                : 'text-gray-600 hover:text-gray-900'
            }`}
          >
            Chats
          </button>
          <button
            onClick={() => setHistoryMode('evaluations')}
            className={`flex-1 px-3 py-1 text-xs font-medium rounded transition-colors ${
              historyMode === 'evaluations'
                ? 'bg-white text-gray-900 shadow-sm'
                : 'text-gray-600 hover:text-gray-900'
            }`}
          >
            Evaluations
          </button>
        </div>
        
        <p className="text-sm text-gray-500">
          {sessions.length} {historyMode === 'chat' ? 'chat' : 'evaluation'}{sessions.length !== 1 ? 's' : ''}
        </p>
      </div>

      {/* Search and Filter */}
      <SearchAndFilter onFilterChange={handleFilterChange} />

      {/* Session List - Simple scrollable list instead of virtualized */}
      <div ref={containerRef} className="flex-1 overflow-y-auto">
        {sessions.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full px-4 text-center py-8">
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
            <p className="text-gray-600 font-medium">No {historyMode === 'chat' ? 'chats' : 'evaluations'} yet</p>
            <p className="text-gray-500 text-sm mt-1">
              Start by {historyMode === 'chat' ? 'sending a message' : 'submitting an evaluation'}
            </p>
          </div>
        ) : (
          <div className="divide-y divide-gray-100">
            {sessions.map((session) => (
              <SessionItem
                key={session.id}
                session={session}
                isSelected={currentSessionId === session.id}
                onSelect={(id) => onSelectSession(id, historyMode === 'evaluations')}
                onDelete={handleDeleteSession}
              />
            ))}
          </div>
        )}
        
        {/* Loading Indicator */}
        {isLoading && (
          <div className="px-4 py-6 text-center">
            <div className="inline-block animate-spin rounded-full h-6 w-6 border-b-2 border-blue-600"></div>
            <p className="text-sm text-gray-500 mt-2">Loading...</p>
          </div>
        )}

        {/* Infinite Scroll Observer Target */}
        <div ref={observerTarget} className="h-4" />
      </div>
    </div>
  );
};

export default HistorySidebar;
