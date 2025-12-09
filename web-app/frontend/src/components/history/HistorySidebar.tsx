import React, { useEffect, useCallback, useRef, useState } from 'react';
import { useHistoryStore } from '../../store/historyStore';
import { evaluationsApi } from '../../api/evaluations';
import SearchAndFilter, { FilterOptions } from './SearchAndFilter';
import { VirtualizedHistoryList } from './VirtualizedHistoryList';

interface HistorySidebarProps {
  currentSessionId: string | null;
  onSelectSession: (sessionId: string) => void;
  className?: string;
}

const HistorySidebar: React.FC<HistorySidebarProps> = ({
  currentSessionId,
  onSelectSession,
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
  const [containerHeight, setContainerHeight] = useState(600);
  const containerRef = useRef<HTMLDivElement>(null);

  // Measure container height
  useEffect(() => {
    const updateHeight = () => {
      if (containerRef.current) {
        setContainerHeight(containerRef.current.clientHeight);
      }
    };
    
    updateHeight();
    window.addEventListener('resize', updateHeight);
    return () => window.removeEventListener('resize', updateHeight);
  }, []);

  // Load initial sessions
  useEffect(() => {
    loadSessions(1, filters);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [filters]);

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

      // Build query parameters
      const params: any = {
        page,
        limit: 20,
        sort_by: currentFilters.sortBy || 'created_at',
        order: currentFilters.sortOrder || 'desc',
      };

      // Add status filter
      if (currentFilters.status) {
        params.status_filter = currentFilters.status;
      }

      // Add score range filters
      if (currentFilters.scoreRange?.min !== null && currentFilters.scoreRange?.min !== undefined) {
        params.min_score = currentFilters.scoreRange.min;
      }
      if (currentFilters.scoreRange?.max !== null && currentFilters.scoreRange?.max !== undefined) {
        params.max_score = currentFilters.scoreRange.max;
      }

      // For search query, we'll use the search endpoint if available
      let response;
      if (currentFilters.searchQuery) {
        // Note: The backend search endpoint would need to support additional filters
        // For now, we'll use the regular endpoint and filter client-side
        response = await evaluationsApi.getHistory(params);
      } else {
        response = await evaluationsApi.getHistory(params);
      }

      // Convert API response to store format
      let formattedSessions = response.sessions.map((session) => ({
        id: session.id,
        timestamp: new Date(session.timestamp),
        sourcePreview: session.source_preview,
        consensusScore: session.consensus_score || 0,
        hallucinationScore: session.hallucination_score || 0,
        status: session.status,
      }));

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
      setHasMore(response.has_more);
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

  return (
    <div className={`flex flex-col h-full bg-white border-r border-gray-200 ${className}`}>
      {/* Header */}
      <div className="px-4 py-3 border-b border-gray-200">
        <h2 className="text-lg font-semibold text-gray-900">History</h2>
        <p className="text-sm text-gray-500 mt-1">
          {sessions.length} session{sessions.length !== 1 ? 's' : ''}
        </p>
      </div>

      {/* Search and Filter */}
      <SearchAndFilter onFilterChange={handleFilterChange} />

      {/* Session List with Virtual Scrolling */}
      <div ref={containerRef} className="flex-1 overflow-hidden">
        <VirtualizedHistoryList
          sessions={sessions}
          currentSessionId={currentSessionId}
          onSelectSession={onSelectSession}
          containerHeight={containerHeight}
        />
        
        {/* Loading Indicator */}
        {isLoading && (
          <div className="px-4 py-6 text-center border-t border-gray-200">
            <div className="inline-block animate-spin rounded-full h-6 w-6 border-b-2 border-blue-600"></div>
            <p className="text-sm text-gray-500 mt-2">Loading more sessions...</p>
          </div>
        )}

        {/* Infinite Scroll Observer Target */}
        <div ref={observerTarget} className="h-4" />
      </div>
    </div>
  );
};

export default HistorySidebar;
