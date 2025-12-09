import React, { useState, useCallback, useEffect, useRef } from 'react';
import { format } from 'date-fns';

export interface FilterOptions {
  dateRange?: {
    start: Date | null;
    end: Date | null;
  };
  scoreRange?: {
    min: number | null;
    max: number | null;
  };
  searchQuery?: string;
  sortBy?: 'created_at' | 'consensus_score' | 'hallucination_score';
  sortOrder?: 'asc' | 'desc';
  status?: 'completed' | 'pending' | 'failed' | 'all';
}

interface SearchAndFilterProps {
  onFilterChange: (filters: FilterOptions) => void;
  className?: string;
}

const SearchAndFilter: React.FC<SearchAndFilterProps> = React.memo(({
  onFilterChange,
  className = '',
}) => {
  const [isExpanded, setIsExpanded] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [dateStart, setDateStart] = useState<string>('');
  const [dateEnd, setDateEnd] = useState<string>('');
  const [scoreMin, setScoreMin] = useState<string>('');
  const [scoreMax, setScoreMax] = useState<string>('');
  const [sortBy, setSortBy] = useState<'created_at' | 'consensus_score' | 'hallucination_score'>(
    'created_at'
  );
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('desc');
  const [status, setStatus] = useState<'completed' | 'pending' | 'failed' | 'all'>('all');
  
  // Debounce timer ref
  const debounceTimerRef = useRef<NodeJS.Timeout | null>(null);

  const handleApplyFilters = useCallback(() => {
    const filters: FilterOptions = {
      searchQuery: searchQuery.trim() || undefined,
      dateRange:
        dateStart || dateEnd
          ? {
              start: dateStart ? new Date(dateStart) : null,
              end: dateEnd ? new Date(dateEnd) : null,
            }
          : undefined,
      scoreRange:
        scoreMin || scoreMax
          ? {
              min: scoreMin ? parseFloat(scoreMin) : null,
              max: scoreMax ? parseFloat(scoreMax) : null,
            }
          : undefined,
      sortBy,
      sortOrder,
      status: status !== 'all' ? status : undefined,
    };

    onFilterChange(filters);
  }, [searchQuery, dateStart, dateEnd, scoreMin, scoreMax, sortBy, sortOrder, status, onFilterChange]);

  // Debounced search handler
  const handleSearchChange = useCallback((value: string) => {
    setSearchQuery(value);
    
    // Clear existing timer
    if (debounceTimerRef.current) {
      clearTimeout(debounceTimerRef.current);
    }
    
    // Set new timer for debounced search (500ms delay)
    debounceTimerRef.current = setTimeout(() => {
      const filters: FilterOptions = {
        searchQuery: value.trim() || undefined,
        dateRange:
          dateStart || dateEnd
            ? {
                start: dateStart ? new Date(dateStart) : null,
                end: dateEnd ? new Date(dateEnd) : null,
              }
            : undefined,
        scoreRange:
          scoreMin || scoreMax
            ? {
                min: scoreMin ? parseFloat(scoreMin) : null,
                max: scoreMax ? parseFloat(scoreMax) : null,
              }
            : undefined,
        sortBy,
        sortOrder,
        status: status !== 'all' ? status : undefined,
      };
      onFilterChange(filters);
    }, 500);
  }, [dateStart, dateEnd, scoreMin, scoreMax, sortBy, sortOrder, status, onFilterChange]);

  // Cleanup debounce timer on unmount
  useEffect(() => {
    return () => {
      if (debounceTimerRef.current) {
        clearTimeout(debounceTimerRef.current);
      }
    };
  }, []);

  const handleClearFilters = useCallback(() => {
    setSearchQuery('');
    setDateStart('');
    setDateEnd('');
    setScoreMin('');
    setScoreMax('');
    setSortBy('created_at');
    setSortOrder('desc');
    setStatus('all');
    onFilterChange({
      sortBy: 'created_at',
      sortOrder: 'desc',
    });
  }, [onFilterChange]);

  const hasActiveFilters =
    searchQuery ||
    dateStart ||
    dateEnd ||
    scoreMin ||
    scoreMax ||
    status !== 'all' ||
    sortBy !== 'created_at' ||
    sortOrder !== 'desc';

  return (
    <div className={`bg-white border-b border-gray-200 ${className}`}>
      {/* Search Bar */}
      <div className="px-4 py-3">
        <div className="relative">
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => handleSearchChange(e.target.value)}
            onKeyPress={(e) => {
              if (e.key === 'Enter') {
                handleApplyFilters();
              }
            }}
            placeholder="Search evaluations..."
            className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          />
          <svg
            className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
            />
          </svg>
        </div>

        {/* Filter Toggle Button */}
        <button
          onClick={() => setIsExpanded(!isExpanded)}
          className="mt-2 flex items-center gap-2 text-sm text-gray-600 hover:text-gray-900"
        >
          <svg
            className={`w-4 h-4 transition-transform ${isExpanded ? 'rotate-180' : ''}`}
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M19 9l-7 7-7-7"
            />
          </svg>
          <span>
            {isExpanded ? 'Hide' : 'Show'} Filters
            {hasActiveFilters && !isExpanded && (
              <span className="ml-1 inline-flex items-center justify-center w-5 h-5 text-xs font-medium text-white bg-blue-600 rounded-full">
                !
              </span>
            )}
          </span>
        </button>
      </div>

      {/* Expanded Filters */}
      {isExpanded && (
        <div className="px-4 pb-4 space-y-4 border-t border-gray-100 pt-4">
          {/* Date Range */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Date Range</label>
            <div className="grid grid-cols-2 gap-2">
              <input
                type="date"
                value={dateStart}
                onChange={(e) => setDateStart(e.target.value)}
                className="px-3 py-2 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                placeholder="Start date"
              />
              <input
                type="date"
                value={dateEnd}
                onChange={(e) => setDateEnd(e.target.value)}
                className="px-3 py-2 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                placeholder="End date"
              />
            </div>
          </div>

          {/* Score Range */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Consensus Score Range
            </label>
            <div className="grid grid-cols-2 gap-2">
              <input
                type="number"
                min="0"
                max="100"
                value={scoreMin}
                onChange={(e) => setScoreMin(e.target.value)}
                className="px-3 py-2 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                placeholder="Min (0)"
              />
              <input
                type="number"
                min="0"
                max="100"
                value={scoreMax}
                onChange={(e) => setScoreMax(e.target.value)}
                className="px-3 py-2 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                placeholder="Max (100)"
              />
            </div>
          </div>

          {/* Status Filter */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Status</label>
            <select
              value={status}
              onChange={(e) =>
                setStatus(e.target.value as 'completed' | 'pending' | 'failed' | 'all')
              }
              className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            >
              <option value="all">All Statuses</option>
              <option value="completed">Completed</option>
              <option value="pending">Pending</option>
              <option value="failed">Failed</option>
            </select>
          </div>

          {/* Sort Options */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Sort By</label>
            <div className="grid grid-cols-2 gap-2">
              <select
                value={sortBy}
                onChange={(e) =>
                  setSortBy(
                    e.target.value as 'created_at' | 'consensus_score' | 'hallucination_score'
                  )
                }
                className="px-3 py-2 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              >
                <option value="created_at">Date Created</option>
                <option value="consensus_score">Consensus Score</option>
                <option value="hallucination_score">Hallucination Score</option>
              </select>
              <select
                value={sortOrder}
                onChange={(e) => setSortOrder(e.target.value as 'asc' | 'desc')}
                className="px-3 py-2 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              >
                <option value="desc">Descending</option>
                <option value="asc">Ascending</option>
              </select>
            </div>
          </div>

          {/* Action Buttons */}
          <div className="flex gap-2 pt-2">
            <button
              onClick={handleApplyFilters}
              className="flex-1 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors text-sm font-medium"
            >
              Apply Filters
            </button>
            <button
              onClick={handleClearFilters}
              className="px-4 py-2 border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50 transition-colors text-sm font-medium"
            >
              Clear
            </button>
          </div>

          {/* Active Filters Summary */}
          {hasActiveFilters && (
            <div className="pt-2 border-t border-gray-100">
              <p className="text-xs text-gray-500 mb-2">Active Filters:</p>
              <div className="flex flex-wrap gap-2">
                {searchQuery && (
                  <span className="inline-flex items-center gap-1 px-2 py-1 bg-blue-100 text-blue-800 rounded text-xs">
                    Search: {searchQuery}
                  </span>
                )}
                {dateStart && (
                  <span className="inline-flex items-center gap-1 px-2 py-1 bg-blue-100 text-blue-800 rounded text-xs">
                    From: {format(new Date(dateStart), 'MMM d, yyyy')}
                  </span>
                )}
                {dateEnd && (
                  <span className="inline-flex items-center gap-1 px-2 py-1 bg-blue-100 text-blue-800 rounded text-xs">
                    To: {format(new Date(dateEnd), 'MMM d, yyyy')}
                  </span>
                )}
                {scoreMin && (
                  <span className="inline-flex items-center gap-1 px-2 py-1 bg-blue-100 text-blue-800 rounded text-xs">
                    Min Score: {scoreMin}
                  </span>
                )}
                {scoreMax && (
                  <span className="inline-flex items-center gap-1 px-2 py-1 bg-blue-100 text-blue-800 rounded text-xs">
                    Max Score: {scoreMax}
                  </span>
                )}
                {status !== 'all' && (
                  <span className="inline-flex items-center gap-1 px-2 py-1 bg-blue-100 text-blue-800 rounded text-xs">
                    Status: {status}
                  </span>
                )}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
});

export default SearchAndFilter;
