import React, { useEffect } from 'react';
import { useHistoryStore } from '../../store/historyStore';

interface SidebarProps {
  isOpen: boolean;
  onClose: () => void;
}

const Sidebar: React.FC<SidebarProps> = ({ isOpen, onClose }) => {
  const { sessions, isLoading, hasMore } = useHistoryStore();

  useEffect(() => {
    // Load initial history when sidebar opens
    // This will be implemented with API calls in task 6.3
  }, []);

  const handleLoadMore = () => {
    // Load more sessions
    // This will be implemented with API calls in task 6.3
  };

  return (
    <aside
      className={`
        fixed lg:static inset-y-0 left-0 z-50
        w-80 bg-white border-r border-gray-200
        transform transition-transform duration-300 ease-in-out
        ${isOpen ? 'translate-x-0' : '-translate-x-full lg:translate-x-0'}
        flex flex-col
      `}
    >
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-gray-200">
        <h2 className="text-lg font-semibold text-gray-900">History</h2>
        <button
          onClick={onClose}
          className="lg:hidden p-2 rounded-md text-gray-600 hover:text-gray-900 hover:bg-gray-100"
          aria-label="Close sidebar"
        >
          <svg
            className="h-5 w-5"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M6 18L18 6M6 6l12 12"
            />
          </svg>
        </button>
      </div>

      {/* Session List */}
      <div className="flex-1 overflow-y-auto p-4">
        {isLoading && sessions.length === 0 ? (
          <div className="flex items-center justify-center py-8">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600"></div>
          </div>
        ) : sessions.length === 0 ? (
          <div className="text-center py-8 text-gray-500">
            <p>No evaluation history yet</p>
            <p className="text-sm mt-2">Start a new evaluation to see it here</p>
          </div>
        ) : (
          <div className="space-y-3">
            {sessions.map((session) => (
              <div
                key={session.id}
                className="p-3 bg-gray-50 rounded-lg hover:bg-gray-100 cursor-pointer transition-colors"
              >
                <div className="flex items-start justify-between mb-2">
                  <span className="text-xs text-gray-500">
                    {new Date(session.timestamp).toLocaleDateString()}
                  </span>
                  <span
                    className={`
                      text-xs px-2 py-1 rounded-full font-medium
                      ${
                        session.status === 'completed'
                          ? 'bg-green-100 text-green-800'
                          : session.status === 'pending'
                          ? 'bg-yellow-100 text-yellow-800'
                          : 'bg-red-100 text-red-800'
                      }
                    `}
                  >
                    {session.status}
                  </span>
                </div>
                <p className="text-sm text-gray-700 line-clamp-2 mb-2">
                  {session.sourcePreview}
                </p>
                <div className="flex items-center justify-between text-xs">
                  <span className="text-gray-600">
                    Score: {session.consensusScore.toFixed(1)}
                  </span>
                  <span
                    className={`
                      font-medium
                      ${
                        session.hallucinationScore < 30
                          ? 'text-green-600'
                          : session.hallucinationScore < 60
                          ? 'text-yellow-600'
                          : 'text-red-600'
                      }
                    `}
                  >
                    Hallucination: {session.hallucinationScore.toFixed(1)}
                  </span>
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Load More Button */}
        {hasMore && sessions.length > 0 && (
          <button
            onClick={handleLoadMore}
            disabled={isLoading}
            className="w-full mt-4 py-2 px-4 bg-primary-50 text-primary-600 rounded-md hover:bg-primary-100 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            {isLoading ? 'Loading...' : 'Load More'}
          </button>
        )}
      </div>
    </aside>
  );
};

export default Sidebar;
