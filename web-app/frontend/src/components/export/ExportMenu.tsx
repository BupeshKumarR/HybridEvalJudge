import React, { useState } from 'react';
import { exportAsJSON, exportAsCSV, exportAsPDF } from '../../utils/exportUtils';

interface ExportMenuProps {
  sessionId: string;
  onExportStart?: () => void;
  onExportComplete?: () => void;
  onExportError?: (error: Error) => void;
}

const ExportMenu: React.FC<ExportMenuProps> = ({
  sessionId,
  onExportStart,
  onExportComplete,
  onExportError,
}) => {
  const [isOpen, setIsOpen] = useState(false);
  const [isExporting, setIsExporting] = useState(false);
  const [exportFormat, setExportFormat] = useState<string | null>(null);

  const handleExport = async (format: 'json' | 'csv' | 'pdf') => {
    setIsExporting(true);
    setExportFormat(format);
    setIsOpen(false);

    if (onExportStart) {
      onExportStart();
    }

    try {
      switch (format) {
        case 'json':
          await exportAsJSON(sessionId);
          break;
        case 'csv':
          await exportAsCSV(sessionId);
          break;
        case 'pdf':
          await exportAsPDF(sessionId);
          break;
      }

      if (onExportComplete) {
        onExportComplete();
      }
    } catch (error) {
      console.error(`Failed to export as ${format}:`, error);
      if (onExportError) {
        onExportError(error as Error);
      }
    } finally {
      setIsExporting(false);
      setExportFormat(null);
    }
  };

  return (
    <div className="relative inline-block text-left">
      <button
        onClick={() => setIsOpen(!isOpen)}
        disabled={isExporting}
        className="inline-flex items-center gap-2 px-3 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-md hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
        aria-label="Export evaluation"
      >
        {isExporting ? (
          <>
            <svg
              className="w-4 h-4 animate-spin"
              fill="none"
              viewBox="0 0 24 24"
            >
              <circle
                className="opacity-25"
                cx="12"
                cy="12"
                r="10"
                stroke="currentColor"
                strokeWidth="4"
              />
              <path
                className="opacity-75"
                fill="currentColor"
                d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
              />
            </svg>
            <span>Exporting {exportFormat?.toUpperCase()}...</span>
          </>
        ) : (
          <>
            <svg
              className="w-4 h-4"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"
              />
            </svg>
            <span>Export</span>
            <svg
              className={`w-4 h-4 transition-transform ${
                isOpen ? 'rotate-180' : ''
              }`}
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
          </>
        )}
      </button>

      {isOpen && !isExporting && (
        <>
          {/* Backdrop */}
          <div
            className="fixed inset-0 z-10"
            onClick={() => setIsOpen(false)}
          />

          {/* Dropdown menu */}
          <div className="absolute right-0 z-20 w-56 mt-2 origin-top-right bg-white border border-gray-200 rounded-md shadow-lg">
            <div className="py-1">
              <button
                onClick={() => handleExport('json')}
                className="flex items-center w-full px-4 py-2 text-sm text-gray-700 hover:bg-gray-100 hover:text-gray-900"
              >
                <svg
                  className="w-5 h-5 mr-3 text-blue-500"
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
                <div className="flex-1 text-left">
                  <div className="font-medium">Export as JSON</div>
                  <div className="text-xs text-gray-500">
                    Complete structured data
                  </div>
                </div>
              </button>

              <button
                onClick={() => handleExport('csv')}
                className="flex items-center w-full px-4 py-2 text-sm text-gray-700 hover:bg-gray-100 hover:text-gray-900"
              >
                <svg
                  className="w-5 h-5 mr-3 text-green-500"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M3 10h18M3 14h18m-9-4v8m-7 0h14a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z"
                  />
                </svg>
                <div className="flex-1 text-left">
                  <div className="font-medium">Export as CSV</div>
                  <div className="text-xs text-gray-500">
                    Tabular data format
                  </div>
                </div>
              </button>

              <button
                onClick={() => handleExport('pdf')}
                className="flex items-center w-full px-4 py-2 text-sm text-gray-700 hover:bg-gray-100 hover:text-gray-900"
              >
                <svg
                  className="w-5 h-5 mr-3 text-red-500"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M7 21h10a2 2 0 002-2V9.414a1 1 0 00-.293-.707l-5.414-5.414A1 1 0 0012.586 3H7a2 2 0 00-2 2v14a2 2 0 002 2z"
                  />
                </svg>
                <div className="flex-1 text-left">
                  <div className="font-medium">Export as PDF</div>
                  <div className="text-xs text-gray-500">
                    Professional report
                  </div>
                </div>
              </button>
            </div>
          </div>
        </>
      )}
    </div>
  );
};

export default ExportMenu;
