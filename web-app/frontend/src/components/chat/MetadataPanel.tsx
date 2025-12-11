import React, { useState } from 'react';

/**
 * Metadata for an evaluation including model info and timestamps
 */
export interface EvaluationMetadata {
  /** Ollama model name used for generation */
  ollamaModel: string;
  /** Ollama model version (if available) */
  ollamaVersion?: string;
  /** List of judge models used for evaluation */
  judgeModels: string[];
  /** Timestamp when generation started */
  generationTimestamp?: string;
  /** Timestamp when evaluation completed */
  evaluationTimestamp?: string;
  /** Generation parameters (optional) */
  generationParams?: {
    temperature?: number;
    topP?: number;
    maxTokens?: number;
  };
}

interface MetadataPanelProps {
  /** The metadata to display */
  metadata: EvaluationMetadata;
  /** Whether to start in expanded state */
  defaultExpanded?: boolean;
}

/**
 * MetadataPanel - Displays metadata about which models and parameters were used
 * 
 * Features:
 * - Show Ollama model name and version used
 * - Show which judge models evaluated the response
 * - Show timestamps for generation and evaluation
 * - Expandable to show generation parameters
 * 
 * Requirements: 10.1, 10.2, 10.3
 */
const MetadataPanel: React.FC<MetadataPanelProps> = ({
  metadata,
  defaultExpanded = false,
}) => {
  const [isExpanded, setIsExpanded] = useState(defaultExpanded);

  /**
   * Format a timestamp string for display
   */
  const formatTimestamp = (timestamp?: string): string => {
    if (!timestamp) return 'N/A';
    try {
      const date = new Date(timestamp);
      return date.toLocaleString(undefined, {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
      });
    } catch {
      return timestamp;
    }
  };


  /**
   * Get a display-friendly model name
   */
  const getModelDisplayName = (modelName: string): string => {
    // Common model name mappings for better display
    const displayNames: Record<string, string> = {
      'llama3.2': 'Llama 3.2',
      'llama3.1': 'Llama 3.1',
      'llama3': 'Llama 3',
      'llama2': 'Llama 2',
      'mistral': 'Mistral',
      'mixtral': 'Mixtral',
      'codellama': 'Code Llama',
      'phi': 'Phi',
      'gemma': 'Gemma',
      'qwen': 'Qwen',
      'groq-llama-3.3-70b': 'Groq Llama 3.3 70B',
      'gemini-2.0-flash': 'Gemini 2.0 Flash',
    };
    return displayNames[modelName.toLowerCase()] || modelName;
  };

  return (
    <div 
      className="bg-gray-50 border border-gray-200 rounded-lg overflow-hidden"
      role="region"
      aria-label="Evaluation metadata"
    >
      {/* Header - Always visible */}
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full px-4 py-3 flex items-center justify-between hover:bg-gray-100 transition-colors"
        aria-expanded={isExpanded}
        aria-controls="metadata-details"
      >
        <div className="flex items-center gap-2">
          {/* Info Icon */}
          <svg 
            className="w-5 h-5 text-gray-500" 
            fill="none" 
            stroke="currentColor" 
            viewBox="0 0 24 24"
          >
            <path 
              strokeLinecap="round" 
              strokeLinejoin="round" 
              strokeWidth={2} 
              d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" 
            />
          </svg>
          <span className="text-sm font-medium text-gray-700">Metadata</span>
        </div>

        {/* Quick Summary */}
        <div className="flex items-center gap-3">
          {/* Ollama Model Badge */}
          <div className="flex items-center gap-1 px-2 py-1 bg-blue-100 rounded-full">
            <svg 
              className="w-3 h-3 text-blue-600" 
              fill="none" 
              stroke="currentColor" 
              viewBox="0 0 24 24"
            >
              <path 
                strokeLinecap="round" 
                strokeLinejoin="round" 
                strokeWidth={2} 
                d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" 
              />
            </svg>
            <span className="text-xs text-blue-700 font-medium">
              {getModelDisplayName(metadata.ollamaModel)}
            </span>
          </div>

          {/* Judge Count Badge */}
          <div className="flex items-center gap-1 px-2 py-1 bg-purple-100 rounded-full">
            <svg 
              className="w-3 h-3 text-purple-600" 
              fill="none" 
              stroke="currentColor" 
              viewBox="0 0 24 24"
            >
              <path 
                strokeLinecap="round" 
                strokeLinejoin="round" 
                strokeWidth={2} 
                d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z" 
              />
            </svg>
            <span className="text-xs text-purple-700 font-medium">
              {metadata.judgeModels.length} judge{metadata.judgeModels.length !== 1 ? 's' : ''}
            </span>
          </div>

          {/* Expand Icon */}
          <svg
            className={`w-4 h-4 text-gray-400 transition-transform ${isExpanded ? 'rotate-180' : ''}`}
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
          </svg>
        </div>
      </button>


      {/* Expanded Details */}
      {isExpanded && (
        <div 
          id="metadata-details"
          className="px-4 py-3 border-t border-gray-200 space-y-4"
        >
          {/* Ollama Model Section */}
          <div>
            <h4 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-2">
              Generation Model
            </h4>
            <div className="bg-white rounded-lg border border-gray-200 p-3">
              <div className="flex items-center gap-3">
                {/* Model Icon */}
                <div className="w-10 h-10 rounded-full bg-blue-100 flex items-center justify-center">
                  <svg 
                    className="w-5 h-5 text-blue-600" 
                    fill="none" 
                    stroke="currentColor" 
                    viewBox="0 0 24 24"
                  >
                    <path 
                      strokeLinecap="round" 
                      strokeLinejoin="round" 
                      strokeWidth={2} 
                      d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" 
                    />
                  </svg>
                </div>
                <div>
                  <div className="text-sm font-medium text-gray-900">
                    {getModelDisplayName(metadata.ollamaModel)}
                  </div>
                  <div className="text-xs text-gray-500">
                    Ollama{metadata.ollamaVersion ? ` v${metadata.ollamaVersion}` : ''}
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Judge Models Section */}
          <div>
            <h4 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-2">
              Judge Models ({metadata.judgeModels.length})
            </h4>
            <div className="bg-white rounded-lg border border-gray-200 p-3">
              <div className="flex flex-wrap gap-2">
                {metadata.judgeModels.map((judge, idx) => (
                  <div 
                    key={idx}
                    className="flex items-center gap-2 px-3 py-1.5 bg-purple-50 rounded-full border border-purple-200"
                  >
                    <svg 
                      className="w-4 h-4 text-purple-600" 
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
                    <span className="text-sm text-purple-800 font-medium">
                      {getModelDisplayName(judge)}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Timestamps Section */}
          <div>
            <h4 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-2">
              Timestamps
            </h4>
            <div className="bg-white rounded-lg border border-gray-200 p-3 space-y-2">
              {/* Generation Timestamp */}
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <svg 
                    className="w-4 h-4 text-gray-400" 
                    fill="none" 
                    stroke="currentColor" 
                    viewBox="0 0 24 24"
                  >
                    <path 
                      strokeLinecap="round" 
                      strokeLinejoin="round" 
                      strokeWidth={2} 
                      d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" 
                    />
                  </svg>
                  <span className="text-sm text-gray-600">Generation</span>
                </div>
                <span className="text-sm text-gray-900 font-medium">
                  {formatTimestamp(metadata.generationTimestamp)}
                </span>
              </div>
              
              {/* Evaluation Timestamp */}
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <svg 
                    className="w-4 h-4 text-gray-400" 
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
                  <span className="text-sm text-gray-600">Evaluation</span>
                </div>
                <span className="text-sm text-gray-900 font-medium">
                  {formatTimestamp(metadata.evaluationTimestamp)}
                </span>
              </div>
            </div>
          </div>


          {/* Generation Parameters Section (if available) */}
          {metadata.generationParams && (
            <div>
              <h4 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-2">
                Generation Parameters
              </h4>
              <div className="bg-white rounded-lg border border-gray-200 p-3">
                <div className="grid grid-cols-3 gap-4">
                  {metadata.generationParams.temperature !== undefined && (
                    <div className="text-center">
                      <div className="text-xs text-gray-500">Temperature</div>
                      <div className="text-sm font-medium text-gray-900">
                        {metadata.generationParams.temperature.toFixed(2)}
                      </div>
                    </div>
                  )}
                  {metadata.generationParams.topP !== undefined && (
                    <div className="text-center">
                      <div className="text-xs text-gray-500">Top-P</div>
                      <div className="text-sm font-medium text-gray-900">
                        {metadata.generationParams.topP.toFixed(2)}
                      </div>
                    </div>
                  )}
                  {metadata.generationParams.maxTokens !== undefined && (
                    <div className="text-center">
                      <div className="text-xs text-gray-500">Max Tokens</div>
                      <div className="text-sm font-medium text-gray-900">
                        {metadata.generationParams.maxTokens}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default MetadataPanel;
