import React, { useState, useEffect } from 'react';
import { getOllamaModels, checkOllamaHealth, OllamaModel } from '../../api/ollama';

interface OllamaModelSelectorProps {
  selectedModel: string;
  onModelChange: (model: string) => void;
}

const OllamaModelSelector: React.FC<OllamaModelSelectorProps> = ({
  selectedModel,
  onModelChange,
}) => {
  const [models, setModels] = useState<OllamaModel[]>([]);
  const [loading, setLoading] = useState(true);
  const [ollamaAvailable, setOllamaAvailable] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadModels();
  }, []);

  const loadModels = async () => {
    setLoading(true);
    setError(null);

    try {
      // First check if Ollama is available
      const health = await checkOllamaHealth();
      setOllamaAvailable(health.ollama_available);

      if (health.ollama_available) {
        const response = await getOllamaModels();
        setModels(response.models);
        
        // If no model is selected and models are available, select the first one
        if (!selectedModel && response.models.length > 0) {
          onModelChange(response.models[0].name);
        }
      }
    } catch (err: any) {
      console.error('Failed to load Ollama models:', err);
      setOllamaAvailable(false);
      setError(err.response?.data?.detail?.message || 'Failed to connect to Ollama');
    } finally {
      setLoading(false);
    }
  };

  const formatSize = (bytes: number | null): string => {
    if (!bytes) return '';
    const gb = bytes / (1024 * 1024 * 1024);
    return `${gb.toFixed(1)} GB`;
  };

  if (loading) {
    return (
      <div className="space-y-3">
        <div>
          <h3 className="text-lg font-semibold text-gray-900">Ollama Model</h3>
          <p className="text-sm text-gray-600">Select the local LLM for response generation</p>
        </div>
        <div className="flex items-center space-x-2 text-gray-500">
          <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-600"></div>
          <span className="text-sm">Checking Ollama status...</span>
        </div>
      </div>
    );
  }

  if (!ollamaAvailable) {
    return (
      <div className="space-y-3">
        <div>
          <h3 className="text-lg font-semibold text-gray-900">Ollama Model</h3>
          <p className="text-sm text-gray-600">Select the local LLM for response generation</p>
        </div>
        <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
          <div className="flex items-start">
            <svg
              className="w-5 h-5 text-yellow-600 mt-0.5 flex-shrink-0"
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
            <div className="ml-3">
              <h4 className="text-sm font-medium text-yellow-800">Ollama Not Available</h4>
              <p className="text-sm text-yellow-700 mt-1">
                {error || 'Ollama is not running or not installed.'}
              </p>
              <div className="mt-3 space-y-2">
                <p className="text-sm text-yellow-700 font-medium">To get started:</p>
                <ol className="text-sm text-yellow-700 list-decimal list-inside space-y-1">
                  <li>
                    Install Ollama from{' '}
                    <a
                      href="https://ollama.ai"
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-yellow-800 underline hover:text-yellow-900"
                    >
                      ollama.ai
                    </a>
                  </li>
                  <li>Start Ollama with: <code className="bg-yellow-100 px-1 rounded">ollama serve</code></li>
                  <li>Pull a model: <code className="bg-yellow-100 px-1 rounded">ollama pull llama3.2</code></li>
                </ol>
              </div>
              <button
                onClick={loadModels}
                className="mt-3 text-sm text-yellow-800 hover:text-yellow-900 underline"
              >
                Retry connection
              </button>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-3">
      <div>
        <h3 className="text-lg font-semibold text-gray-900">Ollama Model</h3>
        <p className="text-sm text-gray-600">Select the local LLM for response generation</p>
      </div>

      {/* Connection status */}
      <div className="flex items-center space-x-2 text-green-600">
        <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
          <path
            fillRule="evenodd"
            d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z"
            clipRule="evenodd"
          />
        </svg>
        <span className="text-sm">Ollama connected</span>
      </div>

      {models.length === 0 ? (
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
          <div className="flex items-start">
            <svg
              className="w-5 h-5 text-blue-600 mt-0.5 flex-shrink-0"
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
            <div className="ml-3">
              <h4 className="text-sm font-medium text-blue-800">No Models Found</h4>
              <p className="text-sm text-blue-700 mt-1">
                Ollama is running but no models are installed.
              </p>
              <p className="text-sm text-blue-700 mt-2">
                Pull a model with: <code className="bg-blue-100 px-1 rounded">ollama pull llama3.2</code>
              </p>
              <button
                onClick={loadModels}
                className="mt-2 text-sm text-blue-800 hover:text-blue-900 underline"
              >
                Refresh models
              </button>
            </div>
          </div>
        </div>
      ) : (
        <div className="space-y-2">
          <select
            value={selectedModel}
            onChange={(e) => onModelChange(e.target.value)}
            className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 bg-white"
          >
            {models.map((model) => (
              <option key={model.name} value={model.name}>
                {model.name} {model.size ? `(${formatSize(model.size)})` : ''}
              </option>
            ))}
          </select>
          <div className="flex items-center justify-between text-sm text-gray-500">
            <span>{models.length} model{models.length !== 1 ? 's' : ''} available</span>
            <button
              onClick={loadModels}
              className="text-blue-600 hover:text-blue-700 underline"
            >
              Refresh
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default OllamaModelSelector;
