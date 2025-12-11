import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import ConfigurationPanel, {
  ConfigurationSettings,
} from '../components/settings/ConfigurationPanel';
import { useAuthStore } from '../store/authStore';
import { getUserPreferences, updateUserPreferences } from '../api/preferences';

const SettingsPage: React.FC = () => {
  const navigate = useNavigate();
  const { user } = useAuthStore();
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState(false);
  const [preferences, setPreferences] = useState<ConfigurationSettings | null>(
    null
  );

  useEffect(() => {
    loadPreferences();
  }, []);

  const loadPreferences = async () => {
    try {
      setLoading(true);
      setError(null);
      const prefs = await getUserPreferences();
      
      // Get stored Ollama model from localStorage or use default
      const storedOllamaModel = localStorage.getItem('ollama_model') || 'llama3.2';
      const storedEnabledJudges = JSON.parse(localStorage.getItem('enabled_judges') || '["groq-llama", "gemini-flash"]');
      
      setPreferences({
        judgeModels: prefs.default_judge_models || ['gpt-4', 'claude-3'],
        enableRetrieval: prefs.default_retrieval_enabled ?? true,
        aggregationStrategy: prefs.default_aggregation_strategy || 'weighted_average',
        ollamaModel: storedOllamaModel,
        enabledJudges: storedEnabledJudges,
      });
    } catch (err: any) {
      console.error('Failed to load preferences:', err);
      setError('Failed to load preferences. Using defaults.');
      
      // Set defaults on error
      setPreferences({
        judgeModels: ['gpt-4', 'claude-3'],
        enableRetrieval: true,
        aggregationStrategy: 'weighted_average',
        ollamaModel: 'llama3.2',
        enabledJudges: ['groq-llama', 'gemini-flash'],
      });
    } finally {
      setLoading(false);
    }
  };

  const handleSave = async (config: ConfigurationSettings) => {
    try {
      setSaving(true);
      setError(null);
      setSuccess(false);

      // Save Ollama model and enabled judges to localStorage
      if (config.ollamaModel) {
        localStorage.setItem('ollama_model', config.ollamaModel);
      }
      if (config.enabledJudges) {
        localStorage.setItem('enabled_judges', JSON.stringify(config.enabledJudges));
      }

      await updateUserPreferences({
        default_judge_models: config.judgeModels,
        default_retrieval_enabled: config.enableRetrieval,
        default_aggregation_strategy: config.aggregationStrategy,
      });

      setSuccess(true);
      
      // Clear success message after 3 seconds
      setTimeout(() => setSuccess(false), 3000);
    } catch (err: any) {
      console.error('Failed to save preferences:', err);
      setError(
        err.response?.data?.message ||
          'Failed to save preferences. Please try again.'
      );
    } finally {
      setSaving(false);
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto"></div>
          <p className="mt-4 text-gray-600">Loading settings...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <button
                onClick={() => navigate('/')}
                className="text-gray-600 hover:text-gray-900 transition-colors"
              >
                <svg
                  className="w-6 h-6"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M10 19l-7-7m0 0l7-7m-7 7h18"
                  />
                </svg>
              </button>
              <h1 className="text-2xl font-bold text-gray-900">Settings</h1>
            </div>
            <div className="text-sm text-gray-600">
              Logged in as <span className="font-medium">{user?.username}</span>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Notifications */}
        {error && (
          <div className="mb-6 bg-red-50 border border-red-200 rounded-lg p-4">
            <div className="flex items-start">
              <svg
                className="w-5 h-5 text-red-600 mt-0.5"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                />
              </svg>
              <div className="ml-3">
                <p className="text-sm text-red-800">{error}</p>
              </div>
              <button
                onClick={() => setError(null)}
                className="ml-auto text-red-600 hover:text-red-800"
              >
                <svg
                  className="w-5 h-5"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
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
          </div>
        )}

        {success && (
          <div className="mb-6 bg-green-50 border border-green-200 rounded-lg p-4">
            <div className="flex items-start">
              <svg
                className="w-5 h-5 text-green-600 mt-0.5"
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
              <div className="ml-3">
                <p className="text-sm text-green-800">
                  Settings saved successfully!
                </p>
              </div>
            </div>
          </div>
        )}

        {/* Configuration Panel */}
        {preferences && (
          <ConfigurationPanel
            initialConfig={preferences}
            onSave={handleSave}
          />
        )}

        {/* Additional Info */}
        <div className="mt-6 bg-blue-50 border border-blue-200 rounded-lg p-4">
          <div className="flex items-start">
            <svg
              className="w-5 h-5 text-blue-600 mt-0.5"
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
              <p className="text-sm text-blue-800">
                These settings will be used as defaults for new evaluations. You
                can still override them on a per-evaluation basis in the chat
                interface.
              </p>
            </div>
          </div>
        </div>

        {saving && (
          <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
            <div className="bg-white rounded-lg p-6 shadow-xl">
              <div className="flex items-center space-x-3">
                <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-600"></div>
                <p className="text-gray-900">Saving preferences...</p>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default SettingsPage;
