import React from 'react';

interface JudgeConfig {
  id: string;
  name: string;
  description: string;
  enabled: boolean;
  apiKeyConfigured: boolean;
  apiKeyEnvVar: string;
}

interface JudgeConfigurationProps {
  enabledJudges: string[];
  onJudgeToggle: (judgeId: string) => void;
}

const AVAILABLE_JUDGES: JudgeConfig[] = [
  {
    id: 'groq-llama',
    name: 'Groq Llama 3.3 70B',
    description: 'Fast inference with Groq API',
    enabled: false,
    apiKeyConfigured: false,
    apiKeyEnvVar: 'GROQ_API_KEY',
  },
  {
    id: 'gemini-flash',
    name: 'Gemini 2.0 Flash',
    description: 'Google Gemini for evaluation',
    enabled: false,
    apiKeyConfigured: false,
    apiKeyEnvVar: 'GOOGLE_API_KEY',
  },
];

const JudgeConfiguration: React.FC<JudgeConfigurationProps> = ({
  enabledJudges,
  onJudgeToggle,
}) => {
  // Check API key status from environment (in real app, this would come from backend)
  const getApiKeyStatus = (_envVar: string): boolean => {
    // This would typically be fetched from the backend
    // For now, we'll show as "check backend" status
    return false;
  };

  const judges = AVAILABLE_JUDGES.map((judge) => ({
    ...judge,
    enabled: enabledJudges.includes(judge.id),
    apiKeyConfigured: getApiKeyStatus(judge.apiKeyEnvVar),
  }));

  const enabledCount = judges.filter((j) => j.enabled).length;

  return (
    <div className="space-y-3">
      <div>
        <h3 className="text-lg font-semibold text-gray-900">API Judge Configuration</h3>
        <p className="text-sm text-gray-600">
          Enable or disable API judges for evaluation
        </p>
      </div>

      <div className="space-y-3">
        {judges.map((judge) => (
          <div
            key={judge.id}
            className={`p-4 rounded-lg border-2 transition-all ${
              judge.enabled
                ? 'border-blue-600 bg-blue-50'
                : 'border-gray-200 bg-white'
            }`}
          >
            <div className="flex items-start justify-between">
              <div className="flex-1">
                <div className="flex items-center space-x-2">
                  <span className="font-medium text-gray-900">{judge.name}</span>
                  {/* API Key Status Badge */}
                  <span
                    className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-medium ${
                      judge.apiKeyConfigured
                        ? 'bg-green-100 text-green-800'
                        : 'bg-gray-100 text-gray-600'
                    }`}
                  >
                    {judge.apiKeyConfigured ? (
                      <>
                        <svg
                          className="w-3 h-3 mr-1"
                          fill="currentColor"
                          viewBox="0 0 20 20"
                        >
                          <path
                            fillRule="evenodd"
                            d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z"
                            clipRule="evenodd"
                          />
                        </svg>
                        API Key Set
                      </>
                    ) : (
                      <>
                        <svg
                          className="w-3 h-3 mr-1"
                          fill="currentColor"
                          viewBox="0 0 20 20"
                        >
                          <path
                            fillRule="evenodd"
                            d="M18 8a6 6 0 01-7.743 5.743L10 14l-1 1-1 1H6v2H2v-4l4.257-4.257A6 6 0 1118 8zm-6-4a1 1 0 100 2 2 2 0 012 2 1 1 0 102 0 4 4 0 00-4-4z"
                            clipRule="evenodd"
                          />
                        </svg>
                        Set {judge.apiKeyEnvVar}
                      </>
                    )}
                  </span>
                </div>
                <p className="text-sm text-gray-600 mt-1">{judge.description}</p>
              </div>

              {/* Toggle Switch */}
              <button
                type="button"
                onClick={() => onJudgeToggle(judge.id)}
                className={`relative inline-flex h-6 w-11 flex-shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 ${
                  judge.enabled ? 'bg-blue-600' : 'bg-gray-200'
                }`}
                role="switch"
                aria-checked={judge.enabled}
                aria-label={`Toggle ${judge.name}`}
              >
                <span
                  className={`pointer-events-none inline-block h-5 w-5 transform rounded-full bg-white shadow ring-0 transition duration-200 ease-in-out ${
                    judge.enabled ? 'translate-x-5' : 'translate-x-0'
                  }`}
                />
              </button>
            </div>
          </div>
        ))}
      </div>

      {/* Summary */}
      <div className="flex items-center justify-between text-sm">
        <span className="text-gray-600">
          {enabledCount} of {judges.length} judges enabled
        </span>
        {enabledCount === 0 && (
          <span className="text-yellow-600">
            Enable at least one judge for evaluation
          </span>
        )}
      </div>

      {/* API Key Help */}
      <div className="bg-gray-50 border border-gray-200 rounded-lg p-4 mt-4">
        <h4 className="text-sm font-medium text-gray-900 mb-2">Setting Up API Keys</h4>
        <p className="text-sm text-gray-600 mb-2">
          API keys should be configured as environment variables on the backend:
        </p>
        <ul className="text-sm text-gray-600 space-y-1">
          <li>
            <code className="bg-gray-100 px-1 rounded">GROQ_API_KEY</code> - Get from{' '}
            <a
              href="https://console.groq.com"
              target="_blank"
              rel="noopener noreferrer"
              className="text-blue-600 hover:underline"
            >
              console.groq.com
            </a>
          </li>
          <li>
            <code className="bg-gray-100 px-1 rounded">GOOGLE_API_KEY</code> - Get from{' '}
            <a
              href="https://aistudio.google.com"
              target="_blank"
              rel="noopener noreferrer"
              className="text-blue-600 hover:underline"
            >
              aistudio.google.com
            </a>
          </li>
        </ul>
      </div>
    </div>
  );
};

export default JudgeConfiguration;
