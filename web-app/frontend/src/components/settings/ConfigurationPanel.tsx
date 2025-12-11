import React, { useState, useEffect } from 'react';
import { useEvaluationStore } from '../../store/evaluationStore';
import OllamaModelSelector from './OllamaModelSelector';
import JudgeConfiguration from './JudgeConfiguration';

interface ConfigurationPanelProps {
  onSave?: (config: ConfigurationSettings) => void;
  initialConfig?: ConfigurationSettings;
}

export interface ConfigurationSettings {
  judgeModels: string[];
  enableRetrieval: boolean;
  aggregationStrategy: string;
  ollamaModel?: string;
  enabledJudges?: string[];
}

const ConfigurationPanel: React.FC<ConfigurationPanelProps> = ({
  onSave,
  initialConfig,
}) => {
  const { config: storeConfig, updateConfig } = useEvaluationStore();
  
  const [localConfig, setLocalConfig] = useState<ConfigurationSettings>(
    initialConfig || storeConfig
  );
  const [hasChanges, setHasChanges] = useState(false);

  useEffect(() => {
    if (initialConfig) {
      setLocalConfig(initialConfig);
    }
  }, [initialConfig]);

  const availableJudgeModels = [
    { id: 'gpt-4', name: 'GPT-4', description: 'OpenAI GPT-4' },
    { id: 'gpt-3.5-turbo', name: 'GPT-3.5 Turbo', description: 'OpenAI GPT-3.5' },
    { id: 'claude-3', name: 'Claude 3', description: 'Anthropic Claude 3' },
    { id: 'claude-2', name: 'Claude 2', description: 'Anthropic Claude 2' },
    { id: 'gemini-pro', name: 'Gemini Pro', description: 'Google Gemini Pro' },
  ];

  const aggregationStrategies = [
    {
      value: 'weighted_average',
      label: 'Weighted Average',
      description: 'Weights scores by judge confidence levels',
    },
    {
      value: 'median',
      label: 'Median',
      description: 'Uses the median score from all judges',
    },
    {
      value: 'majority_vote',
      label: 'Majority Vote',
      description: 'Uses the most common score category',
    },
  ];

  const handleJudgeModelToggle = (modelId: string) => {
    const newModels = localConfig.judgeModels.includes(modelId)
      ? localConfig.judgeModels.filter((m) => m !== modelId)
      : [...localConfig.judgeModels, modelId];

    // Ensure at least one model is selected
    if (newModels.length > 0) {
      setLocalConfig({ ...localConfig, judgeModels: newModels });
      setHasChanges(true);
    }
  };

  const handleRetrievalToggle = (enabled: boolean) => {
    setLocalConfig({ ...localConfig, enableRetrieval: enabled });
    setHasChanges(true);
  };

  const handleAggregationChange = (strategy: string) => {
    setLocalConfig({ ...localConfig, aggregationStrategy: strategy });
    setHasChanges(true);
  };

  const handleSave = () => {
    updateConfig(localConfig);
    if (onSave) {
      onSave(localConfig);
    }
    setHasChanges(false);
  };

  const handleReset = () => {
    setLocalConfig(initialConfig || storeConfig);
    setHasChanges(false);
  };

  return (
    <div className="bg-white rounded-lg shadow-md p-6 space-y-6">
      <div className="border-b border-gray-200 pb-4">
        <h2 className="text-2xl font-bold text-gray-900">
          Evaluation Configuration
        </h2>
        <p className="mt-1 text-sm text-gray-600">
          Customize your evaluation settings and preferences
        </p>
      </div>

      {/* Ollama Model Section - Requirements 7.1, 7.3 */}
      <OllamaModelSelector
        selectedModel={localConfig.ollamaModel || 'llama3.2'}
        onModelChange={(model) => {
          setLocalConfig({ ...localConfig, ollamaModel: model });
          setHasChanges(true);
        }}
      />

      {/* API Judge Configuration Section - Requirements 7.4, 7.5 */}
      <JudgeConfiguration
        enabledJudges={localConfig.enabledJudges || ['groq-llama', 'gemini-flash']}
        onJudgeToggle={(judgeId) => {
          const currentJudges = localConfig.enabledJudges || ['groq-llama', 'gemini-flash'];
          const newJudges = currentJudges.includes(judgeId)
            ? currentJudges.filter((j) => j !== judgeId)
            : [...currentJudges, judgeId];
          setLocalConfig({ ...localConfig, enabledJudges: newJudges });
          setHasChanges(true);
        }}
      />

      {/* Judge Models Section */}
      <div className="space-y-3">
        <div>
          <h3 className="text-lg font-semibold text-gray-900">Judge Models</h3>
          <p className="text-sm text-gray-600">
            Select one or more judge models to evaluate outputs
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          {availableJudgeModels.map((model) => (
            <button
              key={model.id}
              type="button"
              onClick={() => handleJudgeModelToggle(model.id)}
              className={`p-4 rounded-lg border-2 transition-all text-left ${
                localConfig.judgeModels.includes(model.id)
                  ? 'border-blue-600 bg-blue-50'
                  : 'border-gray-200 bg-white hover:border-gray-300'
              }`}
            >
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  <div className="font-medium text-gray-900">{model.name}</div>
                  <div className="text-sm text-gray-600 mt-1">
                    {model.description}
                  </div>
                </div>
                <div
                  className={`flex-shrink-0 ml-3 w-5 h-5 rounded-full border-2 flex items-center justify-center ${
                    localConfig.judgeModels.includes(model.id)
                      ? 'border-blue-600 bg-blue-600'
                      : 'border-gray-300'
                  }`}
                >
                  {localConfig.judgeModels.includes(model.id) && (
                    <svg
                      className="w-3 h-3 text-white"
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={3}
                        d="M5 13l4 4L19 7"
                      />
                    </svg>
                  )}
                </div>
              </div>
            </button>
          ))}
        </div>

        {localConfig.judgeModels.length === 0 && (
          <p className="text-sm text-red-600">
            Please select at least one judge model
          </p>
        )}
      </div>

      {/* Retrieval Section */}
      <div className="space-y-3">
        <div>
          <h3 className="text-lg font-semibold text-gray-900">
            Knowledge Retrieval
          </h3>
          <p className="text-sm text-gray-600">
            Enable external knowledge retrieval for fact-checking
          </p>
        </div>

        <div className="flex items-start space-x-3">
          <button
            type="button"
            onClick={() => handleRetrievalToggle(!localConfig.enableRetrieval)}
            className={`relative inline-flex h-6 w-11 flex-shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 ${
              localConfig.enableRetrieval ? 'bg-blue-600' : 'bg-gray-200'
            }`}
          >
            <span
              className={`pointer-events-none inline-block h-5 w-5 transform rounded-full bg-white shadow ring-0 transition duration-200 ease-in-out ${
                localConfig.enableRetrieval ? 'translate-x-5' : 'translate-x-0'
              }`}
            />
          </button>
          <div className="flex-1">
            <p className="text-sm font-medium text-gray-900">
              {localConfig.enableRetrieval ? 'Enabled' : 'Disabled'}
            </p>
            <p className="text-xs text-gray-500 mt-1">
              When enabled, the system will retrieve relevant passages from
              external sources to verify claims
            </p>
          </div>
        </div>
      </div>

      {/* Aggregation Strategy Section */}
      <div className="space-y-3">
        <div>
          <h3 className="text-lg font-semibold text-gray-900">
            Aggregation Strategy
          </h3>
          <p className="text-sm text-gray-600">
            Choose how to combine scores from multiple judges
          </p>
        </div>

        <div className="space-y-2">
          {aggregationStrategies.map((strategy) => (
            <button
              key={strategy.value}
              type="button"
              onClick={() => handleAggregationChange(strategy.value)}
              className={`w-full p-4 rounded-lg border-2 transition-all text-left ${
                localConfig.aggregationStrategy === strategy.value
                  ? 'border-blue-600 bg-blue-50'
                  : 'border-gray-200 bg-white hover:border-gray-300'
              }`}
            >
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  <div className="font-medium text-gray-900">
                    {strategy.label}
                  </div>
                  <div className="text-sm text-gray-600 mt-1">
                    {strategy.description}
                  </div>
                </div>
                <div
                  className={`flex-shrink-0 ml-3 w-5 h-5 rounded-full border-2 flex items-center justify-center ${
                    localConfig.aggregationStrategy === strategy.value
                      ? 'border-blue-600 bg-blue-600'
                      : 'border-gray-300'
                  }`}
                >
                  {localConfig.aggregationStrategy === strategy.value && (
                    <div className="w-2 h-2 rounded-full bg-white" />
                  )}
                </div>
              </div>
            </button>
          ))}
        </div>
      </div>

      {/* Action Buttons */}
      {hasChanges && (
        <div className="flex items-center justify-end space-x-3 pt-4 border-t border-gray-200">
          <button
            type="button"
            onClick={handleReset}
            className="px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors"
          >
            Reset
          </button>
          <button
            type="button"
            onClick={handleSave}
            disabled={localConfig.judgeModels.length === 0}
            className={`px-4 py-2 text-sm font-medium text-white rounded-lg transition-colors ${
              localConfig.judgeModels.length === 0
                ? 'bg-gray-400 cursor-not-allowed'
                : 'bg-blue-600 hover:bg-blue-700'
            }`}
          >
            Save Configuration
          </button>
        </div>
      )}
    </div>
  );
};

export default ConfigurationPanel;
