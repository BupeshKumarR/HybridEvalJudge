import React, { useState } from 'react';
import { EvaluationConfig } from '../../store/evaluationStore';

interface ChatInputFormProps {
  onSubmit: (sourceText: string, candidateOutput: string) => void;
  isEvaluating: boolean;
  config: EvaluationConfig;
  onConfigChange: (config: Partial<EvaluationConfig>) => void;
}

const ChatInputForm: React.FC<ChatInputFormProps> = ({
  onSubmit,
  isEvaluating,
  config,
  onConfigChange,
}) => {
  const [sourceText, setSourceText] = useState('');
  const [candidateOutput, setCandidateOutput] = useState('');
  const [showConfig, setShowConfig] = useState(false);
  const [errors, setErrors] = useState<{
    sourceText?: string;
    candidateOutput?: string;
  }>({});

  const validateForm = (): boolean => {
    const newErrors: typeof errors = {};

    if (!sourceText.trim()) {
      newErrors.sourceText = 'Source text is required';
    } else if (sourceText.trim().length < 10) {
      newErrors.sourceText = 'Source text must be at least 10 characters';
    }

    if (!candidateOutput.trim()) {
      newErrors.candidateOutput = 'Candidate output is required';
    } else if (candidateOutput.trim().length < 10) {
      newErrors.candidateOutput =
        'Candidate output must be at least 10 characters';
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();

    if (!validateForm()) {
      return;
    }

    onSubmit(sourceText.trim(), candidateOutput.trim());

    // Clear form after submission
    setSourceText('');
    setCandidateOutput('');
    setErrors({});
  };

  const handleJudgeModelToggle = (model: string) => {
    const currentModels = config.judgeModels;
    const newModels = currentModels.includes(model)
      ? currentModels.filter((m) => m !== model)
      : [...currentModels, model];

    // Ensure at least one model is selected
    if (newModels.length > 0) {
      onConfigChange({ judgeModels: newModels });
    }
  };

  const availableJudgeModels = [
    'gpt-4',
    'gpt-3.5-turbo',
    'claude-3',
    'claude-2',
    'gemini-pro',
  ];

  const aggregationStrategies = [
    { value: 'weighted_average', label: 'Weighted Average' },
    { value: 'median', label: 'Median' },
    { value: 'majority_vote', label: 'Majority Vote' },
  ];

  return (
    <div className="bg-white border-t border-gray-200 shadow-lg">
      <form onSubmit={handleSubmit} className="p-4">
        {/* Configuration Panel */}
        <div className="mb-4">
          <button
            type="button"
            onClick={() => setShowConfig(!showConfig)}
            className="flex items-center gap-2 text-sm text-gray-600 hover:text-gray-900 transition-colors"
          >
            <svg
              className={`w-4 h-4 transition-transform ${
                showConfig ? 'rotate-90' : ''
              }`}
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M9 5l7 7-7 7"
              />
            </svg>
            <span className="font-medium">Configuration</span>
          </button>

          {showConfig && (
            <div className="mt-3 p-4 bg-gray-50 rounded-lg space-y-4">
              {/* Judge Models */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Judge Models (select at least one)
                </label>
                <div className="flex flex-wrap gap-2">
                  {availableJudgeModels.map((model) => (
                    <button
                      key={model}
                      type="button"
                      onClick={() => handleJudgeModelToggle(model)}
                      className={`px-3 py-1.5 rounded-full text-sm font-medium transition-colors ${
                        config.judgeModels.includes(model)
                          ? 'bg-blue-600 text-white'
                          : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                      }`}
                    >
                      {model}
                    </button>
                  ))}
                </div>
              </div>

              {/* Retrieval Toggle */}
              <div>
                <label className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={config.enableRetrieval}
                    onChange={(e) =>
                      onConfigChange({ enableRetrieval: e.target.checked })
                    }
                    className="w-4 h-4 text-blue-600 rounded focus:ring-blue-500"
                  />
                  <span className="text-sm font-medium text-gray-700">
                    Enable Retrieval
                  </span>
                </label>
                <p className="text-xs text-gray-500 mt-1 ml-6">
                  Use external knowledge retrieval for fact-checking
                </p>
              </div>

              {/* Aggregation Strategy */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Aggregation Strategy
                </label>
                <select
                  value={config.aggregationStrategy}
                  onChange={(e) =>
                    onConfigChange({ aggregationStrategy: e.target.value })
                  }
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                >
                  {aggregationStrategies.map((strategy) => (
                    <option key={strategy.value} value={strategy.value}>
                      {strategy.label}
                    </option>
                  ))}
                </select>
              </div>
            </div>
          )}
        </div>

        {/* Source Text Input */}
        <div className="mb-4">
          <label
            htmlFor="sourceText"
            className="block text-sm font-medium text-gray-700 mb-2"
          >
            Source Text
          </label>
          <textarea
            id="sourceText"
            value={sourceText}
            onChange={(e) => {
              setSourceText(e.target.value);
              if (errors.sourceText) {
                setErrors((prev) => ({ ...prev, sourceText: undefined }));
              }
            }}
            placeholder="Enter the original source text or reference material..."
            rows={4}
            disabled={isEvaluating}
            className={`w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none ${
              errors.sourceText
                ? 'border-red-500'
                : 'border-gray-300'
            } ${isEvaluating ? 'bg-gray-100 cursor-not-allowed' : ''}`}
          />
          {errors.sourceText && (
            <p className="mt-1 text-sm text-red-600">{errors.sourceText}</p>
          )}
        </div>

        {/* Candidate Output Input */}
        <div className="mb-4">
          <label
            htmlFor="candidateOutput"
            className="block text-sm font-medium text-gray-700 mb-2"
          >
            Candidate Output
          </label>
          <textarea
            id="candidateOutput"
            value={candidateOutput}
            onChange={(e) => {
              setCandidateOutput(e.target.value);
              if (errors.candidateOutput) {
                setErrors((prev) => ({ ...prev, candidateOutput: undefined }));
              }
            }}
            placeholder="Enter the LLM-generated output to evaluate..."
            rows={4}
            disabled={isEvaluating}
            className={`w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none ${
              errors.candidateOutput
                ? 'border-red-500'
                : 'border-gray-300'
            } ${isEvaluating ? 'bg-gray-100 cursor-not-allowed' : ''}`}
          />
          {errors.candidateOutput && (
            <p className="mt-1 text-sm text-red-600">
              {errors.candidateOutput}
            </p>
          )}
        </div>

        {/* Submit Button */}
        <div className="flex items-center justify-between">
          <div className="text-xs text-gray-500">
            {config.judgeModels.length} judge(s) selected
          </div>
          <button
            type="submit"
            disabled={isEvaluating}
            className={`px-6 py-2.5 rounded-lg font-medium transition-colors ${
              isEvaluating
                ? 'bg-gray-400 cursor-not-allowed'
                : 'bg-blue-600 hover:bg-blue-700 text-white'
            }`}
          >
            {isEvaluating ? (
              <span className="flex items-center gap-2">
                <svg
                  className="animate-spin h-5 w-5"
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
                Evaluating...
              </span>
            ) : (
              'Evaluate'
            )}
          </button>
        </div>
      </form>
    </div>
  );
};

export default ChatInputForm;
