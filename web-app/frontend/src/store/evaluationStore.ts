import { create } from 'zustand';
import { persist } from 'zustand/middleware';

export interface EvaluationMessage {
  id: string;
  type: 'user' | 'system' | 'evaluation' | 'error';
  timestamp: Date;
  content: {
    sourceText?: string;
    candidateOutput?: string;
    results?: any;
    message?: string;
  };
}

export interface EvaluationConfig {
  judgeModels: string[];
  enableRetrieval: boolean;
  aggregationStrategy: string;
}

interface EvaluationState {
  currentSessionId: string | null;
  messages: EvaluationMessage[];
  isEvaluating: boolean;
  config: EvaluationConfig;
  preferencesLoaded: boolean;
  addMessage: (message: EvaluationMessage) => void;
  clearMessages: () => void;
  setEvaluating: (isEvaluating: boolean) => void;
  setSessionId: (sessionId: string) => void;
  updateConfig: (config: Partial<EvaluationConfig>) => void;
  loadPreferences: (config: EvaluationConfig) => void;
  setPreferencesLoaded: (loaded: boolean) => void;
}

export const useEvaluationStore = create<EvaluationState>()(
  persist(
    (set) => ({
      currentSessionId: null,
      messages: [],
      isEvaluating: false,
      preferencesLoaded: false,
      config: {
        judgeModels: ['gpt-4', 'claude-3'],
        enableRetrieval: true,
        aggregationStrategy: 'weighted_average',
      },
      addMessage: (message) =>
        set((state) => ({
          messages: [...state.messages, message],
        })),
      clearMessages: () =>
        set({
          messages: [],
          currentSessionId: null,
        }),
      setEvaluating: (isEvaluating) =>
        set({
          isEvaluating,
        }),
      setSessionId: (sessionId) =>
        set({
          currentSessionId: sessionId,
        }),
      updateConfig: (config) =>
        set((state) => ({
          config: { ...state.config, ...config },
        })),
      loadPreferences: (config) =>
        set({
          config,
          preferencesLoaded: true,
        }),
      setPreferencesLoaded: (loaded) =>
        set({
          preferencesLoaded: loaded,
        }),
    }),
    {
      name: 'evaluation-storage',
      partialize: (state) => ({
        config: state.config,
        preferencesLoaded: state.preferencesLoaded,
      }),
    }
  )
);
