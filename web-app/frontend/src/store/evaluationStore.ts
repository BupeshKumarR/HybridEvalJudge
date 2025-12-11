import { create } from 'zustand';
import { persist, createJSONStorage, StateStorage } from 'zustand/middleware';

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
  setSessionId: (sessionId: string | null) => void;
  updateConfig: (config: Partial<EvaluationConfig>) => void;
  loadPreferences: (config: EvaluationConfig) => void;
  setPreferencesLoaded: (loaded: boolean) => void;
  restoreState: () => void;
}

/**
 * Custom storage that handles Date serialization/deserialization
 * for proper state restoration on page reload.
 * 
 * Requirements: 12.4 - State restoration on browser reload
 */
const customStorage: StateStorage = {
  getItem: (name: string): string | null => {
    const str = localStorage.getItem(name);
    if (!str) return null;
    
    try {
      // Parse and restore Date objects in messages
      const parsed = JSON.parse(str);
      if (parsed.state?.messages) {
        parsed.state.messages = parsed.state.messages.map((msg: any) => ({
          ...msg,
          timestamp: new Date(msg.timestamp),
        }));
      }
      return JSON.stringify(parsed);
    } catch {
      return str;
    }
  },
  setItem: (name: string, value: string): void => {
    localStorage.setItem(name, value);
  },
  removeItem: (name: string): void => {
    localStorage.removeItem(name);
  },
};

export const useEvaluationStore = create<EvaluationState>()(
  persist(
    (set, get) => ({
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
      restoreState: () => {
        // This method is called to ensure state is properly restored
        // The persist middleware handles the actual restoration
        const state = get();
        // Ensure timestamps are Date objects after restoration
        if (state.messages.length > 0) {
          const messagesWithDates = state.messages.map((msg) => ({
            ...msg,
            timestamp: msg.timestamp instanceof Date 
              ? msg.timestamp 
              : new Date(msg.timestamp),
          }));
          set({ messages: messagesWithDates });
        }
      },
    }),
    {
      name: 'evaluation-storage',
      storage: createJSONStorage(() => customStorage),
      partialize: (state) => ({
        // Persist conversation history, evaluation results, and config
        // Requirements: 12.4 - Save conversation history and evaluation results
        currentSessionId: state.currentSessionId,
        messages: state.messages,
        config: state.config,
        preferencesLoaded: state.preferencesLoaded,
      }),
      onRehydrateStorage: () => (state) => {
        // Called when state is restored from storage
        if (state) {
          // Ensure isEvaluating is reset on page reload
          // (we don't want to restore an "evaluating" state)
          state.isEvaluating = false;
        }
      },
    }
  )
);
