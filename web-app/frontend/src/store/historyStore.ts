import { create } from 'zustand';

export interface HistorySession {
  id: string;
  timestamp: Date;
  sourcePreview: string;
  consensusScore: number;
  hallucinationScore: number;
  status: 'completed' | 'pending' | 'failed';
}

interface HistoryState {
  sessions: HistorySession[];
  currentPage: number;
  hasMore: boolean;
  isLoading: boolean;
  setSessions: (sessions: HistorySession[]) => void;
  addSessions: (sessions: HistorySession[]) => void;
  setCurrentPage: (page: number) => void;
  setHasMore: (hasMore: boolean) => void;
  setLoading: (isLoading: boolean) => void;
  clearHistory: () => void;
}

export const useHistoryStore = create<HistoryState>((set) => ({
  sessions: [],
  currentPage: 1,
  hasMore: true,
  isLoading: false,
  setSessions: (sessions) =>
    set({
      sessions,
    }),
  addSessions: (sessions) =>
    set((state) => ({
      sessions: [...state.sessions, ...sessions],
    })),
  setCurrentPage: (page) =>
    set({
      currentPage: page,
    }),
  setHasMore: (hasMore) =>
    set({
      hasMore,
    }),
  setLoading: (isLoading) =>
    set({
      isLoading,
    }),
  clearHistory: () =>
    set({
      sessions: [],
      currentPage: 1,
      hasMore: true,
    }),
}));
