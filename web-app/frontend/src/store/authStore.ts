import { create } from 'zustand';
import { persist } from 'zustand/middleware';

interface User {
  id: string;
  username: string;
  email: string;
}

interface AuthState {
  user: User | null;
  token: string | null;
  isAuthenticated: boolean;
  login: (user: User, token: string) => void;
  logout: () => void;
  updateToken: (token: string) => void;
}

export const useAuthStore = create<AuthState>()(
  persist(
    (set) => ({
      user: null,
      token: null,
      isAuthenticated: false,
      login: (user, token) =>
        set({
          user,
          token,
          isAuthenticated: true,
        }),
      logout: () => {
        // Reset auth state
        set({
          user: null,
          token: null,
          isAuthenticated: false,
        });
        
        // Reset preferences loaded flag in evaluation store
        // This will be imported dynamically to avoid circular dependencies
        import('./evaluationStore').then(({ useEvaluationStore }) => {
          useEvaluationStore.getState().setPreferencesLoaded(false);
        });
      },
      updateToken: (token) =>
        set({
          token,
        }),
    }),
    {
      name: 'auth-storage',
    }
  )
);
