import { useEffect, useState } from 'react';
import { useEvaluationStore } from '../store/evaluationStore';
import { useAuthStore } from '../store/authStore';
import { getUserPreferences } from '../api/preferences';

/**
 * Hook to load user preferences on authentication
 */
export const usePreferences = () => {
  const { isAuthenticated } = useAuthStore();
  const { loadPreferences, preferencesLoaded, setPreferencesLoaded } =
    useEvaluationStore();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const loadUserPreferences = async () => {
      // Only load if authenticated and not already loaded
      if (!isAuthenticated || preferencesLoaded) {
        return;
      }

      try {
        setLoading(true);
        setError(null);

        const preferences = await getUserPreferences();

        // Update evaluation store with user preferences
        loadPreferences({
          judgeModels:
            preferences.default_judge_models || ['gpt-4', 'claude-3'],
          enableRetrieval: preferences.default_retrieval_enabled ?? true,
          aggregationStrategy:
            preferences.default_aggregation_strategy || 'weighted_average',
        });
      } catch (err: any) {
        console.error('Failed to load user preferences:', err);
        setError('Failed to load preferences. Using defaults.');
        
        // Mark as loaded even on error to prevent infinite retries
        setPreferencesLoaded(true);
      } finally {
        setLoading(false);
      }
    };

    loadUserPreferences();
  }, [isAuthenticated, preferencesLoaded, loadPreferences, setPreferencesLoaded]);

  return { loading, error };
};
