import apiClient from './client';

export interface UserPreferences {
  user_id: string;
  default_judge_models: string[] | null;
  default_retrieval_enabled: boolean;
  default_aggregation_strategy: string;
  theme: string;
  notifications_enabled: boolean;
  updated_at: string;
}

export interface UserPreferencesUpdate {
  default_judge_models?: string[];
  default_retrieval_enabled?: boolean;
  default_aggregation_strategy?: string;
  theme?: string;
  notifications_enabled?: boolean;
}

/**
 * Get user preferences
 */
export const getUserPreferences = async (): Promise<UserPreferences> => {
  const response = await apiClient.get<UserPreferences>('/preferences');
  return response.data;
};

/**
 * Update user preferences
 */
export const updateUserPreferences = async (
  preferences: UserPreferencesUpdate
): Promise<UserPreferences> => {
  const response = await apiClient.put<UserPreferences>(
    '/preferences',
    preferences
  );
  return response.data;
};

/**
 * Reset user preferences to defaults
 */
export const resetUserPreferences = async (): Promise<UserPreferences> => {
  const response = await apiClient.post<UserPreferences>(
    '/preferences/reset'
  );
  return response.data;
};
