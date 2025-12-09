import apiClient, { retryRequest } from './client';
import {
  EvaluationRequest,
  EvaluationResponse,
  EvaluationSession,
  HistoryResponse,
  AggregateStatistics,
} from './types';

export const evaluationsApi = {
  /**
   * Create a new evaluation
   */
  createEvaluation: async (data: EvaluationRequest): Promise<EvaluationResponse> => {
    const response = await retryRequest(
      () => apiClient.post<EvaluationResponse>('/evaluations', data),
      2 // Retry up to 2 times
    );
    return response.data;
  },

  /**
   * Get evaluation results by session ID
   */
  getEvaluation: async (sessionId: string): Promise<EvaluationSession> => {
    const response = await apiClient.get<EvaluationSession>(`/evaluations/${sessionId}`);
    return response.data;
  },

  /**
   * Get evaluation history with pagination
   */
  getHistory: async (params: {
    page?: number;
    limit?: number;
    sort_by?: string;
    filter_by_score?: string;
  }): Promise<HistoryResponse> => {
    const response = await apiClient.get<HistoryResponse>('/evaluations', {
      params,
    });
    return response.data;
  },

  /**
   * Export evaluation results
   */
  exportEvaluation: async (
    sessionId: string,
    format: 'json' | 'csv' | 'pdf'
  ): Promise<Blob> => {
    const response = await apiClient.get(`/evaluations/${sessionId}/export`, {
      params: { format },
      responseType: 'blob',
    });
    return response.data;
  },

  /**
   * Delete an evaluation
   */
  deleteEvaluation: async (sessionId: string): Promise<void> => {
    await apiClient.delete(`/evaluations/${sessionId}`);
  },

  /**
   * Search evaluations
   */
  searchEvaluations: async (query: string): Promise<HistoryResponse> => {
    const response = await apiClient.get<HistoryResponse>('/evaluations/search', {
      params: { q: query },
    });
    return response.data;
  },

  /**
   * Get aggregate statistics across sessions
   */
  getAggregateStatistics: async (days: number = 30): Promise<AggregateStatistics> => {
    const response = await apiClient.get<AggregateStatistics>('/evaluations/statistics/aggregate', {
      params: { days },
    });
    return response.data;
  },
};
