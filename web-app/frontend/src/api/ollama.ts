import apiClient from './client';

export interface OllamaModel {
  name: string;
  size: number | null;
  digest: string | null;
  modified_at: string | null;
}

export interface OllamaModelsResponse {
  models: OllamaModel[];
  count: number;
}

export interface OllamaHealthResponse {
  status: string;
  ollama_available: boolean;
  host: string;
  message: string;
}

/**
 * Get list of available Ollama models
 */
export const getOllamaModels = async (): Promise<OllamaModelsResponse> => {
  const response = await apiClient.get<OllamaModelsResponse>('/ollama/models');
  return response.data;
};

/**
 * Check Ollama health status
 */
export const checkOllamaHealth = async (): Promise<OllamaHealthResponse> => {
  const response = await apiClient.get<OllamaHealthResponse>('/ollama/health');
  return response.data;
};
