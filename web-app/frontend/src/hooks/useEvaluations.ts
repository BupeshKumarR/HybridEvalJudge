import { useMutation, useQuery, useQueryClient } from 'react-query';
import { evaluationsApi } from '../api/evaluations';
import { EvaluationRequest } from '../api/types';
import { useEvaluationStore } from '../store/evaluationStore';

export const useCreateEvaluation = () => {
  const { setSessionId, setEvaluating } = useEvaluationStore();

  return useMutation(
    (data: EvaluationRequest) => evaluationsApi.createEvaluation(data),
    {
      onSuccess: (data) => {
        setSessionId(data.session_id);
        setEvaluating(true);
      },
      onError: (error: any) => {
        console.error('Evaluation creation failed:', error);
        setEvaluating(false);
      },
    }
  );
};

export const useEvaluation = (sessionId: string | null) => {
  return useQuery(
    ['evaluation', sessionId],
    () => evaluationsApi.getEvaluation(sessionId!),
    {
      enabled: !!sessionId,
      retry: 3,
      retryDelay: (attemptIndex) => Math.min(1000 * 2 ** attemptIndex, 30000),
    }
  );
};

export const useEvaluationHistory = (params: {
  page?: number;
  limit?: number;
  sort_by?: string;
  filter_by_score?: string;
}) => {
  return useQuery(
    ['evaluationHistory', params],
    () => evaluationsApi.getHistory(params),
    {
      keepPreviousData: true,
      staleTime: 60000, // 1 minute
    }
  );
};

export const useExportEvaluation = () => {
  return useMutation(
    ({ sessionId, format }: { sessionId: string; format: 'json' | 'csv' | 'pdf' }) =>
      evaluationsApi.exportEvaluation(sessionId, format),
    {
      onSuccess: (blob, variables) => {
        // Create download link
        const url = window.URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = `evaluation-${variables.sessionId}.${variables.format}`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        window.URL.revokeObjectURL(url);
      },
      onError: (error: any) => {
        console.error('Export failed:', error);
      },
    }
  );
};

export const useDeleteEvaluation = () => {
  const queryClient = useQueryClient();

  return useMutation(
    (sessionId: string) => evaluationsApi.deleteEvaluation(sessionId),
    {
      onSuccess: () => {
        // Invalidate history queries to refresh the list
        queryClient.invalidateQueries('evaluationHistory');
      },
      onError: (error: any) => {
        console.error('Delete failed:', error);
      },
    }
  );
};

export const useSearchEvaluations = (query: string) => {
  return useQuery(
    ['searchEvaluations', query],
    () => evaluationsApi.searchEvaluations(query),
    {
      enabled: query.length > 0,
      staleTime: 30000, // 30 seconds
    }
  );
};
