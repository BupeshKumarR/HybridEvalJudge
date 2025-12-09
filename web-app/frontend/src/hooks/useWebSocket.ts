import { useEffect, useCallback, useState } from 'react';
import websocketService, { WebSocketEvents } from '../services/websocket';
import { useAuthStore } from '../store/authStore';

export const useWebSocket = () => {
  const [isConnected, setIsConnected] = useState(false);
  const { isAuthenticated } = useAuthStore();

  useEffect(() => {
    if (isAuthenticated) {
      // Connect when authenticated
      websocketService.connect();

      // Setup connection status listeners
      const handleConnect = () => setIsConnected(true);
      const handleDisconnect = () => setIsConnected(false);

      websocketService.on('connect', handleConnect);
      websocketService.on('disconnect', handleDisconnect);

      // Check initial connection status
      setIsConnected(websocketService.isConnected());

      return () => {
        websocketService.off('connect', handleConnect);
        websocketService.off('disconnect', handleDisconnect);
      };
    } else {
      // Disconnect when not authenticated
      websocketService.disconnect();
      setIsConnected(false);
      return undefined;
    }
  }, [isAuthenticated]);

  const emit = useCallback(
    <K extends keyof WebSocketEvents>(event: K, data: WebSocketEvents[K]) => {
      websocketService.emit(event, data);
    },
    []
  );

  const on = useCallback(
    <K extends keyof WebSocketEvents>(
      event: K,
      handler: (data: WebSocketEvents[K]) => void
    ) => {
      websocketService.on(event, handler);
    },
    []
  );

  const off = useCallback(
    <K extends keyof WebSocketEvents>(
      event: K,
      handler?: (data: WebSocketEvents[K]) => void
    ) => {
      websocketService.off(event, handler);
    },
    []
  );

  return {
    isConnected,
    emit,
    on,
    off,
  };
};

/**
 * Hook for listening to specific WebSocket events
 */
export const useWebSocketEvent = <K extends keyof WebSocketEvents>(
  event: K,
  handler: (data: WebSocketEvents[K]) => void,
  deps: any[] = []
) => {
  const { on, off } = useWebSocket();

  useEffect(() => {
    on(event, handler);

    return () => {
      off(event, handler);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [event, ...deps]);
};

/**
 * Hook for evaluation streaming
 */
export const useEvaluationStream = () => {
  const { emit, on, off, isConnected } = useWebSocket();
  const [progress, setProgress] = useState<{
    stage: string;
    progress: number;
    message: string;
  } | null>(null);
  const [judgeResults, setJudgeResults] = useState<any[]>([]);
  const [finalResults, setFinalResults] = useState<any | null>(null);
  const [error, setError] = useState<{
    error_type: string;
    message: string;
    recovery_suggestions: string[];
  } | null>(null);

  useEffect(() => {
    const handleProgress = (data: WebSocketEvents['evaluation_progress']) => {
      setProgress(data);
    };

    const handleJudgeResult = (data: WebSocketEvents['judge_result']) => {
      setJudgeResults((prev) => [...prev, data]);
    };

    const handleComplete = (data: WebSocketEvents['evaluation_complete']) => {
      setFinalResults(data);
      setProgress(null);
    };

    const handleError = (data: WebSocketEvents['evaluation_error']) => {
      setError(data);
      setProgress(null);
    };

    on('evaluation_progress', handleProgress);
    on('judge_result', handleJudgeResult);
    on('evaluation_complete', handleComplete);
    on('evaluation_error', handleError);

    return () => {
      off('evaluation_progress', handleProgress);
      off('judge_result', handleJudgeResult);
      off('evaluation_complete', handleComplete);
      off('evaluation_error', handleError);
    };
  }, [on, off]);

  const startEvaluation = useCallback(
    (data: WebSocketEvents['start_evaluation']) => {
      // Reset state
      setProgress(null);
      setJudgeResults([]);
      setFinalResults(null);
      setError(null);

      // Emit start event
      emit('start_evaluation', data);
    },
    [emit]
  );

  const reset = useCallback(() => {
    setProgress(null);
    setJudgeResults([]);
    setFinalResults(null);
    setError(null);
  }, []);

  return {
    isConnected,
    progress,
    judgeResults,
    finalResults,
    error,
    startEvaluation,
    reset,
  };
};
