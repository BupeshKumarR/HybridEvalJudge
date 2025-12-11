import { io, Socket } from 'socket.io-client';
import { useAuthStore } from '../store/authStore';

// WebSocket event types
export interface WebSocketEvents {
  // Client -> Server
  start_evaluation: {
    session_id: string;
    source_text: string;
    candidate_output: string;
    config: any;
  };

  chat_message: {
    chat_session_id: string;
    question: string;
    model?: string;
  };

  join_chat_session: {
    chat_session_id: string;
  };

  leave_chat_session: {
    chat_session_id: string;
  };

  // Server -> Client
  evaluation_progress: {
    stage: 'generation' | 'claim_extraction' | 'verification' | 'scoring' | 'aggregation';
    progress: number;
    message: string;
  };

  judge_result: {
    judge_name: string;
    score: number;
    confidence: number;
    reasoning: string;
    flagged_issues: any[];
  };

  judge_verdict: {
    judge_name: string;
    score: number;
    confidence: number;
    reasoning: string;
    issues: any[];
    status?: 'available' | 'unavailable' | 'failed' | 'timeout' | 'rate_limited';
    error_message?: string;
  };

  evaluation_complete: {
    consensus_score: number;
    hallucination_score: number;
    confidence_interval: [number, number];
    inter_judge_agreement: number;
    full_results: any;
  };

  evaluation_error: {
    error_type: string;
    message: string;
    recovery_suggestions: string[];
  };

  // Ollama-specific events (Requirements 2.4, 9.3)
  generation_started: {
    chat_session_id: string;
    model: string;
  };

  stream_token: {
    token: string;
    done: boolean;
  };

  generation_complete: {
    chat_session_id: string;
    message_id: string;
    response: string;
  };

  generation_error: {
    error: 'connection_error' | 'model_not_found' | 'timeout' | 'service_error';
    message: string;
    suggestions: string[];
  };

  // Chat evaluation events
  chat_evaluation_complete: {
    chat_session_id: string;
    message_id: string;
    evaluation_id: string;
    consensus_score: number;
    hallucination_score: number;
    confidence_interval: [number, number];
    inter_judge_agreement: number;
    claim_verdicts: any[];
    status: string;
    judge_statuses?: Array<{
      judge_name: string;
      status: 'available' | 'unavailable' | 'failed' | 'timeout' | 'rate_limited';
      error_message?: string;
    }>;
  };

  chat_evaluation_error: {
    chat_session_id: string;
    message_id: string;
    error: string;
    message: string;
  };

  message_saved: {
    message_id: string;
    role: 'user' | 'assistant';
    content: string;
    created_at: string;
  };

  chat_session_joined: {
    chat_session_id: string;
    message: string;
  };

  chat_session_left: {
    chat_session_id: string;
    message: string;
  };

  connect: void;
  disconnect: void;
  connect_error: Error;
}

class WebSocketService {
  private socket: Socket | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 1000;
  private eventHandlers: Map<string, Set<Function>> = new Map();

  /**
   * Connect to WebSocket server
   */
  connect(url?: string): void {
    if (this.socket?.connected) {
      console.log('WebSocket already connected');
      return;
    }

    const wsUrl = url || process.env.REACT_APP_WS_URL || 'http://localhost:8000';
    const token = useAuthStore.getState().token;
    
    console.log('WebSocket connecting to:', wsUrl);
    console.log('Token available:', !!token);
    console.log('Token (first 20 chars):', token ? token.substring(0, 20) + '...' : 'null');

    if (!token) {
      console.error('WebSocket: No auth token available, cannot connect');
      return;
    }

    this.socket = io(wsUrl, {
      auth: {
        token,
      },
      path: '/ws/socket.io',  // Match the backend mount path
      transports: ['websocket', 'polling'],
      reconnection: true,
      reconnectionAttempts: this.maxReconnectAttempts,
      reconnectionDelay: this.reconnectDelay,
      reconnectionDelayMax: 5000,
      timeout: 20000,
    });

    this.setupEventListeners();
  }

  /**
   * Disconnect from WebSocket server
   */
  disconnect(): void {
    if (this.socket) {
      this.socket.disconnect();
      this.socket = null;
      this.reconnectAttempts = 0;
    }
  }

  /**
   * Check if connected
   */
  isConnected(): boolean {
    return this.socket?.connected || false;
  }

  /**
   * Emit event to server
   */
  emit<K extends keyof WebSocketEvents>(
    event: K,
    data: WebSocketEvents[K]
  ): void {
    console.log(`WebSocket emit: ${event}`, data);
    console.log('Socket connected:', this.socket?.connected);
    
    if (!this.socket?.connected) {
      console.error('WebSocket not connected, cannot emit:', event);
      return;
    }

    this.socket.emit(event, data);
    console.log(`WebSocket emit successful: ${event}`);
  }

  /**
   * Listen to event from server
   * Note: Handlers are stored in eventHandlers map and called via notifyHandlers
   * We do NOT add handlers directly to socket to avoid duplicate calls
   */
  on<K extends keyof WebSocketEvents>(
    event: K,
    handler: (data: WebSocketEvents[K]) => void
  ): void {
    if (!this.eventHandlers.has(event as string)) {
      this.eventHandlers.set(event as string, new Set());
    }

    this.eventHandlers.get(event as string)!.add(handler);
    // Note: Do NOT add handler directly to socket - setupEventListeners handles
    // routing events to notifyHandlers which calls all registered handlers
  }

  /**
   * Remove event listener
   */
  off<K extends keyof WebSocketEvents>(
    event: K,
    handler?: (data: WebSocketEvents[K]) => void
  ): void {
    if (handler) {
      this.eventHandlers.get(event as string)?.delete(handler);
      // Note: Do NOT remove from socket directly - handlers are managed via eventHandlers map
    } else {
      this.eventHandlers.delete(event as string);
    }
  }

  /**
   * Setup internal event listeners
   */
  private setupEventListeners(): void {
    if (!this.socket) return;

    this.socket.on('connect', () => {
      console.log('WebSocket connected');
      this.reconnectAttempts = 0;
      this.notifyHandlers('connect', undefined);
    });

    this.socket.on('disconnect', (reason) => {
      console.log('WebSocket disconnected:', reason);
      this.notifyHandlers('disconnect', undefined);

      // Attempt to reconnect if not a manual disconnect
      if (reason === 'io server disconnect') {
        // Server initiated disconnect, try to reconnect
        this.handleReconnect();
      }
    });

    this.socket.on('connect_error', (error) => {
      console.error('WebSocket connection error:', error);
      this.notifyHandlers('connect_error', error);
      this.handleReconnect();
    });

    // Setup handlers for evaluation events
    this.socket.on('evaluation_progress', (data) => {
      this.notifyHandlers('evaluation_progress', data);
    });

    this.socket.on('judge_result', (data) => {
      this.notifyHandlers('judge_result', data);
    });

    this.socket.on('judge_verdict', (data) => {
      this.notifyHandlers('judge_verdict', data);
    });

    this.socket.on('evaluation_complete', (data) => {
      this.notifyHandlers('evaluation_complete', data);
    });

    this.socket.on('evaluation_error', (data) => {
      this.notifyHandlers('evaluation_error', data);
    });

    // Chat and Ollama events (Requirements 2.4, 9.1, 9.3)
    this.socket.on('generation_started', (data) => {
      this.notifyHandlers('generation_started', data);
    });

    this.socket.on('stream_token', (data) => {
      this.notifyHandlers('stream_token', data);
    });

    this.socket.on('generation_complete', (data) => {
      this.notifyHandlers('generation_complete', data);
    });

    this.socket.on('generation_error', (data) => {
      this.notifyHandlers('generation_error', data);
    });

    this.socket.on('chat_evaluation_complete', (data) => {
      this.notifyHandlers('chat_evaluation_complete', data);
    });

    this.socket.on('chat_evaluation_error', (data) => {
      this.notifyHandlers('chat_evaluation_error', data);
    });

    this.socket.on('message_saved', (data) => {
      this.notifyHandlers('message_saved', data);
    });

    this.socket.on('chat_session_joined', (data) => {
      this.notifyHandlers('chat_session_joined', data);
    });

    this.socket.on('chat_session_left', (data) => {
      this.notifyHandlers('chat_session_left', data);
    });
  }

  /**
   * Handle reconnection logic
   */
  private handleReconnect(): void {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      console.error('Max reconnection attempts reached');
      return;
    }

    this.reconnectAttempts++;
    const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1);

    console.log(`Attempting to reconnect (${this.reconnectAttempts}/${this.maxReconnectAttempts}) in ${delay}ms`);

    setTimeout(() => {
      if (!this.socket?.connected) {
        this.socket?.connect();
      }
    }, delay);
  }

  /**
   * Notify all handlers for an event
   */
  private notifyHandlers(event: string, data: any): void {
    const handlers = this.eventHandlers.get(event);
    if (handlers) {
      handlers.forEach((handler) => handler(data));
    }
  }

  /**
   * Clear all event handlers
   */
  clearHandlers(): void {
    this.eventHandlers.clear();
    if (this.socket) {
      this.socket.removeAllListeners();
      this.setupEventListeners();
    }
  }
}

// Export singleton instance
export const websocketService = new WebSocketService();
export default websocketService;
