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

  // Server -> Client
  evaluation_progress: {
    stage: 'retrieval' | 'verification' | 'judging' | 'aggregation';
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

    this.socket = io(wsUrl, {
      auth: {
        token,
      },
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
    if (!this.socket?.connected) {
      console.error('WebSocket not connected');
      return;
    }

    this.socket.emit(event, data);
  }

  /**
   * Listen to event from server
   */
  on<K extends keyof WebSocketEvents>(
    event: K,
    handler: (data: WebSocketEvents[K]) => void
  ): void {
    if (!this.eventHandlers.has(event as string)) {
      this.eventHandlers.set(event as string, new Set());
    }

    this.eventHandlers.get(event as string)!.add(handler);

    if (this.socket) {
      this.socket.on(event as string, handler);
    }
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
      if (this.socket) {
        this.socket.off(event as string, handler);
      }
    } else {
      this.eventHandlers.delete(event as string);
      if (this.socket) {
        this.socket.off(event as string);
      }
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

    this.socket.on('evaluation_complete', (data) => {
      this.notifyHandlers('evaluation_complete', data);
    });

    this.socket.on('evaluation_error', (data) => {
      this.notifyHandlers('evaluation_error', data);
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
