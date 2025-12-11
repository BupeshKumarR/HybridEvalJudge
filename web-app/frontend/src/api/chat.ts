import apiClient from './client';

/**
 * Chat session and message types
 */
export interface ChatSession {
  id: string;
  user_id: string;
  ollama_model: string;
  created_at: string;
  updated_at: string;
}

/**
 * Chat session summary returned by list endpoint
 */
export interface ChatSessionSummary {
  id: string;
  ollama_model: string;
  created_at: string;
  updated_at: string;
  message_count: number;
  last_message_preview: string | null;
}

export interface ChatMessage {
  id: string;
  session_id: string;
  role: 'user' | 'assistant';
  content: string;
  created_at: string;
  evaluation_id?: string;
}

export interface CreateSessionRequest {
  ollama_model: string;
}

export interface CreateMessageRequest {
  role: 'user' | 'assistant';
  content: string;
  evaluation_id?: string;
}

export interface ChatMessagesResponse {
  messages: ChatMessage[];
  total: number;
  has_more: boolean;
}

/**
 * Chat API functions for session and message management
 * Requirements: 6.1, 6.2, 6.3
 */
export const chatApi = {
  /**
   * Create a new chat session
   */
  createSession: async (data: CreateSessionRequest): Promise<ChatSession> => {
    const response = await apiClient.post<ChatSession>('/chat/sessions', data);
    return response.data;
  },

  /**
   * Get a chat session by ID
   */
  getSession: async (sessionId: string): Promise<ChatSession> => {
    const response = await apiClient.get<ChatSession>(`/chat/sessions/${sessionId}`);
    return response.data;
  },

  /**
   * Get messages for a chat session
   */
  getMessages: async (
    sessionId: string,
    limit: number = 50,
    offset: number = 0
  ): Promise<ChatMessagesResponse> => {
    // Backend returns array directly, wrap it in expected format
    const response = await apiClient.get<ChatMessage[]>(
      `/chat/sessions/${sessionId}/messages`,
      { params: { limit, offset } }
    );
    const messages = response.data;
    return {
      messages,
      total: messages.length,
      has_more: false,
    };
  },

  /**
   * Add a message to a chat session
   */
  addMessage: async (
    sessionId: string,
    data: CreateMessageRequest
  ): Promise<ChatMessage> => {
    const response = await apiClient.post<ChatMessage>(
      `/chat/sessions/${sessionId}/messages`,
      data
    );
    return response.data;
  },

  /**
   * Delete a chat session
   */
  deleteSession: async (sessionId: string): Promise<void> => {
    await apiClient.delete(`/chat/sessions/${sessionId}`);
  },

  /**
   * List all chat sessions for the current user
   */
  listSessions: async (
    limit: number = 20,
    page: number = 1
  ): Promise<{ sessions: ChatSessionSummary[]; total: number; has_more: boolean }> => {
    const response = await apiClient.get('/chat/sessions', {
      params: { limit, page },
    });
    return response.data;
  },
};

export default chatApi;
