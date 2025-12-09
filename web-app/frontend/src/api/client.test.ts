import axios from 'axios';
import MockAdapter from 'axios-mock-adapter';
import { apiClient, retryRequest } from './client';
import { useAuthStore } from '../store/authStore';

// Mock the auth store
jest.mock('../store/authStore', () => ({
  useAuthStore: {
    getState: jest.fn(() => ({
      token: 'test-token',
      logout: jest.fn(),
    })),
  },
}));

describe('API Client', () => {
  let mock: MockAdapter;

  beforeEach(() => {
    mock = new MockAdapter(apiClient);
    jest.clearAllMocks();
  });

  afterEach(() => {
    mock.restore();
  });

  describe('Request Interceptor', () => {
    it('should add authorization header when token exists', async () => {
      (useAuthStore.getState as jest.Mock).mockReturnValue({
        token: 'test-token',
        logout: jest.fn(),
      });

      mock.onGet('/test').reply(200, { data: 'success' });

      await apiClient.get('/test');

      expect(mock.history.get[0].headers?.Authorization).toBe('Bearer test-token');
    });

    it('should not add authorization header when token is null', async () => {
      (useAuthStore.getState as jest.Mock).mockReturnValue({
        token: null,
        logout: jest.fn(),
      });

      mock.onGet('/test').reply(200, { data: 'success' });

      await apiClient.get('/test');

      expect(mock.history.get[0].headers?.Authorization).toBeUndefined();
    });
  });

  describe('Response Interceptor', () => {
    it('should handle 401 errors and logout', async () => {
      const logoutMock = jest.fn();
      (useAuthStore.getState as jest.Mock).mockReturnValue({
        token: 'test-token',
        logout: logoutMock,
      });

      mock.onGet('/test').reply(401);

      try {
        await apiClient.get('/test');
      } catch (error) {
        expect(logoutMock).toHaveBeenCalled();
      }
    });

    it('should handle 429 rate limit errors', async () => {
      const consoleWarnSpy = jest.spyOn(console, 'warn').mockImplementation();
      
      mock.onGet('/test').reply(429, {}, { 'retry-after': '60' });

      await expect(apiClient.get('/test')).rejects.toThrow();
      expect(consoleWarnSpy).toHaveBeenCalledWith('Rate limited. Retry after 60 seconds');

      consoleWarnSpy.mockRestore();
    });

    it('should handle 500 server errors', async () => {
      const consoleErrorSpy = jest.spyOn(console, 'error').mockImplementation();
      
      mock.onGet('/test').reply(500, { error: 'Internal server error' });

      try {
        await apiClient.get('/test');
      } catch (error) {
        expect(consoleErrorSpy).toHaveBeenCalled();
      }

      consoleErrorSpy.mockRestore();
    });
  });

  describe('retryRequest', () => {
    it('should succeed on first attempt', async () => {
      const requestFn = jest.fn().mockResolvedValue({ data: 'success' });

      const result = await retryRequest(requestFn);

      expect(result).toEqual({ data: 'success' });
      expect(requestFn).toHaveBeenCalledTimes(1);
    });

    it('should retry on 500 errors', async () => {
      const requestFn = jest.fn()
        .mockRejectedValueOnce({ response: { status: 500 } })
        .mockResolvedValueOnce({ data: 'success' });

      const result = await retryRequest(requestFn, 3, 10);

      expect(result).toEqual({ data: 'success' });
      expect(requestFn).toHaveBeenCalledTimes(2);
    });

    it('should not retry on 4xx errors', async () => {
      const error = { response: { status: 400 }, isAxiosError: true };
      const requestFn = jest.fn().mockRejectedValue(error);

      await expect(retryRequest(requestFn, 3, 10)).rejects.toEqual(error);
      expect(requestFn).toHaveBeenCalledTimes(1);
    });

    it('should throw after max retries', async () => {
      const requestFn = jest.fn().mockRejectedValue({ response: { status: 500 } });

      await expect(retryRequest(requestFn, 3, 10)).rejects.toEqual({ response: { status: 500 } });
      expect(requestFn).toHaveBeenCalledTimes(3);
    });
  });
});
