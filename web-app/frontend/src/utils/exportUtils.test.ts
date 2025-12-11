import { 
  exportAsJSON, 
  exportAsCSV, 
  exportAsPDF, 
  generateShareableLink, 
  copyToClipboard,
  exportChatSessionAsJSON,
  exportChatSessionAsCSV
} from './exportUtils';
import apiClient from '../api/client';

jest.mock('../api/client');

describe('Export Utils', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    
    // Mock DOM methods
    document.createElement = jest.fn((_tag) => {
      const element = {
        href: '',
        download: '',
        click: jest.fn(),
        style: {},
        value: '',
        focus: jest.fn(),
        select: jest.fn(),
      } as any;
      return element;
    });
    
    document.body.appendChild = jest.fn();
    document.body.removeChild = jest.fn();
    
    window.URL.createObjectURL = jest.fn(() => 'blob:mock-url');
    window.URL.revokeObjectURL = jest.fn();
  });

  describe('exportAsJSON', () => {
    it('should export evaluation as JSON', async () => {
      const mockData = { evaluation: 'data' };
      (apiClient.get as jest.Mock).mockResolvedValue({
        data: JSON.stringify(mockData),
      });

      await exportAsJSON('test-session-id');

      expect(apiClient.get).toHaveBeenCalledWith('/evaluations/test-session-id/export', {
        params: { format: 'json' },
        responseType: 'blob',
      });
    });

    it('should handle export errors', async () => {
      const consoleErrorSpy = jest.spyOn(console, 'error').mockImplementation();
      (apiClient.get as jest.Mock).mockRejectedValue(new Error('Export failed'));

      await expect(exportAsJSON('test-session-id')).rejects.toThrow('Export failed');
      expect(consoleErrorSpy).toHaveBeenCalled();

      consoleErrorSpy.mockRestore();
    });
  });

  describe('exportAsCSV', () => {
    it('should export evaluation as CSV', async () => {
      const mockData = 'csv,data';
      (apiClient.get as jest.Mock).mockResolvedValue({
        data: mockData,
      });

      await exportAsCSV('test-session-id');

      expect(apiClient.get).toHaveBeenCalledWith('/evaluations/test-session-id/export', {
        params: { format: 'csv' },
        responseType: 'blob',
      });
    });
  });

  describe('exportAsPDF', () => {
    it('should export evaluation as PDF', async () => {
      const mockData = 'pdf-data';
      (apiClient.get as jest.Mock).mockResolvedValue({
        data: mockData,
      });

      await exportAsPDF('test-session-id');

      expect(apiClient.get).toHaveBeenCalledWith('/evaluations/test-session-id/export', {
        params: { format: 'pdf' },
        responseType: 'blob',
      });
    });
  });

  describe('generateShareableLink', () => {
    it('should generate correct shareable link', () => {
      Object.defineProperty(window, 'location', {
        value: { origin: 'http://localhost:3000' },
        writable: true,
      });

      const link = generateShareableLink('test-session-id');

      expect(link).toBe('http://localhost:3000/shared/test-session-id');
    });
  });

  describe('copyToClipboard', () => {
    it('should copy text using clipboard API', async () => {
      const writeTextMock = jest.fn().mockResolvedValue(undefined);
      Object.assign(navigator, {
        clipboard: {
          writeText: writeTextMock,
        },
      });
      Object.defineProperty(window, 'isSecureContext', {
        value: true,
        writable: true,
      });

      const result = await copyToClipboard('test text');

      expect(result).toBe(true);
      expect(writeTextMock).toHaveBeenCalledWith('test text');
    });

    it('should use fallback method when clipboard API unavailable', async () => {
      Object.assign(navigator, {
        clipboard: undefined,
      });

      const execCommandMock = jest.fn().mockReturnValue(true);
      document.execCommand = execCommandMock;

      const result = await copyToClipboard('test text');

      expect(result).toBe(true);
    });

    it('should handle copy errors', async () => {
      const consoleErrorSpy = jest.spyOn(console, 'error').mockImplementation();
      Object.assign(navigator, {
        clipboard: {
          writeText: jest.fn().mockRejectedValue(new Error('Copy failed')),
        },
      });
      Object.defineProperty(window, 'isSecureContext', {
        value: true,
        writable: true,
      });

      const result = await copyToClipboard('test text');

      expect(result).toBe(false);
      expect(consoleErrorSpy).toHaveBeenCalled();

      consoleErrorSpy.mockRestore();
    });
  });

  describe('exportChatSessionAsJSON', () => {
    it('should export chat session as JSON', async () => {
      const mockData = { session: 'data', messages: [] };
      (apiClient.get as jest.Mock).mockResolvedValue({
        data: JSON.stringify(mockData),
      });

      await exportChatSessionAsJSON('test-chat-session-id');

      expect(apiClient.get).toHaveBeenCalledWith('/chat/sessions/test-chat-session-id/export', {
        params: { format: 'json' },
        responseType: 'blob',
      });
    });

    it('should handle chat session export errors', async () => {
      const consoleErrorSpy = jest.spyOn(console, 'error').mockImplementation();
      (apiClient.get as jest.Mock).mockRejectedValue(new Error('Export failed'));

      await expect(exportChatSessionAsJSON('test-chat-session-id')).rejects.toThrow('Export failed');
      expect(consoleErrorSpy).toHaveBeenCalled();

      consoleErrorSpy.mockRestore();
    });
  });

  describe('exportChatSessionAsCSV', () => {
    it('should export chat session as CSV', async () => {
      const mockData = 'csv,data';
      (apiClient.get as jest.Mock).mockResolvedValue({
        data: mockData,
      });

      await exportChatSessionAsCSV('test-chat-session-id');

      expect(apiClient.get).toHaveBeenCalledWith('/chat/sessions/test-chat-session-id/export', {
        params: { format: 'csv' },
        responseType: 'blob',
      });
    });
  });
});
