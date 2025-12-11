/**
 * Utility functions for exporting evaluation results in various formats
 * 
 * Requirements: 11.1, 11.2, 11.3, 11.4
 */

import apiClient from '../api/client';

/**
 * Helper function to trigger file download
 */
const downloadBlob = (blob: Blob, filename: string): void => {
  const url = window.URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  window.URL.revokeObjectURL(url);
};

/**
 * Export evaluation session as JSON
 */
export const exportAsJSON = async (sessionId: string): Promise<void> => {
  try {
    const response = await apiClient.get(`/evaluations/${sessionId}/export`, {
      params: { format: 'json' },
      responseType: 'blob',
    });

    const blob = new Blob([response.data], { type: 'application/json' });
    downloadBlob(blob, `evaluation_${sessionId}.json`);
  } catch (error) {
    console.error('Failed to export as JSON:', error);
    throw error;
  }
};

/**
 * Export evaluation session as CSV
 */
export const exportAsCSV = async (sessionId: string): Promise<void> => {
  try {
    const response = await apiClient.get(`/evaluations/${sessionId}/export`, {
      params: { format: 'csv' },
      responseType: 'blob',
    });

    const blob = new Blob([response.data], { type: 'text/csv' });
    downloadBlob(blob, `evaluation_${sessionId}.csv`);
  } catch (error) {
    console.error('Failed to export as CSV:', error);
    throw error;
  }
};

/**
 * Export evaluation session as PDF
 */
export const exportAsPDF = async (sessionId: string): Promise<void> => {
  try {
    const response = await apiClient.get(`/evaluations/${sessionId}/export`, {
      params: { format: 'pdf' },
      responseType: 'blob',
    });

    const blob = new Blob([response.data], { type: 'application/pdf' });
    downloadBlob(blob, `evaluation_${sessionId}.pdf`);
  } catch (error) {
    console.error('Failed to export as PDF:', error);
    throw error;
  }
};

/**
 * Export chat session as JSON
 * Includes all messages and their evaluations
 * 
 * Requirements: 11.1, 11.3, 11.4
 */
export const exportChatSessionAsJSON = async (sessionId: string): Promise<void> => {
  try {
    const response = await apiClient.get(`/chat/sessions/${sessionId}/export`, {
      params: { format: 'json' },
      responseType: 'blob',
    });

    const blob = new Blob([response.data], { type: 'application/json' });
    downloadBlob(blob, `chat_session_${sessionId}.json`);
  } catch (error) {
    console.error('Failed to export chat session as JSON:', error);
    throw error;
  }
};

/**
 * Export chat session as CSV
 * Includes all messages and their evaluations in tabular format
 * 
 * Requirements: 11.2, 11.3, 11.4
 */
export const exportChatSessionAsCSV = async (sessionId: string): Promise<void> => {
  try {
    const response = await apiClient.get(`/chat/sessions/${sessionId}/export`, {
      params: { format: 'csv' },
      responseType: 'blob',
    });

    const blob = new Blob([response.data], { type: 'text/csv' });
    downloadBlob(blob, `chat_session_${sessionId}.csv`);
  } catch (error) {
    console.error('Failed to export chat session as CSV:', error);
    throw error;
  }
};

/**
 * Generate shareable link for an evaluation session
 */
export const generateShareableLink = (sessionId: string): string => {
  const baseUrl = window.location.origin;
  return `${baseUrl}/shared/${sessionId}`;
};

/**
 * Copy text to clipboard
 */
export const copyToClipboard = async (text: string): Promise<boolean> => {
  try {
    if (navigator.clipboard && window.isSecureContext) {
      await navigator.clipboard.writeText(text);
      return true;
    } else {
      // Fallback for older browsers
      const textArea = document.createElement('textarea');
      textArea.value = text;
      textArea.style.position = 'fixed';
      textArea.style.left = '-999999px';
      document.body.appendChild(textArea);
      textArea.focus();
      textArea.select();
      try {
        document.execCommand('copy');
        document.body.removeChild(textArea);
        return true;
      } catch (error) {
        document.body.removeChild(textArea);
        return false;
      }
    }
  } catch (error) {
    console.error('Failed to copy to clipboard:', error);
    return false;
  }
};
