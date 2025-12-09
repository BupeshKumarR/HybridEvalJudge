/**
 * Utility functions for exporting evaluation results in various formats
 */

import { EvaluationSession } from '../api/types';
import apiClient from '../api/client';

/**
 * Export evaluation session as JSON
 */
export const exportAsJSON = async (sessionId: string): Promise<void> => {
  try {
    const response = await apiClient.get(`/evaluations/${sessionId}/export`, {
      params: { format: 'json' },
      responseType: 'blob',
    });

    // Create blob and download
    const blob = new Blob([response.data], { type: 'application/json' });
    const url = window.URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `evaluation_${sessionId}.json`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    window.URL.revokeObjectURL(url);
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

    // Create blob and download
    const blob = new Blob([response.data], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `evaluation_${sessionId}.csv`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    window.URL.revokeObjectURL(url);
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

    // Create blob and download
    const blob = new Blob([response.data], { type: 'application/pdf' });
    const url = window.URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `evaluation_${sessionId}.pdf`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    window.URL.revokeObjectURL(url);
  } catch (error) {
    console.error('Failed to export as PDF:', error);
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
