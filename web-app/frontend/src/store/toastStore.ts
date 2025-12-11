import { create } from 'zustand';
import { v4 as uuidv4 } from 'uuid';

export type ToastType = 'success' | 'error' | 'info' | 'warning';

export interface Toast {
  id: string;
  type: ToastType;
  title: string;
  message?: string;
  duration?: number;
}

interface ToastState {
  toasts: Toast[];
  addToast: (toast: Omit<Toast, 'id'>) => void;
  removeToast: (id: string) => void;
  clearToasts: () => void;
  // Convenience methods
  success: (title: string, message?: string, duration?: number) => void;
  error: (title: string, message?: string, duration?: number) => void;
  info: (title: string, message?: string, duration?: number) => void;
  warning: (title: string, message?: string, duration?: number) => void;
}

export const useToastStore = create<ToastState>((set) => ({
  toasts: [],

  addToast: (toast) => {
    const id = uuidv4();
    set((state) => ({
      toasts: [...state.toasts, { ...toast, id }],
    }));
  },

  removeToast: (id) => {
    set((state) => ({
      toasts: state.toasts.filter((toast) => toast.id !== id),
    }));
  },

  clearToasts: () => {
    set({ toasts: [] });
  },

  success: (title, message, duration = 5000) => {
    const id = uuidv4();
    set((state) => ({
      toasts: [...state.toasts, { id, type: 'success', title, message, duration }],
    }));
  },

  error: (title, message, duration = 7000) => {
    const id = uuidv4();
    set((state) => ({
      toasts: [...state.toasts, { id, type: 'error', title, message, duration }],
    }));
  },

  info: (title, message, duration = 5000) => {
    const id = uuidv4();
    set((state) => ({
      toasts: [...state.toasts, { id, type: 'info', title, message, duration }],
    }));
  },

  warning: (title, message, duration = 6000) => {
    const id = uuidv4();
    set((state) => ({
      toasts: [...state.toasts, { id, type: 'warning', title, message, duration }],
    }));
  },
}));
