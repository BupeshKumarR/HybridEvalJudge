import { useMutation, useQuery } from 'react-query';
import { authApi } from '../api/auth';
import { useAuthStore } from '../store/authStore';
import { LoginRequest } from '../api/types';

export const useLogin = () => {
  const { login } = useAuthStore();

  return useMutation(
    (credentials: LoginRequest) => authApi.login(credentials),
    {
      onSuccess: (data) => {
        login(data.user, data.access_token);
      },
      onError: (error: any) => {
        console.error('Login failed:', error);
      },
    }
  );
};

export const useRegister = () => {
  const { login } = useAuthStore();

  return useMutation(
    (data: { username: string; email: string; password: string }) =>
      authApi.register(data),
    {
      onSuccess: (data) => {
        login(data.user, data.access_token);
      },
      onError: (error: any) => {
        console.error('Registration failed:', error);
      },
    }
  );
};

export const useLogout = () => {
  const { logout } = useAuthStore();

  return useMutation(() => authApi.logout(), {
    onSuccess: () => {
      logout();
    },
  });
};

export const useCurrentUser = () => {
  const { isAuthenticated } = useAuthStore();

  return useQuery('currentUser', () => authApi.getCurrentUser(), {
    enabled: isAuthenticated,
    retry: false,
  });
};
