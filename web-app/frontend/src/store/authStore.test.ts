import { useAuthStore } from './authStore';

describe('Auth Store', () => {
  beforeEach(() => {
    // Reset store state before each test
    useAuthStore.setState({
      user: null,
      token: null,
      isAuthenticated: false,
    });
  });

  describe('login', () => {
    it('should set user and token on login', () => {
      const user = {
        id: '123',
        username: 'testuser',
        email: 'test@example.com',
      };
      const token = 'test-token';

      useAuthStore.getState().login(user, token);

      const state = useAuthStore.getState();
      expect(state.user).toEqual(user);
      expect(state.token).toBe(token);
      expect(state.isAuthenticated).toBe(true);
    });
  });

  describe('logout', () => {
    it('should clear user and token on logout', () => {
      // First login
      const user = {
        id: '123',
        username: 'testuser',
        email: 'test@example.com',
      };
      useAuthStore.getState().login(user, 'test-token');

      // Then logout
      useAuthStore.getState().logout();

      const state = useAuthStore.getState();
      expect(state.user).toBeNull();
      expect(state.token).toBeNull();
      expect(state.isAuthenticated).toBe(false);
    });
  });

  describe('updateToken', () => {
    it('should update token', () => {
      const newToken = 'new-test-token';

      useAuthStore.getState().updateToken(newToken);

      const state = useAuthStore.getState();
      expect(state.token).toBe(newToken);
    });
  });
});
