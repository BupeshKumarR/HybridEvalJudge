import React from 'react';
import { Link } from 'react-router-dom';
import { useAuthStore } from '../../store/authStore';

interface NavbarProps {
  onToggleSidebar: () => void;
  onToggleMobileMenu: () => void;
  showSidebarToggle?: boolean;
}

const Navbar: React.FC<NavbarProps> = ({
  onToggleSidebar,
  onToggleMobileMenu,
  showSidebarToggle = true,
}) => {
  const { user, logout } = useAuthStore();

  return (
    <nav className="bg-white shadow-sm border-b border-gray-200">
      <div className="px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          {/* Left side */}
          <div className="flex items-center">
            {showSidebarToggle && (
              <button
                onClick={onToggleSidebar}
                className="lg:hidden mr-3 p-2 rounded-md text-gray-600 hover:text-gray-900 hover:bg-gray-100 focus:outline-none focus:ring-2 focus:ring-primary-500"
                aria-label="Toggle sidebar"
              >
                <svg
                  className="h-6 w-6"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M4 6h16M4 12h16M4 18h16"
                  />
                </svg>
              </button>
            )}
            
            <Link to="/" className="flex items-center">
              <span className="text-xl font-bold text-primary-600">
                LLM Judge Auditor
              </span>
            </Link>
          </div>

          {/* Right side */}
          <div className="flex items-center space-x-4">
            {/* Desktop Navigation */}
            <div className="hidden md:flex items-center space-x-4">
              <Link
                to="/"
                className="text-gray-700 hover:text-primary-600 px-3 py-2 rounded-md text-sm font-medium"
              >
                Chat
              </Link>
              <Link
                to="/history"
                className="text-gray-700 hover:text-primary-600 px-3 py-2 rounded-md text-sm font-medium"
              >
                History
              </Link>
              <Link
                to="/settings"
                className="text-gray-700 hover:text-primary-600 px-3 py-2 rounded-md text-sm font-medium"
              >
                Settings
              </Link>
            </div>

            {/* User Menu */}
            {user && (
              <div className="flex items-center space-x-3">
                <span className="hidden sm:block text-sm text-gray-700">
                  {user.username}
                </span>
                <button
                  onClick={logout}
                  className="text-sm text-gray-700 hover:text-primary-600 px-3 py-2 rounded-md font-medium"
                >
                  Logout
                </button>
              </div>
            )}

            {/* Mobile menu button */}
            <button
              onClick={onToggleMobileMenu}
              className="md:hidden p-2 rounded-md text-gray-600 hover:text-gray-900 hover:bg-gray-100 focus:outline-none focus:ring-2 focus:ring-primary-500"
              aria-label="Toggle mobile menu"
            >
              <svg
                className="h-6 w-6"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M12 5v.01M12 12v.01M12 19v.01M12 6a1 1 0 110-2 1 1 0 010 2zm0 7a1 1 0 110-2 1 1 0 010 2zm0 7a1 1 0 110-2 1 1 0 010 2z"
                />
              </svg>
            </button>
          </div>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;
