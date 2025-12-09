import React, { useState } from 'react';
import Navbar from './Navbar';
import Sidebar from './Sidebar';
import MobileMenu from './MobileMenu';

interface MainLayoutProps {
  children: React.ReactNode;
  showSidebar?: boolean;
}

const MainLayout: React.FC<MainLayoutProps> = ({ children, showSidebar = true }) => {
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);

  const toggleSidebar = () => {
    setIsSidebarOpen(!isSidebarOpen);
  };

  const toggleMobileMenu = () => {
    setIsMobileMenuOpen(!isMobileMenuOpen);
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <Navbar
        onToggleSidebar={toggleSidebar}
        onToggleMobileMenu={toggleMobileMenu}
        showSidebarToggle={showSidebar}
      />
      
      <div className="flex">
        {showSidebar && (
          <>
            {/* Desktop Sidebar */}
            <div className="hidden lg:block">
              <Sidebar isOpen={true} onClose={() => {}} />
            </div>
            
            {/* Mobile Sidebar */}
            <div className="lg:hidden">
              {isSidebarOpen && (
                <>
                  <div
                    className="fixed inset-0 bg-black bg-opacity-50 z-40"
                    onClick={toggleSidebar}
                  />
                  <Sidebar isOpen={isSidebarOpen} onClose={toggleSidebar} />
                </>
              )}
            </div>
          </>
        )}
        
        <main className="flex-1 overflow-auto">
          {children}
        </main>
      </div>
      
      {/* Mobile Menu */}
      <MobileMenu isOpen={isMobileMenuOpen} onClose={toggleMobileMenu} />
    </div>
  );
};

export default MainLayout;
