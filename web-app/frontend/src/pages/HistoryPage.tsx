import React from 'react';
import MainLayout from '../components/layout/MainLayout';

const HistoryPage: React.FC = () => {
  return (
    <MainLayout showSidebar={false}>
      <div className="container mx-auto px-4 py-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-4">
          Evaluation History
        </h1>
        <p className="text-gray-600">
          History view will be implemented in the next tasks.
        </p>
      </div>
    </MainLayout>
  );
};

export default HistoryPage;
