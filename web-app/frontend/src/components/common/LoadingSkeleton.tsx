import React from 'react';

export interface LoadingSkeletonProps {
  variant?: 'text' | 'circular' | 'rectangular';
  width?: string | number;
  height?: string | number;
  className?: string;
  count?: number;
}

const LoadingSkeleton: React.FC<LoadingSkeletonProps> = ({
  variant = 'text',
  width,
  height,
  className = '',
  count = 1,
}) => {
  const getVariantClasses = () => {
    switch (variant) {
      case 'circular':
        return 'rounded-full';
      case 'rectangular':
        return 'rounded-md';
      case 'text':
        return 'rounded';
    }
  };

  const getDefaultSize = () => {
    switch (variant) {
      case 'circular':
        return { width: '40px', height: '40px' };
      case 'rectangular':
        return { width: '100%', height: '200px' };
      case 'text':
        return { width: '100%', height: '1em' };
    }
  };

  const defaultSize = getDefaultSize();
  const style = {
    width: width || defaultSize.width,
    height: height || defaultSize.height,
  };

  const skeleton = (
    <div
      className={`bg-gray-200 animate-pulse ${getVariantClasses()} ${className}`}
      style={style}
      aria-hidden="true"
    />
  );

  if (count > 1) {
    return (
      <div className="space-y-2">
        {Array.from({ length: count }).map((_, index) => (
          <div key={index}>{skeleton}</div>
        ))}
      </div>
    );
  }

  return skeleton;
};

export default LoadingSkeleton;

// Preset skeleton components
export const TextSkeleton: React.FC<{ lines?: number }> = ({ lines = 3 }) => (
  <div className="space-y-2">
    {Array.from({ length: lines }).map((_, index) => (
      <LoadingSkeleton
        key={index}
        variant="text"
        width={index === lines - 1 ? '70%' : '100%'}
      />
    ))}
  </div>
);

export const CardSkeleton: React.FC = () => (
  <div className="border rounded-lg p-4 space-y-3">
    <div className="flex items-center gap-3">
      <LoadingSkeleton variant="circular" width={40} height={40} />
      <div className="flex-1 space-y-2">
        <LoadingSkeleton variant="text" width="60%" />
        <LoadingSkeleton variant="text" width="40%" />
      </div>
    </div>
    <LoadingSkeleton variant="rectangular" height={100} />
    <TextSkeleton lines={2} />
  </div>
);

export const ChartSkeleton: React.FC = () => (
  <div className="space-y-3">
    <LoadingSkeleton variant="text" width="40%" height={24} />
    <LoadingSkeleton variant="rectangular" height={300} />
  </div>
);
