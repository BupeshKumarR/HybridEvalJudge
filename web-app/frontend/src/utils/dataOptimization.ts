/**
 * Utility functions for optimizing large datasets in visualizations
 */

/**
 * Sample data points for large datasets to improve rendering performance
 * Uses reservoir sampling algorithm for uniform distribution
 */
export function sampleData<T>(data: T[], maxPoints: number = 100): T[] {
  if (data.length <= maxPoints) {
    return data;
  }

  const sampled: T[] = [];
  
  // Always include first and last points for continuity
  sampled.push(data[0]);
  
  // Reservoir sampling for middle points
  const step = Math.floor(data.length / (maxPoints - 2));
  for (let i = step; i < data.length - 1; i += step) {
    sampled.push(data[i]);
  }
  
  sampled.push(data[data.length - 1]);
  
  return sampled;
}

/**
 * Downsample time series data while preserving trends
 * Uses Largest Triangle Three Buckets (LTTB) algorithm
 */
export function downsampleTimeSeries<T extends { x: number; y: number }>(
  data: T[],
  threshold: number = 100
): T[] {
  if (data.length <= threshold) {
    return data;
  }

  const sampled: T[] = [];
  
  // Always add first point
  sampled.push(data[0]);
  
  // Bucket size
  const bucketSize = (data.length - 2) / (threshold - 2);
  
  let a = 0; // Initially a is the first point in the triangle
  
  for (let i = 0; i < threshold - 2; i++) {
    // Calculate point average for next bucket
    let avgX = 0;
    let avgY = 0;
    const avgRangeStart = Math.floor((i + 1) * bucketSize) + 1;
    const avgRangeEnd = Math.floor((i + 2) * bucketSize) + 1;
    const avgRangeLength = avgRangeEnd - avgRangeStart;
    
    for (let j = avgRangeStart; j < avgRangeEnd; j++) {
      avgX += data[j].x;
      avgY += data[j].y;
    }
    avgX /= avgRangeLength;
    avgY /= avgRangeLength;
    
    // Get the range for this bucket
    const rangeOffs = Math.floor(i * bucketSize) + 1;
    const rangeTo = Math.floor((i + 1) * bucketSize) + 1;
    
    // Point a
    const pointAX = data[a].x;
    const pointAY = data[a].y;
    
    let maxArea = -1;
    let maxAreaPoint = 0;
    
    for (let j = rangeOffs; j < rangeTo; j++) {
      // Calculate triangle area
      const area = Math.abs(
        (pointAX - avgX) * (data[j].y - pointAY) -
        (pointAX - data[j].x) * (avgY - pointAY)
      ) * 0.5;
      
      if (area > maxArea) {
        maxArea = area;
        maxAreaPoint = j;
      }
    }
    
    sampled.push(data[maxAreaPoint]);
    a = maxAreaPoint;
  }
  
  // Always add last point
  sampled.push(data[data.length - 1]);
  
  return sampled;
}

/**
 * Aggregate data into bins for histogram-style visualizations
 */
export function binData(
  data: number[],
  numBins: number = 10
): Array<{ min: number; max: number; count: number; values: number[] }> {
  if (data.length === 0) {
    return [];
  }

  const min = Math.min(...data);
  const max = Math.max(...data);
  const binSize = (max - min) / numBins;
  
  const bins: Array<{ min: number; max: number; count: number; values: number[] }> = [];
  
  for (let i = 0; i < numBins; i++) {
    const binMin = min + i * binSize;
    const binMax = i === numBins - 1 ? max : binMin + binSize;
    
    bins.push({
      min: binMin,
      max: binMax,
      count: 0,
      values: []
    });
  }
  
  // Assign data points to bins
  data.forEach(value => {
    const binIndex = Math.min(
      Math.floor((value - min) / binSize),
      numBins - 1
    );
    bins[binIndex].count++;
    bins[binIndex].values.push(value);
  });
  
  return bins;
}

/**
 * Throttle function calls for performance
 */
export function throttle<T extends (...args: any[]) => any>(
  func: T,
  delay: number
): (...args: Parameters<T>) => void {
  let lastCall = 0;
  let timeoutId: NodeJS.Timeout | null = null;
  
  return function(...args: Parameters<T>) {
    const now = Date.now();
    
    if (now - lastCall >= delay) {
      lastCall = now;
      func(...args);
    } else {
      if (timeoutId) {
        clearTimeout(timeoutId);
      }
      timeoutId = setTimeout(() => {
        lastCall = Date.now();
        func(...args);
      }, delay - (now - lastCall));
    }
  };
}

/**
 * Debounce function calls for performance
 */
export function debounce<T extends (...args: any[]) => any>(
  func: T,
  delay: number
): (...args: Parameters<T>) => void {
  let timeoutId: NodeJS.Timeout | null = null;
  
  return function(...args: Parameters<T>) {
    if (timeoutId) {
      clearTimeout(timeoutId);
    }
    
    timeoutId = setTimeout(() => {
      func(...args);
    }, delay);
  };
}

/**
 * Check if dataset is large and needs optimization
 */
export function needsOptimization(dataSize: number, threshold: number = 100): boolean {
  return dataSize > threshold;
}

/**
 * Calculate optimal sample size based on container dimensions
 */
export function calculateOptimalSampleSize(
  dataSize: number,
  containerWidth: number,
  pixelsPerPoint: number = 2
): number {
  const maxPoints = Math.floor(containerWidth / pixelsPerPoint);
  return Math.min(dataSize, maxPoints);
}
