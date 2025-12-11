import React from 'react';

export type PipelineStage = 'generation' | 'claim_extraction' | 'verification' | 'scoring' | 'aggregation';

export interface PipelineIndicatorProps {
  currentStage: PipelineStage | null;
  completedStages: PipelineStage[];
  progress?: number;
  message?: string;
}

interface StageConfig {
  id: PipelineStage;
  label: string;
  icon: React.ReactNode;
}

const CheckIcon: React.FC<{ className?: string }> = ({ className }) => (
  <svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
  </svg>
);

const GenerationIcon: React.FC<{ className?: string }> = ({ className }) => (
  <svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
  </svg>
);

const ClaimExtractionIcon: React.FC<{ className?: string }> = ({ className }) => (
  <svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-3 7h3m-3 4h3m-6-4h.01M9 16h.01" />
  </svg>
);

const VerificationIcon: React.FC<{ className?: string }> = ({ className }) => (
  <svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
  </svg>
);

const ScoringIcon: React.FC<{ className?: string }> = ({ className }) => (
  <svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
  </svg>
);


const AggregationIcon: React.FC<{ className?: string }> = ({ className }) => (
  <svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
  </svg>
);

const STAGES: StageConfig[] = [
  { id: 'generation', label: 'Response Generation', icon: <GenerationIcon className="w-5 h-5" /> },
  { id: 'claim_extraction', label: 'Claim Extraction', icon: <ClaimExtractionIcon className="w-5 h-5" /> },
  { id: 'verification', label: 'Fact Checking', icon: <VerificationIcon className="w-5 h-5" /> },
  { id: 'scoring', label: 'Judge Scoring', icon: <ScoringIcon className="w-5 h-5" /> },
  { id: 'aggregation', label: 'Aggregation', icon: <AggregationIcon className="w-5 h-5" /> },
];

const PipelineIndicator: React.FC<PipelineIndicatorProps> = ({
  currentStage,
  completedStages,
  progress,
  message,
}) => {
  const getStageStatus = (stageId: PipelineStage): 'completed' | 'current' | 'pending' => {
    if (completedStages.includes(stageId)) return 'completed';
    if (currentStage === stageId) return 'current';
    return 'pending';
  };

  const getStageStyles = (status: 'completed' | 'current' | 'pending') => {
    switch (status) {
      case 'completed':
        return {
          circle: 'bg-green-500 border-green-500 text-white',
          label: 'text-green-700 font-medium',
          connector: 'bg-green-500',
        };
      case 'current':
        return {
          circle: 'bg-blue-500 border-blue-500 text-white animate-pulse',
          label: 'text-blue-700 font-semibold',
          connector: 'bg-gray-300',
        };
      case 'pending':
        return {
          circle: 'bg-white border-gray-300 text-gray-400',
          label: 'text-gray-400',
          connector: 'bg-gray-300',
        };
    }
  };

  return (
    <div className="bg-white border border-gray-200 rounded-lg shadow-sm p-4">
      <div className="flex items-center justify-between mb-3">
        <h4 className="text-sm font-semibold text-gray-700">Evaluation Pipeline</h4>
        {progress !== undefined && currentStage && (
          <span className="text-xs font-medium text-blue-600">{Math.round(progress)}%</span>
        )}
      </div>
      
      {/* Pipeline stages */}
      <div className="flex items-center justify-between">
        {STAGES.map((stage, index) => {
          const status = getStageStatus(stage.id);
          const styles = getStageStyles(status);
          const isLast = index === STAGES.length - 1;

          return (
            <React.Fragment key={stage.id}>
              {/* Stage circle with icon */}
              <div className="flex flex-col items-center">
                <div
                  className={`w-10 h-10 rounded-full border-2 flex items-center justify-center transition-all duration-300 ${styles.circle}`}
                  title={stage.label}
                >
                  {status === 'completed' ? (
                    <CheckIcon className="w-5 h-5" />
                  ) : (
                    stage.icon
                  )}
                </div>
                <span className={`mt-2 text-xs text-center max-w-[80px] leading-tight ${styles.label}`}>
                  {stage.label}
                </span>
              </div>

              {/* Connector line */}
              {!isLast && (
                <div className={`flex-1 h-0.5 mx-2 transition-all duration-300 ${styles.connector}`} />
              )}
            </React.Fragment>
          );
        })}
      </div>

      {/* Progress message */}
      {message && currentStage && (
        <div className="mt-3 pt-3 border-t border-gray-100">
          <p className="text-xs text-gray-600 text-center">{message}</p>
        </div>
      )}
    </div>
  );
};

export default PipelineIndicator;
