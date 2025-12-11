// API Request/Response Types

export interface LoginRequest {
  username: string;
  password: string;
}

/**
 * Metadata for an evaluation including model info and timestamps
 * Requirements: 10.1, 10.2, 10.3
 */
export interface EvaluationMetadata {
  /** Ollama model name used for generation */
  ollamaModel: string;
  /** Ollama model version (if available) */
  ollamaVersion?: string;
  /** List of judge models used for evaluation */
  judgeModels: string[];
  /** Timestamp when generation started */
  generationTimestamp?: string;
  /** Timestamp when evaluation completed */
  evaluationTimestamp?: string;
  /** Generation parameters (optional) */
  generationParams?: {
    temperature?: number;
    topP?: number;
    maxTokens?: number;
  };
}

export interface LoginResponse {
  access_token: string;
  token_type: string;
  user: {
    id: string;
    username: string;
    email: string;
  };
}

export interface EvaluationRequest {
  source_text: string;
  candidate_output: string;
  config: {
    judge_models: string[];
    enable_retrieval: boolean;
    aggregation_strategy: string;
  };
}

export interface EvaluationResponse {
  session_id: string;
  websocket_url: string;
  status: 'pending' | 'completed' | 'failed';
}

export interface JudgeResult {
  id?: string;
  judge_name: string;
  score: number;
  confidence: number;
  reasoning?: string;
  flagged_issues: FlaggedIssue[];
  response_time_ms?: number;
  created_at?: string;
}

export interface FlaggedIssue {
  id?: string;
  issue_type: string;
  severity: string;
  description: string;
  evidence?: Record<string, any>;
  text_span_start?: number;
  text_span_end?: number;
}

export interface VerifierVerdict {
  claim_text: string;
  label: string;
  confidence: number;
  evidence: string[];
  reasoning: string;
}

export type ClaimVerdictType = 'SUPPORTED' | 'REFUTED' | 'NOT_ENOUGH_INFO';
export type ClaimType = 'numerical' | 'temporal' | 'definitional' | 'general';

export interface ClaimVerdict {
  id?: string;
  evaluation_id?: string;
  claim_text: string;
  claim_type: ClaimType;
  verdict: ClaimVerdictType;
  confidence: number;
  judge_name: string;
  text_span_start: number;
  text_span_end: number;
  reasoning?: string;
}

export interface ConfidenceMetrics {
  mean_confidence: number;
  confidence_interval: [number, number];
  confidence_level: number;
  is_low_confidence: boolean;
}

export interface InterJudgeAgreement {
  cohens_kappa?: number;
  fleiss_kappa?: number;
  krippendorff_alpha?: number;
  pairwise_correlations: Record<string, Record<string, number>>;
  interpretation: string;
}

export interface HallucinationMetrics {
  overall_score: number;
  breakdown_by_type: Record<string, number>;
  affected_text_spans: Array<[number, number, string]>;
  severity_distribution: Record<string, number>;
}

export interface EvaluationResults {
  session_id: string;
  consensus_score: number;
  judge_results: JudgeResult[];
  verifier_verdicts: VerifierVerdict[];
  confidence_metrics: ConfidenceMetrics;
  inter_judge_agreement: InterJudgeAgreement;
  hallucination_metrics: HallucinationMetrics;
  variance: number;
  standard_deviation: number;
  processing_time_ms: number;
  timestamp: string;
}

export interface SessionMetadata {
  total_judges: number;
  judges_used: string[];
  aggregation_strategy?: string;
  retrieval_enabled: boolean;
  num_retrieved_passages?: number;
  num_verifier_verdicts?: number;
  processing_time_ms?: number;
  variance?: number;
  standard_deviation?: number;
}

export interface EvaluationSession {
  id: string;
  user_id: string;
  source_text: string;
  candidate_output: string;
  consensus_score?: number;
  hallucination_score?: number;
  confidence_interval_lower?: number;
  confidence_interval_upper?: number;
  inter_judge_agreement?: number;
  status: 'pending' | 'in_progress' | 'completed' | 'failed' | 'cancelled';
  config?: Record<string, any>;
  created_at: string;
  completed_at?: string;
  judge_results: JudgeResult[];
  verifier_verdicts: VerifierVerdict[];
  session_metadata?: SessionMetadata;
}

export interface SessionSummary {
  id: string;
  timestamp: string;
  source_preview: string;
  consensus_score: number;
  hallucination_score: number;
  status: 'completed' | 'pending' | 'failed';
}

export interface HistoryResponse {
  sessions: SessionSummary[];
  total: number;
  page: number;
  has_more: boolean;
}

export interface ApiError {
  error: string;
  message: string;
  request_id?: string;
}

export interface StatisticalMeasures {
  count: number;
  mean: number;
  median: number;
  std_dev: number;
  variance: number;
  min: number;
  max: number;
  q1: number;
  q3: number;
}

export interface JudgePerformance {
  total_evaluations: number;
  score_stats: StatisticalMeasures;
  confidence_stats: StatisticalMeasures;
  response_time_stats?: StatisticalMeasures;
}

export interface DailyTrend {
  date: string;
  count: number;
  avg_consensus_score?: number;
  avg_hallucination_score?: number;
}

export interface TrendAnalysis {
  first_half_average: number;
  second_half_average: number;
  percent_change: number;
  direction: 'improving' | 'declining' | 'stable';
}

export interface AggregateStatistics {
  total_evaluations: number;
  date_range_days: number;
  consensus_score_stats?: StatisticalMeasures;
  hallucination_score_stats?: StatisticalMeasures;
  variance_stats?: StatisticalMeasures;
  std_dev_stats?: StatisticalMeasures;
  processing_time_stats?: StatisticalMeasures;
  consensus_score_trend?: TrendAnalysis;
  judge_performance: Record<string, JudgePerformance>;
  daily_trends: DailyTrend[];
  message?: string;
}
