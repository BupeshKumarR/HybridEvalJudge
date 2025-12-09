import React, { useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import apiClient from '../api/client';
import { EvaluationSession } from '../api/types';
import EvaluationResultMessage from '../components/chat/EvaluationResultMessage';

const SharedEvaluationView: React.FC = () => {
  const { sessionId } = useParams<{ sessionId: string }>();
  const navigate = useNavigate();
  const [evaluation, setEvaluation] = useState<EvaluationSession | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchSharedEvaluation = async () => {
      if (!sessionId) {
        setError('Invalid session ID');
        setLoading(false);
        return;
      }

      try {
        const response = await apiClient.get(`/evaluations/shared/${sessionId}`);
        setEvaluation(response.data);
      } catch (err: any) {
        console.error('Failed to fetch shared evaluation:', err);
        if (err.response?.status === 404) {
          setError('Evaluation not found');
        } else if (err.response?.status === 403) {
          setError('This evaluation is not available for sharing');
        } else {
          setError('Failed to load evaluation');
        }
      } finally {
        setLoading(false);
      }
    };

    fetchSharedEvaluation();
  }, [sessionId]);

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <svg
            className="w-12 h-12 mx-auto mb-4 text-blue-600 animate-spin"
            fill="none"
            viewBox="0 0 24 24"
          >
            <circle
              className="opacity-25"
              cx="12"
              cy="12"
              r="10"
              stroke="currentColor"
              strokeWidth="4"
            />
            <path
              className="opacity-75"
              fill="currentColor"
              d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
            />
          </svg>
          <p className="text-gray-600">Loading evaluation...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="max-w-md w-full bg-white rounded-lg shadow-md p-6">
          <div className="flex items-center gap-3 mb-4">
            <svg
              className="w-12 h-12 text-red-500"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
              />
            </svg>
            <div>
              <h2 className="text-xl font-semibold text-gray-900">
                Unable to Load Evaluation
              </h2>
              <p className="text-gray-600 mt-1">{error}</p>
            </div>
          </div>
          <button
            onClick={() => navigate('/')}
            className="w-full px-4 py-2 text-sm font-medium text-white bg-blue-600 rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
          >
            Go to Home
          </button>
        </div>
      </div>
    );
  }

  if (!evaluation) {
    return null;
  }

  // Convert evaluation session to results format for display
  const results = {
    session_id: evaluation.id,
    consensus_score: evaluation.consensus_score || 0,
    judge_results: evaluation.judge_results,
    verifier_verdicts: evaluation.verifier_verdicts,
    confidence_metrics: {
      mean_confidence:
        evaluation.judge_results.reduce((sum, jr) => sum + jr.confidence, 0) /
        evaluation.judge_results.length,
      confidence_interval: [
        evaluation.confidence_interval_lower || 0,
        evaluation.confidence_interval_upper || 0,
      ] as [number, number],
      confidence_level: 0.95,
      is_low_confidence:
        (evaluation.confidence_interval_upper || 0) -
          (evaluation.confidence_interval_lower || 0) >
        20,
    },
    inter_judge_agreement: {
      cohens_kappa: null,
      fleiss_kappa: null,
      krippendorff_alpha: null,
      pairwise_correlations: {},
      interpretation: 'N/A',
    },
    hallucination_metrics: {
      overall_score: evaluation.hallucination_score || 0,
      breakdown_by_type: {},
      affected_text_spans: [] as Array<[number, number, string]>,
      severity_distribution: {},
    },
    variance: evaluation.session_metadata?.variance || 0,
    standard_deviation: evaluation.session_metadata?.standard_deviation || 0,
    processing_time_ms: evaluation.session_metadata?.processing_time_ms || 0,
    timestamp: evaluation.created_at,
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white border-b border-gray-200 shadow-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <svg
                className="w-8 h-8 text-blue-600"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"
                />
              </svg>
              <div>
                <h1 className="text-xl font-semibold text-gray-900">
                  LLM Judge Auditor
                </h1>
                <p className="text-sm text-gray-500">Shared Evaluation</p>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <span className="px-3 py-1 text-xs font-medium text-blue-800 bg-blue-100 rounded-full">
                Read-Only
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Content */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4 mb-6">
          <div className="flex items-start gap-3">
            <svg
              className="w-5 h-5 text-yellow-600 flex-shrink-0 mt-0.5"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
              />
            </svg>
            <div className="text-sm text-yellow-800">
              <p className="font-medium mb-1">Viewing Shared Evaluation</p>
              <p>
                This is a read-only view of an evaluation shared with you. You
                can view all results but cannot modify or delete this
                evaluation.
              </p>
            </div>
          </div>
        </div>

        {/* Source Text */}
        <div className="bg-white rounded-lg shadow-md p-6 mb-6">
          <h2 className="text-lg font-semibold text-gray-900 mb-3">
            Source Text
          </h2>
          <p className="text-gray-700 whitespace-pre-wrap">
            {evaluation.source_text}
          </p>
        </div>

        {/* Candidate Output */}
        <div className="bg-white rounded-lg shadow-md p-6 mb-6">
          <h2 className="text-lg font-semibold text-gray-900 mb-3">
            Candidate Output
          </h2>
          <p className="text-gray-700 whitespace-pre-wrap">
            {evaluation.candidate_output}
          </p>
        </div>

        {/* Evaluation Results */}
        <EvaluationResultMessage
          results={results}
          timestamp={new Date(evaluation.created_at)}
        />
      </div>
    </div>
  );
};

export default SharedEvaluationView;
