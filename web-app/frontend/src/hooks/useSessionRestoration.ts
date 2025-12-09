import { useCallback } from 'react';
import { useEvaluationStore } from '../store/evaluationStore';
import { evaluationsApi } from '../api/evaluations';
import { EvaluationResults } from '../api/types';
import { v4 as uuidv4 } from 'uuid';

/**
 * Hook for restoring evaluation sessions from history
 */
export const useSessionRestoration = () => {
  const { setSessionId, addMessage, clearMessages } = useEvaluationStore();

  const restoreSession = useCallback(
    async (sessionId: string): Promise<boolean> => {
      try {
        // Fetch full evaluation session data
        const sessionData = await evaluationsApi.getEvaluation(sessionId);

        // Clear current messages
        clearMessages();

        // Set the session ID
        setSessionId(sessionData.id);

        // Add user message with source and candidate
        addMessage({
          id: uuidv4(),
          type: 'user',
          timestamp: new Date(sessionData.created_at),
          content: {
            sourceText: sessionData.source_text,
            candidateOutput: sessionData.candidate_output,
          },
        });

        // If evaluation is completed, add the results message
        if (sessionData.status === 'completed' && sessionData.judge_results.length > 0) {
          // Build evaluation results object
          const evaluationResults: EvaluationResults = {
            session_id: sessionData.id,
            consensus_score: sessionData.consensus_score || 0,
            judge_results: sessionData.judge_results.map((jr) => ({
              judge_name: jr.judge_name,
              score: jr.score,
              confidence: jr.confidence,
              reasoning: jr.reasoning || '',
              flagged_issues: jr.flagged_issues.map((issue) => ({
                issue_type: issue.issue_type,
                severity: issue.severity,
                description: issue.description,
                evidence: issue.evidence,
                text_span_start: issue.text_span_start,
                text_span_end: issue.text_span_end,
              })),
              response_time_ms: jr.response_time_ms || 0,
            })),
            verifier_verdicts: sessionData.verifier_verdicts.map((vv) => ({
              claim_text: vv.claim_text,
              label: vv.label,
              confidence: vv.confidence,
              evidence: vv.evidence ? Object.values(vv.evidence) : [],
              reasoning: vv.reasoning || '',
            })),
            confidence_metrics: {
              mean_confidence:
                sessionData.judge_results.reduce((sum, jr) => sum + jr.confidence, 0) /
                sessionData.judge_results.length,
              confidence_interval: [
                sessionData.confidence_interval_lower || 0,
                sessionData.confidence_interval_upper || 0,
              ],
              confidence_level: 0.95,
              is_low_confidence:
                (sessionData.confidence_interval_upper || 0) -
                  (sessionData.confidence_interval_lower || 0) >
                20,
            },
            inter_judge_agreement: {
              cohens_kappa:
                sessionData.judge_results.length === 2
                  ? sessionData.inter_judge_agreement
                  : undefined,
              fleiss_kappa:
                sessionData.judge_results.length > 2
                  ? sessionData.inter_judge_agreement
                  : undefined,
              krippendorff_alpha: undefined,
              pairwise_correlations: {},
              interpretation: getAgreementInterpretation(
                sessionData.inter_judge_agreement || 0
              ),
            },
            hallucination_metrics: {
              overall_score: sessionData.hallucination_score || 0,
              breakdown_by_type: {},
              affected_text_spans: [],
              severity_distribution: {},
            },
            variance: sessionData.session_metadata?.variance || 0,
            standard_deviation: sessionData.session_metadata?.standard_deviation || 0,
            processing_time_ms: sessionData.session_metadata?.processing_time_ms || 0,
            timestamp: sessionData.created_at,
          };

          // Add evaluation result message
          addMessage({
            id: uuidv4(),
            type: 'evaluation',
            timestamp: new Date(sessionData.completed_at || sessionData.created_at),
            content: {
              results: evaluationResults,
            },
          });
        } else if (sessionData.status === 'failed') {
          // Add error message for failed evaluations
          addMessage({
            id: uuidv4(),
            type: 'error',
            timestamp: new Date(sessionData.completed_at || sessionData.created_at),
            content: {
              message: 'This evaluation failed to complete. Please try again.',
            },
          });
        } else if (sessionData.status === 'pending' || sessionData.status === 'in_progress') {
          // Add system message for pending evaluations
          addMessage({
            id: uuidv4(),
            type: 'system',
            timestamp: new Date(),
            content: {
              message: 'This evaluation is still in progress or pending.',
            },
          });
        }

        return true;
      } catch (error) {
        console.error('Failed to restore session:', error);
        return false;
      }
    },
    [setSessionId, addMessage, clearMessages]
  );

  return { restoreSession };
};

/**
 * Get human-readable interpretation of inter-judge agreement score
 */
function getAgreementInterpretation(kappa: number): string {
  if (kappa < 0) return 'poor';
  if (kappa < 0.2) return 'slight';
  if (kappa < 0.4) return 'fair';
  if (kappa < 0.6) return 'moderate';
  if (kappa < 0.8) return 'substantial';
  return 'almost_perfect';
}
