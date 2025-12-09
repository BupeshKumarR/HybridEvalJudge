"""
Evaluation service for processing evaluations with streaming updates.
"""
import asyncio
import logging
import time
from typing import Dict, List, Optional, Any
from uuid import UUID
from datetime import datetime
from sqlalchemy.orm import Session

from ..models import (
    EvaluationSession,
    JudgeResult,
    FlaggedIssue,
    VerifierVerdict,
    SessionMetadata
)
from ..schemas import (
    EvaluationStatus,
    IssueType,
    IssueSeverity,
    VerifierLabel
)
from ..websocket import (
    emit_evaluation_progress,
    emit_judge_result,
    emit_evaluation_complete,
    emit_evaluation_error
)
from .metrics_calculator import MetricsCalculator

logger = logging.getLogger(__name__)


class EvaluationService:
    """Service for handling evaluation processing with streaming updates."""
    
    def __init__(self, db: Session):
        """
        Initialize evaluation service.
        
        Args:
            db: Database session
        """
        self.db = db
    
    async def process_evaluation(
        self,
        session_id: UUID,
        source_text: str,
        candidate_output: str,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Process an evaluation with streaming updates.
        
        Args:
            session_id: Evaluation session UUID
            source_text: Source text for evaluation
            candidate_output: Candidate output to evaluate
            config: Optional configuration dictionary
        """
        session_id_str = str(session_id)
        start_time = time.time()
        
        try:
            # Update session status
            session = self.db.query(EvaluationSession).filter(
                EvaluationSession.id == session_id
            ).first()
            
            if not session:
                logger.error(f"Session not found: {session_id}")
                return
            
            session.status = EvaluationStatus.IN_PROGRESS
            self.db.commit()
            
            # Extract configuration
            judge_models = config.get('judge_models', ['gpt-4', 'claude-3']) if config else ['gpt-4', 'claude-3']
            enable_retrieval = config.get('enable_retrieval', True) if config else True
            aggregation_strategy = config.get('aggregation_strategy', 'weighted_average') if config else 'weighted_average'
            
            # Stage 1: Retrieval (if enabled)
            if enable_retrieval:
                await emit_evaluation_progress(
                    session_id_str,
                    'retrieval',
                    10.0,
                    'Retrieving relevant context...'
                )
                
                # Simulate retrieval
                await asyncio.sleep(0.5)
                num_retrieved = await self._simulate_retrieval(source_text)
                
                await emit_evaluation_progress(
                    session_id_str,
                    'retrieval',
                    20.0,
                    f'Retrieved {num_retrieved} relevant passages'
                )
            else:
                num_retrieved = 0
            
            # Stage 2: Verification
            await emit_evaluation_progress(
                session_id_str,
                'verification',
                30.0,
                'Extracting and verifying claims...'
            )
            
            # Simulate claim verification
            await asyncio.sleep(0.5)
            verifier_verdicts = await self._simulate_verification(
                session_id,
                candidate_output
            )
            
            await emit_evaluation_progress(
                session_id_str,
                'verification',
                40.0,
                f'Verified {len(verifier_verdicts)} claims'
            )
            
            # Stage 3: Judge evaluation
            await emit_evaluation_progress(
                session_id_str,
                'judging',
                50.0,
                f'Evaluating with {len(judge_models)} judges...'
            )
            
            judge_results = []
            progress_per_judge = 30.0 / len(judge_models)
            
            for i, judge_model in enumerate(judge_models):
                # Simulate judge evaluation
                await asyncio.sleep(0.8)
                
                judge_result = await self._simulate_judge_evaluation(
                    session_id,
                    judge_model,
                    source_text,
                    candidate_output
                )
                judge_results.append(judge_result)
                
                # Emit individual judge result
                await emit_judge_result(session_id_str, {
                    'judge_name': judge_result.judge_name,
                    'score': judge_result.score,
                    'confidence': judge_result.confidence,
                    'reasoning': judge_result.reasoning,
                    'flagged_issues': [
                        {
                            'type': issue.issue_type,
                            'severity': issue.severity,
                            'description': issue.description
                        }
                        for issue in judge_result.flagged_issues
                    ]
                })
                
                current_progress = 50.0 + (i + 1) * progress_per_judge
                await emit_evaluation_progress(
                    session_id_str,
                    'judging',
                    current_progress,
                    f'Judge {i + 1}/{len(judge_models)} completed: {judge_model}'
                )
            
            # Stage 4: Aggregation
            await emit_evaluation_progress(
                session_id_str,
                'aggregation',
                85.0,
                'Aggregating results and calculating metrics...'
            )
            
            # Calculate metrics
            await asyncio.sleep(0.3)
            metrics = await self._calculate_metrics(
                judge_results,
                verifier_verdicts,
                aggregation_strategy
            )
            
            # Update session with results
            session.consensus_score = metrics['consensus_score']
            session.hallucination_score = metrics['hallucination_score']
            session.confidence_interval_lower = metrics['confidence_interval'][0]
            session.confidence_interval_upper = metrics['confidence_interval'][1]
            session.inter_judge_agreement = metrics['inter_judge_agreement']
            session.status = EvaluationStatus.COMPLETED
            session.completed_at = datetime.utcnow()
            
            # Create session metadata
            processing_time_ms = int((time.time() - start_time) * 1000)
            metadata = SessionMetadata(
                session_id=session_id,
                total_judges=len(judge_models),
                judges_used=judge_models,
                aggregation_strategy=aggregation_strategy,
                retrieval_enabled=enable_retrieval,
                num_retrieved_passages=num_retrieved if enable_retrieval else None,
                num_verifier_verdicts=len(verifier_verdicts),
                processing_time_ms=processing_time_ms,
                variance=metrics['variance'],
                standard_deviation=metrics['standard_deviation']
            )
            self.db.add(metadata)
            
            self.db.commit()
            
            await emit_evaluation_progress(
                session_id_str,
                'aggregation',
                100.0,
                'Evaluation complete!'
            )
            
            # Emit completion event
            await emit_evaluation_complete(session_id_str, {
                'session_id': session_id_str,
                'consensus_score': metrics['consensus_score'],
                'hallucination_score': metrics['hallucination_score'],
                'confidence_interval': metrics['confidence_interval'],
                'inter_judge_agreement': metrics['inter_judge_agreement'],
                'status': EvaluationStatus.COMPLETED,
                'processing_time_ms': processing_time_ms
            })
            
            logger.info(f"Evaluation completed: {session_id} in {processing_time_ms}ms")
            
        except Exception as e:
            logger.error(f"Evaluation failed: {session_id} - {e}", exc_info=True)
            
            # Update session status
            try:
                session = self.db.query(EvaluationSession).filter(
                    EvaluationSession.id == session_id
                ).first()
                if session:
                    session.status = EvaluationStatus.FAILED
                    self.db.commit()
            except Exception as db_error:
                logger.error(f"Failed to update session status: {db_error}")
            
            # Emit error event
            await emit_evaluation_error(
                session_id_str,
                'evaluation_error',
                str(e),
                [
                    'Check your configuration settings',
                    'Verify that all required services are available',
                    'Try again with different parameters'
                ]
            )
    
    async def _simulate_retrieval(self, source_text: str) -> int:
        """
        Simulate retrieval of relevant passages.
        
        Args:
            source_text: Source text for retrieval
            
        Returns:
            Number of retrieved passages
        """
        # In production, this would call the actual retrieval component
        return min(5, len(source_text.split()) // 20)
    
    async def _simulate_verification(
        self,
        session_id: UUID,
        candidate_output: str
    ) -> List[VerifierVerdict]:
        """
        Simulate claim verification.
        
        Args:
            session_id: Evaluation session UUID
            candidate_output: Candidate output to verify
            
        Returns:
            List of verifier verdicts
        """
        # In production, this would call the actual verifier
        verdicts = []
        
        # Simulate extracting claims
        sentences = candidate_output.split('.')[:3]  # Take first 3 sentences as claims
        
        for i, claim in enumerate(sentences):
            if not claim.strip():
                continue
            
            # Simulate verification
            import random
            labels = [VerifierLabel.SUPPORTED, VerifierLabel.REFUTED, VerifierLabel.NOT_ENOUGH_INFO]
            label = random.choice(labels)
            confidence = random.uniform(0.7, 0.95)
            
            verdict = VerifierVerdict(
                session_id=session_id,
                claim_text=claim.strip(),
                label=label,
                confidence=confidence,
                evidence={'sources': [f'source_{i}']},
                reasoning=f'Claim {i+1} was {label.lower()} based on available evidence'
            )
            self.db.add(verdict)
            verdicts.append(verdict)
        
        self.db.commit()
        return verdicts
    
    async def _simulate_judge_evaluation(
        self,
        session_id: UUID,
        judge_name: str,
        source_text: str,
        candidate_output: str
    ) -> JudgeResult:
        """
        Simulate judge evaluation.
        
        Args:
            session_id: Evaluation session UUID
            judge_name: Name of the judge model
            source_text: Source text
            candidate_output: Candidate output
            
        Returns:
            Judge result
        """
        # In production, this would call the actual judge
        import random
        
        # Simulate scoring
        score = random.uniform(60, 95)
        confidence = random.uniform(0.75, 0.95)
        response_time_ms = random.randint(500, 2000)
        
        # Create judge result
        judge_result = JudgeResult(
            session_id=session_id,
            judge_name=judge_name,
            score=score,
            confidence=confidence,
            reasoning=f'{judge_name} evaluated the output and found it to be {"accurate" if score > 80 else "partially accurate"}.',
            response_time_ms=response_time_ms
        )
        self.db.add(judge_result)
        self.db.flush()  # Get the ID
        
        # Simulate flagged issues
        if score < 80:
            num_issues = random.randint(1, 3)
            issue_types = [IssueType.FACTUAL_ERROR, IssueType.UNSUPPORTED_CLAIM, IssueType.HALLUCINATION]
            severities = [IssueSeverity.LOW, IssueSeverity.MEDIUM, IssueSeverity.HIGH]
            
            for i in range(num_issues):
                issue = FlaggedIssue(
                    judge_result_id=judge_result.id,
                    issue_type=random.choice(issue_types),
                    severity=random.choice(severities),
                    description=f'Issue {i+1} detected by {judge_name}',
                    evidence={'details': f'evidence_{i}'}
                )
                self.db.add(issue)
                judge_result.flagged_issues.append(issue)
        
        self.db.commit()
        self.db.refresh(judge_result)
        return judge_result
    
    async def _calculate_metrics(
        self,
        judge_results: List[JudgeResult],
        verifier_verdicts: List[VerifierVerdict],
        aggregation_strategy: str
    ) -> Dict[str, Any]:
        """
        Calculate evaluation metrics using MetricsCalculator.
        
        Args:
            judge_results: List of judge results
            verifier_verdicts: List of verifier verdicts
            aggregation_strategy: Aggregation strategy to use
            
        Returns:
            Dictionary of calculated metrics
        """
        import numpy as np
        
        scores = [jr.score for jr in judge_results]
        confidences = [jr.confidence for jr in judge_results]
        
        # Consensus score
        if aggregation_strategy == 'weighted_average':
            consensus_score = float(np.average(scores, weights=confidences))
        else:
            consensus_score = float(np.mean(scores))
        
        # Calculate hallucination metrics
        hallucination_metrics = MetricsCalculator.calculate_hallucination_score(
            judge_results,
            verifier_verdicts,
            consensus_score
        )
        
        # Calculate confidence metrics
        confidence_metrics = MetricsCalculator.calculate_confidence_metrics(
            judge_results,
            confidence_level=0.95
        )
        
        # Calculate inter-judge agreement
        agreement_metrics = MetricsCalculator.calculate_inter_judge_agreement(
            judge_results
        )
        
        # Calculate statistical metrics
        statistical_metrics = MetricsCalculator.calculate_statistical_metrics(
            judge_results
        )
        
        return {
            'consensus_score': consensus_score,
            'hallucination_score': hallucination_metrics.overall_score,
            'hallucination_breakdown': hallucination_metrics.breakdown_by_type,
            'hallucination_severity_distribution': hallucination_metrics.severity_distribution,
            'affected_text_spans': hallucination_metrics.affected_text_spans,
            'confidence_interval': confidence_metrics.confidence_interval,
            'mean_confidence': confidence_metrics.mean_confidence,
            'is_low_confidence': confidence_metrics.is_low_confidence,
            'inter_judge_agreement': agreement_metrics.fleiss_kappa or agreement_metrics.cohens_kappa or 0.0,
            'agreement_interpretation': agreement_metrics.interpretation,
            'pairwise_correlations': agreement_metrics.pairwise_correlations,
            'variance': statistical_metrics['variance'],
            'standard_deviation': statistical_metrics['standard_deviation'],
            'score_distribution': statistical_metrics['score_distribution']
        }
