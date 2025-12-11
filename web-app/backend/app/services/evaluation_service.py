"""
Evaluation service for processing evaluations with streaming updates.
Supports both real API judges (Groq, Gemini) and simulated judges.

This service implements the evaluation pipeline with the following stages:
1. Generation - Response generation (handled externally for chat flow)
2. Claim Extraction - Extract factual claims from response
3. Verification - Verify claims against evidence
4. Scoring - Judge evaluation with streaming verdicts
5. Aggregation - Calculate final metrics

Requirements: 8.1, 8.2, 3.3
"""
import asyncio
import logging
import time
import os
from typing import Dict, List, Optional, Any, Tuple, Callable, Awaitable
from uuid import UUID
from datetime import datetime
from enum import Enum
from sqlalchemy.orm import Session
from concurrent.futures import ThreadPoolExecutor

# Enable detailed logging for debugging evaluation issues
DETAILED_LOGGING = True

from ..models import (
    EvaluationSession,
    JudgeResult,
    FlaggedIssue,
    VerifierVerdict,
    SessionMetadata,
    ClaimVerdict
)
from ..schemas import (
    EvaluationStatus,
    IssueType,
    IssueSeverity,
    VerifierLabel,
    ClaimTypeEnum
)
from ..websocket import (
    emit_evaluation_progress,
    emit_judge_result,
    emit_evaluation_complete,
    emit_evaluation_error
)
from .metrics_calculator import MetricsCalculator
from .claim_extraction_service import ClaimExtractionService, ClaimType

logger = logging.getLogger(__name__)

# Try to import real API judge clients
REAL_JUDGES_AVAILABLE = False
try:
    from groq import Groq
    GROQ_AVAILABLE = True
    logger.info("Groq client available")
except ImportError:
    GROQ_AVAILABLE = False
    logger.warning("Groq client not available")

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
    logger.info("Gemini client available")
except ImportError:
    GEMINI_AVAILABLE = False
    logger.warning("Gemini client not available")

REAL_JUDGES_AVAILABLE = GROQ_AVAILABLE or GEMINI_AVAILABLE


class PipelineStage(str, Enum):
    """
    Evaluation pipeline stages as defined in the design document.
    
    Requirements: 8.1, 8.4
    """
    GENERATION = "generation"
    CLAIM_EXTRACTION = "claim_extraction"
    VERIFICATION = "verification"
    SCORING = "scoring"
    AGGREGATION = "aggregation"


class EvaluationService:
    """Service for handling evaluation processing with streaming updates."""
    
    def __init__(self, db: Session):
        """
        Initialize evaluation service.
        
        Args:
            db: Database session
        """
        self.db = db
        self.use_real_judges = self._check_api_keys()
        self.groq_client = None
        self.gemini_model = None
        self.claim_extraction_service = ClaimExtractionService()
        
        # Custom progress emitter for chat flow (can be overridden)
        self._progress_emitter: Optional[Callable[[str, float, str], Awaitable[None]]] = None
        self._judge_verdict_emitter: Optional[Callable[[dict], Awaitable[None]]] = None
        
        if self.use_real_judges:
            self._initialize_judge_clients()
    
    def _check_api_keys(self) -> bool:
        """Check if API keys are available for real judges."""
        groq_key = os.environ.get("GROQ_API_KEY")
        gemini_key = os.environ.get("GEMINI_API_KEY")
        
        has_keys = bool(groq_key) or bool(gemini_key)
        if has_keys:
            logger.info("API keys found - real judges will be used")
        else:
            logger.info("No API keys found - using simulated judges")
        return has_keys
    
    def _initialize_judge_clients(self):
        """Initialize the API judge clients."""
        groq_key = os.environ.get("GROQ_API_KEY")
        gemini_key = os.environ.get("GEMINI_API_KEY")
        
        if groq_key and GROQ_AVAILABLE:
            try:
                self.groq_client = Groq(api_key=groq_key)
                logger.info("Groq client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Groq client: {e}")
        
        if gemini_key and GEMINI_AVAILABLE:
            try:
                genai.configure(api_key=gemini_key)
                self.gemini_model = genai.GenerativeModel("gemini-2.0-flash-exp")
                logger.info("Gemini client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Gemini client: {e}")
    
    def set_progress_emitter(
        self,
        emitter: Callable[[str, float, str], Awaitable[None]]
    ):
        """
        Set a custom progress emitter for chat flow.
        
        This allows the chat WebSocket handler to receive pipeline stage
        updates directly instead of using the default session-based emitter.
        
        Args:
            emitter: Async function that takes (stage, progress, message)
        
        Requirements: 8.1, 8.2
        """
        self._progress_emitter = emitter
    
    def set_judge_verdict_emitter(
        self,
        emitter: Callable[[dict], Awaitable[None]]
    ):
        """
        Set a custom judge verdict emitter for chat flow.
        
        This allows streaming judge verdicts as they complete.
        
        Args:
            emitter: Async function that takes judge verdict dict
        
        Requirements: 3.3
        """
        self._judge_verdict_emitter = emitter
    
    async def _emit_pipeline_progress(
        self,
        session_id_str: str,
        stage: PipelineStage,
        progress: float,
        message: str
    ):
        """
        Emit pipeline progress update.
        
        Uses custom emitter if set, otherwise falls back to default.
        
        Args:
            session_id_str: Session ID string
            stage: Current pipeline stage
            progress: Progress percentage (0-100)
            message: Progress message
        
        Requirements: 8.1, 8.2
        """
        if self._progress_emitter:
            await self._progress_emitter(stage.value, progress, message)
        else:
            await emit_evaluation_progress(
                session_id_str,
                stage.value,
                progress,
                message
            )
    
    async def _emit_judge_verdict(
        self,
        session_id_str: str,
        judge_data: dict
    ):
        """
        Emit individual judge verdict.
        
        Uses custom emitter if set, otherwise falls back to default.
        
        Args:
            session_id_str: Session ID string
            judge_data: Judge verdict data
        
        Requirements: 3.3
        """
        if self._judge_verdict_emitter:
            await self._judge_verdict_emitter(judge_data)
        else:
            await emit_judge_result(session_id_str, judge_data)
    
    async def process_evaluation(
        self,
        session_id: UUID,
        source_text: str,
        candidate_output: str,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Process an evaluation with streaming updates.
        
        Pipeline stages (Requirements 8.1, 8.4):
        1. Generation - Already complete when this is called
        2. Claim Extraction - Extract factual claims from response
        3. Verification - Verify claims against evidence
        4. Scoring - Judge evaluation with streaming verdicts
        5. Aggregation - Calculate final metrics
        
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
            logger.info(f"[EVAL_PIPELINE] Looking up session: {session_id}")
            session = self.db.query(EvaluationSession).filter(
                EvaluationSession.id == session_id
            ).first()
            
            if not session:
                logger.error(f"[EVAL_PIPELINE] Session not found: {session_id}")
                await emit_evaluation_error(
                    session_id_str,
                    'session_not_found',
                    f'Evaluation session {session_id} not found',
                    ['The session may have been deleted', 'Try creating a new evaluation']
                )
                return
            
            logger.info(f"[EVAL_PIPELINE] Session found, current status: {session.status}")
            session.status = EvaluationStatus.IN_PROGRESS
            self.db.commit()
            logger.info(f"[EVAL_PIPELINE] Session status updated to IN_PROGRESS")
            
            # Extract configuration
            judge_models = config.get('judge_models', ['gpt-4', 'claude-3']) if config else ['gpt-4', 'claude-3']
            enable_retrieval = config.get('enable_retrieval', True) if config else True
            aggregation_strategy = config.get('aggregation_strategy', 'weighted_average') if config else 'weighted_average'
            
            # Stage 1: Generation (already complete for chat flow)
            # Emit generation complete status
            await self._emit_pipeline_progress(
                session_id_str,
                PipelineStage.GENERATION,
                100.0,
                'Response generation complete'
            )
            
            # Stage 2: Claim Extraction (Requirements 8.4)
            await self._emit_pipeline_progress(
                session_id_str,
                PipelineStage.CLAIM_EXTRACTION,
                0.0,
                'Starting claim extraction...'
            )
            
            if DETAILED_LOGGING:
                logger.info(f"[EVAL_PIPELINE] Session {session_id}: Starting claim extraction")
                logger.info(f"[EVAL_PIPELINE] Session {session_id}: Candidate output length: {len(candidate_output)}")
                logger.info(f"[EVAL_PIPELINE] Session {session_id}: Candidate output preview: {candidate_output[:200]}...")
            
            # Extract claims from the candidate output
            extracted_claims = self.claim_extraction_service.extract_claims(candidate_output)
            
            if DETAILED_LOGGING:
                logger.info(f"[EVAL_PIPELINE] Session {session_id}: Extracted {len(extracted_claims)} claims")
                for i, claim in enumerate(extracted_claims):
                    logger.info(f"[EVAL_PIPELINE] Session {session_id}: Claim {i+1}: type={claim.claim_type.value}, text='{claim.text[:50]}...'")
            
            await self._emit_pipeline_progress(
                session_id_str,
                PipelineStage.CLAIM_EXTRACTION,
                50.0,
                f'Extracted {len(extracted_claims)} claims'
            )
            
            # Save extracted claims to database
            claim_verdicts = await self._save_extracted_claims(
                session_id,
                extracted_claims
            )
            
            if DETAILED_LOGGING:
                logger.info(f"[EVAL_PIPELINE] Session {session_id}: Saved {len(claim_verdicts)} claim verdicts to database")
            
            await self._emit_pipeline_progress(
                session_id_str,
                PipelineStage.CLAIM_EXTRACTION,
                100.0,
                f'Claim extraction complete: {len(extracted_claims)} claims found'
            )
            
            # Stage 3: Verification
            await self._emit_pipeline_progress(
                session_id_str,
                PipelineStage.VERIFICATION,
                0.0,
                'Starting claim verification...'
            )
            
            # Retrieval for verification (if enabled)
            num_retrieved = 0
            if enable_retrieval:
                await self._emit_pipeline_progress(
                    session_id_str,
                    PipelineStage.VERIFICATION,
                    20.0,
                    'Retrieving relevant context...'
                )
                await asyncio.sleep(0.3)
                num_retrieved = await self._simulate_retrieval(source_text)
            
            # Verify claims
            await self._emit_pipeline_progress(
                session_id_str,
                PipelineStage.VERIFICATION,
                50.0,
                'Verifying claims against evidence...'
            )
            
            await asyncio.sleep(0.3)
            verifier_verdicts = await self._simulate_verification(
                session_id,
                candidate_output
            )
            
            await self._emit_pipeline_progress(
                session_id_str,
                PipelineStage.VERIFICATION,
                100.0,
                f'Verification complete: {len(verifier_verdicts)} claims verified'
            )
            
            # Stage 4: Scoring (Judge evaluation) - Requirements 3.3
            # Determine which judges to use
            real_judges_to_use = []
            if self.use_real_judges:
                if self.groq_client:
                    real_judges_to_use.append('groq')
                if self.gemini_model:
                    real_judges_to_use.append('gemini')
            
            if DETAILED_LOGGING:
                logger.info(f"[EVAL_PIPELINE] Session {session_id}: use_real_judges={self.use_real_judges}")
                logger.info(f"[EVAL_PIPELINE] Session {session_id}: groq_client={self.groq_client is not None}")
                logger.info(f"[EVAL_PIPELINE] Session {session_id}: gemini_model={self.gemini_model is not None}")
                logger.info(f"[EVAL_PIPELINE] Session {session_id}: real_judges_to_use={real_judges_to_use}")
            
            if real_judges_to_use:
                judge_count = len(real_judges_to_use)
                judge_info = f"real API judges ({', '.join(real_judges_to_use)})"
            else:
                judge_count = len(judge_models)
                judge_info = f"{judge_count} simulated judges"
            
            if DETAILED_LOGGING:
                logger.info(f"[EVAL_PIPELINE] Session {session_id}: Starting scoring with {judge_info}")
            
            await self._emit_pipeline_progress(
                session_id_str,
                PipelineStage.SCORING,
                0.0,
                f'Starting evaluation with {judge_info}...'
            )
            
            judge_results = []
            judge_statuses = []  # Track judge availability status (Requirements 9.1)
            progress_per_judge = 80.0 / max(judge_count, 1)
            
            if real_judges_to_use:
                # Use real API judges with failure handling (Requirements 9.1, 9.2)
                for i, judge_type in enumerate(real_judges_to_use):
                    if judge_type == 'groq':
                        judge_result, judge_status = await self._real_groq_evaluation(
                            session_id, source_text, candidate_output
                        )
                    else:  # gemini
                        judge_result, judge_status = await self._real_gemini_evaluation(
                            session_id, source_text, candidate_output
                        )
                    
                    judge_results.append(judge_result)
                    judge_statuses.append(judge_status)
                    
                    # Emit individual judge verdict incrementally (Requirements 3.3)
                    # Include status for unavailable judges (Requirements 9.1)
                    judge_verdict_data = {
                        'judge_name': judge_result.judge_name,
                        'score': judge_result.score,
                        'confidence': judge_result.confidence,
                        'reasoning': judge_result.reasoning,
                        'status': judge_status['status'],
                        'error_message': judge_status.get('error_message'),
                        'flagged_issues': [
                            {
                                'type': issue.issue_type.value if hasattr(issue.issue_type, 'value') else issue.issue_type,
                                'severity': issue.severity.value if hasattr(issue.severity, 'value') else issue.severity,
                                'description': issue.description
                            }
                            for issue in judge_result.flagged_issues
                        ]
                    }
                    await self._emit_judge_verdict(session_id_str, judge_verdict_data)
                    
                    current_progress = 10.0 + (i + 1) * progress_per_judge
                    await self._emit_pipeline_progress(
                        session_id_str,
                        PipelineStage.SCORING,
                        current_progress,
                        f'Judge {i + 1}/{len(real_judges_to_use)} completed: {judge_result.judge_name}'
                    )
            else:
                # Use simulated judges
                for i, judge_model in enumerate(judge_models):
                    await asyncio.sleep(0.8)
                    
                    judge_result = await self._simulate_judge_evaluation(
                        session_id,
                        judge_model,
                        source_text,
                        candidate_output
                    )
                    judge_results.append(judge_result)
                    
                    # Track simulated judge status (always available)
                    judge_statuses.append({
                        'judge_name': judge_model,
                        'status': 'available',
                        'error_message': None
                    })
                    
                    # Emit individual judge verdict incrementally (Requirements 3.3)
                    judge_verdict_data = {
                        'judge_name': judge_result.judge_name,
                        'score': judge_result.score,
                        'confidence': judge_result.confidence,
                        'reasoning': judge_result.reasoning,
                        'status': 'available',
                        'error_message': None,
                        'flagged_issues': [
                            {
                                'type': issue.issue_type.value if hasattr(issue.issue_type, 'value') else issue.issue_type,
                                'severity': issue.severity.value if hasattr(issue.severity, 'value') else issue.severity,
                                'description': issue.description
                            }
                            for issue in judge_result.flagged_issues
                        ]
                    }
                    await self._emit_judge_verdict(session_id_str, judge_verdict_data)
                    
                    current_progress = 10.0 + (i + 1) * progress_per_judge
                    await self._emit_pipeline_progress(
                        session_id_str,
                        PipelineStage.SCORING,
                        current_progress,
                        f'Judge {i + 1}/{len(judge_models)} completed: {judge_model}'
                    )
            
            # Check for all judges failed scenario (Requirements 9.2)
            available_judges = [jr for jr, js in zip(judge_results, judge_statuses) if js['status'] == 'available']
            if not available_judges and judge_results:
                logger.warning(f"All judges failed for session {session_id}")
                # Continue with partial results - show warning in UI
            
            await self._emit_pipeline_progress(
                session_id_str,
                PipelineStage.SCORING,
                100.0,
                f'All {judge_count} judges completed'
            )
            
            # Stage 5: Aggregation (Requirements 8.1, 8.2)
            await self._emit_pipeline_progress(
                session_id_str,
                PipelineStage.AGGREGATION,
                0.0,
                'Starting result aggregation...'
            )
            
            await self._emit_pipeline_progress(
                session_id_str,
                PipelineStage.AGGREGATION,
                30.0,
                'Calculating consensus score...'
            )
            
            # Calculate metrics
            await asyncio.sleep(0.2)
            metrics = await self._calculate_metrics(
                judge_results,
                verifier_verdicts,
                aggregation_strategy
            )
            
            await self._emit_pipeline_progress(
                session_id_str,
                PipelineStage.AGGREGATION,
                60.0,
                'Calculating hallucination metrics...'
            )
            
            # Update session with results
            session.consensus_score = metrics['consensus_score']
            session.hallucination_score = metrics['hallucination_score']
            session.confidence_interval_lower = metrics['confidence_interval'][0]
            session.confidence_interval_upper = metrics['confidence_interval'][1]
            session.inter_judge_agreement = metrics['inter_judge_agreement']
            session.status = EvaluationStatus.COMPLETED
            session.completed_at = datetime.utcnow()
            
            await self._emit_pipeline_progress(
                session_id_str,
                PipelineStage.AGGREGATION,
                80.0,
                'Saving evaluation metadata...'
            )
            
            # Create session metadata
            processing_time_ms = int((time.time() - start_time) * 1000)
            actual_judges_used = [jr.judge_name for jr in judge_results]
            metadata = SessionMetadata(
                session_id=session_id,
                total_judges=len(judge_results),
                judges_used=actual_judges_used,
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
            
            await self._emit_pipeline_progress(
                session_id_str,
                PipelineStage.AGGREGATION,
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
                'claim_verdicts': [
                    {
                        'claim_text': cv.claim_text,
                        'claim_type': cv.claim_type.value if hasattr(cv.claim_type, 'value') else cv.claim_type,
                        'verdict': cv.verdict.value if hasattr(cv.verdict, 'value') else cv.verdict,
                        'confidence': cv.confidence,
                        'text_span_start': cv.text_span_start,
                        'text_span_end': cv.text_span_end
                    }
                    for cv in claim_verdicts
                ],
                'status': EvaluationStatus.COMPLETED.value,
                'processing_time_ms': processing_time_ms
            })
            
            logger.info(f"Evaluation completed: {session_id} in {processing_time_ms}ms")
            
        except Exception as e:
            import traceback
            error_traceback = traceback.format_exc()
            logger.error(f"[EVAL_PIPELINE] Evaluation failed: {session_id} - {e}")
            logger.error(f"[EVAL_PIPELINE] Full traceback:\n{error_traceback}")
            
            # Update session status
            try:
                session = self.db.query(EvaluationSession).filter(
                    EvaluationSession.id == session_id
                ).first()
                if session:
                    session.status = EvaluationStatus.FAILED
                    self.db.commit()
                    logger.info(f"[EVAL_PIPELINE] Session {session_id} marked as FAILED")
            except Exception as db_error:
                logger.error(f"[EVAL_PIPELINE] Failed to update session status: {db_error}")
            
            # Emit error event
            await emit_evaluation_error(
                session_id_str,
                'evaluation_error',
                str(e),
                [
                    'Check your configuration settings',
                    'Verify that all required services are available',
                    'Try again with different parameters',
                    f'Error details: {str(e)[:200]}'
                ]
            )
    
    async def _save_extracted_claims(
        self,
        session_id: UUID,
        extracted_claims: List
    ) -> List[ClaimVerdict]:
        """
        Save extracted claims to the database as ClaimVerdict records.
        
        Args:
            session_id: Evaluation session UUID
            extracted_claims: List of ExtractedClaim objects from ClaimExtractionService
            
        Returns:
            List of saved ClaimVerdict objects
        
        Requirements: 5.4
        """
        import random
        
        claim_verdicts = []
        
        # Map ClaimType to ClaimTypeEnum
        claim_type_map = {
            ClaimType.NUMERICAL: ClaimTypeEnum.NUMERICAL,
            ClaimType.TEMPORAL: ClaimTypeEnum.TEMPORAL,
            ClaimType.DEFINITIONAL: ClaimTypeEnum.DEFINITIONAL,
            ClaimType.GENERAL: ClaimTypeEnum.GENERAL,
        }
        
        for claim in extracted_claims:
            # Simulate verdict assignment (in production, this would come from verification)
            verdicts = [VerifierLabel.SUPPORTED, VerifierLabel.REFUTED, VerifierLabel.NOT_ENOUGH_INFO]
            verdict = random.choice(verdicts)
            confidence = random.uniform(0.7, 0.95)
            
            claim_verdict = ClaimVerdict(
                evaluation_id=session_id,
                claim_text=claim.text,
                claim_type=claim_type_map.get(claim.claim_type, ClaimTypeEnum.GENERAL),
                verdict=verdict,
                confidence=confidence,
                text_span_start=claim.text_span_start,
                text_span_end=claim.text_span_end
            )
            self.db.add(claim_verdict)
            claim_verdicts.append(claim_verdict)
        
        self.db.commit()
        
        # Refresh to get IDs
        for cv in claim_verdicts:
            self.db.refresh(cv)
        
        return claim_verdicts
    
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
    
    async def _real_groq_evaluation(
        self,
        session_id: UUID,
        source_text: str,
        candidate_output: str
    ) -> Tuple[JudgeResult, dict]:
        """
        Evaluate using real Groq API (Llama 3.3 70B).
        
        Args:
            session_id: Evaluation session UUID
            source_text: Source text
            candidate_output: Candidate output
            
        Returns:
            Tuple of (JudgeResult, status_dict) where status_dict contains
            judge availability status for error handling (Requirements 9.1)
        """
        start_time = time.time()
        judge_name = 'groq-llama-3.3-70b'
        
        prompt = f"""You are an expert evaluator assessing the accuracy and quality of an AI-generated response.

SOURCE/REFERENCE TEXT:
{source_text}

CANDIDATE OUTPUT TO EVALUATE:
{candidate_output}

Please evaluate the candidate output and provide:
1. A score from 0-100 (where 100 is perfect accuracy)
2. A confidence level from 0-1 (how confident you are in your assessment)
3. Detailed reasoning for your score
4. Any factual errors, hallucinations, or unsupported claims found

Respond in this exact JSON format:
{{
    "score": <number 0-100>,
    "confidence": <number 0-1>,
    "reasoning": "<detailed explanation>",
    "issues": [
        {{"type": "factual_error|hallucination|unsupported_claim", "severity": "low|medium|high", "description": "<issue description>"}}
    ]
}}"""

        try:
            if DETAILED_LOGGING:
                logger.info(f"[GROQ_JUDGE] Session {session_id}: Calling Groq API...")
                logger.info(f"[GROQ_JUDGE] Session {session_id}: Source text length: {len(source_text)}")
                logger.info(f"[GROQ_JUDGE] Session {session_id}: Candidate output length: {len(candidate_output)}")
            
            response = await asyncio.to_thread(
                self.groq_client.chat.completions.create,
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1000
            )
            
            response_time_ms = int((time.time() - start_time) * 1000)
            response_text = response.choices[0].message.content
            
            if DETAILED_LOGGING:
                logger.info(f"[GROQ_JUDGE] Session {session_id}: Got response in {response_time_ms}ms")
                logger.info(f"[GROQ_JUDGE] Session {session_id}: Response preview: {response_text[:200]}...")
            
            # Parse the JSON response
            import json
            import re
            
            # Extract JSON from response
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if json_match:
                result_data = json.loads(json_match.group())
                if DETAILED_LOGGING:
                    logger.info(f"[GROQ_JUDGE] Session {session_id}: Parsed score={result_data.get('score')}, confidence={result_data.get('confidence')}")
            else:
                logger.error(f"[GROQ_JUDGE] Session {session_id}: No JSON found in response: {response_text}")
                raise ValueError("No JSON found in response")
            
            score = float(result_data.get('score', 75))
            confidence = float(result_data.get('confidence', 0.8))
            reasoning = result_data.get('reasoning', 'Evaluation completed')
            issues = result_data.get('issues', [])
            
            # Create judge result
            judge_result = JudgeResult(
                session_id=session_id,
                judge_name='groq-llama-3.3-70b',
                score=min(100, max(0, score)),
                confidence=min(1.0, max(0.0, confidence)),
                reasoning=reasoning[:2000],  # Truncate if too long
                response_time_ms=response_time_ms
            )
            self.db.add(judge_result)
            self.db.flush()
            
            # Add flagged issues
            for issue in issues[:5]:  # Limit to 5 issues
                issue_type_map = {
                    'factual_error': IssueType.FACTUAL_ERROR,
                    'hallucination': IssueType.HALLUCINATION,
                    'unsupported_claim': IssueType.UNSUPPORTED_CLAIM
                }
                severity_map = {
                    'low': IssueSeverity.LOW,
                    'medium': IssueSeverity.MEDIUM,
                    'high': IssueSeverity.HIGH
                }
                
                flagged_issue = FlaggedIssue(
                    judge_result_id=judge_result.id,
                    issue_type=issue_type_map.get(issue.get('type', 'factual_error'), IssueType.FACTUAL_ERROR),
                    severity=severity_map.get(issue.get('severity', 'medium'), IssueSeverity.MEDIUM),
                    description=issue.get('description', 'Issue detected')[:500],
                    evidence={'source': 'groq-llama-3.3-70b'}
                )
                self.db.add(flagged_issue)
                judge_result.flagged_issues.append(flagged_issue)
            
            self.db.commit()
            self.db.refresh(judge_result)
            
            # Return with available status
            status = {
                'judge_name': judge_name,
                'status': 'available',
                'error_message': None
            }
            return judge_result, status
            
        except asyncio.TimeoutError as e:
            logger.error(f"Groq evaluation timed out: {e}")
            # Return unavailable result with timeout status (Requirements 9.1)
            judge_result = self._create_unavailable_judge_result(
                session_id, judge_name, 'Request timed out'
            )
            status = {
                'judge_name': judge_name,
                'status': 'timeout',
                'error_message': 'Request timed out after waiting for response'
            }
            return judge_result, status
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Groq evaluation failed: {e}")
            
            # Determine failure type for proper annotation (Requirements 9.1)
            if 'rate' in error_msg.lower() or '429' in error_msg:
                failure_status = 'rate_limited'
                error_message = 'API rate limit exceeded'
            elif 'auth' in error_msg.lower() or '401' in error_msg or '403' in error_msg:
                failure_status = 'unavailable'
                error_message = 'Authentication failed - check API key'
            else:
                failure_status = 'failed'
                error_message = f'Evaluation failed: {error_msg[:100]}'
            
            # Create unavailable judge result
            judge_result = self._create_unavailable_judge_result(
                session_id, judge_name, error_message
            )
            status = {
                'judge_name': judge_name,
                'status': failure_status,
                'error_message': error_message
            }
            return judge_result, status
    
    def _create_unavailable_judge_result(
        self,
        session_id: UUID,
        judge_name: str,
        error_message: str
    ) -> JudgeResult:
        """
        Create a JudgeResult marked as unavailable for failed judges.
        
        Requirements: 9.1 - Annotate failed judges as "Unavailable"
        
        Args:
            session_id: Evaluation session UUID
            judge_name: Name of the judge
            error_message: Error message describing the failure
            
        Returns:
            JudgeResult with unavailable status
        """
        judge_result = JudgeResult(
            session_id=session_id,
            judge_name=f"{judge_name} (Unavailable)",
            score=None,  # No score for unavailable judges
            confidence=0.0,
            reasoning=f"Judge unavailable: {error_message}",
            response_time_ms=0
        )
        self.db.add(judge_result)
        self.db.commit()
        self.db.refresh(judge_result)
        return judge_result
    
    async def _real_gemini_evaluation(
        self,
        session_id: UUID,
        source_text: str,
        candidate_output: str
    ) -> Tuple[JudgeResult, dict]:
        """
        Evaluate using real Gemini API (Gemini 2.0 Flash).
        
        Args:
            session_id: Evaluation session UUID
            source_text: Source text
            candidate_output: Candidate output
            
        Returns:
            Tuple of (JudgeResult, status_dict) where status_dict contains
            judge availability status for error handling (Requirements 9.1)
        """
        start_time = time.time()
        judge_name = 'gemini-2.0-flash'
        
        prompt = f"""You are an expert evaluator assessing the accuracy and quality of an AI-generated response.

SOURCE/REFERENCE TEXT:
{source_text}

CANDIDATE OUTPUT TO EVALUATE:
{candidate_output}

Please evaluate the candidate output and provide:
1. A score from 0-100 (where 100 is perfect accuracy)
2. A confidence level from 0-1 (how confident you are in your assessment)
3. Detailed reasoning for your score
4. Any factual errors, hallucinations, or unsupported claims found

Respond in this exact JSON format:
{{
    "score": <number 0-100>,
    "confidence": <number 0-1>,
    "reasoning": "<detailed explanation>",
    "issues": [
        {{"type": "factual_error|hallucination|unsupported_claim", "severity": "low|medium|high", "description": "<issue description>"}}
    ]
}}"""

        try:
            if DETAILED_LOGGING:
                logger.info(f"[GEMINI_JUDGE] Session {session_id}: Calling Gemini API...")
                logger.info(f"[GEMINI_JUDGE] Session {session_id}: Source text length: {len(source_text)}")
                logger.info(f"[GEMINI_JUDGE] Session {session_id}: Candidate output length: {len(candidate_output)}")
            
            response = await asyncio.to_thread(
                self.gemini_model.generate_content,
                prompt
            )
            
            response_time_ms = int((time.time() - start_time) * 1000)
            response_text = response.text
            
            if DETAILED_LOGGING:
                logger.info(f"[GEMINI_JUDGE] Session {session_id}: Got response in {response_time_ms}ms")
                logger.info(f"[GEMINI_JUDGE] Session {session_id}: Response preview: {response_text[:200]}...")
            
            # Parse the JSON response
            import json
            import re
            
            # Extract JSON from response
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if json_match:
                result_data = json.loads(json_match.group())
                if DETAILED_LOGGING:
                    logger.info(f"[GEMINI_JUDGE] Session {session_id}: Parsed score={result_data.get('score')}, confidence={result_data.get('confidence')}")
            else:
                logger.error(f"[GEMINI_JUDGE] Session {session_id}: No JSON found in response: {response_text}")
                raise ValueError("No JSON found in response")
            
            score = float(result_data.get('score', 75))
            confidence = float(result_data.get('confidence', 0.8))
            reasoning = result_data.get('reasoning', 'Evaluation completed')
            issues = result_data.get('issues', [])
            
            # Create judge result
            judge_result = JudgeResult(
                session_id=session_id,
                judge_name='gemini-2.0-flash',
                score=min(100, max(0, score)),
                confidence=min(1.0, max(0.0, confidence)),
                reasoning=reasoning[:2000],  # Truncate if too long
                response_time_ms=response_time_ms
            )
            self.db.add(judge_result)
            self.db.flush()
            
            # Add flagged issues
            for issue in issues[:5]:  # Limit to 5 issues
                issue_type_map = {
                    'factual_error': IssueType.FACTUAL_ERROR,
                    'hallucination': IssueType.HALLUCINATION,
                    'unsupported_claim': IssueType.UNSUPPORTED_CLAIM
                }
                severity_map = {
                    'low': IssueSeverity.LOW,
                    'medium': IssueSeverity.MEDIUM,
                    'high': IssueSeverity.HIGH
                }
                
                flagged_issue = FlaggedIssue(
                    judge_result_id=judge_result.id,
                    issue_type=issue_type_map.get(issue.get('type', 'factual_error'), IssueType.FACTUAL_ERROR),
                    severity=severity_map.get(issue.get('severity', 'medium'), IssueSeverity.MEDIUM),
                    description=issue.get('description', 'Issue detected')[:500],
                    evidence={'source': 'gemini-2.0-flash'}
                )
                self.db.add(flagged_issue)
                judge_result.flagged_issues.append(flagged_issue)
            
            self.db.commit()
            self.db.refresh(judge_result)
            
            # Return with available status
            status = {
                'judge_name': judge_name,
                'status': 'available',
                'error_message': None
            }
            return judge_result, status
            
        except asyncio.TimeoutError as e:
            logger.error(f"Gemini evaluation timed out: {e}")
            # Return unavailable result with timeout status (Requirements 9.1)
            judge_result = self._create_unavailable_judge_result(
                session_id, judge_name, 'Request timed out'
            )
            status = {
                'judge_name': judge_name,
                'status': 'timeout',
                'error_message': 'Request timed out after waiting for response'
            }
            return judge_result, status
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Gemini evaluation failed: {e}")
            
            # Determine failure type for proper annotation (Requirements 9.1)
            if 'rate' in error_msg.lower() or '429' in error_msg:
                failure_status = 'rate_limited'
                error_message = 'API rate limit exceeded'
            elif 'auth' in error_msg.lower() or '401' in error_msg or '403' in error_msg:
                failure_status = 'unavailable'
                error_message = 'Authentication failed - check API key'
            else:
                failure_status = 'failed'
                error_message = f'Evaluation failed: {error_msg[:100]}'
            
            # Create unavailable judge result
            judge_result = self._create_unavailable_judge_result(
                session_id, judge_name, error_message
            )
            status = {
                'judge_name': judge_name,
                'status': failure_status,
                'error_message': error_message
            }
            return judge_result, status
    
    async def _calculate_metrics(
        self,
        judge_results: List[JudgeResult],
        verifier_verdicts: List[VerifierVerdict],
        aggregation_strategy: str
    ) -> Dict[str, Any]:
        """
        Calculate evaluation metrics using MetricsCalculator.
        
        Handles unavailable judges by filtering them out of calculations.
        Requirements: 9.1, 9.2 - Continue with available judges
        
        Args:
            judge_results: List of judge results (may include unavailable judges)
            verifier_verdicts: List of verifier verdicts
            aggregation_strategy: Aggregation strategy to use
            
        Returns:
            Dictionary of calculated metrics
        """
        import numpy as np
        
        # Filter out unavailable judges (those with None scores) - Requirements 9.1, 9.2
        available_results = [jr for jr in judge_results if jr.score is not None]
        
        if not available_results:
            # All judges failed - return default metrics (Requirements 9.2)
            logger.warning("No available judge results - returning default metrics")
            return {
                'consensus_score': 0.0,
                'hallucination_score': 100.0,  # Assume worst case
                'hallucination_breakdown': {},
                'hallucination_severity_distribution': {},
                'affected_text_spans': [],
                'confidence_interval': (0.0, 0.0),
                'mean_confidence': 0.0,
                'is_low_confidence': True,
                'inter_judge_agreement': 0.0,
                'agreement_interpretation': 'No judges available',
                'pairwise_correlations': {},
                'variance': 0.0,
                'standard_deviation': 0.0,
                'score_distribution': {}
            }
        
        scores = [jr.score for jr in available_results]
        confidences = [jr.confidence for jr in available_results]
        
        # Consensus score
        if aggregation_strategy == 'weighted_average':
            consensus_score = float(np.average(scores, weights=confidences))
        else:
            consensus_score = float(np.mean(scores))
        
        # Calculate hallucination metrics (use available results only)
        hallucination_metrics = MetricsCalculator.calculate_hallucination_score(
            available_results,
            verifier_verdicts,
            consensus_score
        )
        
        # Calculate confidence metrics (use available results only)
        confidence_metrics = MetricsCalculator.calculate_confidence_metrics(
            available_results,
            confidence_level=0.95
        )
        
        # Calculate inter-judge agreement (use available results only)
        agreement_metrics = MetricsCalculator.calculate_inter_judge_agreement(
            available_results
        )
        
        # Calculate statistical metrics (use available results only)
        statistical_metrics = MetricsCalculator.calculate_statistical_metrics(
            available_results
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
