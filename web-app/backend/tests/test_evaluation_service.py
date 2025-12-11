"""
Tests for evaluation service.
"""
import pytest
import asyncio
from unittest.mock import patch, AsyncMock, MagicMock
from uuid import uuid4

from app.services.evaluation_service import EvaluationService
from app.models import EvaluationSession, JudgeResult, VerifierVerdict, FlaggedIssue
from app.schemas import EvaluationStatus, VerifierLabel, IssueType, IssueSeverity


@pytest.mark.asyncio
class TestEvaluationService:
    """Tests for EvaluationService."""
    
    async def test_process_evaluation_success(self, db_session, test_evaluation_session):
        """Test successful evaluation processing."""
        service = EvaluationService(db_session)
        
        # Mock WebSocket emitters
        with patch('app.services.evaluation_service.emit_evaluation_progress') as mock_progress, \
             patch('app.services.evaluation_service.emit_judge_result') as mock_judge, \
             patch('app.services.evaluation_service.emit_evaluation_complete') as mock_complete:
            
            await service.process_evaluation(
                session_id=test_evaluation_session.id,
                source_text=test_evaluation_session.source_text,
                candidate_output=test_evaluation_session.candidate_output,
                config={'judge_models': ['gpt-4'], 'enable_retrieval': True}
            )
            
            # Verify session was updated
            db_session.refresh(test_evaluation_session)
            assert test_evaluation_session.status == EvaluationStatus.COMPLETED
            assert test_evaluation_session.consensus_score is not None
            assert test_evaluation_session.hallucination_score is not None
            assert test_evaluation_session.completed_at is not None
            
            # Verify progress was emitted
            assert mock_progress.call_count > 0
            
            # Verify judge result was emitted
            assert mock_judge.call_count > 0
            
            # Verify completion was emitted
            mock_complete.assert_called_once()
    
    async def test_process_evaluation_with_retrieval(self, db_session, test_evaluation_session):
        """Test evaluation processing with retrieval enabled."""
        service = EvaluationService(db_session)
        # Disable real judges for this test
        service.use_real_judges = False
        service.groq_client = None
        service.gemini_model = None
        
        with patch('app.services.evaluation_service.emit_evaluation_progress') as mock_progress, \
             patch('app.services.evaluation_service.emit_judge_result'), \
             patch('app.services.evaluation_service.emit_evaluation_complete'):
            
            await service.process_evaluation(
                session_id=test_evaluation_session.id,
                source_text=test_evaluation_session.source_text,
                candidate_output=test_evaluation_session.candidate_output,
                config={'enable_retrieval': True}
            )
            
            # Check that verification stage was called (which includes retrieval)
            # Pipeline stages: generation, claim_extraction, verification, scoring, aggregation
            progress_calls = [call[0] for call in mock_progress.call_args_list]
            verification_calls = [call for call in progress_calls if len(call) > 1 and call[1] == 'verification']
            assert len(verification_calls) > 0
    
    async def test_process_evaluation_without_retrieval(self, db_session, test_evaluation_session):
        """Test evaluation processing without retrieval."""
        service = EvaluationService(db_session)
        # Disable real judges for this test
        service.use_real_judges = False
        service.groq_client = None
        service.gemini_model = None
        
        with patch('app.services.evaluation_service.emit_evaluation_progress') as mock_progress, \
             patch('app.services.evaluation_service.emit_judge_result'), \
             patch('app.services.evaluation_service.emit_evaluation_complete'):
            
            await service.process_evaluation(
                session_id=test_evaluation_session.id,
                source_text=test_evaluation_session.source_text,
                candidate_output=test_evaluation_session.candidate_output,
                config={'enable_retrieval': False}
            )
            
            # Verification stage is still called, but retrieval is skipped within it
            # Check that all pipeline stages are present
            progress_calls = [call[0] for call in mock_progress.call_args_list]
            stages_called = set(call[1] for call in progress_calls if len(call) > 1)
            # Should have: generation, claim_extraction, verification, scoring, aggregation
            assert 'verification' in stages_called
    
    async def test_process_evaluation_multiple_judges(self, db_session, test_evaluation_session):
        """Test evaluation with multiple simulated judges."""
        service = EvaluationService(db_session)
        # Disable real judges to use simulated judges
        service.use_real_judges = False
        service.groq_client = None
        service.gemini_model = None
        
        with patch('app.services.evaluation_service.emit_evaluation_progress'), \
             patch('app.services.evaluation_service.emit_judge_result') as mock_judge, \
             patch('app.services.evaluation_service.emit_evaluation_complete'):
            
            await service.process_evaluation(
                session_id=test_evaluation_session.id,
                source_text=test_evaluation_session.source_text,
                candidate_output=test_evaluation_session.candidate_output,
                config={'judge_models': ['gpt-4', 'claude-3', 'gemini-pro']}
            )
            
            # Verify judge results were emitted for each judge
            assert mock_judge.call_count == 3
            
            # Verify judge results were saved
            judge_results = db_session.query(JudgeResult).filter(
                JudgeResult.session_id == test_evaluation_session.id
            ).all()
            assert len(judge_results) == 3
    
    async def test_process_evaluation_error_handling(self, db_session, test_evaluation_session):
        """Test error handling during evaluation."""
        service = EvaluationService(db_session)
        
        # Mock an error during judge evaluation
        with patch.object(service, '_simulate_judge_evaluation', side_effect=Exception("Test error")), \
             patch('app.services.evaluation_service.emit_evaluation_progress'), \
             patch('app.services.evaluation_service.emit_evaluation_error') as mock_error:
            
            await service.process_evaluation(
                session_id=test_evaluation_session.id,
                source_text=test_evaluation_session.source_text,
                candidate_output=test_evaluation_session.candidate_output,
                config={'judge_models': ['gpt-4']}
            )
            
            # Verify error was emitted
            mock_error.assert_called_once()
            
            # Verify session status was updated to failed
            db_session.refresh(test_evaluation_session)
            assert test_evaluation_session.status == EvaluationStatus.FAILED
    
    async def test_simulate_retrieval(self, db_session):
        """Test retrieval simulation."""
        service = EvaluationService(db_session)
        
        # Use longer text to ensure retrieval returns results
        source_text = " ".join(["word"] * 50)  # 50 words
        num_retrieved = await service._simulate_retrieval(source_text)
        
        assert num_retrieved > 0
        assert num_retrieved <= 5
    
    async def test_simulate_verification(self, db_session, test_evaluation_session):
        """Test verification simulation."""
        service = EvaluationService(db_session)
        
        candidate_output = "First claim. Second claim. Third claim."
        verdicts = await service._simulate_verification(
            test_evaluation_session.id,
            candidate_output
        )
        
        assert len(verdicts) > 0
        assert all(isinstance(v, VerifierVerdict) for v in verdicts)
        assert all(v.label in [VerifierLabel.SUPPORTED, VerifierLabel.REFUTED, VerifierLabel.NOT_ENOUGH_INFO] for v in verdicts)
    
    async def test_simulate_judge_evaluation(self, db_session, test_evaluation_session):
        """Test judge evaluation simulation."""
        service = EvaluationService(db_session)
        
        judge_result = await service._simulate_judge_evaluation(
            test_evaluation_session.id,
            'gpt-4',
            'Source text',
            'Candidate output'
        )
        
        assert isinstance(judge_result, JudgeResult)
        assert judge_result.judge_name == 'gpt-4'
        assert 0 <= judge_result.score <= 100
        assert 0 <= judge_result.confidence <= 1
        assert judge_result.response_time_ms > 0
    
    async def test_calculate_metrics(self, db_session, test_evaluation_session):
        """Test metrics calculation."""
        service = EvaluationService(db_session)
        
        # Create mock judge results
        judge_results = [
            JudgeResult(
                session_id=test_evaluation_session.id,
                judge_name='judge1',
                score=85.0,
                confidence=0.9,
                reasoning='Good',
                response_time_ms=1000
            ),
            JudgeResult(
                session_id=test_evaluation_session.id,
                judge_name='judge2',
                score=90.0,
                confidence=0.85,
                reasoning='Very good',
                response_time_ms=1200
            )
        ]
        
        verifier_verdicts = []
        
        metrics = await service._calculate_metrics(
            judge_results,
            verifier_verdicts,
            'weighted_average'
        )
        
        assert 'consensus_score' in metrics
        assert 'hallucination_score' in metrics
        assert 'confidence_interval' in metrics
        assert 'inter_judge_agreement' in metrics
        assert 'variance' in metrics
        assert 'standard_deviation' in metrics
        
        assert 0 <= metrics['consensus_score'] <= 100
        assert 0 <= metrics['hallucination_score'] <= 100
        assert 0 <= metrics['inter_judge_agreement'] <= 1
    
    async def test_calculate_hallucination_score(self, db_session, test_evaluation_session):
        """Test hallucination score calculation via _calculate_metrics."""
        service = EvaluationService(db_session)
        
        # Create judge results with issues
        judge_result = JudgeResult(
            session_id=test_evaluation_session.id,
            judge_name='judge1',
            score=70.0,
            confidence=0.8,
            reasoning='Some issues found',
            response_time_ms=1000
        )
        db_session.add(judge_result)
        db_session.flush()
        
        # Add a real flagged issue
        flagged_issue = FlaggedIssue(
            judge_result_id=judge_result.id,
            issue_type=IssueType.HALLUCINATION,
            severity=IssueSeverity.HIGH,
            description='Test hallucination issue'
        )
        db_session.add(flagged_issue)
        db_session.commit()
        db_session.refresh(judge_result)
        
        # Create verifier verdicts with refutations
        verifier_verdict = VerifierVerdict(
            session_id=test_evaluation_session.id,
            claim_text='Test claim',
            label=VerifierLabel.REFUTED,
            confidence=0.9,
            reasoning='Claim was refuted'
        )
        db_session.add(verifier_verdict)
        db_session.commit()
        
        # Calculate metrics which includes hallucination score
        metrics = await service._calculate_metrics(
            [judge_result],
            [verifier_verdict],
            'weighted_average'
        )
        
        assert 0 <= metrics['hallucination_score'] <= 100
        # With refuted claims and issues, hallucination score should be higher
        assert metrics['hallucination_score'] > 0

    async def test_pipeline_stages_emitted(self, db_session, test_evaluation_session):
        """Test that all pipeline stages are emitted during evaluation.
        
        Requirements: 8.1, 8.2, 8.4
        Pipeline stages: generation, claim_extraction, verification, scoring, aggregation
        """
        service = EvaluationService(db_session)
        # Disable real judges for this test
        service.use_real_judges = False
        service.groq_client = None
        service.gemini_model = None
        
        emitted_stages = []
        
        async def capture_progress(stage: str, progress: float, message: str):
            emitted_stages.append(stage)
        
        service.set_progress_emitter(capture_progress)
        
        with patch('app.services.evaluation_service.emit_judge_result'), \
             patch('app.services.evaluation_service.emit_evaluation_complete'):
            
            await service.process_evaluation(
                session_id=test_evaluation_session.id,
                source_text=test_evaluation_session.source_text,
                candidate_output=test_evaluation_session.candidate_output,
                config={'judge_models': ['gpt-4'], 'enable_retrieval': True}
            )
        
        # Verify all pipeline stages were emitted
        expected_stages = ['generation', 'claim_extraction', 'verification', 'scoring', 'aggregation']
        for stage in expected_stages:
            assert stage in emitted_stages, f"Stage '{stage}' was not emitted"
    
    async def test_judge_verdicts_streamed_incrementally(self, db_session, test_evaluation_session):
        """Test that judge verdicts are streamed as they complete.
        
        Requirements: 3.3
        """
        service = EvaluationService(db_session)
        # Disable real judges for this test
        service.use_real_judges = False
        service.groq_client = None
        service.gemini_model = None
        
        emitted_verdicts = []
        
        async def capture_verdict(verdict_data: dict):
            emitted_verdicts.append(verdict_data)
        
        service.set_judge_verdict_emitter(capture_verdict)
        
        with patch('app.services.evaluation_service.emit_evaluation_progress'), \
             patch('app.services.evaluation_service.emit_evaluation_complete'):
            
            await service.process_evaluation(
                session_id=test_evaluation_session.id,
                source_text=test_evaluation_session.source_text,
                candidate_output=test_evaluation_session.candidate_output,
                config={'judge_models': ['gpt-4', 'claude-3'], 'enable_retrieval': False}
            )
        
        # Verify verdicts were emitted for each judge
        assert len(emitted_verdicts) == 2
        
        # Verify verdict structure
        for verdict in emitted_verdicts:
            assert 'judge_name' in verdict
            assert 'score' in verdict
            assert 'confidence' in verdict
            assert 'reasoning' in verdict
            assert 'flagged_issues' in verdict
    
    async def test_claim_extraction_in_pipeline(self, db_session, test_evaluation_session):
        """Test that claims are extracted during the pipeline.
        
        Requirements: 5.4, 8.4
        """
        service = EvaluationService(db_session)
        # Disable real judges for this test
        service.use_real_judges = False
        service.groq_client = None
        service.gemini_model = None
        
        with patch('app.services.evaluation_service.emit_evaluation_progress'), \
             patch('app.services.evaluation_service.emit_judge_result'), \
             patch('app.services.evaluation_service.emit_evaluation_complete'):
            
            # Use a longer candidate output to ensure claims are extracted
            test_evaluation_session.candidate_output = (
                "The Earth is the third planet from the Sun. "
                "It was formed approximately 4.5 billion years ago. "
                "The population of Earth is about 8 billion people."
            )
            db_session.commit()
            
            await service.process_evaluation(
                session_id=test_evaluation_session.id,
                source_text=test_evaluation_session.source_text,
                candidate_output=test_evaluation_session.candidate_output,
                config={'judge_models': ['gpt-4'], 'enable_retrieval': False}
            )
        
        # Verify claims were extracted and saved
        from app.models import ClaimVerdict
        claim_verdicts = db_session.query(ClaimVerdict).filter(
            ClaimVerdict.evaluation_id == test_evaluation_session.id
        ).all()
        
        # Should have extracted at least one claim
        assert len(claim_verdicts) > 0
        
        # Verify claim structure
        for cv in claim_verdicts:
            assert cv.claim_text is not None
            assert cv.claim_type is not None
            assert cv.verdict is not None
            assert cv.text_span_start >= 0
            assert cv.text_span_end > cv.text_span_start
