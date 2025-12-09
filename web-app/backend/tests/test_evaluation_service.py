"""
Tests for evaluation service.
"""
import pytest
import asyncio
from unittest.mock import patch, AsyncMock, MagicMock
from uuid import uuid4

from app.services.evaluation_service import EvaluationService
from app.models import EvaluationSession, JudgeResult, VerifierVerdict
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
        
        with patch('app.services.evaluation_service.emit_evaluation_progress') as mock_progress, \
             patch('app.services.evaluation_service.emit_judge_result'), \
             patch('app.services.evaluation_service.emit_evaluation_complete'):
            
            await service.process_evaluation(
                session_id=test_evaluation_session.id,
                source_text=test_evaluation_session.source_text,
                candidate_output=test_evaluation_session.candidate_output,
                config={'enable_retrieval': True}
            )
            
            # Check that retrieval stage was called
            progress_calls = [call[0] for call in mock_progress.call_args_list]
            retrieval_calls = [call for call in progress_calls if len(call) > 1 and call[1] == 'retrieval']
            assert len(retrieval_calls) > 0
    
    async def test_process_evaluation_without_retrieval(self, db_session, test_evaluation_session):
        """Test evaluation processing without retrieval."""
        service = EvaluationService(db_session)
        
        with patch('app.services.evaluation_service.emit_evaluation_progress') as mock_progress, \
             patch('app.services.evaluation_service.emit_judge_result'), \
             patch('app.services.evaluation_service.emit_evaluation_complete'):
            
            await service.process_evaluation(
                session_id=test_evaluation_session.id,
                source_text=test_evaluation_session.source_text,
                candidate_output=test_evaluation_session.candidate_output,
                config={'enable_retrieval': False}
            )
            
            # Check that retrieval stage was not called
            progress_calls = [call[0] for call in mock_progress.call_args_list]
            retrieval_calls = [call for call in progress_calls if len(call) > 1 and call[1] == 'retrieval']
            assert len(retrieval_calls) == 0
    
    async def test_process_evaluation_multiple_judges(self, db_session, test_evaluation_session):
        """Test evaluation with multiple judges."""
        service = EvaluationService(db_session)
        
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
        """Test hallucination score calculation."""
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
        
        # Add flagged issues
        issue = MagicMock()
        issue.severity = IssueSeverity.HIGH
        judge_result.flagged_issues = [issue]
        
        # Create verifier verdicts with refutations
        verdict = MagicMock()
        verdict.label = VerifierLabel.REFUTED
        verifier_verdicts = [verdict]
        
        hallucination_score = await service._calculate_hallucination_score(
            [judge_result],
            verifier_verdicts,
            70.0
        )
        
        assert 0 <= hallucination_score <= 100
        # With refuted claims and issues, score should be higher
        assert hallucination_score > 20
