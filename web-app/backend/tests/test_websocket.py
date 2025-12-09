"""
Tests for WebSocket functionality.
"""
import pytest
import asyncio
from uuid import uuid4
from unittest.mock import Mock, patch, AsyncMock
from sqlalchemy.orm import Session

from app.websocket import (
    authenticate_socket,
    verify_session_access,
    emit_evaluation_progress,
    emit_judge_result,
    emit_evaluation_complete,
    emit_evaluation_error,
    get_active_connections_count
)
from app.models import User, EvaluationSession
from app.schemas import EvaluationStatus


@pytest.mark.asyncio
class TestWebSocketAuthentication:
    """Tests for WebSocket authentication."""
    
    async def test_authenticate_socket_valid_token(self, created_user, db_session):
        """Test authentication with valid token."""
        from app.auth import create_access_token
        
        # Create valid token
        token = create_access_token(data={"sub": str(created_user.id), "username": created_user.username})
        
        # Authenticate
        user = await authenticate_socket("test_sid", token)
        
        assert user is not None
        assert user.id == created_user.id
        assert user.username == created_user.username
    
    async def test_authenticate_socket_invalid_token(self):
        """Test authentication with invalid token."""
        user = await authenticate_socket("test_sid", "invalid_token")
        assert user is None
    
    async def test_authenticate_socket_expired_token(self, created_user):
        """Test authentication with expired token."""
        from app.auth import create_access_token
        from datetime import timedelta
        
        # Create expired token
        token = create_access_token(
            data={"sub": str(created_user.id), "username": created_user.username},
            expires_delta=timedelta(seconds=-1)
        )
        
        # Wait a moment to ensure expiration
        await asyncio.sleep(0.1)
        
        user = await authenticate_socket("test_sid", token)
        assert user is None


@pytest.mark.asyncio
class TestWebSocketSessionAccess:
    """Tests for WebSocket session access verification."""
    
    async def test_verify_session_access_valid(self, created_user, test_evaluation_session):
        """Test session access verification with valid access."""
        has_access = await verify_session_access(
            created_user.id,
            str(test_evaluation_session.id)
        )
        assert has_access is True
    
    async def test_verify_session_access_invalid_user(self, test_evaluation_session):
        """Test session access verification with invalid user."""
        random_user_id = uuid4()
        has_access = await verify_session_access(
            random_user_id,
            str(test_evaluation_session.id)
        )
        assert has_access is False
    
    async def test_verify_session_access_invalid_session(self, created_user):
        """Test session access verification with invalid session."""
        random_session_id = str(uuid4())
        has_access = await verify_session_access(
            created_user.id,
            random_session_id
        )
        assert has_access is False


@pytest.mark.asyncio
class TestWebSocketEmitters:
    """Tests for WebSocket event emitters."""
    
    @patch('app.websocket.sio.emit')
    async def test_emit_evaluation_progress(self, mock_emit):
        """Test emitting evaluation progress."""
        session_id = str(uuid4())
        
        await emit_evaluation_progress(
            session_id,
            'judging',
            50.0,
            'Evaluating with judges...'
        )
        
        mock_emit.assert_called_once()
        call_args = mock_emit.call_args
        assert call_args[0][0] == 'evaluation_progress'
        assert call_args[0][1]['stage'] == 'judging'
        assert call_args[0][1]['progress'] == 50.0
        assert call_args[0][1]['message'] == 'Evaluating with judges...'
    
    @patch('app.websocket.sio.emit')
    async def test_emit_judge_result(self, mock_emit):
        """Test emitting judge result."""
        session_id = str(uuid4())
        judge_data = {
            'judge_name': 'gpt-4',
            'score': 85.5,
            'confidence': 0.9,
            'reasoning': 'Test reasoning'
        }
        
        await emit_judge_result(session_id, judge_data)
        
        mock_emit.assert_called_once()
        call_args = mock_emit.call_args
        assert call_args[0][0] == 'judge_result'
        assert call_args[0][1] == judge_data
    
    @patch('app.websocket.sio.emit')
    async def test_emit_evaluation_complete(self, mock_emit):
        """Test emitting evaluation complete."""
        session_id = str(uuid4())
        results = {
            'consensus_score': 85.0,
            'hallucination_score': 15.0,
            'status': 'completed'
        }
        
        await emit_evaluation_complete(session_id, results)
        
        mock_emit.assert_called_once()
        call_args = mock_emit.call_args
        assert call_args[0][0] == 'evaluation_complete'
        assert call_args[0][1] == results
    
    @patch('app.websocket.sio.emit')
    async def test_emit_evaluation_error(self, mock_emit):
        """Test emitting evaluation error."""
        session_id = str(uuid4())
        
        await emit_evaluation_error(
            session_id,
            'timeout_error',
            'Evaluation timed out',
            ['Try again', 'Check configuration']
        )
        
        mock_emit.assert_called_once()
        call_args = mock_emit.call_args
        assert call_args[0][0] == 'evaluation_error'
        assert call_args[0][1]['error_type'] == 'timeout_error'
        assert call_args[0][1]['message'] == 'Evaluation timed out'
        assert len(call_args[0][1]['recovery_suggestions']) == 2


class TestActiveConnections:
    """Tests for active connection tracking."""
    
    def test_get_active_connections_count_empty(self):
        """Test getting connection count for non-existent session."""
        session_id = str(uuid4())
        count = get_active_connections_count(session_id)
        assert count == 0
    
    def test_get_active_connections_count_with_connections(self):
        """Test getting connection count with active connections."""
        from app.websocket import active_connections
        
        session_id = str(uuid4())
        active_connections[session_id] = {
            'sid1': uuid4(),
            'sid2': uuid4(),
            'sid3': uuid4()
        }
        
        count = get_active_connections_count(session_id)
        assert count == 3
        
        # Cleanup
        del active_connections[session_id]
