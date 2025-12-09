"""
WebSocket server implementation using Socket.IO for real-time evaluation streaming.
"""
import socketio
import logging
from typing import Dict, Optional
from uuid import UUID
from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from .database import SessionLocal
from .models import User, EvaluationSession
from .auth import decode_access_token

logger = logging.getLogger(__name__)

# Create Socket.IO server with async mode
sio = socketio.AsyncServer(
    async_mode='asgi',
    cors_allowed_origins=[
        "http://localhost:3000",
        "http://localhost:80",
        "http://localhost:8000",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:80",
        "http://127.0.0.1:8000"
    ],
    logger=True,
    engineio_logger=False
)

# Create ASGI app for Socket.IO
socket_app = socketio.ASGIApp(sio, socketio_path='/socket.io')

# Store active connections: {session_id: {sid: user_id}}
active_connections: Dict[str, Dict[str, UUID]] = {}


async def authenticate_socket(sid: str, token: str) -> Optional[User]:
    """
    Authenticate a WebSocket connection using JWT token.
    
    Args:
        sid: Socket.IO session ID
        token: JWT access token
        
    Returns:
        User object if authentication successful, None otherwise
    """
    try:
        # Decode JWT token
        token_data = decode_access_token(token)
        if not token_data or not token_data.user_id:
            logger.warning(f"Invalid token for socket {sid}")
            return None
        
        # Get user from database
        db = SessionLocal()
        try:
            user = db.query(User).filter(User.id == token_data.user_id).first()
            if not user:
                logger.warning(f"User not found for socket {sid}: {token_data.user_id}")
                return None
            
            logger.info(f"Socket authenticated: {sid} for user {user.username}")
            return user
        finally:
            db.close()
            
    except Exception as e:
        logger.error(f"Authentication error for socket {sid}: {e}")
        return None


async def verify_session_access(user_id: UUID, session_id: str) -> bool:
    """
    Verify that a user has access to an evaluation session.
    
    Args:
        user_id: User UUID
        session_id: Evaluation session UUID
        
    Returns:
        True if user has access, False otherwise
    """
    try:
        db = SessionLocal()
        try:
            session = (
                db.query(EvaluationSession)
                .filter(
                    EvaluationSession.id == UUID(session_id),
                    EvaluationSession.user_id == user_id
                )
                .first()
            )
            return session is not None
        finally:
            db.close()
    except Exception as e:
        logger.error(f"Error verifying session access: {e}")
        return False


@sio.event
async def connect(sid, environ, auth):
    """
    Handle client connection.
    
    Args:
        sid: Socket.IO session ID
        environ: ASGI environment
        auth: Authentication data containing JWT token
    """
    logger.info(f"Client connecting: {sid}")
    
    # Check for authentication token
    if not auth or 'token' not in auth:
        logger.warning(f"Connection rejected - no token: {sid}")
        return False
    
    # Authenticate user
    user = await authenticate_socket(sid, auth['token'])
    if not user:
        logger.warning(f"Connection rejected - authentication failed: {sid}")
        return False
    
    # Store user info in session
    await sio.save_session(sid, {'user_id': str(user.id), 'username': user.username})
    
    logger.info(f"Client connected: {sid} (user: {user.username})")
    return True


@sio.event
async def disconnect(sid):
    """
    Handle client disconnection.
    
    Args:
        sid: Socket.IO session ID
    """
    # Get session data
    session = await sio.get_session(sid)
    username = session.get('username', 'unknown') if session else 'unknown'
    
    logger.info(f"Client disconnected: {sid} (user: {username})")
    
    # Remove from all rooms
    for session_id, connections in list(active_connections.items()):
        if sid in connections:
            del connections[sid]
            logger.info(f"Removed {sid} from session room {session_id}")
            
            # Clean up empty rooms
            if not connections:
                del active_connections[session_id]
                logger.info(f"Removed empty session room {session_id}")


@sio.event
async def join_session(sid, data):
    """
    Join an evaluation session room for receiving updates.
    
    Args:
        sid: Socket.IO session ID
        data: Dictionary containing 'session_id'
    """
    try:
        # Get user session
        session = await sio.get_session(sid)
        if not session:
            await sio.emit('error', {
                'error': 'authentication_required',
                'message': 'Not authenticated'
            }, to=sid)
            return
        
        user_id = UUID(session['user_id'])
        session_id = data.get('session_id')
        
        if not session_id:
            await sio.emit('error', {
                'error': 'invalid_request',
                'message': 'session_id is required'
            }, to=sid)
            return
        
        # Verify user has access to this session
        has_access = await verify_session_access(user_id, session_id)
        if not has_access:
            await sio.emit('error', {
                'error': 'access_denied',
                'message': 'You do not have access to this evaluation session'
            }, to=sid)
            return
        
        # Join the room
        await sio.enter_room(sid, session_id)
        
        # Track connection
        if session_id not in active_connections:
            active_connections[session_id] = {}
        active_connections[session_id][sid] = user_id
        
        logger.info(f"Socket {sid} joined session room {session_id}")
        
        # Confirm join
        await sio.emit('session_joined', {
            'session_id': session_id,
            'message': 'Successfully joined evaluation session'
        }, to=sid)
        
    except Exception as e:
        logger.error(f"Error joining session: {e}", exc_info=True)
        await sio.emit('error', {
            'error': 'internal_error',
            'message': 'Failed to join session'
        }, to=sid)


@sio.event
async def leave_session(sid, data):
    """
    Leave an evaluation session room.
    
    Args:
        sid: Socket.IO session ID
        data: Dictionary containing 'session_id'
    """
    try:
        session_id = data.get('session_id')
        
        if not session_id:
            await sio.emit('error', {
                'error': 'invalid_request',
                'message': 'session_id is required'
            }, to=sid)
            return
        
        # Leave the room
        await sio.leave_room(sid, session_id)
        
        # Remove from tracking
        if session_id in active_connections and sid in active_connections[session_id]:
            del active_connections[session_id][sid]
            
            # Clean up empty rooms
            if not active_connections[session_id]:
                del active_connections[session_id]
        
        logger.info(f"Socket {sid} left session room {session_id}")
        
        # Confirm leave
        await sio.emit('session_left', {
            'session_id': session_id,
            'message': 'Successfully left evaluation session'
        }, to=sid)
        
    except Exception as e:
        logger.error(f"Error leaving session: {e}", exc_info=True)
        await sio.emit('error', {
            'error': 'internal_error',
            'message': 'Failed to leave session'
        }, to=sid)


@sio.event
async def start_evaluation(sid, data):
    """
    Start an evaluation with streaming updates.
    
    Args:
        sid: Socket.IO session ID
        data: Dictionary containing evaluation parameters
    """
    try:
        # Get user session
        session = await sio.get_session(sid)
        if not session:
            await sio.emit('error', {
                'error': 'authentication_required',
                'message': 'Not authenticated'
            }, to=sid)
            return
        
        user_id = UUID(session['user_id'])
        session_id = data.get('session_id')
        
        if not session_id:
            await sio.emit('error', {
                'error': 'invalid_request',
                'message': 'session_id is required'
            }, to=sid)
            return
        
        # Verify user has access to this session
        has_access = await verify_session_access(user_id, session_id)
        if not has_access:
            await sio.emit('error', {
                'error': 'access_denied',
                'message': 'You do not have access to this evaluation session'
            }, to=sid)
            return
        
        # Import here to avoid circular dependency
        from .services.evaluation_service import EvaluationService
        from .database import SessionLocal
        import asyncio
        
        # Get session from database
        db = SessionLocal()
        try:
            from .models import EvaluationSession as EvalSessionModel
            eval_session = db.query(EvalSessionModel).filter(
                EvalSessionModel.id == UUID(session_id)
            ).first()
            
            if not eval_session:
                await sio.emit('error', {
                    'error': 'session_not_found',
                    'message': 'Evaluation session not found'
                }, to=sid)
                return
            
            # Start evaluation
            evaluation_service = EvaluationService(db)
            asyncio.create_task(
                evaluation_service.process_evaluation(
                    session_id=eval_session.id,
                    source_text=eval_session.source_text,
                    candidate_output=eval_session.candidate_output,
                    config=eval_session.config
                )
            )
            
            logger.info(f"Evaluation started via WebSocket: {session_id} by {session['username']}")
            
            # Confirm start
            await sio.emit('evaluation_started', {
                'session_id': session_id,
                'message': 'Evaluation started successfully'
            }, to=sid)
            
        finally:
            db.close()
        
    except Exception as e:
        logger.error(f"Error starting evaluation: {e}", exc_info=True)
        await sio.emit('error', {
            'error': 'internal_error',
            'message': 'Failed to start evaluation'
        }, to=sid)


@sio.event
async def ping(sid):
    """
    Handle ping event for connection health check.
    
    Args:
        sid: Socket.IO session ID
    """
    await sio.emit('pong', {}, to=sid)


# Helper functions for emitting events to session rooms

async def emit_to_session(session_id: str, event: str, data: dict):
    """
    Emit an event to all clients in a session room.
    
    Args:
        session_id: Evaluation session UUID
        event: Event name
        data: Event data
    """
    try:
        await sio.emit(event, data, room=session_id)
        logger.debug(f"Emitted {event} to session {session_id}")
    except Exception as e:
        logger.error(f"Error emitting to session {session_id}: {e}")


async def emit_evaluation_progress(session_id: str, stage: str, progress: float, message: str):
    """
    Emit evaluation progress update.
    
    Args:
        session_id: Evaluation session UUID
        stage: Current evaluation stage
        progress: Progress percentage (0-100)
        message: Progress message
    """
    await emit_to_session(session_id, 'evaluation_progress', {
        'stage': stage,
        'progress': progress,
        'message': message
    })


async def emit_judge_result(session_id: str, judge_data: dict):
    """
    Emit individual judge result.
    
    Args:
        session_id: Evaluation session UUID
        judge_data: Judge result data
    """
    await emit_to_session(session_id, 'judge_result', judge_data)


async def emit_evaluation_complete(session_id: str, results: dict):
    """
    Emit evaluation completion event.
    
    Args:
        session_id: Evaluation session UUID
        results: Complete evaluation results
    """
    await emit_to_session(session_id, 'evaluation_complete', results)


async def emit_evaluation_error(session_id: str, error_type: str, message: str, recovery_suggestions: list):
    """
    Emit evaluation error event.
    
    Args:
        session_id: Evaluation session UUID
        error_type: Type of error
        message: Error message
        recovery_suggestions: List of recovery suggestions
    """
    await emit_to_session(session_id, 'evaluation_error', {
        'error_type': error_type,
        'message': message,
        'recovery_suggestions': recovery_suggestions
    })


def get_active_connections_count(session_id: str) -> int:
    """
    Get the number of active connections for a session.
    
    Args:
        session_id: Evaluation session UUID
        
    Returns:
        Number of active connections
    """
    return len(active_connections.get(session_id, {}))
