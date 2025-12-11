"""
WebSocket server implementation using Socket.IO for real-time evaluation streaming
and chat message handling with Ollama integration.
"""
import socketio
import logging
import asyncio
from typing import Dict, Optional, List
from uuid import UUID
from datetime import datetime
from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from .database import SessionLocal
from .models import User, EvaluationSession, ChatSession, ChatMessage
from .auth import decode_access_token

logger = logging.getLogger(__name__)

# Create Socket.IO server with async mode
# Allow all origins in development for easier testing
import os
cors_origins = os.environ.get('CORS_ORIGINS', '').split(',') if os.environ.get('CORS_ORIGINS') else '*'
if cors_origins == '*' or os.environ.get('ENVIRONMENT', 'development') == 'development':
    cors_origins = '*'  # Allow all origins in development

sio = socketio.AsyncServer(
    async_mode='asgi',
    cors_allowed_origins=cors_origins,
    logger=True,
    engineio_logger=True  # Enable engine.io logging for debugging
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
    logger.info(f"Auth data received: {auth}")
    logger.info(f"Environ keys: {list(environ.keys()) if environ else 'None'}")
    
    # Check for authentication token
    if not auth or 'token' not in auth:
        logger.warning(f"Connection rejected - no token: {sid}, auth={auth}")
        return False
    
    token = auth['token']
    logger.info(f"Token received (first 20 chars): {token[:20] if token else 'None'}...")
    
    # Authenticate user
    user = await authenticate_socket(sid, token)
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


async def verify_chat_session_access(user_id: UUID, chat_session_id: str) -> bool:
    """
    Verify that a user has access to a chat session.
    
    Args:
        user_id: User UUID
        chat_session_id: Chat session UUID
        
    Returns:
        True if user has access, False otherwise
    """
    try:
        db = SessionLocal()
        try:
            session = (
                db.query(ChatSession)
                .filter(
                    ChatSession.id == UUID(chat_session_id),
                    ChatSession.user_id == user_id
                )
                .first()
            )
            return session is not None
        finally:
            db.close()
    except Exception as e:
        logger.error(f"Error verifying chat session access: {e}")
        return False


@sio.event
async def join_chat_session(sid, data):
    """
    Join a chat session room for receiving streaming updates.
    
    Args:
        sid: Socket.IO session ID
        data: Dictionary containing 'chat_session_id'
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
        chat_session_id = data.get('chat_session_id')
        
        if not chat_session_id:
            await sio.emit('error', {
                'error': 'invalid_request',
                'message': 'chat_session_id is required'
            }, to=sid)
            return
        
        # Verify user has access to this chat session
        has_access = await verify_chat_session_access(user_id, chat_session_id)
        if not has_access:
            await sio.emit('error', {
                'error': 'access_denied',
                'message': 'You do not have access to this chat session'
            }, to=sid)
            return
        
        # Join the room (prefix with 'chat_' to distinguish from evaluation sessions)
        room_id = f"chat_{chat_session_id}"
        await sio.enter_room(sid, room_id)
        
        # Track connection
        if room_id not in active_connections:
            active_connections[room_id] = {}
        active_connections[room_id][sid] = user_id
        
        logger.info(f"Socket {sid} joined chat session room {chat_session_id}")
        
        # Confirm join
        await sio.emit('chat_session_joined', {
            'chat_session_id': chat_session_id,
            'message': 'Successfully joined chat session'
        }, to=sid)
        
    except Exception as e:
        logger.error(f"Error joining chat session: {e}", exc_info=True)
        await sio.emit('error', {
            'error': 'internal_error',
            'message': 'Failed to join chat session'
        }, to=sid)


@sio.event
async def leave_chat_session(sid, data):
    """
    Leave a chat session room.
    
    Args:
        sid: Socket.IO session ID
        data: Dictionary containing 'chat_session_id'
    """
    try:
        chat_session_id = data.get('chat_session_id')
        
        if not chat_session_id:
            await sio.emit('error', {
                'error': 'invalid_request',
                'message': 'chat_session_id is required'
            }, to=sid)
            return
        
        room_id = f"chat_{chat_session_id}"
        
        # Leave the room
        await sio.leave_room(sid, room_id)
        
        # Remove from tracking
        if room_id in active_connections and sid in active_connections[room_id]:
            del active_connections[room_id][sid]
            
            # Clean up empty rooms
            if not active_connections[room_id]:
                del active_connections[room_id]
        
        logger.info(f"Socket {sid} left chat session room {chat_session_id}")
        
        # Confirm leave
        await sio.emit('chat_session_left', {
            'chat_session_id': chat_session_id,
            'message': 'Successfully left chat session'
        }, to=sid)
        
    except Exception as e:
        logger.error(f"Error leaving chat session: {e}", exc_info=True)
        await sio.emit('error', {
            'error': 'internal_error',
            'message': 'Failed to leave chat session'
        }, to=sid)


@sio.event
async def chat_message(sid, data):
    """
    Handle incoming chat message from client.
    Streams response from Ollama and triggers evaluation.
    
    Args:
        sid: Socket.IO session ID
        data: Dictionary containing:
            - chat_session_id: Chat session UUID
            - question: User's question
            - model: Optional Ollama model override
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
        chat_session_id = data.get('chat_session_id')
        question = data.get('question', '').strip()
        model_override = data.get('model')
        
        if not chat_session_id:
            await sio.emit('error', {
                'error': 'invalid_request',
                'message': 'chat_session_id is required'
            }, to=sid)
            return
        
        if not question:
            await sio.emit('error', {
                'error': 'invalid_request',
                'message': 'question is required and cannot be empty'
            }, to=sid)
            return
        
        # Verify user has access to this chat session
        has_access = await verify_chat_session_access(user_id, chat_session_id)
        if not has_access:
            await sio.emit('error', {
                'error': 'access_denied',
                'message': 'You do not have access to this chat session'
            }, to=sid)
            return
        
        # Import services here to avoid circular imports
        from .services.ollama_service import (
            OllamaService,
            Message,
            OllamaConnectionError,
            OllamaModelNotFoundError,
            OllamaServiceError,
            get_ollama_service
        )
        
        db = SessionLocal()
        try:
            # Get chat session
            chat_session = db.query(ChatSession).filter(
                ChatSession.id == UUID(chat_session_id)
            ).first()
            
            if not chat_session:
                await sio.emit('error', {
                    'error': 'session_not_found',
                    'message': 'Chat session not found'
                }, to=sid)
                return
            
            # Determine which model to use
            model = model_override or chat_session.ollama_model
            
            # Save user message to database
            user_message = ChatMessage(
                session_id=chat_session.id,
                role='user',
                content=question
            )
            db.add(user_message)
            db.commit()
            db.refresh(user_message)
            
            # Emit user message confirmation
            room_id = f"chat_{chat_session_id}"
            await sio.emit('message_saved', {
                'message_id': str(user_message.id),
                'role': 'user',
                'content': question,
                'created_at': user_message.created_at.isoformat()
            }, room=room_id)
            
            # Get conversation history for context
            previous_messages = (
                db.query(ChatMessage)
                .filter(ChatMessage.session_id == chat_session.id)
                .order_by(ChatMessage.created_at)
                .all()
            )
            
            # Convert to Message objects for Ollama (exclude the just-added message)
            conversation_history = [
                Message(role=msg.role, content=msg.content)
                for msg in previous_messages[:-1]  # Exclude the current question
            ]
            
            # Start streaming from Ollama
            ollama_service = get_ollama_service()
            
            # Emit generation started
            await sio.emit('generation_started', {
                'chat_session_id': chat_session_id,
                'model': model
            }, room=room_id)
            
            # Stream tokens
            full_response = []
            try:
                async for token in ollama_service.generate_stream(
                    question=question,
                    model=model,
                    conversation_history=conversation_history
                ):
                    full_response.append(token)
                    
                    # Emit stream token event
                    await sio.emit('stream_token', {
                        'token': token,
                        'done': False
                    }, room=room_id)
                
                # Emit stream completion
                await sio.emit('stream_token', {
                    'token': '',
                    'done': True
                }, room=room_id)
                
            except OllamaConnectionError as e:
                logger.error(f"Ollama connection error: {e}")
                await sio.emit('generation_error', {
                    'error': 'connection_error',
                    'message': str(e),
                    'suggestions': [
                        'Ensure Ollama is running: ollama serve',
                        'Check if Ollama is accessible at the configured host',
                        'Verify network connectivity'
                    ]
                }, room=room_id)
                return
                
            except OllamaModelNotFoundError as e:
                logger.error(f"Ollama model not found: {e}")
                await sio.emit('generation_error', {
                    'error': 'model_not_found',
                    'message': str(e),
                    'suggestions': [
                        f'Pull the model: ollama pull {model}',
                        'Check available models in settings'
                    ]
                }, room=room_id)
                return
                
            except OllamaServiceError as e:
                logger.error(f"Ollama service error: {e}")
                await sio.emit('generation_error', {
                    'error': 'service_error',
                    'message': str(e)
                }, room=room_id)
                return
            
            # Save assistant response to database
            response_text = ''.join(full_response)
            assistant_message = ChatMessage(
                session_id=chat_session.id,
                role='assistant',
                content=response_text
            )
            db.add(assistant_message)
            
            # Update chat session timestamp
            chat_session.updated_at = datetime.utcnow()
            
            db.commit()
            db.refresh(assistant_message)
            
            # Emit response saved confirmation
            await sio.emit('message_saved', {
                'message_id': str(assistant_message.id),
                'role': 'assistant',
                'content': response_text,
                'created_at': assistant_message.created_at.isoformat()
            }, room=room_id)
            
            # Emit generation complete
            await sio.emit('generation_complete', {
                'chat_session_id': chat_session_id,
                'message_id': str(assistant_message.id),
                'response': response_text
            }, room=room_id)
            
            logger.info(
                f"Chat message processed: session={chat_session_id}, "
                f"user={session['username']}, response_length={len(response_text)}"
            )
            
            # Trigger automatic evaluation
            asyncio.create_task(
                _trigger_chat_evaluation(
                    chat_session_id=chat_session_id,
                    question=question,
                    response=response_text,
                    assistant_message_id=str(assistant_message.id),
                    room_id=room_id
                )
            )
            
        finally:
            db.close()
        
    except Exception as e:
        logger.error(f"Error processing chat message: {e}", exc_info=True)
        await sio.emit('error', {
            'error': 'internal_error',
            'message': 'Failed to process chat message'
        }, to=sid)


async def _trigger_chat_evaluation(
    chat_session_id: str,
    question: str,
    response: str,
    assistant_message_id: str,
    room_id: str
):
    """
    Trigger evaluation of the chat response by API judges.
    
    This function sets up custom emitters to stream pipeline progress
    and judge verdicts to the chat room.
    
    Pipeline stages (Requirements 8.1, 8.2, 8.4):
    1. Generation - Already complete
    2. Claim Extraction - Extract factual claims
    3. Verification - Verify claims
    4. Scoring - Judge evaluation with streaming verdicts (Requirements 3.3)
    5. Aggregation - Calculate final metrics
    
    Args:
        chat_session_id: Chat session UUID
        question: User's question
        response: LLM response to evaluate
        assistant_message_id: ID of the assistant message
        room_id: WebSocket room ID for emitting events
    """
    logger.info(f"[CHAT_EVAL] Starting evaluation for chat_session={chat_session_id}")
    logger.info(f"[CHAT_EVAL] Question length: {len(question)}, Response length: {len(response)}")
    logger.info(f"[CHAT_EVAL] Response preview: {response[:200]}...")
    
    try:
        from .services.evaluation_service import EvaluationService
        
        db = SessionLocal()
        try:
            # Create an evaluation session for this chat response
            from .models import EvaluationSession as EvalSessionModel
            
            chat_session = db.query(ChatSession).filter(
                ChatSession.id == UUID(chat_session_id)
            ).first()
            
            if not chat_session:
                logger.error(f"Chat session not found: {chat_session_id}")
                return
            
            logger.info(f"[CHAT_EVAL] Creating evaluation session for chat_session={chat_session_id}")
            eval_session = EvalSessionModel(
                user_id=chat_session.user_id,
                source_text=question,
                candidate_output=response,
                status='pending',
                config={'judge_models': ['groq-llama-3.3-70b', 'gemini-2.0-flash']}
            )
            db.add(eval_session)
            db.commit()
            db.refresh(eval_session)
            logger.info(f"[CHAT_EVAL] Created evaluation session: {eval_session.id}")
            
            # Link evaluation to chat message
            assistant_msg = db.query(ChatMessage).filter(
                ChatMessage.id == UUID(assistant_message_id)
            ).first()
            if assistant_msg:
                assistant_msg.evaluation_id = eval_session.id
                db.commit()
            
            # Create evaluation service with custom emitters for chat room
            evaluation_service = EvaluationService(db)
            
            # Set up custom progress emitter to stream to chat room (Requirements 8.1, 8.2)
            async def chat_progress_emitter(stage: str, progress: float, message: str):
                await emit_chat_evaluation_progress(room_id, stage, progress, message)
            
            # Set up custom judge verdict emitter to stream verdicts (Requirements 3.3, 9.1)
            async def chat_judge_verdict_emitter(judge_data: dict):
                # Include status for unavailable judges (Requirements 9.1)
                await sio.emit('judge_verdict', {
                    'judge_name': judge_data.get('judge_name', 'Unknown'),
                    'score': judge_data.get('score', 0),
                    'confidence': judge_data.get('confidence', 0),
                    'reasoning': judge_data.get('reasoning', ''),
                    'issues': judge_data.get('flagged_issues', []),
                    'status': judge_data.get('status', 'available'),
                    'error_message': judge_data.get('error_message')
                }, room=room_id)
            
            evaluation_service.set_progress_emitter(chat_progress_emitter)
            evaluation_service.set_judge_verdict_emitter(chat_judge_verdict_emitter)
            
            # Process evaluation with streaming updates
            logger.info(f"[CHAT_EVAL] Starting process_evaluation for eval_session={eval_session.id}")
            await evaluation_service.process_evaluation(
                session_id=eval_session.id,
                source_text=question,
                candidate_output=response,
                config=eval_session.config
            )
            logger.info(f"[CHAT_EVAL] process_evaluation completed for eval_session={eval_session.id}")
            
            # Emit evaluation complete to chat room
            db.refresh(eval_session)
            
            # Get claim verdicts for the response
            from .models import ClaimVerdict
            claim_verdicts = db.query(ClaimVerdict).filter(
                ClaimVerdict.evaluation_id == eval_session.id
            ).all()
            
            logger.info(
                f"[CHAT_EVAL] Evaluation results: consensus_score={eval_session.consensus_score}, "
                f"hallucination_score={eval_session.hallucination_score}, "
                f"claim_verdicts={len(claim_verdicts)}, status={eval_session.status}"
            )
            
            await sio.emit('chat_evaluation_complete', {
                'chat_session_id': chat_session_id,
                'message_id': assistant_message_id,
                'evaluation_id': str(eval_session.id),
                'consensus_score': eval_session.consensus_score,
                'hallucination_score': eval_session.hallucination_score,
                'confidence_interval': [
                    eval_session.confidence_interval_lower,
                    eval_session.confidence_interval_upper
                ],
                'inter_judge_agreement': eval_session.inter_judge_agreement,
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
                'status': eval_session.status.value if hasattr(eval_session.status, 'value') else eval_session.status
            }, room=room_id)
            
            logger.info(
                f"[CHAT_EVAL] Chat evaluation completed: chat_session={chat_session_id}, "
                f"evaluation={eval_session.id}, claims={len(claim_verdicts)}"
            )
            
        finally:
            db.close()
            
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        logger.error(f"[CHAT_EVAL] Error triggering chat evaluation: {e}")
        logger.error(f"[CHAT_EVAL] Full traceback:\n{error_traceback}")
        await sio.emit('chat_evaluation_error', {
            'chat_session_id': chat_session_id,
            'message_id': assistant_message_id,
            'error': 'evaluation_failed',
            'message': str(e),
            'traceback': error_traceback[:500]  # Include partial traceback for debugging
        }, room=room_id)


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


# Chat-specific helper functions

async def emit_to_chat_session(chat_session_id: str, event: str, data: dict):
    """
    Emit an event to all clients in a chat session room.
    
    Args:
        chat_session_id: Chat session UUID
        event: Event name
        data: Event data
    """
    try:
        room_id = f"chat_{chat_session_id}"
        await sio.emit(event, data, room=room_id)
        logger.debug(f"Emitted {event} to chat session {chat_session_id}")
    except Exception as e:
        logger.error(f"Error emitting to chat session {chat_session_id}: {e}")


async def emit_chat_evaluation_progress(
    room_id: str,
    stage: str,
    progress: float,
    message: str
):
    """
    Emit chat evaluation progress update.
    
    Args:
        room_id: WebSocket room ID
        stage: Current evaluation stage (generation, claim_extraction, verification, scoring, aggregation)
        progress: Progress percentage (0-100)
        message: Progress message
    """
    await sio.emit('evaluation_progress', {
        'stage': stage,
        'progress': progress,
        'message': message
    }, room=room_id)


async def emit_chat_judge_verdict(
    room_id: str,
    judge_name: str,
    score: float,
    confidence: float,
    reasoning: str,
    issues: List[dict],
    status: str = 'available',
    error_message: Optional[str] = None
):
    """
    Emit individual judge verdict for chat evaluation.
    
    Requirements: 3.3, 9.1 - Stream judge verdicts with availability status
    
    Args:
        room_id: WebSocket room ID
        judge_name: Name of the judge
        score: Judge's score (0-100)
        confidence: Confidence level (0-1)
        reasoning: Judge's reasoning
        issues: List of flagged issues
        status: Judge availability status (available, unavailable, failed, timeout, rate_limited)
        error_message: Error message if judge failed
    """
    await sio.emit('judge_verdict', {
        'judge_name': judge_name,
        'score': score,
        'confidence': confidence,
        'reasoning': reasoning,
        'issues': issues,
        'status': status,
        'error_message': error_message
    }, room=room_id)


async def emit_stream_token(chat_session_id: str, token: str, done: bool = False):
    """
    Emit a streaming token to a chat session.
    
    Args:
        chat_session_id: Chat session UUID
        token: Token text
        done: Whether streaming is complete
    """
    await emit_to_chat_session(chat_session_id, 'stream_token', {
        'token': token,
        'done': done
    })
