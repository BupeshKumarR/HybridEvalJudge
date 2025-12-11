"""
Chat router for managing chat sessions and messages.
"""
from fastapi import APIRouter, Depends, HTTPException, status, Query, Response
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session, selectinload
from sqlalchemy import desc
from typing import Optional
from uuid import UUID
from datetime import datetime
import logging
import io

from ..database import get_db
from ..models import User, ChatSession, ChatMessage, EvaluationSession
from ..schemas import (
    ChatSessionCreate,
    ChatSessionResponse,
    ChatSessionSummary,
    ChatSessionListResponse,
    ChatMessageCreate,
    ChatMessageResponse,
    MessageRole
)
from ..auth import get_current_active_user
from ..services.export_service import ExportService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/chat", tags=["chat"])


@router.post("/sessions", response_model=ChatSessionResponse, status_code=status.HTTP_201_CREATED)
async def create_chat_session(
    session_data: ChatSessionCreate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Create a new chat session.
    
    Args:
        session_data: Chat session creation data with ollama_model
        current_user: Current authenticated user
        db: Database session
        
    Returns:
        Created chat session
    """
    new_session = ChatSession(
        user_id=current_user.id,
        ollama_model=session_data.ollama_model
    )
    
    db.add(new_session)
    db.commit()
    db.refresh(new_session)
    
    logger.info(
        f"Chat session created: {new_session.id} by user {current_user.username} "
        f"with model {session_data.ollama_model}"
    )
    
    return new_session



@router.get("/sessions", response_model=ChatSessionListResponse)
async def list_chat_sessions(
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(20, ge=1, le=100, description="Items per page"),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    List user's chat sessions with pagination.
    
    Args:
        page: Page number (1-indexed)
        limit: Number of items per page
        current_user: Current authenticated user
        db: Database session
        
    Returns:
        Paginated list of chat session summaries
    """
    # Build base query
    query = db.query(ChatSession).filter(
        ChatSession.user_id == current_user.id
    )
    
    # Get total count
    total = query.count()
    
    # Apply sorting (newest first)
    query = query.order_by(desc(ChatSession.updated_at))
    
    # Apply pagination
    offset = (page - 1) * limit
    sessions = query.offset(offset).limit(limit).all()
    
    # Build session summaries
    summaries = []
    for session in sessions:
        message_count = len(session.messages)
        last_message_preview = None
        
        if session.messages:
            last_message = session.messages[-1]
            last_message_preview = (
                last_message.content[:100] + "..."
                if len(last_message.content) > 100
                else last_message.content
            )
        
        summary = ChatSessionSummary(
            id=session.id,
            ollama_model=session.ollama_model,
            created_at=session.created_at,
            updated_at=session.updated_at,
            message_count=message_count,
            last_message_preview=last_message_preview
        )
        summaries.append(summary)
    
    # Calculate if there are more pages
    has_more = (offset + limit) < total
    
    return ChatSessionListResponse(
        sessions=summaries,
        total=total,
        page=page,
        limit=limit,
        has_more=has_more
    )


@router.get("/sessions/{session_id}", response_model=ChatSessionResponse)
async def get_chat_session(
    session_id: UUID,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Get a chat session by ID with all messages.
    
    Args:
        session_id: Chat session UUID
        current_user: Current authenticated user
        db: Database session
        
    Returns:
        Chat session with all messages
        
    Raises:
        HTTPException: If session not found or access denied
    """
    session = (
        db.query(ChatSession)
        .options(selectinload(ChatSession.messages))
        .filter(
            ChatSession.id == session_id,
            ChatSession.user_id == current_user.id
        )
        .first()
    )
    
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chat session not found"
        )
    
    return session


@router.get("/sessions/{session_id}/messages", response_model=list[ChatMessageResponse])
async def get_chat_messages(
    session_id: UUID,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Get all messages for a chat session in chronological order.
    
    Args:
        session_id: Chat session UUID
        current_user: Current authenticated user
        db: Database session
        
    Returns:
        List of chat messages in chronological order
        
    Raises:
        HTTPException: If session not found or access denied
    """
    # Verify session exists and belongs to user
    session = (
        db.query(ChatSession)
        .filter(
            ChatSession.id == session_id,
            ChatSession.user_id == current_user.id
        )
        .first()
    )
    
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chat session not found"
        )
    
    # Get messages ordered by created_at
    messages = (
        db.query(ChatMessage)
        .filter(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.created_at)
        .all()
    )
    
    return messages


@router.post("/sessions/{session_id}/messages", response_model=ChatMessageResponse, status_code=status.HTTP_201_CREATED)
async def add_chat_message(
    session_id: UUID,
    message_data: ChatMessageCreate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Add a message to a chat session.
    
    Args:
        session_id: Chat session UUID
        message_data: Message content and role
        current_user: Current authenticated user
        db: Database session
        
    Returns:
        Created chat message
        
    Raises:
        HTTPException: If session not found or access denied
    """
    # Verify session exists and belongs to user
    session = (
        db.query(ChatSession)
        .filter(
            ChatSession.id == session_id,
            ChatSession.user_id == current_user.id
        )
        .first()
    )
    
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chat session not found"
        )
    
    # Create new message
    new_message = ChatMessage(
        session_id=session_id,
        role=message_data.role.value,
        content=message_data.content
    )
    
    db.add(new_message)
    
    # Update session's updated_at timestamp
    session.updated_at = datetime.utcnow()
    
    db.commit()
    db.refresh(new_message)
    
    logger.info(
        f"Message added to session {session_id}: role={message_data.role.value}"
    )
    
    return new_message


@router.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_chat_session(
    session_id: UUID,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Delete a chat session and all its messages.
    
    Args:
        session_id: Chat session UUID
        current_user: Current authenticated user
        db: Database session
        
    Raises:
        HTTPException: If session not found or access denied
    """
    session = (
        db.query(ChatSession)
        .filter(
            ChatSession.id == session_id,
            ChatSession.user_id == current_user.id
        )
        .first()
    )
    
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chat session not found"
        )
    
    db.delete(session)
    db.commit()
    
    logger.info(
        f"Chat session deleted: {session_id} by user {current_user.username}"
    )


@router.get("/sessions/{session_id}/export")
async def export_chat_session(
    session_id: UUID,
    format: str = Query("json", description="Export format (json, csv)"),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Export chat session with all messages and evaluations.
    
    Args:
        session_id: Chat session UUID
        format: Export format (json or csv)
        current_user: Current authenticated user
        db: Database session
        
    Returns:
        File download in requested format
        
    Raises:
        HTTPException: If session not found or format not supported
        
    Requirements: 11.1, 11.2, 11.3, 11.4
    """
    # Query session with all relationships
    session = (
        db.query(ChatSession)
        .options(
            selectinload(ChatSession.messages).selectinload(ChatMessage.evaluation)
        )
        .filter(
            ChatSession.id == session_id,
            ChatSession.user_id == current_user.id
        )
        .first()
    )
    
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chat session not found"
        )
    
    export_service = ExportService(db)
    
    if format.lower() == "json":
        content = export_service.export_chat_session_json(session)
        return Response(
            content=content,
            media_type="application/json",
            headers={
                "Content-Disposition": f"attachment; filename=chat_session_{session.id}.json"
            }
        )
    elif format.lower() == "csv":
        content = export_service.export_chat_session_csv(session)
        return StreamingResponse(
            iter([content]),
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename=chat_session_{session.id}.csv"
            }
        )
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported export format: {format}. Supported formats: json, csv"
        )
    
    logger.info(
        f"Chat session exported: {session_id} as {format} by user {current_user.username}"
    )
