"""
Evaluation router for creating and managing evaluation sessions.
"""
from fastapi import APIRouter, Depends, HTTPException, status, Query, Response
from fastapi.responses import JSONResponse, StreamingResponse
from sqlalchemy.orm import Session, joinedload, selectinload
from sqlalchemy import desc, func
from typing import Optional, List
from uuid import UUID
from datetime import datetime
import logging
import json
import csv
import io

from ..database import get_db
from ..models import (
    User,
    EvaluationSession,
    JudgeResult,
    FlaggedIssue,
    VerifierVerdict,
    SessionMetadata
)
from ..schemas import (
    EvaluationSessionCreate,
    EvaluationSessionResponse,
    SessionSummary,
    SessionListResponse,
    EvaluationStatus
)
from ..auth import get_current_active_user
from ..cache import get_cached, set_cached, invalidate_cache

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/evaluations", tags=["evaluations"])


@router.post("", response_model=EvaluationSessionResponse, status_code=status.HTTP_201_CREATED)
async def create_evaluation(
    evaluation_data: EvaluationSessionCreate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Create a new evaluation session.
    
    Args:
        evaluation_data: Evaluation request data
        current_user: Current authenticated user
        db: Database session
        
    Returns:
        Created evaluation session with session_id and websocket_url
    """
    # Create new evaluation session
    new_session = EvaluationSession(
        user_id=current_user.id,
        source_text=evaluation_data.source_text,
        candidate_output=evaluation_data.candidate_output,
        status=EvaluationStatus.PENDING,
        config=evaluation_data.config.model_dump() if evaluation_data.config else None
    )
    
    db.add(new_session)
    db.commit()
    db.refresh(new_session)
    
    logger.info(
        f"Evaluation session created: {new_session.id} by user {current_user.username}"
    )
    
    # Return session with websocket URL for streaming updates
    response_data = EvaluationSessionResponse.model_validate(new_session)
    
    return response_data


@router.post("/{session_id}/start", status_code=status.HTTP_202_ACCEPTED)
async def start_evaluation(
    session_id: UUID,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Start processing an evaluation session with streaming updates.
    
    Args:
        session_id: Evaluation session UUID
        current_user: Current authenticated user
        db: Database session
        
    Returns:
        Acknowledgment that evaluation has started
        
    Raises:
        HTTPException: If session not found or already processing
    """
    from ..services.evaluation_service import EvaluationService
    import asyncio
    
    # Query session
    session = (
        db.query(EvaluationSession)
        .filter(
            EvaluationSession.id == session_id,
            EvaluationSession.user_id == current_user.id
        )
        .first()
    )
    
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Evaluation session not found"
        )
    
    if session.status == EvaluationStatus.IN_PROGRESS:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Evaluation is already in progress"
        )
    
    if session.status == EvaluationStatus.COMPLETED:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Evaluation has already been completed"
        )
    
    # Start evaluation in background
    evaluation_service = EvaluationService(db)
    asyncio.create_task(
        evaluation_service.process_evaluation(
            session_id=session.id,
            source_text=session.source_text,
            candidate_output=session.candidate_output,
            config=session.config
        )
    )
    
    logger.info(f"Evaluation started: {session_id} by user {current_user.username}")
    
    return {
        "message": "Evaluation started",
        "session_id": str(session_id),
        "status": "processing",
        "websocket_url": f"/ws/socket.io"
    }


@router.get("/{session_id}", response_model=EvaluationSessionResponse)
async def get_evaluation(
    session_id: UUID,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Get evaluation session results by ID.
    
    Args:
        session_id: Evaluation session UUID
        current_user: Current authenticated user
        db: Database session
        
    Returns:
        Complete evaluation session with all results
        
    Raises:
        HTTPException: If session not found or access denied
    """
    # Try to get from cache first (only for completed sessions)
    cache_key = f"evaluation:session:{session_id}:{current_user.id}"
    cached_data = get_cached(cache_key)
    
    if cached_data:
        logger.debug(f"Cache hit for evaluation session: {session_id}")
        return EvaluationSessionResponse(**cached_data)
    
    # Query session with all relationships using eager loading
    session = (
        db.query(EvaluationSession)
        .options(
            selectinload(EvaluationSession.judge_results).selectinload(JudgeResult.flagged_issues),
            selectinload(EvaluationSession.verifier_verdicts),
            selectinload(EvaluationSession.session_metadata)
        )
        .filter(
            EvaluationSession.id == session_id,
            EvaluationSession.user_id == current_user.id
        )
        .first()
    )
    
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Evaluation session not found"
        )
    
    # Cache completed sessions for 5 minutes
    if session.status == EvaluationStatus.COMPLETED:
        session_data = EvaluationSessionResponse.model_validate(session)
        set_cached(cache_key, session_data.model_dump(), ttl=300)
    
    return session


@router.get("", response_model=SessionListResponse)
async def list_evaluations(
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(20, ge=1, le=100, description="Items per page"),
    sort_by: str = Query("created_at", description="Sort field"),
    order: str = Query("desc", description="Sort order (asc/desc)"),
    status_filter: Optional[EvaluationStatus] = Query(None, description="Filter by status"),
    min_score: Optional[float] = Query(None, ge=0, le=100, description="Minimum consensus score"),
    max_score: Optional[float] = Query(None, ge=0, le=100, description="Maximum consensus score"),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    List user's evaluation history with pagination and filtering.
    
    Args:
        page: Page number (1-indexed)
        limit: Number of items per page
        sort_by: Field to sort by
        order: Sort order (asc/desc)
        status_filter: Optional status filter
        min_score: Optional minimum consensus score filter
        max_score: Optional maximum consensus score filter
        current_user: Current authenticated user
        db: Database session
        
    Returns:
        Paginated list of evaluation session summaries
    """
    # Build base query
    query = db.query(EvaluationSession).filter(
        EvaluationSession.user_id == current_user.id
    )
    
    # Apply filters
    if status_filter:
        query = query.filter(EvaluationSession.status == status_filter)
    
    if min_score is not None:
        query = query.filter(EvaluationSession.consensus_score >= min_score)
    
    if max_score is not None:
        query = query.filter(EvaluationSession.consensus_score <= max_score)
    
    # Get total count
    total = query.count()
    
    # Apply sorting
    sort_field = getattr(EvaluationSession, sort_by, EvaluationSession.created_at)
    if order.lower() == "desc":
        query = query.order_by(desc(sort_field))
    else:
        query = query.order_by(sort_field)
    
    # Apply pagination
    offset = (page - 1) * limit
    sessions = query.offset(offset).limit(limit).all()
    
    # Build session summaries
    summaries = []
    for session in sessions:
        # Count related records
        num_judge_results = len(session.judge_results)
        num_verifier_verdicts = len(session.verifier_verdicts)
        num_flagged_issues = sum(len(jr.flagged_issues) for jr in session.judge_results)
        
        # Create preview text (first 100 characters)
        source_preview = (
            session.source_text[:100] + "..."
            if len(session.source_text) > 100
            else session.source_text
        )
        candidate_preview = (
            session.candidate_output[:100] + "..."
            if len(session.candidate_output) > 100
            else session.candidate_output
        )
        
        summary = SessionSummary(
            id=session.id,
            consensus_score=session.consensus_score,
            hallucination_score=session.hallucination_score,
            status=session.status,
            created_at=session.created_at,
            completed_at=session.completed_at,
            source_preview=source_preview,
            candidate_preview=candidate_preview,
            num_judge_results=num_judge_results,
            num_verifier_verdicts=num_verifier_verdicts,
            num_flagged_issues=num_flagged_issues
        )
        summaries.append(summary)
    
    # Calculate if there are more pages
    has_more = (offset + limit) < total
    
    return SessionListResponse(
        sessions=summaries,
        total=total,
        page=page,
        limit=limit,
        has_more=has_more
    )


@router.get("/{session_id}/export")
async def export_evaluation(
    session_id: UUID,
    format: str = Query("json", description="Export format (json, csv, pdf)"),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Export evaluation results in various formats.
    
    Args:
        session_id: Evaluation session UUID
        format: Export format (json, csv, pdf)
        current_user: Current authenticated user
        db: Database session
        
    Returns:
        File download in requested format
        
    Raises:
        HTTPException: If session not found or format not supported
    """
    # Query session
    session = (
        db.query(EvaluationSession)
        .filter(
            EvaluationSession.id == session_id,
            EvaluationSession.user_id == current_user.id
        )
        .first()
    )
    
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Evaluation session not found"
        )
    
    if format.lower() == "json":
        return export_as_json(session)
    elif format.lower() == "csv":
        return export_as_csv(session)
    elif format.lower() == "pdf":
        return export_as_pdf(session)
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported export format: {format}"
        )


def export_as_json(session: EvaluationSession) -> Response:
    """Export evaluation session as JSON."""
    # Convert to response model
    session_data = EvaluationSessionResponse.model_validate(session)
    
    # Convert to JSON
    json_data = session_data.model_dump_json(indent=2)
    
    # Return as downloadable file
    return Response(
        content=json_data,
        media_type="application/json",
        headers={
            "Content-Disposition": f"attachment; filename=evaluation_{session.id}.json"
        }
    )


def export_as_pdf(session: EvaluationSession) -> StreamingResponse:
    """Export evaluation session as PDF."""
    from ..services.pdf_export_service import PDFExportService
    
    pdf_service = PDFExportService()
    pdf_buffer = pdf_service.generate_pdf(session)
    
    return StreamingResponse(
        pdf_buffer,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f"attachment; filename=evaluation_{session.id}.pdf"
        }
    )


def export_as_csv(session: EvaluationSession) -> StreamingResponse:
    """Export evaluation session as CSV."""
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Write header
    writer.writerow([
        "Session ID",
        "Status",
        "Consensus Score",
        "Hallucination Score",
        "Confidence Lower",
        "Confidence Upper",
        "Inter-Judge Agreement",
        "Created At",
        "Completed At"
    ])
    
    # Write session data
    writer.writerow([
        str(session.id),
        session.status,
        session.consensus_score,
        session.hallucination_score,
        session.confidence_interval_lower,
        session.confidence_interval_upper,
        session.inter_judge_agreement,
        session.created_at.isoformat() if session.created_at else "",
        session.completed_at.isoformat() if session.completed_at else ""
    ])
    
    # Write judge results
    writer.writerow([])
    writer.writerow(["Judge Results"])
    writer.writerow(["Judge Name", "Score", "Confidence", "Response Time (ms)", "Reasoning"])
    
    for judge_result in session.judge_results:
        writer.writerow([
            judge_result.judge_name,
            judge_result.score,
            judge_result.confidence,
            judge_result.response_time_ms,
            judge_result.reasoning
        ])
    
    # Write flagged issues
    writer.writerow([])
    writer.writerow(["Flagged Issues"])
    writer.writerow(["Judge", "Issue Type", "Severity", "Description"])
    
    for judge_result in session.judge_results:
        for issue in judge_result.flagged_issues:
            writer.writerow([
                judge_result.judge_name,
                issue.issue_type,
                issue.severity,
                issue.description
            ])
    
    # Write verifier verdicts
    writer.writerow([])
    writer.writerow(["Verifier Verdicts"])
    writer.writerow(["Claim", "Label", "Confidence", "Reasoning"])
    
    for verdict in session.verifier_verdicts:
        writer.writerow([
            verdict.claim_text,
            verdict.label,
            verdict.confidence,
            verdict.reasoning
        ])
    
    # Get CSV content
    output.seek(0)
    
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={
            "Content-Disposition": f"attachment; filename=evaluation_{session.id}.csv"
        }
    )


@router.get("/statistics/aggregate")
async def get_aggregate_statistics(
    days: int = Query(30, ge=1, le=365, description="Number of days to include"),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Get aggregate statistics across all user's evaluation sessions.
    
    Args:
        days: Number of days to include in statistics (default: 30)
        current_user: Current authenticated user
        db: Database session
        
    Returns:
        Aggregate statistics including trends and judge performance
    """
    from datetime import datetime, timedelta
    import statistics as stats
    
    # Calculate date threshold
    date_threshold = datetime.utcnow() - timedelta(days=days)
    
    # Query completed sessions within time range
    sessions = (
        db.query(EvaluationSession)
        .filter(
            EvaluationSession.user_id == current_user.id,
            EvaluationSession.status == EvaluationStatus.COMPLETED,
            EvaluationSession.created_at >= date_threshold
        )
        .order_by(EvaluationSession.created_at)
        .all()
    )
    
    if not sessions:
        return {
            "total_evaluations": 0,
            "date_range_days": days,
            "message": "No completed evaluations found in the specified time range"
        }
    
    # Extract metrics
    consensus_scores = [s.consensus_score for s in sessions if s.consensus_score is not None]
    hallucination_scores = [s.hallucination_score for s in sessions if s.hallucination_score is not None]
    variances = [s.session_metadata.variance for s in sessions if s.session_metadata and s.session_metadata.variance is not None]
    std_devs = [s.session_metadata.standard_deviation for s in sessions if s.session_metadata and s.session_metadata.standard_deviation is not None]
    processing_times = [s.session_metadata.processing_time_ms for s in sessions if s.session_metadata and s.session_metadata.processing_time_ms is not None]
    
    # Calculate aggregate statistics
    aggregate_stats = {
        "total_evaluations": len(sessions),
        "date_range_days": days,
        "consensus_score_stats": calculate_stats(consensus_scores) if consensus_scores else None,
        "hallucination_score_stats": calculate_stats(hallucination_scores) if hallucination_scores else None,
        "variance_stats": calculate_stats(variances) if variances else None,
        "std_dev_stats": calculate_stats(std_devs) if std_devs else None,
        "processing_time_stats": calculate_stats(processing_times) if processing_times else None,
    }
    
    # Calculate trends (compare first half vs second half)
    if len(consensus_scores) >= 4:
        mid_point = len(consensus_scores) // 2
        first_half_avg = stats.mean(consensus_scores[:mid_point])
        second_half_avg = stats.mean(consensus_scores[mid_point:])
        trend = ((second_half_avg - first_half_avg) / first_half_avg) * 100 if first_half_avg != 0 else 0
        
        aggregate_stats["consensus_score_trend"] = {
            "first_half_average": first_half_avg,
            "second_half_average": second_half_avg,
            "percent_change": trend,
            "direction": "improving" if trend > 0 else "declining" if trend < 0 else "stable"
        }
    
    # Judge performance statistics
    judge_stats = {}
    for session in sessions:
        for judge_result in session.judge_results:
            judge_name = judge_result.judge_name
            if judge_name not in judge_stats:
                judge_stats[judge_name] = {
                    "scores": [],
                    "confidences": [],
                    "response_times": [],
                    "total_evaluations": 0
                }
            
            judge_stats[judge_name]["scores"].append(judge_result.score)
            judge_stats[judge_name]["confidences"].append(judge_result.confidence)
            if judge_result.response_time_ms:
                judge_stats[judge_name]["response_times"].append(judge_result.response_time_ms)
            judge_stats[judge_name]["total_evaluations"] += 1
    
    # Calculate per-judge statistics
    judge_performance = {}
    for judge_name, data in judge_stats.items():
        judge_performance[judge_name] = {
            "total_evaluations": data["total_evaluations"],
            "score_stats": calculate_stats(data["scores"]),
            "confidence_stats": calculate_stats(data["confidences"]),
            "response_time_stats": calculate_stats(data["response_times"]) if data["response_times"] else None
        }
    
    aggregate_stats["judge_performance"] = judge_performance
    
    # Time series data for trends (daily aggregates)
    time_series = {}
    for session in sessions:
        date_key = session.created_at.date().isoformat()
        if date_key not in time_series:
            time_series[date_key] = {
                "date": date_key,
                "count": 0,
                "consensus_scores": [],
                "hallucination_scores": []
            }
        
        time_series[date_key]["count"] += 1
        if session.consensus_score is not None:
            time_series[date_key]["consensus_scores"].append(session.consensus_score)
        if session.hallucination_score is not None:
            time_series[date_key]["hallucination_scores"].append(session.hallucination_score)
    
    # Calculate daily averages
    daily_trends = []
    for date_key in sorted(time_series.keys()):
        data = time_series[date_key]
        daily_trends.append({
            "date": data["date"],
            "count": data["count"],
            "avg_consensus_score": stats.mean(data["consensus_scores"]) if data["consensus_scores"] else None,
            "avg_hallucination_score": stats.mean(data["hallucination_scores"]) if data["hallucination_scores"] else None
        })
    
    aggregate_stats["daily_trends"] = daily_trends
    
    return aggregate_stats


def calculate_stats(values: List[float]) -> dict:
    """Calculate statistical measures for a list of values."""
    import statistics as stats
    
    if not values:
        return None
    
    sorted_values = sorted(values)
    n = len(values)
    
    return {
        "count": n,
        "mean": stats.mean(values),
        "median": stats.median(values),
        "std_dev": stats.stdev(values) if n > 1 else 0,
        "variance": stats.variance(values) if n > 1 else 0,
        "min": min(values),
        "max": max(values),
        "q1": sorted_values[n // 4] if n >= 4 else sorted_values[0],
        "q3": sorted_values[3 * n // 4] if n >= 4 else sorted_values[-1]
    }


@router.post("/{session_id}/share")
async def create_shareable_link(
    session_id: UUID,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Generate a shareable link for an evaluation session.
    
    Args:
        session_id: Evaluation session UUID
        current_user: Current authenticated user
        db: Database session
        
    Returns:
        Shareable link URL
        
    Raises:
        HTTPException: If session not found or access denied
    """
    # Query session
    session = (
        db.query(EvaluationSession)
        .filter(
            EvaluationSession.id == session_id,
            EvaluationSession.user_id == current_user.id
        )
        .first()
    )
    
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Evaluation session not found"
        )
    
    # Generate shareable link
    # In production, you might want to use a URL shortener or token-based system
    shareable_url = f"/shared/{session_id}"
    
    logger.info(
        f"Shareable link created for session: {session_id} by user {current_user.username}"
    )
    
    return {
        "session_id": str(session_id),
        "shareable_url": shareable_url,
        "full_url": f"{shareable_url}",  # Frontend will prepend the base URL
        "created_at": datetime.utcnow().isoformat()
    }


@router.get("/shared/{session_id}", response_model=EvaluationSessionResponse)
async def get_shared_evaluation(
    session_id: UUID,
    db: Session = Depends(get_db)
):
    """
    Get evaluation session via shareable link (read-only, no authentication required).
    
    Args:
        session_id: Evaluation session UUID
        db: Database session
        
    Returns:
        Complete evaluation session with all results
        
    Raises:
        HTTPException: If session not found
    """
    # Query session with all relationships
    session = (
        db.query(EvaluationSession)
        .filter(EvaluationSession.id == session_id)
        .first()
    )
    
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Evaluation session not found"
        )
    
    # Only allow access to completed sessions
    if session.status != EvaluationStatus.COMPLETED:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only completed evaluations can be shared"
        )
    
    logger.info(f"Shared evaluation accessed: {session_id}")
    
    return session


@router.delete("/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_evaluation(
    session_id: UUID,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Delete an evaluation session.
    
    Args:
        session_id: Evaluation session UUID
        current_user: Current authenticated user
        db: Database session
        
    Raises:
        HTTPException: If session not found or access denied
    """
    # Query session
    session = (
        db.query(EvaluationSession)
        .filter(
            EvaluationSession.id == session_id,
            EvaluationSession.user_id == current_user.id
        )
        .first()
    )
    
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Evaluation session not found"
        )
    
    # Delete session (cascade will delete related records)
    db.delete(session)
    db.commit()
    
    logger.info(
        f"Evaluation session deleted: {session_id} by user {current_user.username}"
    )
    
    return Response(status_code=status.HTTP_204_NO_CONTENT)
