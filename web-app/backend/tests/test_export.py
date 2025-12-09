"""
Tests for export functionality.
"""
import pytest
from uuid import uuid4
from datetime import datetime
from io import BytesIO

from app.models import (
    EvaluationSession,
    JudgeResult,
    FlaggedIssue,
    VerifierVerdict,
    SessionMetadata,
    User
)
from app.routers.evaluations import export_as_json, export_as_csv
from app.services.pdf_export_service import PDFExportService


@pytest.fixture
def sample_user(db_session):
    """Create a sample user."""
    user = User(
        username="testuser",
        email="test@example.com",
        password_hash="hashed_password"
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


@pytest.fixture
def sample_evaluation_session(db_session, sample_user):
    """Create a sample evaluation session with results."""
    session = EvaluationSession(
        user_id=sample_user.id,
        source_text="This is the source text for testing.",
        candidate_output="This is the candidate output for testing.",
        consensus_score=85.5,
        hallucination_score=15.2,
        confidence_interval_lower=80.0,
        confidence_interval_upper=90.0,
        inter_judge_agreement=0.85,
        status="completed",
        config={"judge_models": ["gpt-4", "claude-3"], "enable_retrieval": True}
    )
    db_session.add(session)
    db_session.commit()
    db_session.refresh(session)
    
    # Add judge results
    judge1 = JudgeResult(
        session_id=session.id,
        judge_name="gpt-4",
        score=87.0,
        confidence=0.92,
        reasoning="The output is mostly accurate with minor issues.",
        response_time_ms=1500
    )
    judge2 = JudgeResult(
        session_id=session.id,
        judge_name="claude-3",
        score=84.0,
        confidence=0.88,
        reasoning="Good quality output with some concerns.",
        response_time_ms=1200
    )
    db_session.add(judge1)
    db_session.add(judge2)
    db_session.commit()
    db_session.refresh(judge1)
    db_session.refresh(judge2)
    
    # Add flagged issue
    issue = FlaggedIssue(
        judge_result_id=judge1.id,
        issue_type="factual_error",
        severity="low",
        description="Minor factual inconsistency detected.",
        evidence={"detail": "test evidence"}
    )
    db_session.add(issue)
    
    # Add verifier verdict
    verdict = VerifierVerdict(
        session_id=session.id,
        claim_text="Test claim",
        label="SUPPORTED",
        confidence=0.95,
        evidence={"sources": ["source1"]},
        reasoning="Strong evidence supports this claim."
    )
    db_session.add(verdict)
    
    # Add session metadata
    metadata = SessionMetadata(
        session_id=session.id,
        total_judges=2,
        judges_used=["gpt-4", "claude-3"],
        aggregation_strategy="weighted_average",
        retrieval_enabled=True,
        num_retrieved_passages=5,
        num_verifier_verdicts=1,
        processing_time_ms=3000,
        variance=2.25,
        standard_deviation=1.5
    )
    db_session.add(metadata)
    
    db_session.commit()
    db_session.refresh(session)
    
    return session


def test_export_as_json(sample_evaluation_session):
    """Test JSON export functionality."""
    response = export_as_json(sample_evaluation_session)
    
    assert response.status_code == 200
    assert response.media_type == "application/json"
    assert "attachment" in response.headers["Content-Disposition"]
    assert f"evaluation_{sample_evaluation_session.id}.json" in response.headers["Content-Disposition"]
    
    # Verify content is valid JSON
    import json
    content = response.body.decode('utf-8')
    data = json.loads(content)
    
    assert data["id"] == str(sample_evaluation_session.id)
    assert data["consensus_score"] == 85.5
    assert data["hallucination_score"] == 15.2
    assert len(data["judge_results"]) == 2
    assert len(data["verifier_verdicts"]) == 1


def test_export_as_csv(sample_evaluation_session):
    """Test CSV export functionality."""
    response = export_as_csv(sample_evaluation_session)
    
    assert response.status_code == 200
    assert response.media_type == "text/csv"
    assert "attachment" in response.headers["Content-Disposition"]
    assert f"evaluation_{sample_evaluation_session.id}.csv" in response.headers["Content-Disposition"]
    
    # Verify content contains expected data
    # For StreamingResponse, we need to iterate through the body
    import asyncio
    
    async def get_content():
        chunks = []
        async for chunk in response.body_iterator:
            # Handle both bytes and strings
            if isinstance(chunk, bytes):
                chunks.append(chunk.decode('utf-8'))
            else:
                chunks.append(chunk)
        return "".join(chunks)
    
    content = asyncio.run(get_content())
    
    assert "Session ID" in content
    assert str(sample_evaluation_session.id) in content
    assert "85.5" in content  # consensus score
    assert "Judge Results" in content
    assert "gpt-4" in content
    assert "claude-3" in content


def test_pdf_export_service(sample_evaluation_session):
    """Test PDF export service."""
    pdf_service = PDFExportService()
    pdf_buffer = pdf_service.generate_pdf(sample_evaluation_session)
    
    assert isinstance(pdf_buffer, BytesIO)
    
    # Read PDF content
    pdf_content = pdf_buffer.read()
    
    # Verify it's a PDF (starts with PDF magic bytes)
    assert pdf_content.startswith(b'%PDF')
    
    # Verify PDF has content (not empty)
    assert len(pdf_content) > 1000  # PDFs are typically larger than 1KB


def test_pdf_export_with_minimal_data(db_session, sample_user):
    """Test PDF export with minimal evaluation data."""
    session = EvaluationSession(
        user_id=sample_user.id,
        source_text="Minimal source text.",
        candidate_output="Minimal candidate output.",
        status="completed"
    )
    db_session.add(session)
    db_session.commit()
    db_session.refresh(session)
    
    pdf_service = PDFExportService()
    pdf_buffer = pdf_service.generate_pdf(session)
    
    assert isinstance(pdf_buffer, BytesIO)
    pdf_content = pdf_buffer.read()
    assert pdf_content.startswith(b'%PDF')


def test_json_export_includes_all_fields(sample_evaluation_session):
    """Test that JSON export includes all expected fields."""
    response = export_as_json(sample_evaluation_session)
    
    import json
    content = response.body.decode('utf-8')
    data = json.loads(content)
    
    # Check top-level fields
    assert "id" in data
    assert "user_id" in data
    assert "source_text" in data
    assert "candidate_output" in data
    assert "consensus_score" in data
    assert "hallucination_score" in data
    assert "confidence_interval_lower" in data
    assert "confidence_interval_upper" in data
    assert "inter_judge_agreement" in data
    assert "status" in data
    assert "created_at" in data
    
    # Check nested structures
    assert "judge_results" in data
    assert len(data["judge_results"]) > 0
    assert "judge_name" in data["judge_results"][0]
    assert "score" in data["judge_results"][0]
    assert "confidence" in data["judge_results"][0]
    
    assert "verifier_verdicts" in data
    assert len(data["verifier_verdicts"]) > 0
    
    assert "session_metadata" in data
    assert data["session_metadata"]["total_judges"] == 2
