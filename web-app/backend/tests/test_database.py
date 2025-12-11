"""
Tests for database models and migrations.
"""
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import uuid

from app.database import Base
from app.models import (
    User, EvaluationSession, JudgeResult, FlaggedIssue,
    VerifierVerdict, SessionMetadata, UserPreference
)


@pytest.fixture
def db_engine():
    """Create an in-memory SQLite database for testing."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    yield engine
    Base.metadata.drop_all(engine)


@pytest.fixture
def db_session(db_engine):
    """Create a database session for testing."""
    Session = sessionmaker(bind=db_engine)
    session = Session()
    yield session
    session.close()


class TestUserModel:
    """Tests for User model."""
    
    def test_create_user(self, db_session):
        """Test creating a user."""
        user = User(
            username="testuser",
            email="test@example.com",
            password_hash="hashed_password"
        )
        db_session.add(user)
        db_session.commit()
        
        assert user.id is not None
        assert user.username == "testuser"
        assert user.email == "test@example.com"
        assert user.created_at is not None
    
    def test_username_validation(self, db_session):
        """Test username length validation."""
        with pytest.raises(ValueError, match="at least 3 characters"):
            user = User(
                username="ab",  # Too short
                email="test@example.com",
                password_hash="hashed_password"
            )
    
    def test_email_validation(self, db_session):
        """Test email format validation."""
        with pytest.raises(ValueError, match="Invalid email format"):
            user = User(
                username="testuser",
                email="invalid-email",
                password_hash="hashed_password"
            )


class TestEvaluationSessionModel:
    """Tests for EvaluationSession model."""
    
    def test_create_evaluation_session(self, db_session):
        """Test creating an evaluation session."""
        user = User(
            username="testuser",
            email="test@example.com",
            password_hash="hashed_password"
        )
        db_session.add(user)
        db_session.commit()
        
        session = EvaluationSession(
            user_id=user.id,
            source_text="Source text",
            candidate_output="Candidate output",
            status="pending"
        )
        db_session.add(session)
        db_session.commit()
        
        assert session.id is not None
        assert session.user_id == user.id
        assert session.status == "pending"
        assert session.created_at is not None
    
    def test_status_validation(self, db_session):
        """Test status validation."""
        user = User(
            username="testuser",
            email="test@example.com",
            password_hash="hashed_password"
        )
        db_session.add(user)
        db_session.commit()
        
        with pytest.raises(ValueError, match="Status must be one of"):
            session = EvaluationSession(
                user_id=user.id,
                source_text="Source text",
                candidate_output="Candidate output",
                status="invalid_status"
            )
    
    def test_score_validation(self, db_session):
        """Test score range validation."""
        user = User(
            username="testuser",
            email="test@example.com",
            password_hash="hashed_password"
        )
        db_session.add(user)
        db_session.commit()
        
        with pytest.raises(ValueError, match="must be between 0 and 100"):
            session = EvaluationSession(
                user_id=user.id,
                source_text="Source text",
                candidate_output="Candidate output",
                consensus_score=150  # Invalid: > 100
            )


class TestJudgeResultModel:
    """Tests for JudgeResult model."""
    
    def test_create_judge_result(self, db_session):
        """Test creating a judge result."""
        user = User(
            username="testuser",
            email="test@example.com",
            password_hash="hashed_password"
        )
        db_session.add(user)
        db_session.commit()
        
        session = EvaluationSession(
            user_id=user.id,
            source_text="Source text",
            candidate_output="Candidate output"
        )
        db_session.add(session)
        db_session.commit()
        
        judge_result = JudgeResult(
            session_id=session.id,
            judge_name="test_judge",
            score=85.5,
            confidence=0.9,
            reasoning="Test reasoning"
        )
        db_session.add(judge_result)
        db_session.commit()
        
        assert judge_result.id is not None
        assert judge_result.session_id == session.id
        assert judge_result.score == 85.5
        assert judge_result.confidence == 0.9
    
    def test_score_validation(self, db_session):
        """Test score validation."""
        user = User(
            username="testuser",
            email="test@example.com",
            password_hash="hashed_password"
        )
        db_session.add(user)
        db_session.commit()
        
        session = EvaluationSession(
            user_id=user.id,
            source_text="Source text",
            candidate_output="Candidate output"
        )
        db_session.add(session)
        db_session.commit()
        
        with pytest.raises(ValueError, match="Score must be between 0 and 100"):
            judge_result = JudgeResult(
                session_id=session.id,
                judge_name="test_judge",
                score=150,  # Invalid
                confidence=0.9
            )
    
    def test_confidence_validation(self, db_session):
        """Test confidence validation."""
        user = User(
            username="testuser",
            email="test@example.com",
            password_hash="hashed_password"
        )
        db_session.add(user)
        db_session.commit()
        
        session = EvaluationSession(
            user_id=user.id,
            source_text="Source text",
            candidate_output="Candidate output"
        )
        db_session.add(session)
        db_session.commit()
        
        with pytest.raises(ValueError, match="Confidence must be between 0 and 1"):
            judge_result = JudgeResult(
                session_id=session.id,
                judge_name="test_judge",
                score=85,
                confidence=1.5  # Invalid
            )


class TestFlaggedIssueModel:
    """Tests for FlaggedIssue model."""
    
    def test_create_flagged_issue(self, db_session):
        """Test creating a flagged issue."""
        user = User(
            username="testuser",
            email="test@example.com",
            password_hash="hashed_password"
        )
        db_session.add(user)
        db_session.commit()
        
        session = EvaluationSession(
            user_id=user.id,
            source_text="Source text",
            candidate_output="Candidate output"
        )
        db_session.add(session)
        db_session.commit()
        
        judge_result = JudgeResult(
            session_id=session.id,
            judge_name="test_judge",
            score=85,
            confidence=0.9
        )
        db_session.add(judge_result)
        db_session.commit()
        
        issue = FlaggedIssue(
            judge_result_id=judge_result.id,
            issue_type="hallucination",
            severity="high",
            description="Test issue"
        )
        db_session.add(issue)
        db_session.commit()
        
        assert issue.id is not None
        assert issue.issue_type == "hallucination"
        assert issue.severity == "high"
    
    def test_issue_type_validation(self, db_session):
        """Test issue type validation."""
        user = User(
            username="testuser",
            email="test@example.com",
            password_hash="hashed_password"
        )
        db_session.add(user)
        db_session.commit()
        
        session = EvaluationSession(
            user_id=user.id,
            source_text="Source text",
            candidate_output="Candidate output"
        )
        db_session.add(session)
        db_session.commit()
        
        judge_result = JudgeResult(
            session_id=session.id,
            judge_name="test_judge",
            score=85,
            confidence=0.9
        )
        db_session.add(judge_result)
        db_session.commit()
        
        with pytest.raises(ValueError, match="Issue type must be one of"):
            issue = FlaggedIssue(
                judge_result_id=judge_result.id,
                issue_type="invalid_type",
                severity="high",
                description="Test issue"
            )


class TestCascadeDeletes:
    """Tests for cascade delete behavior."""
    
    def test_delete_user_cascades(self, db_session):
        """Test that deleting a user cascades to sessions."""
        user = User(
            username="testuser",
            email="test@example.com",
            password_hash="hashed_password"
        )
        db_session.add(user)
        db_session.commit()
        
        session = EvaluationSession(
            user_id=user.id,
            source_text="Source text",
            candidate_output="Candidate output"
        )
        db_session.add(session)
        db_session.commit()
        
        session_id = session.id
        
        # Delete user
        db_session.delete(user)
        db_session.commit()
        
        # Session should be deleted
        deleted_session = db_session.query(EvaluationSession).filter_by(id=session_id).first()
        assert deleted_session is None
    
    def test_delete_session_cascades(self, db_session):
        """Test that deleting a session cascades to judge results."""
        user = User(
            username="testuser",
            email="test@example.com",
            password_hash="hashed_password"
        )
        db_session.add(user)
        db_session.commit()
        
        session = EvaluationSession(
            user_id=user.id,
            source_text="Source text",
            candidate_output="Candidate output"
        )
        db_session.add(session)
        db_session.commit()
        
        judge_result = JudgeResult(
            session_id=session.id,
            judge_name="test_judge",
            score=85,
            confidence=0.9
        )
        db_session.add(judge_result)
        db_session.commit()
        
        judge_result_id = judge_result.id
        
        # Delete session
        db_session.delete(session)
        db_session.commit()
        
        # Judge result should be deleted
        deleted_result = db_session.query(JudgeResult).filter_by(id=judge_result_id).first()
        assert deleted_result is None
