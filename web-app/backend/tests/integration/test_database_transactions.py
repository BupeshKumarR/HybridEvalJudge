"""
Integration tests for database transactions.
Tests transaction handling, rollbacks, and data integrity.
"""
import pytest
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from app.models import User, EvaluationSession, JudgeResult, FlaggedIssue
from tests.conftest import db_session, created_user


class TestDatabaseTransactions:
    """Test database transaction handling."""

    def test_cascade_delete_evaluation_session(
        self, db_session: Session, created_user: User
    ):
        """Test that deleting a session cascades to related records."""
        # Create evaluation session
        session = EvaluationSession(
            user_id=created_user.id,
            source_text="Test source",
            candidate_output="Test output",
            status="completed",
            consensus_score=85.0
        )
        db_session.add(session)
        db_session.commit()
        db_session.refresh(session)

        # Add judge result
        judge_result = JudgeResult(
            session_id=session.id,
            judge_name="gpt-4",
            score=85.0,
            confidence=0.9,
            reasoning="Test reasoning"
        )
        db_session.add(judge_result)
        db_session.commit()
        db_session.refresh(judge_result)

        # Add flagged issue
        issue = FlaggedIssue(
            judge_result_id=judge_result.id,
            issue_type="factual_error",
            severity="medium",
            description="Test issue"
        )
        db_session.add(issue)
        db_session.commit()

        # Delete session
        db_session.delete(session)
        db_session.commit()

        # Verify cascade delete
        assert db_session.query(JudgeResult).filter(
            JudgeResult.id == judge_result.id
        ).first() is None

        assert db_session.query(FlaggedIssue).filter(
            FlaggedIssue.id == issue.id
        ).first() is None

    def test_unique_constraint_enforcement(
        self, db_session: Session
    ):
        """Test that unique constraints are enforced."""
        # Create first user
        user1 = User(
            username="uniqueuser",
            email="unique@test.com",
            password_hash="hashed_password"
        )
        db_session.add(user1)
        db_session.commit()

        # Try to create user with same username
        user2 = User(
            username="uniqueuser",
            email="different@test.com",
            password_hash="hashed_password"
        )
        db_session.add(user2)

        with pytest.raises(IntegrityError):
            db_session.commit()

        db_session.rollback()

    def test_foreign_key_constraint(
        self, db_session: Session
    ):
        """Test that foreign key constraints are enforced."""
        # Try to create evaluation session with invalid user_id
        session = EvaluationSession(
            user_id="00000000-0000-0000-0000-000000000000",  # Non-existent user
            source_text="Test",
            candidate_output="Test",
            status="pending"
        )
        db_session.add(session)

        with pytest.raises(IntegrityError):
            db_session.commit()

        db_session.rollback()

    def test_transaction_rollback(
        self, db_session: Session, created_user: User
    ):
        """Test that transactions can be rolled back."""
        # Create session
        session = EvaluationSession(
            user_id=created_user.id,
            source_text="Rollback test",
            candidate_output="Rollback test",
            status="pending"
        )
        db_session.add(session)
        db_session.flush()  # Flush but don't commit

        session_id = session.id

        # Rollback
        db_session.rollback()

        # Verify session was not persisted
        assert db_session.query(EvaluationSession).filter(
            EvaluationSession.id == session_id
        ).first() is None

    def test_concurrent_updates(
        self, db_session: Session, created_user: User
    ):
        """Test handling of concurrent updates."""
        # Create session
        session = EvaluationSession(
            user_id=created_user.id,
            source_text="Concurrent test",
            candidate_output="Concurrent test",
            status="pending",
            consensus_score=None
        )
        db_session.add(session)
        db_session.commit()
        db_session.refresh(session)

        # Simulate concurrent update
        session.consensus_score = 85.0
        session.status = "completed"
        db_session.commit()

        # Verify update
        updated_session = db_session.query(EvaluationSession).filter(
            EvaluationSession.id == session.id
        ).first()

        assert updated_session.consensus_score == 85.0
        assert updated_session.status == "completed"

    def test_bulk_insert_performance(
        self, db_session: Session, created_user: User
    ):
        """Test bulk insert operations."""
        # Create multiple sessions
        sessions = []
        for i in range(10):
            session = EvaluationSession(
                user_id=created_user.id,
                source_text=f"Bulk test {i}",
                candidate_output=f"Bulk output {i}",
                status="pending"
            )
            sessions.append(session)

        # Bulk insert
        db_session.bulk_save_objects(sessions)
        db_session.commit()

        # Verify all were inserted
        count = db_session.query(EvaluationSession).filter(
            EvaluationSession.user_id == created_user.id
        ).count()

        assert count >= 10

    def test_query_filtering_and_ordering(
        self, db_session: Session, created_user: User
    ):
        """Test complex queries with filtering and ordering."""
        # Create sessions with different scores
        scores = [90.0, 75.0, 85.0, 60.0, 95.0]
        for score in scores:
            session = EvaluationSession(
                user_id=created_user.id,
                source_text="Query test",
                candidate_output="Query test",
                status="completed",
                consensus_score=score
            )
            db_session.add(session)

        db_session.commit()

        # Query with filtering and ordering
        high_score_sessions = db_session.query(EvaluationSession).filter(
            EvaluationSession.user_id == created_user.id,
            EvaluationSession.consensus_score >= 80.0
        ).order_by(EvaluationSession.consensus_score.desc()).all()

        assert len(high_score_sessions) == 3
        assert high_score_sessions[0].consensus_score == 95.0
        assert high_score_sessions[1].consensus_score == 90.0
        assert high_score_sessions[2].consensus_score == 85.0
