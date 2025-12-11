"""Pytest configuration and fixtures."""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event, TypeDecorator, CHAR, TEXT
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
import uuid
import json

from app.main import app
from app.database import Base, get_db
from app.models import User
from app.auth import get_password_hash


# Custom UUID type for SQLite
class GUID(TypeDecorator):
    """Platform-independent GUID type.
    Uses PostgreSQL's UUID type, otherwise uses CHAR(36), storing as stringified hex values.
    """
    impl = CHAR
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == 'postgresql':
            return dialect.type_descriptor(UUID())
        else:
            return dialect.type_descriptor(CHAR(36))

    def process_bind_param(self, value, dialect):
        if value is None:
            return value
        elif dialect.name == 'postgresql':
            return str(value)
        else:
            if isinstance(value, uuid.UUID):
                return str(value)
            else:
                return str(uuid.UUID(value))

    def process_result_value(self, value, dialect):
        if value is None:
            return value
        else:
            if not isinstance(value, uuid.UUID):
                value = uuid.UUID(value)
            return value


# Custom JSON type for SQLite
class JSONType(TypeDecorator):
    """Platform-independent JSON type.
    Uses PostgreSQL's JSONB type, otherwise uses TEXT with JSON serialization.
    """
    impl = TEXT
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == 'postgresql':
            return dialect.type_descriptor(JSONB())
        else:
            return dialect.type_descriptor(TEXT())

    def process_bind_param(self, value, dialect):
        if value is None:
            return value
        if dialect.name == 'postgresql':
            return value
        else:
            return json.dumps(value)

    def process_result_value(self, value, dialect):
        if value is None:
            return value
        if dialect.name == 'postgresql':
            return value
        else:
            return json.loads(value)


# Create in-memory SQLite database for testing
SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)

# Replace UUID and JSONB types for SQLite and remove PostgreSQL-specific constraints
@event.listens_for(Base.metadata, "before_create")
def receive_before_create(target, connection, **kw):
    """Replace UUID/JSONB columns and remove PostgreSQL constraints for SQLite."""
    if connection.dialect.name == "sqlite":
        for table in target.tables.values():
            # Replace UUID and JSONB columns
            for column in table.columns:
                if isinstance(column.type, UUID):
                    column.type = GUID()
                elif isinstance(column.type, JSONB):
                    column.type = JSONType()
            
            # Remove all check constraints for SQLite (they use PostgreSQL-specific functions)
            constraints_to_remove = []
            for constraint in table.constraints:
                # Keep only primary key and foreign key constraints
                if constraint.__class__.__name__ == 'CheckConstraint':
                    constraints_to_remove.append(constraint)
            
            for constraint in constraints_to_remove:
                table.constraints.remove(constraint)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def override_get_db():
    """Override database dependency for testing."""
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()


@pytest.fixture(scope="function")
def db_session():
    """Create a fresh database session for each test."""
    Base.metadata.create_all(bind=engine)
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=engine)


@pytest.fixture
def client(db_session):
    """Create a test client with database override."""
    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture
def test_user_data():
    """Test user data."""
    return {
        "username": "testuser",
        "email": "test@example.com",
        "password": "testpass123"
    }


@pytest.fixture
def created_user(db_session, test_user_data):
    """Create a test user in the database."""
    user = User(
        username=test_user_data["username"],
        email=test_user_data["email"],
        password_hash=get_password_hash(test_user_data["password"])
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


@pytest.fixture
def auth_token(client, test_user_data, created_user):
    """Get authentication token for test user."""
    response = client.post(
        "/api/v1/auth/login/json",
        json={
            "username": test_user_data["username"],
            "password": test_user_data["password"]
        }
    )
    return response.json()["access_token"]


@pytest.fixture
def auth_headers(auth_token):
    """Get authorization headers with token."""
    return {"Authorization": f"Bearer {auth_token}"}


@pytest.fixture
def test_evaluation_session(db_session, created_user):
    """Create a test evaluation session."""
    from app.models import EvaluationSession
    from app.schemas import EvaluationStatus
    
    session = EvaluationSession(
        user_id=created_user.id,
        source_text="This is a test source text for evaluation.",
        candidate_output="This is a test candidate output to be evaluated.",
        status=EvaluationStatus.PENDING,
        config={"judge_models": ["gpt-4"], "enable_retrieval": True}
    )
    db_session.add(session)
    db_session.commit()
    db_session.refresh(session)
    return session


@pytest.fixture
def test_chat_session(db_session, created_user):
    """Create a test chat session."""
    from app.models import ChatSession
    
    session = ChatSession(
        user_id=created_user.id,
        ollama_model="llama3.2"
    )
    db_session.add(session)
    db_session.commit()
    db_session.refresh(session)
    return session


@pytest.fixture
def test_chat_message(db_session, test_chat_session):
    """Create a test chat message."""
    from app.models import ChatMessage
    
    message = ChatMessage(
        session_id=test_chat_session.id,
        role="user",
        content="What is the capital of France?"
    )
    db_session.add(message)
    db_session.commit()
    db_session.refresh(message)
    return message
