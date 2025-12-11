"""
Database configuration and session management.
"""
from sqlalchemy import create_engine, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from typing import Generator
import os
import logging

logger = logging.getLogger(__name__)

# Database URL from environment variable
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/llm_judge_auditor"
)

# Check if we should use SQLite fallback for development
USE_SQLITE = os.getenv("USE_SQLITE", "false").lower() == "true"

def create_db_engine():
    """Create database engine with fallback to SQLite if PostgreSQL unavailable."""
    global DATABASE_URL
    
    if USE_SQLITE or DATABASE_URL.startswith("sqlite"):
        # Use SQLite for development
        sqlite_url = "sqlite:///./llm_judge_auditor.db"
        logger.info(f"Using SQLite database: {sqlite_url}")
        return create_engine(
            sqlite_url,
            connect_args={"check_same_thread": False},
            echo=False
        )
    
    # Try PostgreSQL first
    try:
        pg_engine = create_engine(
            DATABASE_URL,
            pool_pre_ping=True,
            pool_size=20,
            max_overflow=40,
            pool_recycle=3600,
            pool_timeout=30,
            echo=False,
            connect_args={
                "connect_timeout": 10,
                "options": "-c statement_timeout=30000"
            }
        )
        # Test connection
        with pg_engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info("Connected to PostgreSQL database")
        return pg_engine
    except Exception as e:
        logger.warning(f"PostgreSQL unavailable ({e}), falling back to SQLite")
        sqlite_url = "sqlite:///./llm_judge_auditor.db"
        return create_engine(
            sqlite_url,
            connect_args={"check_same_thread": False},
            echo=False
        )

# Create engine with fallback
engine = create_db_engine()

# Create SessionLocal class
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create Base class for declarative models
Base = declarative_base()


def get_db() -> Generator:
    """
    Dependency function to get database session.
    Yields a database session and ensures it's closed after use.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """
    Initialize database by creating all tables.
    This should be called on application startup.
    """
    # Import all models to ensure they're registered with Base
    from . import models  # noqa: F401
    
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created successfully")
