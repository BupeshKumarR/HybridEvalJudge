# Database Setup Documentation

This document describes the database setup for the LLM Judge Auditor web application.

## Overview

The database setup consists of three main components:

1. **SQL Schema** (`migrations/schema.sql`) - Raw SQL schema with all tables, indexes, and constraints
2. **SQLAlchemy Models** (`app/models.py`) - Python ORM models matching the schema
3. **Alembic Migrations** (`alembic/versions/`) - Version-controlled database migrations

## Components

### 1. Database Configuration (`app/database.py`)

Provides:
- SQLAlchemy engine configuration
- Session management
- Database connection pooling
- Dependency injection for FastAPI

Key features:
- Connection pool: 10 base connections, 20 max overflow
- Automatic session cleanup
- Environment-based configuration

### 2. SQL Schema (`migrations/schema.sql`)

Complete PostgreSQL schema including:

**Tables:**
- `users` - User accounts with authentication
- `evaluation_sessions` - Evaluation sessions with results
- `judge_results` - Individual judge verdicts
- `flagged_issues` - Issues flagged by judges
- `verifier_verdicts` - Claim verification results
- `session_metadata` - Session statistics and metadata
- `user_preferences` - User configuration preferences

**Indexes:**
- Primary key indexes on all tables
- Foreign key indexes for relationships
- Composite indexes for common queries
- Full-text search indexes on text fields

**Constraints:**
- Foreign key constraints with CASCADE delete
- Check constraints for valid ranges
- Unique constraints on usernames and emails
- Text span validation

**Views:**
- `session_summaries` - Optimized view for history queries
- `user_statistics` - Materialized view for dashboard

### 3. SQLAlchemy Models (`app/models.py`)

ORM models with:
- Proper relationships and cascade deletes
- Field validation using `@validates` decorators
- Type hints for better IDE support
- Comprehensive docstrings

**Model Hierarchy:**
```
User
├── EvaluationSession
│   ├── JudgeResult
│   │   └── FlaggedIssue
│   ├── VerifierVerdict
│   └── SessionMetadata
└── UserPreference
```

**Validation:**
- Username: minimum 3 characters
- Email: valid email format
- Scores: 0-100 range
- Confidence: 0-1 range
- Status: enum validation
- Issue types: enum validation
- Severity levels: enum validation

### 4. Pydantic Schemas (`app/schemas.py`)

API request/response validation:
- Input validation for API requests
- Response serialization
- Type safety with Pydantic v2
- Enum definitions for constants

### 5. Alembic Migrations (`alembic/`)

Database version control:
- Initial migration creates all tables
- Upgrade/downgrade support
- Autogenerate capability
- Environment-based configuration

## Setup Instructions

### Prerequisites

1. PostgreSQL 15+ installed and running
2. Python 3.11+ with dependencies installed
3. Environment variables configured

### Installation

1. **Install dependencies:**
```bash
cd web-app/backend
pip install -r requirements.txt
```

2. **Set database URL:**
```bash
export DATABASE_URL="postgresql://user:password@localhost:5432/llm_judge_auditor"
```

3. **Create database:**
```bash
createdb llm_judge_auditor
```

4. **Run migrations:**
```bash
alembic upgrade head
```

### Alternative: Manual Schema Creation

If you prefer to create the schema manually:

```bash
psql -U postgres -d llm_judge_auditor -f migrations/schema.sql
```

## Verification

### Check Tables

```sql
\dt
```

Expected output:
```
 Schema |        Name         | Type  |  Owner   
--------+---------------------+-------+----------
 public | evaluation_sessions | table | postgres
 public | flagged_issues      | table | postgres
 public | judge_results       | table | postgres
 public | session_metadata    | table | postgres
 public | user_preferences    | table | postgres
 public | users               | table | postgres
 public | verifier_verdicts   | table | postgres
```

### Check Indexes

```sql
\di
```

Should show all indexes including:
- Primary key indexes
- Foreign key indexes
- Composite indexes
- Full-text search indexes

### Check Constraints

```sql
SELECT conname, contype FROM pg_constraint WHERE conrelid = 'users'::regclass;
```

## Testing

### Unit Tests

Run model tests:
```bash
pytest tests/test_database.py -v
```

Tests cover:
- Model creation
- Field validation
- Relationship integrity
- Cascade deletes
- Constraint enforcement

### Integration Tests

Test with actual database:
```bash
# Create test database
createdb llm_judge_auditor_test

# Set test database URL
export DATABASE_URL="postgresql://postgres:postgres@localhost:5432/llm_judge_auditor_test"

# Run migrations
alembic upgrade head

# Run tests
pytest tests/ -v

# Clean up
dropdb llm_judge_auditor_test
```

## Performance Considerations

### Indexes

All common query patterns are indexed:
- User lookups by username/email
- Session queries by user and date
- Score-based filtering
- Status filtering
- Full-text search on text fields

### Connection Pooling

Configured for optimal performance:
- Pool size: 10 connections
- Max overflow: 20 connections
- Pre-ping enabled for connection health checks

### Query Optimization

- Use of composite indexes for multi-column queries
- Materialized view for expensive aggregations
- Proper foreign key indexes for joins

## Maintenance

### Backup

Regular backups recommended:
```bash
pg_dump llm_judge_auditor > backup_$(date +%Y%m%d).sql
```

### Refresh Materialized Views

Refresh user statistics periodically:
```sql
REFRESH MATERIALIZED VIEW CONCURRENTLY user_statistics;
```

Or use the helper function:
```sql
SELECT refresh_user_statistics();
```

### Cleanup Old Data

Archive old sessions if needed:
```sql
-- Archive sessions older than 1 year
DELETE FROM evaluation_sessions 
WHERE created_at < NOW() - INTERVAL '1 year';
```

## Troubleshooting

### Connection Issues

If you can't connect to the database:
1. Check PostgreSQL is running: `pg_isready`
2. Verify DATABASE_URL is correct
3. Check user permissions
4. Verify database exists

### Migration Issues

If migrations fail:
1. Check current revision: `alembic current`
2. View history: `alembic history`
3. Rollback if needed: `alembic downgrade -1`
4. Check database logs for errors

### Performance Issues

If queries are slow:
1. Check indexes are being used: `EXPLAIN ANALYZE <query>`
2. Verify connection pool settings
3. Check for missing indexes
4. Consider adding more specific indexes

## Schema Diagram

```
┌─────────────┐
│    users    │
├─────────────┤
│ id (PK)     │
│ username    │
│ email       │
│ password    │
└──────┬──────┘
       │
       │ 1:N
       │
┌──────▼──────────────────┐
│  evaluation_sessions    │
├─────────────────────────┤
│ id (PK)                 │
│ user_id (FK)            │
│ source_text             │
│ candidate_output        │
│ consensus_score         │
│ hallucination_score     │
│ status                  │
└──────┬──────────────────┘
       │
       ├─────────────────┐
       │ 1:N             │ 1:N
       │                 │
┌──────▼──────────┐ ┌────▼─────────────┐
│  judge_results  │ │ verifier_verdicts│
├─────────────────┤ ├──────────────────┤
│ id (PK)         │ │ id (PK)          │
│ session_id (FK) │ │ session_id (FK)  │
│ judge_name      │ │ claim_text       │
│ score           │ │ label            │
│ confidence      │ │ confidence       │
└──────┬──────────┘ └──────────────────┘
       │
       │ 1:N
       │
┌──────▼──────────┐
│ flagged_issues  │
├─────────────────┤
│ id (PK)         │
│ judge_result_id │
│ issue_type      │
│ severity        │
│ description     │
└─────────────────┘
```

## Summary

The database setup is complete with:
- ✅ Comprehensive SQL schema with all tables, indexes, and constraints
- ✅ SQLAlchemy ORM models with validation and relationships
- ✅ Pydantic schemas for API validation
- ✅ Alembic migration system for version control
- ✅ Test suite for model validation
- ✅ Documentation and setup instructions

All requirements from task 2 have been implemented:
- 2.1: PostgreSQL schema with indexes and constraints ✅
- 2.2: SQLAlchemy models with relationships and validation ✅
- 2.3: Alembic migration system with initial migration ✅
