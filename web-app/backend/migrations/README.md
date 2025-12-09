# Database Migrations

This directory contains database migration scripts for the LLM Judge Auditor web application.

## Schema Overview

The database schema includes the following tables:

### Core Tables

- **users**: User account information
- **evaluation_sessions**: Evaluation session data with source text, candidate output, and results
- **judge_results**: Individual judge verdicts for each evaluation session
- **flagged_issues**: Issues flagged by judges during evaluation
- **verifier_verdicts**: Claim verification results from the verifier component
- **session_metadata**: Metadata and statistics for evaluation sessions
- **user_preferences**: User-specific configuration preferences

### Views

- **session_summaries**: Optimized view for history queries with session previews
- **user_statistics**: Materialized view for user dashboard statistics

## Indexes

The schema includes comprehensive indexes for:
- User lookups (username, email)
- Session queries (user_id + created_at, status, scores)
- Full-text search (source_text, candidate_output)
- Judge results (session_id, judge_name)
- Issue tracking (type, severity)

## Constraints

All tables include appropriate constraints:
- Foreign key constraints with CASCADE delete
- Check constraints for valid ranges (scores 0-100, confidence 0-1)
- Enum-like constraints for status and type fields
- Text span validation

## Running Migrations

### Manual Schema Creation

To create the schema manually:

```bash
psql -U postgres -d llm_judge_auditor -f migrations/schema.sql
```

### Using Alembic

Alembic migrations are managed in the `alembic/` directory. See the main README for Alembic usage.

## Performance Considerations

- Connection pooling configured in `database.py`
- Indexes optimized for common query patterns
- Materialized view for expensive aggregations
- Full-text search indexes for text queries

## Maintenance

### Refresh User Statistics

The `user_statistics` materialized view should be refreshed periodically:

```sql
SELECT refresh_user_statistics();
```

Or via cron job:

```bash
# Refresh every hour
0 * * * * psql -U postgres -d llm_judge_auditor -c "SELECT refresh_user_statistics();"
```
