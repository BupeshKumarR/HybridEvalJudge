# Alembic Database Migrations

This directory contains Alembic database migrations for the LLM Judge Auditor web application.

## Setup

Alembic is already configured and ready to use. The configuration is in `alembic.ini` and the environment setup is in `alembic/env.py`.

## Database URL

The database URL is read from the `DATABASE_URL` environment variable. Set it before running migrations:

```bash
export DATABASE_URL="postgresql://user:password@localhost:5432/llm_judge_auditor"
```

Or use the default from `app/database.py`:
```
postgresql://postgres:postgres@localhost:5432/llm_judge_auditor
```

## Common Commands

### Apply Migrations

Apply all pending migrations:

```bash
alembic upgrade head
```

### Rollback Migrations

Rollback the last migration:

```bash
alembic downgrade -1
```

Rollback to a specific revision:

```bash
alembic downgrade <revision_id>
```

Rollback all migrations:

```bash
alembic downgrade base
```

### Create New Migration

Create a new migration with autogenerate (requires database connection):

```bash
alembic revision --autogenerate -m "Description of changes"
```

Create an empty migration:

```bash
alembic revision -m "Description of changes"
```

### View Migration History

Show current revision:

```bash
alembic current
```

Show migration history:

```bash
alembic history
```

Show pending migrations:

```bash
alembic history --verbose
```

## Migration Files

Migration files are stored in `alembic/versions/`. Each file contains:

- `upgrade()`: Function to apply the migration
- `downgrade()`: Function to rollback the migration
- Revision identifiers for tracking

## Initial Migration

The initial migration (`d959cb7f9873_initial_schema.py`) creates all tables:

- `users`: User accounts
- `evaluation_sessions`: Evaluation sessions with results
- `judge_results`: Individual judge verdicts
- `flagged_issues`: Issues flagged by judges
- `verifier_verdicts`: Claim verification results
- `session_metadata`: Session metadata and statistics
- `user_preferences`: User configuration preferences

## Testing Migrations

Test migrations in a development environment before applying to production:

```bash
# Create a test database
createdb llm_judge_auditor_test

# Set test database URL
export DATABASE_URL="postgresql://postgres:postgres@localhost:5432/llm_judge_auditor_test"

# Apply migrations
alembic upgrade head

# Test rollback
alembic downgrade base

# Clean up
dropdb llm_judge_auditor_test
```

## Production Deployment

For production deployments:

1. Backup the database before applying migrations
2. Test migrations in a staging environment
3. Apply migrations during a maintenance window
4. Monitor for errors and be prepared to rollback

```bash
# Backup database
pg_dump llm_judge_auditor > backup_$(date +%Y%m%d_%H%M%S).sql

# Apply migrations
alembic upgrade head

# If issues occur, rollback
alembic downgrade -1
```

## Troubleshooting

### Connection Errors

If you get connection errors, verify:
- PostgreSQL is running
- Database exists
- DATABASE_URL is correct
- User has proper permissions

### Migration Conflicts

If you have migration conflicts:
1. Check `alembic history` to see the current state
2. Use `alembic downgrade` to rollback if needed
3. Resolve conflicts in migration files
4. Re-apply migrations

### Autogenerate Issues

If autogenerate doesn't detect changes:
- Ensure models are imported in `env.py`
- Check that `target_metadata` is set correctly
- Verify database connection is working
