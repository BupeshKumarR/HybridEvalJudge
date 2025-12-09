"""Initial schema

Revision ID: d959cb7f9873
Revises: 
Create Date: 2025-12-08 23:32:54.587137

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'd959cb7f9873'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Enable UUID extension
    op.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"')
    
    # Create users table
    op.create_table(
        'users',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('uuid_generate_v4()')),
        sa.Column('username', sa.String(255), nullable=False, unique=True),
        sa.Column('email', sa.String(255), nullable=False, unique=True),
        sa.Column('password_hash', sa.String(255), nullable=False),
        sa.Column('created_at', sa.TIMESTAMP, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('last_login', sa.TIMESTAMP, nullable=True),
        sa.CheckConstraint("char_length(username) >= 3", name='users_username_check'),
        sa.CheckConstraint("email ~* '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}$'", name='users_email_check')
    )
    op.create_index('idx_users_username', 'users', ['username'])
    op.create_index('idx_users_email', 'users', ['email'])
    op.create_index('idx_users_created_at', 'users', ['created_at'])
    
    # Create evaluation_sessions table
    op.create_table(
        'evaluation_sessions',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('uuid_generate_v4()')),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=False),
        sa.Column('source_text', sa.Text, nullable=False),
        sa.Column('candidate_output', sa.Text, nullable=False),
        sa.Column('consensus_score', sa.Float, nullable=True),
        sa.Column('hallucination_score', sa.Float, nullable=True),
        sa.Column('confidence_interval_lower', sa.Float, nullable=True),
        sa.Column('confidence_interval_upper', sa.Float, nullable=True),
        sa.Column('inter_judge_agreement', sa.Float, nullable=True),
        sa.Column('status', sa.String(50), nullable=False, server_default='pending'),
        sa.Column('config', postgresql.JSONB, nullable=True),
        sa.Column('created_at', sa.TIMESTAMP, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('completed_at', sa.TIMESTAMP, nullable=True),
        sa.CheckConstraint("status IN ('pending', 'in_progress', 'completed', 'failed', 'cancelled')", name='evaluation_sessions_status_check'),
        sa.CheckConstraint("consensus_score IS NULL OR (consensus_score >= 0 AND consensus_score <= 100)", name='evaluation_sessions_consensus_score_check'),
        sa.CheckConstraint("hallucination_score IS NULL OR (hallucination_score >= 0 AND hallucination_score <= 100)", name='evaluation_sessions_hallucination_score_check'),
        sa.CheckConstraint(
            "(confidence_interval_lower IS NULL AND confidence_interval_upper IS NULL) OR "
            "(confidence_interval_lower IS NOT NULL AND confidence_interval_upper IS NOT NULL AND "
            "confidence_interval_lower <= confidence_interval_upper)",
            name='evaluation_sessions_confidence_check'
        )
    )
    op.create_index('idx_evaluation_sessions_user_created', 'evaluation_sessions', ['user_id', 'created_at'])
    op.create_index('idx_evaluation_sessions_consensus_score', 'evaluation_sessions', ['consensus_score'])
    op.create_index('idx_evaluation_sessions_hallucination_score', 'evaluation_sessions', ['hallucination_score'])
    op.create_index('idx_evaluation_sessions_status', 'evaluation_sessions', ['status'])
    op.create_index('idx_evaluation_sessions_created_at', 'evaluation_sessions', ['created_at'])
    op.create_index('idx_evaluation_sessions_user_status', 'evaluation_sessions', ['user_id', 'status'])
    
    # Create judge_results table
    op.create_table(
        'judge_results',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('uuid_generate_v4()')),
        sa.Column('session_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('evaluation_sessions.id', ondelete='CASCADE'), nullable=False),
        sa.Column('judge_name', sa.String(255), nullable=False),
        sa.Column('score', sa.Float, nullable=False),
        sa.Column('confidence', sa.Float, nullable=False),
        sa.Column('reasoning', sa.Text, nullable=True),
        sa.Column('response_time_ms', sa.Integer, nullable=True),
        sa.Column('created_at', sa.TIMESTAMP, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.CheckConstraint("score >= 0 AND score <= 100", name='judge_results_score_check'),
        sa.CheckConstraint("confidence >= 0 AND confidence <= 1", name='judge_results_confidence_check'),
        sa.CheckConstraint("response_time_ms IS NULL OR response_time_ms >= 0", name='judge_results_response_time_check')
    )
    op.create_index('idx_judge_results_session', 'judge_results', ['session_id'])
    op.create_index('idx_judge_results_judge_name', 'judge_results', ['judge_name'])
    op.create_index('idx_judge_results_score', 'judge_results', ['score'])
    op.create_index('idx_judge_results_session_judge', 'judge_results', ['session_id', 'judge_name'])
    
    # Create flagged_issues table
    op.create_table(
        'flagged_issues',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('uuid_generate_v4()')),
        sa.Column('judge_result_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('judge_results.id', ondelete='CASCADE'), nullable=False),
        sa.Column('issue_type', sa.String(100), nullable=False),
        sa.Column('severity', sa.String(50), nullable=False),
        sa.Column('description', sa.Text, nullable=False),
        sa.Column('evidence', postgresql.JSONB, nullable=True),
        sa.Column('text_span_start', sa.Integer, nullable=True),
        sa.Column('text_span_end', sa.Integer, nullable=True),
        sa.CheckConstraint(
            "issue_type IN ('factual_error', 'hallucination', 'unsupported_claim', "
            "'temporal_inconsistency', 'numerical_error', 'bias')",
            name='flagged_issues_issue_type_check'
        ),
        sa.CheckConstraint("severity IN ('low', 'medium', 'high', 'critical')", name='flagged_issues_severity_check'),
        sa.CheckConstraint(
            "(text_span_start IS NULL AND text_span_end IS NULL) OR "
            "(text_span_start IS NOT NULL AND text_span_end IS NOT NULL AND text_span_start <= text_span_end)",
            name='flagged_issues_text_span_check'
        )
    )
    op.create_index('idx_flagged_issues_judge_result', 'flagged_issues', ['judge_result_id'])
    op.create_index('idx_flagged_issues_issue_type', 'flagged_issues', ['issue_type'])
    op.create_index('idx_flagged_issues_severity', 'flagged_issues', ['severity'])
    op.create_index('idx_flagged_issues_type_severity', 'flagged_issues', ['issue_type', 'severity'])
    
    # Create verifier_verdicts table
    op.create_table(
        'verifier_verdicts',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('uuid_generate_v4()')),
        sa.Column('session_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('evaluation_sessions.id', ondelete='CASCADE'), nullable=False),
        sa.Column('claim_text', sa.Text, nullable=False),
        sa.Column('label', sa.String(50), nullable=False),
        sa.Column('confidence', sa.Float, nullable=False),
        sa.Column('evidence', postgresql.JSONB, nullable=True),
        sa.Column('reasoning', sa.Text, nullable=True),
        sa.CheckConstraint("label IN ('SUPPORTED', 'REFUTED', 'NOT_ENOUGH_INFO')", name='verifier_verdicts_label_check'),
        sa.CheckConstraint("confidence >= 0 AND confidence <= 1", name='verifier_verdicts_confidence_check')
    )
    op.create_index('idx_verifier_verdicts_session', 'verifier_verdicts', ['session_id'])
    op.create_index('idx_verifier_verdicts_label', 'verifier_verdicts', ['label'])
    op.create_index('idx_verifier_verdicts_session_label', 'verifier_verdicts', ['session_id', 'label'])
    
    # Create session_metadata table
    op.create_table(
        'session_metadata',
        sa.Column('session_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('evaluation_sessions.id', ondelete='CASCADE'), primary_key=True),
        sa.Column('total_judges', sa.Integer, nullable=False),
        sa.Column('judges_used', postgresql.JSONB, nullable=False),
        sa.Column('aggregation_strategy', sa.String(100), nullable=True),
        sa.Column('retrieval_enabled', sa.Boolean, server_default='false'),
        sa.Column('num_retrieved_passages', sa.Integer, nullable=True),
        sa.Column('num_verifier_verdicts', sa.Integer, nullable=True),
        sa.Column('processing_time_ms', sa.Integer, nullable=True),
        sa.Column('variance', sa.Float, nullable=True),
        sa.Column('standard_deviation', sa.Float, nullable=True),
        sa.CheckConstraint("total_judges > 0", name='session_metadata_total_judges_check'),
        sa.CheckConstraint("num_retrieved_passages IS NULL OR num_retrieved_passages >= 0", name='session_metadata_num_passages_check'),
        sa.CheckConstraint("num_verifier_verdicts IS NULL OR num_verifier_verdicts >= 0", name='session_metadata_num_verdicts_check'),
        sa.CheckConstraint("processing_time_ms IS NULL OR processing_time_ms >= 0", name='session_metadata_processing_time_check')
    )
    op.create_index('idx_session_metadata_aggregation_strategy', 'session_metadata', ['aggregation_strategy'])
    op.create_index('idx_session_metadata_retrieval_enabled', 'session_metadata', ['retrieval_enabled'])
    
    # Create user_preferences table
    op.create_table(
        'user_preferences',
        sa.Column('user_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('users.id', ondelete='CASCADE'), primary_key=True),
        sa.Column('default_judge_models', postgresql.JSONB, nullable=True),
        sa.Column('default_retrieval_enabled', sa.Boolean, server_default='true'),
        sa.Column('default_aggregation_strategy', sa.String(100), server_default='weighted_average'),
        sa.Column('theme', sa.String(50), server_default='light'),
        sa.Column('notifications_enabled', sa.Boolean, server_default='true'),
        sa.Column('updated_at', sa.TIMESTAMP, server_default=sa.text('CURRENT_TIMESTAMP'))
    )


def downgrade() -> None:
    # Drop tables in reverse order
    op.drop_table('user_preferences')
    op.drop_table('session_metadata')
    op.drop_table('verifier_verdicts')
    op.drop_table('flagged_issues')
    op.drop_table('judge_results')
    op.drop_table('evaluation_sessions')
    op.drop_table('users')
    
    # Drop UUID extension
    op.execute('DROP EXTENSION IF EXISTS "uuid-ossp"')
