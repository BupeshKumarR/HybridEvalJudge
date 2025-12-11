"""add_claim_verdicts_table

Revision ID: b2c3d4e5f6g7
Revises: a1b2c3d4e5f6
Create Date: 2025-12-10 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = 'b2c3d4e5f6g7'
down_revision: Union[str, None] = 'a1b2c3d4e5f6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create claim_verdicts table
    op.create_table(
        'claim_verdicts',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('evaluation_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('claim_text', sa.Text(), nullable=False),
        sa.Column('claim_type', sa.String(length=50), nullable=False),
        sa.Column('verdict', sa.String(length=50), nullable=False),
        sa.Column('confidence', sa.Float(), nullable=False),
        sa.Column('judge_name', sa.String(length=255), nullable=True),
        sa.Column('text_span_start', sa.Integer(), nullable=False),
        sa.Column('text_span_end', sa.Integer(), nullable=False),
        sa.Column('reasoning', sa.Text(), nullable=True),
        sa.Column('created_at', sa.TIMESTAMP(), nullable=True),
        sa.ForeignKeyConstraint(['evaluation_id'], ['evaluation_sessions.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.CheckConstraint(
            "claim_type IN ('numerical', 'temporal', 'definitional', 'general')",
            name='claim_verdicts_claim_type_check'
        ),
        sa.CheckConstraint(
            "verdict IN ('SUPPORTED', 'REFUTED', 'NOT_ENOUGH_INFO')",
            name='claim_verdicts_verdict_check'
        ),
        sa.CheckConstraint(
            "confidence >= 0 AND confidence <= 1",
            name='claim_verdicts_confidence_check'
        ),
        sa.CheckConstraint(
            "text_span_start >= 0 AND text_span_end >= text_span_start",
            name='claim_verdicts_text_span_check'
        )
    )
    
    # Create indexes for claim_verdicts
    op.create_index('ix_claim_verdicts_evaluation_id', 'claim_verdicts', ['evaluation_id'])
    op.create_index('ix_claim_verdicts_claim_type', 'claim_verdicts', ['claim_type'])
    op.create_index('ix_claim_verdicts_verdict', 'claim_verdicts', ['verdict'])
    op.create_index('ix_claim_verdicts_judge_name', 'claim_verdicts', ['judge_name'])
    op.create_index('idx_claim_verdicts_evaluation_verdict', 'claim_verdicts', ['evaluation_id', 'verdict'])


def downgrade() -> None:
    # Drop indexes for claim_verdicts
    op.drop_index('idx_claim_verdicts_evaluation_verdict', table_name='claim_verdicts')
    op.drop_index('ix_claim_verdicts_judge_name', table_name='claim_verdicts')
    op.drop_index('ix_claim_verdicts_verdict', table_name='claim_verdicts')
    op.drop_index('ix_claim_verdicts_claim_type', table_name='claim_verdicts')
    op.drop_index('ix_claim_verdicts_evaluation_id', table_name='claim_verdicts')
    
    # Drop claim_verdicts table
    op.drop_table('claim_verdicts')
