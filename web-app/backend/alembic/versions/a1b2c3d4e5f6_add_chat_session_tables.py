"""add_chat_session_tables

Revision ID: a1b2c3d4e5f6
Revises: 04dac41cf0df
Create Date: 2025-12-10 10:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = 'a1b2c3d4e5f6'
down_revision: Union[str, None] = '04dac41cf0df'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create chat_sessions table
    op.create_table(
        'chat_sessions',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('ollama_model', sa.String(length=255), nullable=False, server_default='llama3.2'),
        sa.Column('created_at', sa.TIMESTAMP(), nullable=True),
        sa.Column('updated_at', sa.TIMESTAMP(), nullable=True),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create indexes for chat_sessions
    op.create_index('ix_chat_sessions_user_id', 'chat_sessions', ['user_id'])
    op.create_index('ix_chat_sessions_created_at', 'chat_sessions', ['created_at'])
    op.create_index('idx_chat_sessions_user_created', 'chat_sessions', ['user_id', 'created_at'])
    
    # Create chat_messages table
    op.create_table(
        'chat_messages',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('session_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('role', sa.String(length=50), nullable=False),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('evaluation_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('created_at', sa.TIMESTAMP(), nullable=True),
        sa.ForeignKeyConstraint(['session_id'], ['chat_sessions.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['evaluation_id'], ['evaluation_sessions.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id'),
        sa.CheckConstraint("role IN ('user', 'assistant')", name='chat_messages_role_check')
    )
    
    # Create indexes for chat_messages
    op.create_index('ix_chat_messages_session_id', 'chat_messages', ['session_id'])
    op.create_index('ix_chat_messages_role', 'chat_messages', ['role'])
    op.create_index('ix_chat_messages_evaluation_id', 'chat_messages', ['evaluation_id'])
    op.create_index('ix_chat_messages_created_at', 'chat_messages', ['created_at'])
    op.create_index('idx_chat_messages_session_created', 'chat_messages', ['session_id', 'created_at'])


def downgrade() -> None:
    # Drop indexes for chat_messages
    op.drop_index('idx_chat_messages_session_created', table_name='chat_messages')
    op.drop_index('ix_chat_messages_created_at', table_name='chat_messages')
    op.drop_index('ix_chat_messages_evaluation_id', table_name='chat_messages')
    op.drop_index('ix_chat_messages_role', table_name='chat_messages')
    op.drop_index('ix_chat_messages_session_id', table_name='chat_messages')
    
    # Drop chat_messages table
    op.drop_table('chat_messages')
    
    # Drop indexes for chat_sessions
    op.drop_index('idx_chat_sessions_user_created', table_name='chat_sessions')
    op.drop_index('ix_chat_sessions_created_at', table_name='chat_sessions')
    op.drop_index('ix_chat_sessions_user_id', table_name='chat_sessions')
    
    # Drop chat_sessions table
    op.drop_table('chat_sessions')
