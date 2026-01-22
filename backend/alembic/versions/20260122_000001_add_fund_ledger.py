"""Add fund_ledger table for double-entry accounting

Revision ID: 20260122_000001
Revises: 20260110_000001
Create Date: 2026-01-22 19:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '20260122_000001'
down_revision = '20260110_000001'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # إنشاء جدول سجل المحاسبة
    op.create_table(
        'fund_ledger',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('entry_type', sa.String(50), nullable=False),
        sa.Column('amount', sa.Numeric(precision=20, scale=8), nullable=False),
        sa.Column('units_delta', sa.Numeric(precision=20, scale=8), nullable=False, server_default='0'),
        sa.Column('nav_at_entry', sa.Numeric(precision=20, scale=8), nullable=False),
        sa.Column('running_total_capital', sa.Numeric(precision=20, scale=8), nullable=False),
        sa.Column('running_total_units', sa.Numeric(precision=20, scale=8), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=True),
        sa.Column('transaction_id', sa.Integer(), nullable=True),
        sa.Column('trade_id', sa.String(100), nullable=True),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('metadata', sa.JSON(), nullable=True),
        sa.Column('timestamp', sa.DateTime(), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.PrimaryKeyConstraint('id')
    )
    
    # إنشاء فهارس للبحث السريع
    op.create_index('ix_fund_ledger_entry_type', 'fund_ledger', ['entry_type'])
    op.create_index('ix_fund_ledger_user_id', 'fund_ledger', ['user_id'])
    op.create_index('ix_fund_ledger_timestamp', 'fund_ledger', ['timestamp'])
    op.create_index('ix_fund_ledger_transaction_id', 'fund_ledger', ['transaction_id'])
    
    # إضافة Foreign Keys
    op.create_foreign_key(
        'fk_fund_ledger_user_id',
        'fund_ledger', 'users',
        ['user_id'], ['id'],
        ondelete='SET NULL'
    )
    op.create_foreign_key(
        'fk_fund_ledger_transaction_id',
        'fund_ledger', 'transactions',
        ['transaction_id'], ['id'],
        ondelete='SET NULL'
    )


def downgrade() -> None:
    # حذف Foreign Keys
    op.drop_constraint('fk_fund_ledger_transaction_id', 'fund_ledger', type_='foreignkey')
    op.drop_constraint('fk_fund_ledger_user_id', 'fund_ledger', type_='foreignkey')
    
    # حذف الفهارس
    op.drop_index('ix_fund_ledger_transaction_id', table_name='fund_ledger')
    op.drop_index('ix_fund_ledger_timestamp', table_name='fund_ledger')
    op.drop_index('ix_fund_ledger_user_id', table_name='fund_ledger')
    op.drop_index('ix_fund_ledger_entry_type', table_name='fund_ledger')
    
    # حذف الجدول
    op.drop_table('fund_ledger')
