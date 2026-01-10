"""Initial migration - Create all tables

Revision ID: 001
Revises: 
Create Date: 2026-01-10

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '001'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ============ Users Table ============
    op.create_table(
        'users',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('email', sa.String(255), nullable=False),
        sa.Column('password_hash', sa.String(255), nullable=False),
        sa.Column('sub_account_email', sa.String(255), nullable=True),
        sa.Column('full_name', sa.String(255), nullable=True),
        sa.Column('phone', sa.String(50), nullable=True),
        sa.Column('status', sa.String(50), server_default='active', nullable=True),
        sa.Column('is_admin', sa.Boolean(), server_default='false', nullable=True),
        sa.Column('is_verified', sa.Boolean(), server_default='false', nullable=True),
        sa.Column('two_factor_enabled', sa.Boolean(), server_default='false', nullable=True),
        sa.Column('two_factor_secret', sa.String(255), nullable=True),
        sa.Column('balance', sa.Float(), server_default='0.0', nullable=True),
        sa.Column('units', sa.Float(), server_default='0.0', nullable=True),
        sa.Column('total_deposited', sa.Float(), server_default='0.0', nullable=True),
        sa.Column('total_withdrawn', sa.Float(), server_default='0.0', nullable=True),
        sa.Column('referral_code', sa.String(20), nullable=True),
        sa.Column('referred_by', sa.Integer(), nullable=True),
        sa.Column('vip_level', sa.String(20), server_default='bronze', nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('last_login', sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(['referred_by'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_users_email'), 'users', ['email'], unique=True)
    op.create_index(op.f('ix_users_id'), 'users', ['id'], unique=False)
    op.create_index(op.f('ix_users_referral_code'), 'users', ['referral_code'], unique=True)

    # ============ Balances Table ============
    op.create_table(
        'balances',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('units', sa.Float(), server_default='0.0', nullable=True),
        sa.Column('balance_usd', sa.Float(), server_default='0.0', nullable=True),
        sa.Column('total_deposited', sa.Float(), server_default='0.0', nullable=True),
        sa.Column('total_withdrawn', sa.Float(), server_default='0.0', nullable=True),
        sa.Column('last_deposit_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('user_id')
    )
    op.create_index(op.f('ix_balances_id'), 'balances', ['id'], unique=False)

    # ============ Trusted Addresses Table ============
    op.create_table(
        'trusted_addresses',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('address', sa.String(255), nullable=False),
        sa.Column('network', sa.String(50), nullable=False),
        sa.Column('label', sa.String(100), nullable=True),
        sa.Column('is_active', sa.Boolean(), server_default='false', nullable=True),
        sa.Column('activated_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_trusted_addresses_id'), 'trusted_addresses', ['id'], unique=False)

    # ============ Transactions Table ============
    op.create_table(
        'transactions',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('type', sa.String(50), nullable=False),
        sa.Column('amount_usd', sa.Float(), nullable=False),
        sa.Column('units_transacted', sa.Float(), nullable=True),
        sa.Column('nav_at_transaction', sa.Float(), nullable=True),
        sa.Column('coin', sa.String(20), server_default='USDC', nullable=True),
        sa.Column('currency', sa.String(20), nullable=True),
        sa.Column('network', sa.String(50), nullable=True),
        sa.Column('tx_hash', sa.String(255), nullable=True),
        sa.Column('external_id', sa.String(100), nullable=True),
        sa.Column('payment_address', sa.String(255), nullable=True),
        sa.Column('metadata', sa.JSON(), nullable=True),
        sa.Column('status', sa.String(50), server_default='pending', nullable=True),
        sa.Column('from_address', sa.String(255), nullable=True),
        sa.Column('to_address', sa.String(255), nullable=True),
        sa.Column('notes', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('confirmed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_transactions_id'), 'transactions', ['id'], unique=False)
    op.create_index(op.f('ix_transactions_external_id'), 'transactions', ['external_id'], unique=False)

    # ============ Withdrawal Requests Table ============
    op.create_table(
        'withdrawal_requests',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('amount', sa.Float(), nullable=False),
        sa.Column('units_to_withdraw', sa.Float(), nullable=False),
        sa.Column('to_address', sa.String(255), nullable=False),
        sa.Column('network', sa.String(50), nullable=False),
        sa.Column('coin', sa.String(20), server_default='USDC', nullable=True),
        sa.Column('status', sa.String(50), server_default='pending_approval', nullable=True),
        sa.Column('confirmation_token', sa.String(255), nullable=True),
        sa.Column('email_confirmed', sa.DateTime(timezone=True), nullable=True),
        sa.Column('reviewed_by', sa.Integer(), nullable=True),
        sa.Column('reviewed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('rejection_reason', sa.Text(), nullable=True),
        sa.Column('requested_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(['reviewed_by'], ['users.id'], ),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_withdrawal_requests_id'), 'withdrawal_requests', ['id'], unique=False)

    # ============ NAV History Table ============
    op.create_table(
        'nav_history',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('nav_value', sa.Float(), nullable=False),
        sa.Column('total_assets_usd', sa.Float(), nullable=False),
        sa.Column('total_units', sa.Float(), nullable=False),
        sa.Column('timestamp', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_nav_history_id'), 'nav_history', ['id'], unique=False)

    # ============ Trading History Table ============
    op.create_table(
        'trading_history',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('symbol', sa.String(20), nullable=False),
        sa.Column('side', sa.String(10), nullable=False),
        sa.Column('order_type', sa.String(20), nullable=False),
        sa.Column('price', sa.Float(), nullable=False),
        sa.Column('quantity', sa.Float(), nullable=False),
        sa.Column('total_value', sa.Float(), nullable=False),
        sa.Column('order_id', sa.String(100), nullable=True),
        sa.Column('pnl', sa.Float(), nullable=True),
        sa.Column('pnl_percent', sa.Float(), nullable=True),
        sa.Column('executed_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_trading_history_id'), 'trading_history', ['id'], unique=False)

    # ============ Platform Stats Table ============
    op.create_table(
        'platform_stats',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('high_water_mark', sa.Float(), server_default='1.0', nullable=True),
        sa.Column('last_fee_calculation', sa.DateTime(timezone=True), nullable=True),
        sa.Column('total_fees_collected', sa.Float(), server_default='0.0', nullable=True),
        sa.Column('emergency_mode', sa.String(10), server_default='off', nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_platform_stats_id'), 'platform_stats', ['id'], unique=False)

    # ============ Support Tickets Table ============
    op.create_table(
        'support_tickets',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('subject', sa.String(255), nullable=False),
        sa.Column('category', sa.String(50), nullable=True),
        sa.Column('priority', sa.String(20), server_default='normal', nullable=True),
        sa.Column('status', sa.String(50), server_default='open', nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('closed_at', sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_support_tickets_id'), 'support_tickets', ['id'], unique=False)

    # ============ Ticket Messages Table ============
    op.create_table(
        'ticket_messages',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('ticket_id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=True),
        sa.Column('message', sa.Text(), nullable=False),
        sa.Column('is_admin', sa.Boolean(), server_default='false', nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.ForeignKeyConstraint(['ticket_id'], ['support_tickets.id'], ),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_ticket_messages_id'), 'ticket_messages', ['id'], unique=False)

    # ============ Referrals Table ============
    op.create_table(
        'referrals',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('referrer_id', sa.Integer(), nullable=False),
        sa.Column('referred_id', sa.Integer(), nullable=False),
        sa.Column('bonus_amount', sa.Float(), server_default='0.0', nullable=True),
        sa.Column('status', sa.String(50), server_default='pending', nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.ForeignKeyConstraint(['referrer_id'], ['users.id'], ),
        sa.ForeignKeyConstraint(['referred_id'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_referrals_id'), 'referrals', ['id'], unique=False)

    # ============ Notifications Table ============
    op.create_table(
        'notifications',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('title', sa.String(255), nullable=False),
        sa.Column('message', sa.Text(), nullable=False),
        sa.Column('type', sa.String(50), nullable=True),
        sa.Column('is_read', sa.Boolean(), server_default='false', nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_notifications_id'), 'notifications', ['id'], unique=False)


def downgrade() -> None:
    op.drop_table('notifications')
    op.drop_table('referrals')
    op.drop_table('ticket_messages')
    op.drop_table('support_tickets')
    op.drop_table('platform_stats')
    op.drop_table('trading_history')
    op.drop_table('nav_history')
    op.drop_table('withdrawal_requests')
    op.drop_table('transactions')
    op.drop_table('trusted_addresses')
    op.drop_table('balances')
    op.drop_table('users')
