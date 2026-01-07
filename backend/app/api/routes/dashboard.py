from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from datetime import datetime, timedelta

from app.core.database import get_db
from app.core.security import get_current_user
from app.core.config import settings
from app.models import User, Balance, Transaction, WithdrawalRequest, TradingHistory, NAVHistory
from app.schemas import (
    UserDashboard,
    TransactionResponse,
    WithdrawalRequestResponse,
    NAVResponse,
    NAVHistoryItem,
    TradeResponse
)
from app.services import nav_service

router = APIRouter(prefix="/dashboard", tags=["Dashboard"])


@router.get("/", response_model=UserDashboard)
async def get_user_dashboard(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get user dashboard with all relevant information"""
    # Get balance
    result = await db.execute(
        select(Balance).where(Balance.user_id == current_user.id)
    )
    balance = result.scalar_one_or_none()
    units = balance.units if balance else 0
    
    # Get current NAV
    current_nav = await nav_service.get_current_nav(db)
    current_value = units * current_nav
    
    # Calculate total deposited
    result = await db.execute(
        select(func.sum(Transaction.amount_usd))
        .where(Transaction.user_id == current_user.id)
        .where(Transaction.type == "deposit")
        .where(Transaction.status == "completed")
    )
    total_deposited = result.scalar() or 0
    
    # Calculate profit/loss
    profit_loss = current_value - total_deposited
    profit_loss_percent = (profit_loss / total_deposited * 100) if total_deposited > 0 else 0
    
    # Check withdrawal eligibility
    can_withdraw = True
    lock_period_ends = None
    if balance and balance.last_deposit_at:
        lock_end = balance.last_deposit_at + timedelta(days=settings.LOCK_PERIOD_DAYS)
        if datetime.utcnow() < lock_end:
            can_withdraw = False
            lock_period_ends = lock_end
    
    # Get recent transactions
    result = await db.execute(
        select(Transaction)
        .where(Transaction.user_id == current_user.id)
        .order_by(Transaction.created_at.desc())
        .limit(10)
    )
    recent_transactions = result.scalars().all()
    
    # Get pending withdrawals
    result = await db.execute(
        select(WithdrawalRequest)
        .where(WithdrawalRequest.user_id == current_user.id)
        .where(WithdrawalRequest.status.in_(["pending_approval", "approved", "processing"]))
    )
    pending_withdrawals = result.scalars().all()
    
    return UserDashboard(
        balance=current_value,
        units=units,
        current_nav=current_nav,
        total_deposited=total_deposited,
        current_value=current_value,
        profit_loss=profit_loss,
        profit_loss_percent=profit_loss_percent,
        can_withdraw=can_withdraw,
        lock_period_ends=lock_period_ends,
        recent_transactions=[
            TransactionResponse(
                id=t.id,
                type=t.type,
                amount_usd=t.amount_usd,
                units_transacted=t.units_transacted,
                nav_at_transaction=t.nav_at_transaction,
                coin=t.coin,
                status=t.status,
                tx_hash=t.tx_hash,
                created_at=t.created_at,
                completed_at=t.completed_at
            )
            for t in recent_transactions
        ],
        pending_withdrawals=[
            WithdrawalRequestResponse(
                id=w.id,
                amount=w.amount,
                units_to_withdraw=w.units_to_withdraw,
                to_address=w.to_address,
                network=w.network,
                coin=w.coin,
                status=w.status,
                requested_at=w.requested_at,
                reviewed_at=w.reviewed_at,
                rejection_reason=w.rejection_reason,
                completed_at=w.completed_at
            )
            for w in pending_withdrawals
        ]
    )


@router.get("/nav", response_model=NAVResponse)
async def get_nav_info(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get current NAV and changes"""
    current_nav = await nav_service.get_current_nav(db)
    total_units = await nav_service.get_total_units(db)
    total_assets = current_nav * total_units
    
    # Get changes
    change_24h = await nav_service.get_nav_change(db, 1)
    change_7d = await nav_service.get_nav_change(db, 7)
    change_30d = await nav_service.get_nav_change(db, 30)
    
    return NAVResponse(
        current_nav=current_nav,
        total_assets_usd=total_assets,
        total_units=total_units,
        change_24h=change_24h,
        change_7d=change_7d,
        change_30d=change_30d
    )


@router.get("/nav/history", response_model=list[NAVHistoryItem])
async def get_nav_history(
    days: int = 30,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get NAV history for chart"""
    history = await nav_service.get_nav_history(db, days)
    return [
        NAVHistoryItem(
            nav_value=h.nav_value,
            total_assets_usd=h.total_assets_usd,
            timestamp=h.timestamp
        )
        for h in history
    ]


@router.get("/trades", response_model=list[TradeResponse])
async def get_public_trades(
    limit: int = 50,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get recent bot trades (public for transparency)"""
    result = await db.execute(
        select(TradingHistory)
        .order_by(TradingHistory.executed_at.desc())
        .limit(limit)
    )
    trades = result.scalars().all()
    
    return [
        TradeResponse(
            id=t.id,
            symbol=t.symbol,
            side=t.side,
            order_type=t.order_type,
            price=t.price,
            quantity=t.quantity,
            total_value=t.total_value,
            pnl=t.pnl,
            pnl_percent=t.pnl_percent,
            executed_at=t.executed_at
        )
        for t in trades
    ]
