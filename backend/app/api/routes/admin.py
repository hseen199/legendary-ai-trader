from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from datetime import datetime, timedelta
from pydantic import BaseModel
from typing import Optional
import secrets

from app.core.database import get_db
from app.core.security import get_current_admin
from app.core.config import settings
from app.models import (
    User, Balance, Transaction, WithdrawalRequest as WithdrawalRequestModel,
    NAVHistory, TradingHistory, PlatformStats
)
from app.schemas import (
    AdminWithdrawalReview,
    AdminStats,
    AdminUserResponse,
    WithdrawalRequestResponse,
    NAVResponse,
    TradeResponse
)
from app.services import binance_service, nav_service, email_service

router = APIRouter(prefix="/admin", tags=["Admin"])


# ============ Additional Schemas ============

class WithdrawalCompleteRequest(BaseModel):
    tx_hash: Optional[str] = None


class WithdrawalDetailResponse(BaseModel):
    id: int
    user_id: int
    user_email: Optional[str] = None
    user_name: Optional[str] = None
    amount: float
    amount_usd: float
    units_to_withdraw: float
    to_address: str
    wallet_address: str
    network: str
    coin: str
    status: str
    requested_at: datetime
    created_at: datetime
    reviewed_by: Optional[int] = None
    reviewed_at: Optional[datetime] = None
    rejection_reason: Optional[str] = None
    completed_at: Optional[datetime] = None
    tx_hash: Optional[str] = None

    class Config:
        from_attributes = True


# ============ Dashboard Stats ============

@router.get("/stats", response_model=AdminStats)
async def get_admin_stats(
    admin: User = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """Get platform statistics for admin dashboard"""
    # Total users
    result = await db.execute(select(func.count(User.id)))
    total_users = result.scalar()
    
    # Active users (logged in last 30 days)
    thirty_days_ago = datetime.utcnow() - timedelta(days=30)
    result = await db.execute(
        select(func.count(User.id))
        .where(User.last_login >= thirty_days_ago)
    )
    active_users = result.scalar()
    
    # Total assets and NAV
    total_assets = await binance_service.get_total_assets_usd()
    total_units = await nav_service.get_total_units(db)
    current_nav = await nav_service.get_current_nav(db)
    
    # Pending withdrawals
    result = await db.execute(
        select(func.count(WithdrawalRequestModel.id))
        .where(WithdrawalRequestModel.status == "pending_approval")
    )
    pending_withdrawals = result.scalar()
    
    # Today's deposits and withdrawals
    today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    
    result = await db.execute(
        select(func.sum(Transaction.amount_usd))
        .where(Transaction.type == "deposit")
        .where(Transaction.status == "completed")
        .where(Transaction.created_at >= today_start)
    )
    deposits_today = result.scalar() or 0
    
    result = await db.execute(
        select(func.sum(WithdrawalRequestModel.amount))
        .where(WithdrawalRequestModel.status == "completed")
        .where(WithdrawalRequestModel.completed_at >= today_start)
    )
    withdrawals_today = result.scalar() or 0
    
    # Platform stats
    result = await db.execute(select(PlatformStats).limit(1))
    platform_stats = result.scalar_one_or_none()
    
    return AdminStats(
        total_users=total_users or 0,
        active_users=active_users or 0,
        total_assets_usd=total_assets,
        total_units=total_units,
        current_nav=current_nav,
        pending_withdrawals=pending_withdrawals or 0,
        total_deposits_today=deposits_today,
        total_withdrawals_today=withdrawals_today,
        high_water_mark=platform_stats.high_water_mark if platform_stats else settings.INITIAL_NAV,
        total_fees_collected=platform_stats.total_fees_collected if platform_stats else 0,
        emergency_mode=platform_stats.emergency_mode if platform_stats else "off"
    )


# ============ User Management ============

@router.get("/users", response_model=list[AdminUserResponse])
async def get_all_users(
    skip: int = 0,
    limit: int = 100,
    admin: User = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """Get all users with their balances"""
    result = await db.execute(
        select(User)
        .offset(skip)
        .limit(limit)
        .order_by(User.created_at.desc())
    )
    users = result.scalars().all()
    
    current_nav = await nav_service.get_current_nav(db)
    
    response = []
    for user in users:
        # Get balance
        result = await db.execute(
            select(Balance).where(Balance.user_id == user.id)
        )
        balance = result.scalar_one_or_none()
        units = balance.units if balance else 0
        
        # Get total deposited
        result = await db.execute(
            select(func.sum(Transaction.amount_usd))
            .where(Transaction.user_id == user.id)
            .where(Transaction.type == "deposit")
            .where(Transaction.status == "completed")
        )
        total_deposited = result.scalar() or 0
        
        # Get total withdrawn
        result = await db.execute(
            select(func.sum(WithdrawalRequestModel.amount))
            .where(WithdrawalRequestModel.user_id == user.id)
            .where(WithdrawalRequestModel.status == "completed")
        )
        total_withdrawn = result.scalar() or 0
        
        response.append(AdminUserResponse(
            id=user.id,
            email=user.email,
            full_name=user.full_name,
            status=user.status,
            is_admin=user.is_admin,
            units=units,
            current_value_usd=units * current_nav,
            total_deposited=total_deposited,
            total_withdrawn=total_withdrawn,
            created_at=user.created_at,
            last_login=user.last_login
        ))
    
    return response


@router.post("/users/{user_id}/suspend")
async def suspend_user(
    user_id: int,
    admin: User = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """Suspend a user account"""
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    if user.is_admin:
        raise HTTPException(status_code=400, detail="Cannot suspend admin")
    
    user.status = "suspended"
    await db.commit()
    
    return {"message": f"User {user.email} suspended"}


@router.post("/users/{user_id}/activate")
async def activate_user(
    user_id: int,
    admin: User = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """Activate a user account"""
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    user.status = "active"
    await db.commit()
    
    return {"message": f"User {user.email} activated"}


# ============ Withdrawal Management ============

@router.get("/withdrawals", response_model=list[WithdrawalDetailResponse])
async def get_all_withdrawals(
    skip: int = 0,
    limit: int = 500,
    status: Optional[str] = None,
    admin: User = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """Get all withdrawal requests with optional status filter"""
    query = select(WithdrawalRequestModel).order_by(WithdrawalRequestModel.requested_at.desc())
    
    if status:
        query = query.where(WithdrawalRequestModel.status == status)
    
    query = query.offset(skip).limit(limit)
    result = await db.execute(query)
    withdrawals = result.scalars().all()
    
    # Get user info for each withdrawal
    response = []
    for w in withdrawals:
        user_result = await db.execute(select(User).where(User.id == w.user_id))
        user = user_result.scalar_one_or_none()
        
        response.append(WithdrawalDetailResponse(
            id=w.id,
            user_id=w.user_id,
            user_email=user.email if user else None,
            user_name=user.full_name if user else None,
            amount=w.amount,
            amount_usd=w.amount,
            units_to_withdraw=w.units_to_withdraw,
            to_address=w.to_address,
            wallet_address=w.to_address,
            network=w.network,
            coin=w.coin,
            status=w.status,
            requested_at=w.requested_at,
            created_at=w.requested_at,
            reviewed_by=w.reviewed_by,
            reviewed_at=w.reviewed_at,
            rejection_reason=w.rejection_reason,
            completed_at=w.completed_at,
            tx_hash=getattr(w, 'tx_hash', None)
        ))
    
    return response


@router.get("/withdrawals/pending", response_model=list[WithdrawalDetailResponse])
async def get_pending_withdrawals(
    admin: User = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """Get all pending withdrawal requests"""
    result = await db.execute(
        select(WithdrawalRequestModel)
        .where(WithdrawalRequestModel.status == "pending_approval")
        .order_by(WithdrawalRequestModel.requested_at.asc())
    )
    withdrawals = result.scalars().all()
    
    response = []
    for w in withdrawals:
        user_result = await db.execute(select(User).where(User.id == w.user_id))
        user = user_result.scalar_one_or_none()
        
        response.append(WithdrawalDetailResponse(
            id=w.id,
            user_id=w.user_id,
            user_email=user.email if user else None,
            user_name=user.full_name if user else None,
            amount=w.amount,
            amount_usd=w.amount,
            units_to_withdraw=w.units_to_withdraw,
            to_address=w.to_address,
            wallet_address=w.to_address,
            network=w.network,
            coin=w.coin,
            status=w.status,
            requested_at=w.requested_at,
            created_at=w.requested_at,
            reviewed_by=w.reviewed_by,
            reviewed_at=w.reviewed_at,
            rejection_reason=w.rejection_reason,
            completed_at=w.completed_at,
            tx_hash=getattr(w, 'tx_hash', None)
        ))
    
    return response


@router.post("/withdrawals/{withdrawal_id}/review")
async def review_withdrawal(
    withdrawal_id: int,
    review: AdminWithdrawalReview,
    admin: User = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """Approve or reject a withdrawal request"""
    result = await db.execute(
        select(WithdrawalRequestModel)
        .where(WithdrawalRequestModel.id == withdrawal_id)
    )
    withdrawal = result.scalar_one_or_none()
    
    if not withdrawal:
        raise HTTPException(status_code=404, detail="Withdrawal not found")
    
    if withdrawal.status != "pending_approval":
        raise HTTPException(status_code=400, detail="Withdrawal already reviewed")
    
    # Get user
    result = await db.execute(select(User).where(User.id == withdrawal.user_id))
    user = result.scalar_one_or_none()
    
    if review.action == "approve":
        # Generate confirmation token
        withdrawal.confirmation_token = secrets.token_urlsafe(32)
        withdrawal.status = "approved"
        withdrawal.reviewed_by = admin.id
        withdrawal.reviewed_at = datetime.utcnow()
        
        await db.commit()
        
        # Send confirmation email
        confirmation_link = f"{settings.API_V1_PREFIX}/wallet/withdraw/confirm/{withdrawal.confirmation_token}"
        await email_service.send_withdrawal_confirmation(
            user.email,
            withdrawal.amount,
            withdrawal.to_address,
            confirmation_link
        )
        
        return {"message": "Withdrawal approved, confirmation email sent"}
    
    elif review.action == "reject":
        withdrawal.status = "rejected"
        withdrawal.reviewed_by = admin.id
        withdrawal.reviewed_at = datetime.utcnow()
        withdrawal.rejection_reason = review.reason
        
        await db.commit()
        
        # Send rejection email
        await email_service.send_withdrawal_rejected(
            user.email,
            withdrawal.amount,
            review.reason or "No reason provided"
        )
        
        return {"message": "Withdrawal rejected"}
    
    else:
        raise HTTPException(status_code=400, detail="Invalid action")


@router.post("/withdrawals/{withdrawal_id}/complete")
async def complete_withdrawal(
    withdrawal_id: int,
    data: WithdrawalCompleteRequest,
    admin: User = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """Mark withdrawal as completed after manual transfer"""
    result = await db.execute(
        select(WithdrawalRequestModel)
        .where(WithdrawalRequestModel.id == withdrawal_id)
    )
    withdrawal = result.scalar_one_or_none()
    
    if not withdrawal:
        raise HTTPException(status_code=404, detail="Withdrawal not found")
    
    if withdrawal.status not in ["approved", "processing"]:
        raise HTTPException(status_code=400, detail="Withdrawal must be approved first")
    
    # Get user balance
    result = await db.execute(
        select(Balance).where(Balance.user_id == withdrawal.user_id)
    )
    balance = result.scalar_one_or_none()
    
    if balance:
        # Deduct units from balance
        balance.units -= withdrawal.units_to_withdraw
        if balance.units < 0:
            balance.units = 0
    
    # Update withdrawal status
    withdrawal.status = "completed"
    withdrawal.completed_at = datetime.utcnow()
    if data.tx_hash:
        withdrawal.tx_hash = data.tx_hash
    
    # Create transaction record
    transaction = Transaction(
        user_id=withdrawal.user_id,
        type="withdrawal",
        amount_usd=withdrawal.amount,
        units=withdrawal.units_to_withdraw,
        status="completed",
        description=f"Withdrawal to {withdrawal.to_address[:10]}..."
    )
    db.add(transaction)
    
    await db.commit()
    
    # Get user for notification
    result = await db.execute(select(User).where(User.id == withdrawal.user_id))
    user = result.scalar_one_or_none()
    
    # Send completion email
    if user:
        await email_service.send_withdrawal_completed(
            user.email,
            withdrawal.amount,
            withdrawal.to_address,
            data.tx_hash
        )
    
    return {"message": "Withdrawal completed successfully", "tx_hash": data.tx_hash}


# ============ Trading History ============

@router.get("/trades", response_model=list[TradeResponse])
async def get_trading_history(
    limit: int = 100,
    admin: User = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """Get bot trading history"""
    result = await db.execute(
        select(TradingHistory)
        .order_by(TradingHistory.executed_at.desc())
        .limit(limit)
    )
    return result.scalars().all()


# ============ NAV Management ============

@router.post("/nav/snapshot")
async def create_nav_snapshot(
    admin: User = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """Manually create NAV snapshot"""
    nav_record = await nav_service.record_nav_snapshot(db)
    return {
        "message": "NAV snapshot created",
        "nav": nav_record.nav_value,
        "total_assets": nav_record.total_assets_usd,
        "total_units": nav_record.total_units
    }


# ============ Emergency Controls ============

@router.post("/emergency/enable")
async def enable_emergency_mode(
    admin: User = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """Enable emergency mode - stops all withdrawals"""
    result = await db.execute(select(PlatformStats).limit(1))
    stats = result.scalar_one_or_none()
    
    if not stats:
        stats = PlatformStats()
        db.add(stats)
    
    stats.emergency_mode = "on"
    await db.commit()
    
    return {"message": "Emergency mode enabled - all withdrawals stopped"}


@router.post("/emergency/disable")
async def disable_emergency_mode(
    admin: User = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """Disable emergency mode"""
    result = await db.execute(select(PlatformStats).limit(1))
    stats = result.scalar_one_or_none()
    
    if stats:
        stats.emergency_mode = "off"
        await db.commit()
    
    return {"message": "Emergency mode disabled"}


# ============ Impersonation ============

class ImpersonateResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: int
    user_email: str
    user_name: Optional[str] = None


@router.post("/users/{user_id}/impersonate", response_model=ImpersonateResponse)
async def impersonate_user(
    user_id: int,
    admin: User = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """
    Generate a token to impersonate a user.
    Admin can use this token to view the platform as the user.
    """
    from app.core.security import create_access_token
    from datetime import timedelta
    
    # Get the user to impersonate
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    if user.is_admin:
        raise HTTPException(status_code=400, detail="Cannot impersonate another admin")
    
    if user.status != "active":
        raise HTTPException(status_code=400, detail="Cannot impersonate inactive user")
    
    # Create a short-lived token for the user (30 minutes)
    access_token = create_access_token(
        data={"sub": str(user.id)},
        expires_delta=timedelta(minutes=30)
    )
    
    return ImpersonateResponse(
        access_token=access_token,
        user_id=user.id,
        user_email=user.email,
        user_name=user.full_name
    )
