from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from datetime import datetime, timedelta
import secrets

from app.core.database import get_db
from app.core.security import get_current_user
from app.core.config import settings
from app.models import User, Balance, Transaction, WithdrawalRequest as WithdrawalRequestModel, TrustedAddress
from app.schemas import (
    DepositAddressResponse,
    DepositHistoryItem,
    WithdrawalRequest,
    WithdrawalRequestResponse,
    WithdrawalConfirm,
    TransactionResponse,
    BalanceResponse,
    TrustedAddressCreate,
    TrustedAddressResponse
)
from app.services import binance_service, nav_service, email_service

router = APIRouter(prefix="/wallet", tags=["Wallet"])


# ============ Balance ============

@router.get("/balance", response_model=BalanceResponse)
async def get_balance(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get user balance and portfolio value"""
    result = await db.execute(
        select(Balance).where(Balance.user_id == current_user.id)
    )
    balance = result.scalar_one_or_none()
    
    if not balance:
        raise HTTPException(status_code=404, detail="Balance not found")
    
    # Calculate current value
    current_nav = await nav_service.get_current_nav(db)
    current_value = balance.units * current_nav
    
    # Calculate profit/loss (simplified - would need deposit history for accurate calculation)
    result = await db.execute(
        select(Transaction)
        .where(Transaction.user_id == current_user.id)
        .where(Transaction.type == "deposit")
        .where(Transaction.status == "completed")
    )
    deposits = result.scalars().all()
    total_deposited = sum(d.amount_usd for d in deposits)
    
    profit_loss = current_value - total_deposited
    profit_loss_percent = (profit_loss / total_deposited * 100) if total_deposited > 0 else 0
    
    # Check if can withdraw (lock period)
    can_withdraw = True
    if balance.last_deposit_at:
        lock_end = balance.last_deposit_at + timedelta(days=settings.LOCK_PERIOD_DAYS)
        can_withdraw = datetime.utcnow() >= lock_end
    
    return BalanceResponse(
        units=balance.units,
        current_value_usd=current_value,
        nav=current_nav,
        profit_loss=profit_loss,
        profit_loss_percent=profit_loss_percent,
        last_deposit_at=balance.last_deposit_at,
        can_withdraw=can_withdraw
    )


# ============ Deposit ============

@router.get("/deposit/address", response_model=DepositAddressResponse)
async def get_deposit_address(
    network: str = "TRC20",
    coin: str = "USDT",
    current_user: User = Depends(get_current_user)
):
    """Get deposit address for user"""
    if not current_user.sub_account_email:
        raise HTTPException(
            status_code=400, 
            detail="Sub-account not configured"
        )
    
    address_info = await binance_service.get_deposit_address(
        email=current_user.sub_account_email,
        coin=coin,
        network=network
    )
    
    if not address_info:
        raise HTTPException(
            status_code=500, 
            detail="Failed to get deposit address"
        )
    
    return DepositAddressResponse(
        address=address_info["address"],
        network=network,
        coin=coin
    )


@router.get("/deposit/history", response_model=list[DepositHistoryItem])
async def get_deposit_history(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get user deposit history"""
    result = await db.execute(
        select(Transaction)
        .where(Transaction.user_id == current_user.id)
        .where(Transaction.type == "deposit")
        .order_by(Transaction.created_at.desc())
    )
    deposits = result.scalars().all()
    
    return [
        DepositHistoryItem(
            id=d.id,
            amount=d.amount_usd,
            coin=d.coin,
            network=d.network or "TRC20",
            status=d.status,
            tx_hash=d.tx_hash,
            units_received=d.units_transacted,
            nav_at_deposit=d.nav_at_transaction,
            created_at=d.created_at,
            completed_at=d.completed_at
        )
        for d in deposits
    ]


# ============ Withdrawal ============

@router.post("/withdraw/request", response_model=WithdrawalRequestResponse)
async def request_withdrawal(
    request: WithdrawalRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Request a withdrawal (requires admin approval)"""
    # Get balance
    result = await db.execute(
        select(Balance).where(Balance.user_id == current_user.id)
    )
    balance = result.scalar_one_or_none()
    
    if not balance:
        raise HTTPException(status_code=404, detail="Balance not found")
    
    # Check lock period
    if balance.last_deposit_at:
        lock_end = balance.last_deposit_at + timedelta(days=settings.LOCK_PERIOD_DAYS)
        if datetime.utcnow() < lock_end:
            raise HTTPException(
                status_code=400,
                detail=f"Withdrawal locked until {lock_end.isoformat()}"
            )
    
    # Calculate units needed
    current_nav = await nav_service.get_current_nav(db)
    units_needed = request.amount / current_nav
    
    if units_needed > balance.units:
        raise HTTPException(
            status_code=400,
            detail="Insufficient balance"
        )
    
    # Check if address is trusted (optional)
    result = await db.execute(
        select(TrustedAddress)
        .where(TrustedAddress.user_id == current_user.id)
        .where(TrustedAddress.address == request.to_address)
        .where(TrustedAddress.is_active == True)
    )
    trusted = result.scalar_one_or_none()
    
    # Create withdrawal request
    withdrawal = WithdrawalRequestModel(
        user_id=current_user.id,
        amount=request.amount,
        units_to_withdraw=units_needed,
        to_address=request.to_address,
        network=request.network,
        coin=request.coin,
        status="pending_approval"
    )
    db.add(withdrawal)
    await db.commit()
    await db.refresh(withdrawal)
    
    return WithdrawalRequestResponse(
        id=withdrawal.id,
        amount=withdrawal.amount,
        units_to_withdraw=withdrawal.units_to_withdraw,
        to_address=withdrawal.to_address,
        network=withdrawal.network,
        coin=withdrawal.coin,
        status=withdrawal.status,
        requested_at=withdrawal.requested_at
    )


@router.get("/withdraw/confirm/{token}")
async def confirm_withdrawal(
    token: str,
    db: AsyncSession = Depends(get_db)
):
    """Confirm withdrawal via email link"""
    result = await db.execute(
        select(WithdrawalRequestModel)
        .where(WithdrawalRequestModel.confirmation_token == token)
        .where(WithdrawalRequestModel.status == "approved")
    )
    withdrawal = result.scalar_one_or_none()
    
    if not withdrawal:
        raise HTTPException(status_code=404, detail="Invalid or expired token")
    
    # Mark as confirmed
    withdrawal.email_confirmed = datetime.utcnow()
    withdrawal.status = "processing"
    await db.commit()
    
    # TODO: Trigger actual withdrawal process
    
    return {"message": "Withdrawal confirmed and processing"}


@router.get("/withdraw/history", response_model=list[WithdrawalRequestResponse])
async def get_withdrawal_history(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get user withdrawal history"""
    result = await db.execute(
        select(WithdrawalRequestModel)
        .where(WithdrawalRequestModel.user_id == current_user.id)
        .order_by(WithdrawalRequestModel.requested_at.desc())
    )
    withdrawals = result.scalars().all()
    
    return [
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
        for w in withdrawals
    ]


# ============ Trusted Addresses ============

@router.post("/trusted-addresses", response_model=TrustedAddressResponse)
async def add_trusted_address(
    address_data: TrustedAddressCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Add a trusted withdrawal address (24h activation delay)"""
    # Check if address already exists
    result = await db.execute(
        select(TrustedAddress)
        .where(TrustedAddress.user_id == current_user.id)
        .where(TrustedAddress.address == address_data.address)
    )
    if result.scalar_one_or_none():
        raise HTTPException(status_code=400, detail="Address already added")
    
    trusted = TrustedAddress(
        user_id=current_user.id,
        address=address_data.address,
        network=address_data.network,
        label=address_data.label,
        is_active=False,  # Will be activated after 24h
        activated_at=datetime.utcnow() + timedelta(hours=24)
    )
    db.add(trusted)
    await db.commit()
    await db.refresh(trusted)
    
    return trusted


@router.get("/trusted-addresses", response_model=list[TrustedAddressResponse])
async def get_trusted_addresses(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get user's trusted addresses"""
    result = await db.execute(
        select(TrustedAddress)
        .where(TrustedAddress.user_id == current_user.id)
        .order_by(TrustedAddress.created_at.desc())
    )
    return result.scalars().all()


# ============ Transactions ============

@router.get("/transactions", response_model=list[TransactionResponse])
async def get_transactions(
    limit: int = 50,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get user transaction history"""
    result = await db.execute(
        select(Transaction)
        .where(Transaction.user_id == current_user.id)
        .order_by(Transaction.created_at.desc())
        .limit(limit)
    )
    return result.scalars().all()
