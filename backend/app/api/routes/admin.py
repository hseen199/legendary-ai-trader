"""
إصلاحات شاملة لـ admin.py
يُستبدل في /opt/asinax/backend/app/api/routes/admin.py
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, update
from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel

from app.core.database import get_db
from app.core.security import get_current_admin
from app.models.user import User, Balance
from app.models.transaction import Transaction, WithdrawalRequest, NAVHistory
from app.models.investor import Deposit, Investor
from app.services import nav_service

router = APIRouter(prefix="/admin", tags=["Admin"])


# ============ Schemas ============

class AdminStatsResponse(BaseModel):
    total_users: int
    active_users: int
    suspended_users: int
    total_deposits: float
    total_withdrawals: float
    pending_withdrawals: int
    pending_deposits: int


class DepositResponse(BaseModel):
    id: int
    user_id: int
    user_email: Optional[str] = None
    amount_usd: float
    currency: str
    status: str
    payment_id: Optional[str] = None
    created_at: datetime
    
    class Config:
        from_attributes = True


class WithdrawalResponse(BaseModel):
    id: int
    user_id: int
    user_email: Optional[str] = None
    amount: float
    to_address: str
    currency: str
    status: str
    created_at: datetime
    
    class Config:
        from_attributes = True


# ============ Statistics Endpoints ============

@router.get("/stats", response_model=AdminStatsResponse)
async def get_admin_stats(
    current_user: dict = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """Get admin dashboard statistics."""
    # Total users
    total_users_result = await db.execute(
        select(func.count(User.id)).where(User.is_admin == False)
    )
    total_users = total_users_result.scalar_one_or_none() or 0
    
    # Active users
    active_users_result = await db.execute(
        select(func.count(User.id))
        .where(User.is_admin == False)
        .where(User.status == 'active')
    )
    active_users = active_users_result.scalar_one_or_none() or 0
    
    # Suspended users
    suspended_users_result = await db.execute(
        select(func.count(User.id))
        .where(User.is_admin == False)
        .where(User.status == 'suspended')
    )
    suspended_users = suspended_users_result.scalar_one_or_none() or 0
    
    # Total deposits
    total_deposits_result = await db.execute(
        select(func.sum(Transaction.amount_usd))
        .where(Transaction.type == 'deposit')
        .where(Transaction.status == 'completed')
    )
    total_deposits = total_deposits_result.scalar_one_or_none() or 0.0
    
    # Total withdrawals
    total_withdrawals_result = await db.execute(
        select(func.sum(Transaction.amount_usd))
        .where(Transaction.type == 'withdrawal')
        .where(Transaction.status == 'completed')
    )
    total_withdrawals = abs(total_withdrawals_result.scalar_one_or_none() or 0.0)
    
    # Pending withdrawals
    pending_withdrawals_result = await db.execute(
        select(func.count(WithdrawalRequest.id))
        .where(WithdrawalRequest.status == 'pending')
    )
    pending_withdrawals = pending_withdrawals_result.scalar_one_or_none() or 0
    
    # Pending deposits
    pending_deposits_result = await db.execute(
        select(func.count(Deposit.id))
        .where(Deposit.status == 'pending')
    )
    pending_deposits = pending_deposits_result.scalar_one_or_none() or 0
    
    return AdminStatsResponse(
        total_users=total_users,
        active_users=active_users,
        suspended_users=suspended_users,
        total_deposits=float(total_deposits),
        total_withdrawals=float(total_withdrawals),
        pending_withdrawals=pending_withdrawals,
        pending_deposits=pending_deposits
    )


# ============ Deposit Management Endpoints ============

@router.get("/deposits/pending")
async def get_pending_deposits(
    current_user: dict = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """Get all pending deposits waiting for approval."""
    result = await db.execute(
        select(Deposit, Investor, User)
        .join(Investor, Deposit.investor_id == Investor.id)
        .join(User, Investor.user_id == User.id)
        .where(Deposit.status == 'pending')
        .order_by(Deposit.created_at.desc())
    )
    deposits = result.all()
    
    return [
        {
            "id": deposit.id,
            "user_id": user.id,
            "investor_id": investor.id,
            "user_email": user.email,
            "user_name": user.full_name,
            "amount_usd": float(deposit.amount),
            "currency": deposit.coin,
            "network": deposit.network,
            "status": deposit.status.value if hasattr(deposit.status, 'value') else str(deposit.status),
            "tx_hash": deposit.tx_hash,
            "created_at": deposit.created_at
        }
        for deposit, investor, user in deposits
    ]


@router.get("/deposits/all")
async def get_all_deposits(
    status: Optional[str] = None,
    limit: int = 100,
    current_user: dict = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """Get all deposits with optional status filter."""
    query = (
        select(Deposit, Investor, User)
        .join(Investor, Deposit.investor_id == Investor.id)
        .join(User, Investor.user_id == User.id)
    )
    
    if status:
        query = query.where(Deposit.status == status)
    
    query = query.order_by(Deposit.created_at.desc()).limit(limit)
    
    result = await db.execute(query)
    deposits = result.all()
    
    return [
        {
            "id": deposit.id,
            "user_id": user.id,
            "investor_id": investor.id,
            "user_email": user.email,
            "user_name": user.full_name,
            "amount_usd": float(deposit.amount),
            "currency": deposit.coin,
            "network": deposit.network,
            "status": deposit.status.value if hasattr(deposit.status, 'value') else str(deposit.status),
            "tx_hash": deposit.tx_hash,
            "units_credited": float(deposit.units_credited) if deposit.units_credited else 0,
            "nav_at_deposit": float(deposit.nav_at_deposit) if deposit.nav_at_deposit else 0,
            "created_at": deposit.created_at,
            "confirmed_at": deposit.confirmed_at
        }
        for deposit, investor, user in deposits
    ]


@router.post("/deposits/{deposit_id}/approve")
async def approve_deposit(
    deposit_id: int,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """
    Approve a pending deposit and add balance to user.
    """
    # Get deposit with investor and user
    result = await db.execute(
        select(Deposit, Investor, User)
        .join(Investor, Deposit.investor_id == Investor.id)
        .join(User, Investor.user_id == User.id)
        .where(Deposit.id == deposit_id)
    )
    row = result.first()
    
    if not row:
        raise HTTPException(status_code=404, detail="الإيداع غير موجود")
    
    deposit, investor, user = row
    
    status_value = deposit.status.value if hasattr(deposit.status, 'value') else str(deposit.status)
    if status_value not in ['pending', 'confirmed']:
        raise HTTPException(
            status_code=400, 
            detail=f"لا يمكن الموافقة على إيداع بحالة: {status_value}"
        )
    
    # Get current NAV
    current_nav = await nav_service.get_current_nav(db)
    
    # Calculate units
    amount_usd = float(deposit.amount)
    units_to_add = amount_usd / current_nav
    
    # Get or create balance
    balance_result = await db.execute(
        select(Balance).where(Balance.user_id == user.id)
    )
    balance = balance_result.scalar_one_or_none()
    
    if not balance:
        balance = Balance(user_id=user.id, units=0)
        db.add(balance)
    
    # Update balance
    old_units = balance.units
    balance.units += units_to_add
    
    # Update investor units too
    investor.total_units = (investor.total_units or 0) + units_to_add
    
    # Update deposit status
    from app.models.investor import DepositStatus
    deposit.status = DepositStatus.CREDITED
    deposit.confirmed_at = datetime.utcnow()
    deposit.units_credited = units_to_add
    deposit.nav_at_deposit = current_nav
    
    # Create transaction record
    transaction = Transaction(
        user_id=user.id,
        type='deposit',
        amount_usd=amount_usd,
        units=units_to_add,
        nav_at_time=current_nav,
        status='completed',
        description=f"إيداع معتمد - {deposit.coin}",
        reference_id=f"DEP-{deposit.id}"
    )
    db.add(transaction)
    
    # Update user's total_deposited if exists
    if hasattr(user, 'total_deposited'):
        user.total_deposited = (user.total_deposited or 0) + amount_usd
    
    await db.commit()
    
    # Send confirmation email
    try:
        from app.services.email_service import email_service
        background_tasks.add_task(
            email_service.send_deposit_confirmation,
            user.email,
            user.full_name or user.email,
            amount_usd,
            units_to_add,
            current_nav
        )
    except Exception as e:
        print(f"Failed to send deposit confirmation email: {e}")
    
    return {
        "success": True,
        "message": f"تم الموافقة على الإيداع وإضافة ${amount_usd:.2f} للمستخدم",
        "deposit_id": deposit_id,
        "user_id": user.id,
        "user_email": user.email,
        "amount_usd": amount_usd,
        "units_added": units_to_add,
        "nav_used": current_nav,
        "old_balance_units": old_units,
        "new_balance_units": balance.units,
        "new_balance_usd": balance.units * current_nav
    }


@router.post("/deposits/{deposit_id}/reject")
async def reject_deposit(
    deposit_id: int,
    reason: str,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """Reject a pending deposit."""
    # Get deposit with investor and user
    result = await db.execute(
        select(Deposit, Investor, User)
        .join(Investor, Deposit.investor_id == Investor.id)
        .join(User, Investor.user_id == User.id)
        .where(Deposit.id == deposit_id)
    )
    row = result.first()
    
    if not row:
        raise HTTPException(status_code=404, detail="الإيداع غير موجود")
    
    deposit, investor, user = row
    
    status_value = deposit.status.value if hasattr(deposit.status, 'value') else str(deposit.status)
    if status_value not in ['pending', 'confirmed']:
        raise HTTPException(
            status_code=400, 
            detail=f"لا يمكن رفض إيداع بحالة: {status_value}"
        )
    
    # Update deposit status
    from app.models.investor import DepositStatus
    deposit.status = DepositStatus.FAILED
    
    await db.commit()
    
    # Send rejection email
    try:
        from app.services.email_service import email_service
        background_tasks.add_task(
            email_service.send_deposit_rejection,
            user.email,
            user.full_name or user.email,
            float(deposit.amount),
            reason
        )
    except Exception as e:
        print(f"Failed to send deposit rejection email: {e}")
    
    return {
        "success": True,
        "message": "تم رفض الإيداع",
        "deposit_id": deposit_id,
        "reason": reason
    }


# ============ Withdrawal Management Endpoints ============

@router.get("/withdrawals/pending")
async def get_pending_withdrawals(
    current_user: dict = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """Get all pending withdrawal requests."""
    result = await db.execute(
        select(WithdrawalRequest, User)
        .join(User, WithdrawalRequest.user_id == User.id)
        .where(WithdrawalRequest.status == 'pending')
        .order_by(WithdrawalRequest.created_at.desc())
    )
    withdrawals = result.all()
    
    return [
        {
            "id": wr.id,
            "user_id": wr.user_id,
            "user_email": user.email,
            "user_name": user.full_name,
            "amount": float(wr.amount),
            "to_address": wr.to_address,
            "currency": wr.currency if hasattr(wr, 'currency') else "USDT",
            "status": wr.status,
            "created_at": wr.created_at
        }
        for wr, user in withdrawals
    ]


@router.get("/withdrawals/all")
async def get_all_withdrawals(
    status: Optional[str] = None,
    limit: int = 100,
    current_user: dict = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """Get all withdrawals with optional status filter."""
    query = select(WithdrawalRequest, User).join(User, WithdrawalRequest.user_id == User.id)
    
    if status:
        query = query.where(WithdrawalRequest.status == status)
    
    query = query.order_by(WithdrawalRequest.created_at.desc()).limit(limit)
    
    result = await db.execute(query)
    withdrawals = result.all()
    
    return [
        {
            "id": wr.id,
            "user_id": wr.user_id,
            "user_email": user.email,
            "user_name": user.full_name,
            "amount": float(wr.amount),
            "to_address": wr.to_address,
            "currency": wr.currency if hasattr(wr, 'currency') else "USDT",
            "status": wr.status,
            "created_at": wr.created_at,
            "tx_hash": wr.tx_hash if hasattr(wr, 'tx_hash') else None
        }
        for wr, user in withdrawals
    ]


@router.post("/withdrawals/{withdrawal_id}/approve")
async def approve_withdrawal(
    withdrawal_id: int,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """Approve a withdrawal request."""
    result = await db.execute(
        select(WithdrawalRequest, User)
        .join(User, WithdrawalRequest.user_id == User.id)
        .where(WithdrawalRequest.id == withdrawal_id)
    )
    row = result.first()
    
    if not row:
        raise HTTPException(status_code=404, detail="طلب السحب غير موجود")
    
    withdrawal, user = row
    
    if withdrawal.status != 'pending':
        raise HTTPException(
            status_code=400, 
            detail=f"لا يمكن الموافقة على طلب سحب بحالة: {withdrawal.status}"
        )
    
    withdrawal.status = 'approved'
    await db.commit()
    
    # Send approval email
    try:
        from app.services.email_service import email_service
        background_tasks.add_task(
            email_service.send_withdrawal_approved,
            user.email,
            user.full_name or user.email,
            float(withdrawal.amount),
            withdrawal.to_address
        )
    except Exception as e:
        print(f"Failed to send withdrawal approval email: {e}")
    
    return {
        "success": True,
        "message": "تمت الموافقة على طلب السحب",
        "withdrawal_id": withdrawal_id,
        "status": "approved"
    }


@router.post("/withdrawals/{withdrawal_id}/complete")
async def complete_withdrawal(
    withdrawal_id: int,
    tx_hash: str,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """Mark withdrawal as completed after sending funds."""
    result = await db.execute(
        select(WithdrawalRequest, User)
        .join(User, WithdrawalRequest.user_id == User.id)
        .where(WithdrawalRequest.id == withdrawal_id)
    )
    row = result.first()
    
    if not row:
        raise HTTPException(status_code=404, detail="طلب السحب غير موجود")
    
    withdrawal, user = row
    
    if withdrawal.status not in ['pending', 'approved']:
        raise HTTPException(
            status_code=400, 
            detail=f"لا يمكن إتمام طلب سحب بحالة: {withdrawal.status}"
        )
    
    withdrawal.status = 'completed'
    if hasattr(withdrawal, 'tx_hash'):
        withdrawal.tx_hash = tx_hash
    
    # Update transaction record if exists
    tx_result = await db.execute(
        select(Transaction)
        .where(Transaction.reference_id == f"WD-{withdrawal.id}")
    )
    transaction = tx_result.scalar_one_or_none()
    if transaction:
        transaction.status = 'completed'
    
    await db.commit()
    
    # Send completion email
    try:
        from app.services.email_service import email_service
        background_tasks.add_task(
            email_service.send_withdrawal_completed,
            user.email,
            user.full_name or user.email,
            float(withdrawal.amount),
            withdrawal.to_address,
            tx_hash
        )
    except Exception as e:
        print(f"Failed to send withdrawal completion email: {e}")
    
    return {
        "success": True,
        "message": "تم إتمام طلب السحب",
        "withdrawal_id": withdrawal_id,
        "tx_hash": tx_hash,
        "status": "completed"
    }


@router.post("/withdrawals/{withdrawal_id}/reject")
async def reject_withdrawal(
    withdrawal_id: int,
    reason: str,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """Reject a withdrawal request and refund the balance."""
    result = await db.execute(
        select(WithdrawalRequest, User)
        .join(User, WithdrawalRequest.user_id == User.id)
        .where(WithdrawalRequest.id == withdrawal_id)
    )
    row = result.first()
    
    if not row:
        raise HTTPException(status_code=404, detail="طلب السحب غير موجود")
    
    withdrawal, user = row
    
    if withdrawal.status not in ['pending', 'approved']:
        raise HTTPException(
            status_code=400, 
            detail=f"لا يمكن رفض طلب سحب بحالة: {withdrawal.status}"
        )
    
    # Get current NAV
    current_nav = await nav_service.get_current_nav(db)
    
    # Refund the balance
    balance_result = await db.execute(
        select(Balance).where(Balance.user_id == withdrawal.user_id)
    )
    balance = balance_result.scalar_one_or_none()
    
    if balance:
        units_to_refund = float(withdrawal.amount) / current_nav
        balance.units += units_to_refund
    
    withdrawal.status = 'rejected'
    if hasattr(withdrawal, 'rejection_reason'):
        withdrawal.rejection_reason = reason
    
    # Update transaction record if exists
    tx_result = await db.execute(
        select(Transaction)
        .where(Transaction.reference_id == f"WD-{withdrawal.id}")
    )
    transaction = tx_result.scalar_one_or_none()
    if transaction:
        transaction.status = 'cancelled'
    
    await db.commit()
    
    # Send rejection email
    try:
        from app.services.email_service import email_service
        background_tasks.add_task(
            email_service.send_withdrawal_rejected,
            user.email,
            user.full_name or user.email,
            float(withdrawal.amount),
            reason
        )
    except Exception as e:
        print(f"Failed to send withdrawal rejection email: {e}")
    
    return {
        "success": True,
        "message": "تم رفض طلب السحب وإرجاع الرصيد",
        "withdrawal_id": withdrawal_id,
        "reason": reason,
        "refunded_amount": float(withdrawal.amount)
    }


# ============ User Management Endpoints ============

@router.get("/users")
async def get_all_users(
    skip: int = 0,
    limit: int = 100,
    status: Optional[str] = None,
    search: Optional[str] = None,
    current_user: dict = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """Get all users with filters."""
    query = select(User).where(User.is_admin == False)
    
    if status:
        query = query.where(User.status == status)
    
    if search:
        query = query.where(
            User.email.ilike(f"%{search}%") | 
            User.full_name.ilike(f"%{search}%")
        )
    
    query = query.order_by(User.created_at.desc()).offset(skip).limit(limit)
    
    result = await db.execute(query)
    users = result.scalars().all()
    
    # Get balances
    current_nav = await nav_service.get_current_nav(db)
    
    response = []
    for user in users:
        balance_result = await db.execute(select(Balance).where(Balance.user_id == user.id))
        balance = balance_result.scalar_one_or_none()
        current_value = (balance.units * current_nav) if balance else 0
        
        response.append({
            "id": user.id,
            "user_id": f"ASX-{user.id:05d}",
            "email": user.email,
            "full_name": user.full_name,
            "status": user.status,
            "vip_level": user.vip_level if hasattr(user, 'vip_level') else "bronze",
            "is_verified": user.is_verified,
            "two_factor_enabled": user.two_factor_enabled if hasattr(user, 'two_factor_enabled') else False,
            "created_at": user.created_at,
            "total_deposited": float(user.total_deposited or 0) if hasattr(user, 'total_deposited') else 0,
            "current_balance_usd": current_value,
            "units": balance.units if balance else 0
        })
    
    return response


@router.get("/users/{user_id}")
async def get_user_details(
    user_id: int,
    current_user: dict = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """Get detailed user information."""
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(status_code=404, detail="المستخدم غير موجود")
    
    # Get balance
    balance_result = await db.execute(select(Balance).where(Balance.user_id == user_id))
    balance = balance_result.scalar_one_or_none()
    
    current_nav = await nav_service.get_current_nav(db)
    current_value = (balance.units * current_nav) if balance else 0
    
    # Get recent transactions
    transactions_result = await db.execute(
        select(Transaction)
        .where(Transaction.user_id == user_id)
        .order_by(Transaction.created_at.desc())
        .limit(20)
    )
    transactions = transactions_result.scalars().all()
    
    return {
        "user": {
            "id": user.id,
            "user_id": f"ASX-{user.id:05d}",
            "email": user.email,
            "full_name": user.full_name,
            "phone_number": user.phone_number if hasattr(user, 'phone_number') else None,
            "status": user.status,
            "vip_level": user.vip_level if hasattr(user, 'vip_level') else "bronze",
            "is_verified": user.is_verified,
            "is_admin": user.is_admin,
            "two_factor_enabled": user.two_factor_enabled if hasattr(user, 'two_factor_enabled') else False,
            "created_at": user.created_at,
            "last_login_at": user.last_login_at if hasattr(user, 'last_login_at') else None,
            "total_deposited": float(user.total_deposited or 0) if hasattr(user, 'total_deposited') else 0
        },
        "balance": {
            "units": balance.units if balance else 0,
            "current_value_usd": current_value,
            "nav": current_nav
        },
        "recent_transactions": [
            {
                "id": t.id,
                "type": t.type,
                "amount_usd": float(t.amount_usd),
                "units": float(t.units) if t.units else 0,
                "status": t.status,
                "created_at": t.created_at
            }
            for t in transactions
        ]
    }


@router.post("/users/{user_id}/update-status")
async def update_user_status(
    user_id: int,
    new_status: str,
    reason: Optional[str] = None,
    background_tasks: BackgroundTasks = None,
    current_user: dict = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """Update user status (active, suspended, banned)."""
    valid_statuses = ['active', 'suspended', 'banned']
    if new_status not in valid_statuses:
        raise HTTPException(status_code=400, detail=f"الحالة غير صالحة. يجب أن تكون: {valid_statuses}")
    
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(status_code=404, detail="المستخدم غير موجود")
    
    if user.is_admin:
        raise HTTPException(status_code=400, detail="لا يمكن تغيير حالة المدير")
    
    old_status = user.status
    user.status = new_status
    
    if new_status == 'suspended' and hasattr(user, 'suspension_reason'):
        user.suspension_reason = reason
        user.suspended_at = datetime.utcnow()
    elif new_status == 'active':
        if hasattr(user, 'suspension_reason'):
            user.suspension_reason = None
        if hasattr(user, 'suspended_at'):
            user.suspended_at = None
    
    await db.commit()
    
    return {
        "success": True,
        "message": f"تم تغيير حالة المستخدم من {old_status} إلى {new_status}",
        "user_id": user_id,
        "old_status": old_status,
        "new_status": new_status
    }


@router.post("/users/{user_id}/adjust-balance")
async def adjust_user_balance(
    user_id: int,
    amount_usd: float,
    adjustment_type: str,
    reason: str,
    current_user: dict = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """Adjust user balance (add or remove)."""
    if adjustment_type not in ['add', 'remove']:
        raise HTTPException(status_code=400, detail="نوع التعديل يجب أن يكون 'add' أو 'remove'")
    
    if amount_usd <= 0:
        raise HTTPException(status_code=400, detail="المبلغ يجب أن يكون موجباً")
    
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(status_code=404, detail="المستخدم غير موجود")
    
    # Get or create balance
    balance_result = await db.execute(select(Balance).where(Balance.user_id == user_id))
    balance = balance_result.scalar_one_or_none()
    
    if not balance:
        balance = Balance(user_id=user_id, units=0)
        db.add(balance)
    
    current_nav = await nav_service.get_current_nav(db)
    units_change = amount_usd / current_nav
    
    old_units = balance.units
    
    if adjustment_type == 'add':
        balance.units += units_change
        transaction_amount = amount_usd
        transaction_units = units_change
    else:
        if balance.units < units_change:
            raise HTTPException(status_code=400, detail="الرصيد غير كافٍ")
        balance.units -= units_change
        transaction_amount = -amount_usd
        transaction_units = -units_change
    
    # Create transaction record
    admin_id = current_user.id
    transaction = Transaction(
        user_id=user_id,
        type='admin_adjustment',
        amount_usd=transaction_amount,
        units=transaction_units,
        nav_at_time=current_nav,
        status='completed',
        description=f"تعديل إداري: {reason}",
        reference_id=f"ADJ-{admin_id}-{int(datetime.utcnow().timestamp())}"
    )
    db.add(transaction)
    
    await db.commit()
    
    return {
        "success": True,
        "message": f"تم {'إضافة' if adjustment_type == 'add' else 'خصم'} ${amount_usd:.2f}",
        "user_id": user_id,
        "adjustment_type": adjustment_type,
        "amount_usd": amount_usd,
        "units_changed": units_change,
        "old_balance_units": old_units,
        "new_balance_units": balance.units,
        "new_balance_usd": balance.units * current_nav
    }


@router.post("/users/{user_id}/upgrade-vip")
async def upgrade_user_vip(
    user_id: int,
    new_level: str,
    current_user: dict = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """Upgrade user VIP level."""
    valid_levels = ['bronze', 'silver', 'gold', 'platinum', 'diamond']
    if new_level not in valid_levels:
        raise HTTPException(status_code=400, detail=f"المستوى غير صالح. يجب أن يكون: {valid_levels}")
    
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(status_code=404, detail="المستخدم غير موجود")
    
    old_level = user.vip_level if hasattr(user, 'vip_level') else "bronze"
    user.vip_level = new_level
    
    await db.commit()
    
    return {
        "success": True,
        "message": f"تم ترقية المستخدم من {old_level} إلى {new_level}",
        "user_id": user_id,
        "old_level": old_level,
        "new_level": new_level
    }


@router.post("/users/{user_id}/impersonate")
async def impersonate_user(
    user_id: int,
    current_user: dict = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """Generate a token to impersonate a user (for support purposes)."""
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(status_code=404, detail="المستخدم غير موجود")
    
    if user.is_admin:
        raise HTTPException(status_code=400, detail="لا يمكن انتحال صفة مدير")
    
    from app.core.security import create_access_token
    from datetime import timedelta
    
    admin_id = current_user.id
    
    impersonation_token = create_access_token(
        data={
            "sub": str(user.id),
            "impersonated_by": admin_id,
            "read_only": True
        },
        expires_delta=timedelta(minutes=15)
    )
    
    return {
        "success": True,
        "message": f"تم إنشاء رمز انتحال لـ {user.email}",
        "impersonation_token": impersonation_token,
        "expires_in_minutes": 15,
        "user_email": user.email
    }

# ============ Suspend User Endpoint ============
@router.post("/users/{user_id}/suspend")
async def suspend_user(
    user_id: int,
    reason: Optional[str] = None,
    background_tasks: BackgroundTasks = None,
    current_user: User = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """Suspend a user account."""
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(status_code=404, detail="المستخدم غير موجود")
    
    if user.is_admin:
        raise HTTPException(status_code=400, detail="لا يمكن إيقاف حساب المدير")
    
    if user.status == 'suspended':
        raise HTTPException(status_code=400, detail="المستخدم موقوف بالفعل")
    
    old_status = user.status
    user.status = 'suspended'
    
    if hasattr(user, 'suspension_reason'):
        user.suspension_reason = reason
    if hasattr(user, 'suspended_at'):
        user.suspended_at = datetime.utcnow()
    
    await db.commit()
    
    return {
        "success": True,
        "message": "تم إيقاف المستخدم بنجاح",
        "user_id": user_id,
        "old_status": old_status,
        "new_status": "suspended"
    }

# ============ Activate User Endpoint ============
@router.post("/users/{user_id}/activate")
async def activate_user(
    user_id: int,
    background_tasks: BackgroundTasks = None,
    current_user: User = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """Activate a suspended user account."""
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(status_code=404, detail="المستخدم غير موجود")
    
    if user.status == 'active':
        raise HTTPException(status_code=400, detail="المستخدم مفعّل بالفعل")
    
    old_status = user.status
    user.status = 'active'
    
    if hasattr(user, 'suspension_reason'):
        user.suspension_reason = None
    if hasattr(user, 'suspended_at'):
        user.suspended_at = None
    
    await db.commit()
    
    return {
        "success": True,
        "message": "تم تفعيل المستخدم بنجاح",
        "user_id": user_id,
        "old_status": old_status,
        "new_status": "active"
    }
