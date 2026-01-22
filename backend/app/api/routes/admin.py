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
from app.models.notification import Notification, NotificationType
from app.schemas import (
    AdminWithdrawalReview,
    AdminStats,
    AdminUserResponse,
    WithdrawalRequestResponse,
    NAVResponse,
    TradeResponse
)
from app.services import binance_service, nav_service, email_service
from app.services.ledger_service import LedgerService

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


# ============ Helper Functions ============

async def create_withdrawal_notification(
    db: AsyncSession,
    user_id: int,
    amount: float,
    status: str,
    to_address: str = "",
    rejection_reason: str = "",
    tx_hash: str = ""
):
    """
    إنشاء إشعار للمستخدم عند تغيير حالة السحب
    """
    # تحديد نوع الإشعار والرسالة بناءً على الحالة
    if status == "approved":
        title_ar = "تمت الموافقة على السحب"
        title_en = "Withdrawal Approved"
        message_ar = f"تمت الموافقة على طلب سحبك بمبلغ ${amount:.2f}. يرجى تأكيد السحب من بريدك الإلكتروني."
        message_en = f"Your withdrawal request of ${amount:.2f} has been approved. Please confirm via email."
    elif status == "rejected":
        title_ar = "تم رفض السحب"
        title_en = "Withdrawal Rejected"
        reason_text = rejection_reason if rejection_reason else "لم يتم تحديد السبب"
        message_ar = f"تم رفض طلب سحبك بمبلغ ${amount:.2f}. السبب: {reason_text}"
        message_en = f"Your withdrawal request of ${amount:.2f} has been rejected. Reason: {reason_text}"
    elif status == "completed":
        title_ar = "تم إتمام السحب"
        title_en = "Withdrawal Completed"
        address_short = to_address[:10] + "..." if len(to_address) > 10 else to_address
        message_ar = f"تم إرسال ${amount:.2f} إلى محفظتك ({address_short}) بنجاح."
        message_en = f"${amount:.2f} has been sent to your wallet ({address_short}) successfully."
        if tx_hash:
            message_ar += f" رقم المعاملة: {tx_hash[:20]}..."
            message_en += f" TX: {tx_hash[:20]}..."
    elif status == "pending":
        title_ar = "طلب سحب جديد"
        title_en = "New Withdrawal Request"
        message_ar = f"تم استلام طلب سحبك بمبلغ ${amount:.2f}. في انتظار المراجعة."
        message_en = f"Your withdrawal request of ${amount:.2f} has been received. Pending review."
    else:
        # حالة غير معروفة
        return None
    
    try:
        notification = Notification(
            user_id=user_id,
            type=NotificationType.WITHDRAWAL,
            title=title_ar,  # استخدام العربية كافتراضي
            message=message_ar,
            data={
                "amount": amount,
                "status": status,
                "to_address": to_address,
                "tx_hash": tx_hash,
                "rejection_reason": rejection_reason,
                "title_en": title_en,
                "message_en": message_en
            }
        )
        db.add(notification)
        await db.flush()  # لا نستخدم commit هنا لأننا داخل transaction أكبر
        return notification
    except Exception as e:
        print(f"Error creating withdrawal notification: {str(e)}")
        return None


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
            is_active=user.is_active,
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
        
        # إنشاء إشعار بالموافقة
        await create_withdrawal_notification(
            db=db,
            user_id=withdrawal.user_id,
            amount=float(withdrawal.amount),
            status="approved",
            to_address=withdrawal.to_address
        )
        
        await db.commit()
        
        # Send confirmation email
        confirmation_link = f"{settings.API_V1_PREFIX}/wallet/withdraw/confirm/{withdrawal.confirmation_token}"
        await email_service.send_withdrawal_confirmation(
            user.email,
            user.full_name or "مستثمر",
            float(withdrawal.amount),
            withdrawal.confirmation_token,
            withdrawal.id
        )
        
        return {"message": "Withdrawal approved, confirmation email sent"}
    
    elif review.action == "reject":
        withdrawal.status = "rejected"
        withdrawal.reviewed_by = admin.id
        withdrawal.reviewed_at = datetime.utcnow()
        withdrawal.rejection_reason = review.reason
        
        # إنشاء إشعار بالرفض
        await create_withdrawal_notification(
            db=db,
            user_id=withdrawal.user_id,
            amount=float(withdrawal.amount),
            status="rejected",
            to_address=withdrawal.to_address,
            rejection_reason=review.reason or "لم يتم تحديد السبب"
        )
        
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
    
    # ═══════════════════════════════════════════════════════════════
    # استخدام نظام المحاسبة المزدوجة الجديد
    # ═══════════════════════════════════════════════════════════════
    ledger = LedgerService(db)
    
    # تسجيل السحب في سجل المحاسبة
    ledger_entry = await ledger.record_withdrawal(
        user_id=withdrawal.user_id,
        amount=float(withdrawal.amount),
        description=f"Withdrawal to {withdrawal.to_address[:10]}... (TX: {data.tx_hash or 'pending'})"
    )
    
    # Get user balance
    result = await db.execute(
        select(Balance).where(Balance.user_id == withdrawal.user_id)
    )
    balance = result.scalar_one_or_none()
    
    if balance:
        # Deduct units from balance (using ledger calculated units)
        balance.units -= abs(ledger_entry.units_delta)
        if balance.units < 0:
            balance.units = 0
        balance.balance_usd -= float(withdrawal.amount)
        if balance.balance_usd < 0:
            balance.balance_usd = 0
    
    # Update User table as well
    await db.execute(
        select(User).where(User.id == withdrawal.user_id)
    )
    user_result = await db.execute(select(User).where(User.id == withdrawal.user_id))
    user_to_update = user_result.scalar_one_or_none()
    if user_to_update:
        user_to_update.units -= abs(ledger_entry.units_delta)
        if user_to_update.units < 0:
            user_to_update.units = 0
        user_to_update.balance -= float(withdrawal.amount)
        if user_to_update.balance < 0:
            user_to_update.balance = 0
    
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
        units=abs(ledger_entry.units_delta),
        units_transacted=abs(ledger_entry.units_delta),
        nav_at_transaction=ledger_entry.nav_at_entry,
        status="completed",
        description=f"Withdrawal to {withdrawal.to_address[:10]}..."
    )
    db.add(transaction)
    
    print(f"✅ Withdrawal recorded in ledger: ${withdrawal.amount:.2f} -> {abs(ledger_entry.units_delta):.6f} units @ NAV ${ledger_entry.nav_at_entry:.6f}")
    
    # إنشاء إشعار بإتمام السحب
    await create_withdrawal_notification(
        db=db,
        user_id=withdrawal.user_id,
        amount=float(withdrawal.amount),
        status="completed",
        to_address=withdrawal.to_address,
        tx_hash=data.tx_hash or ""
    )
    
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



# ============ Balance Management ============

class AdjustBalanceRequest(BaseModel):
    amount_usd: float
    reason: str
    operation: str  # "add" or "deduct"


class AdjustBalanceResponse(BaseModel):
    message: str
    user_id: int
    user_email: str
    previous_units: float
    new_units: float
    previous_value_usd: float
    new_value_usd: float
    amount_adjusted_usd: float
    units_adjusted: float
    operation: str
    reason: str


@router.post("/users/{user_id}/adjust-balance", response_model=AdjustBalanceResponse)
async def adjust_user_balance(
    user_id: int,
    data: AdjustBalanceRequest,
    admin: User = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """
    إضافة أو خصم رصيد من حساب المستخدم
    يتم تحويل المبلغ بالدولار إلى وحدات NAV
    """
    # التحقق من صحة البيانات
    if data.amount_usd <= 0:
        raise HTTPException(status_code=400, detail="المبلغ يجب أن يكون أكبر من صفر")
    
    if data.operation not in ["add", "deduct"]:
        raise HTTPException(status_code=400, detail="العملية يجب أن تكون 'add' أو 'deduct'")
    
    if not data.reason or len(data.reason.strip()) < 3:
        raise HTTPException(status_code=400, detail="يجب تحديد سبب التعديل")
    
    # الحصول على المستخدم
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(status_code=404, detail="المستخدم غير موجود")
    
    # الحصول على NAV الحالي
    current_nav = await nav_service.get_current_nav(db)
    
    # حساب الوحدات
    units_to_adjust = data.amount_usd / current_nav
    
    # الحصول على رصيد المستخدم
    result = await db.execute(
        select(Balance).where(Balance.user_id == user_id)
    )
    balance = result.scalar_one_or_none()
    
    if not balance:
        # إنشاء رصيد جديد إذا لم يكن موجوداً
        balance = Balance(user_id=user_id, units=0)
        db.add(balance)
        await db.flush()
    
    previous_units = balance.units
    previous_value_usd = previous_units * current_nav
    
    # تنفيذ العملية
    if data.operation == "add":
        balance.units += units_to_adjust
        transaction_type = "admin_credit"
        description = f"إضافة رصيد بواسطة الأدمن: {data.reason}"
    else:
        if balance.units < units_to_adjust:
            raise HTTPException(
                status_code=400, 
                detail=f"رصيد المستخدم غير كافٍ. الرصيد الحالي: ${previous_value_usd:.2f}"
            )
        balance.units -= units_to_adjust
        transaction_type = "admin_debit"
        description = f"خصم رصيد بواسطة الأدمن: {data.reason}"
    
    new_units = balance.units
    new_value_usd = new_units * current_nav
    
    # إنشاء سجل المعاملة
    transaction = Transaction(
        user_id=user_id,
        type=transaction_type,
        amount_usd=data.amount_usd if data.operation == "add" else -data.amount_usd,
        units=units_to_adjust if data.operation == "add" else -units_to_adjust,
        status="completed",
        description=description
    )
    db.add(transaction)
    
    # إنشاء إشعار للمستخدم
    notification_title = "تعديل الرصيد" if data.operation == "add" else "خصم من الرصيد"
    notification_message = f"تم {'إضافة' if data.operation == 'add' else 'خصم'} ${data.amount_usd:.2f} {'إلى' if data.operation == 'add' else 'من'} رصيدك. السبب: {data.reason}"
    
    notification = Notification(
        user_id=user_id,
        type=NotificationType.SYSTEM,
        title=notification_title,
        message=notification_message,
        data={
            "amount_usd": data.amount_usd,
            "operation": data.operation,
            "reason": data.reason,
            "previous_value": previous_value_usd,
            "new_value": new_value_usd
        }
    )
    db.add(notification)
    
    await db.commit()
    
    return AdjustBalanceResponse(
        message=f"تم {'إضافة' if data.operation == 'add' else 'خصم'} الرصيد بنجاح",
        user_id=user_id,
        user_email=user.email,
        previous_units=previous_units,
        new_units=new_units,
        previous_value_usd=previous_value_usd,
        new_value_usd=new_value_usd,
        amount_adjusted_usd=data.amount_usd,
        units_adjusted=units_to_adjust,
        operation=data.operation,
        reason=data.reason
    )


# ============ Platform Settings ============

class PlatformSettingsResponse(BaseModel):
    min_deposit: float
    min_withdrawal: float
    withdrawal_fee_percent: float
    deposit_fee_percent: float
    referral_bonus_percent: float
    emergency_mode: str
    maintenance_mode: bool
    max_daily_withdrawal: float
    withdrawal_cooldown_hours: int
    auto_approve_withdrawals: bool
    auto_approve_max_amount: float


class UpdatePlatformSettingsRequest(BaseModel):
    min_deposit: Optional[float] = None
    min_withdrawal: Optional[float] = None
    withdrawal_fee_percent: Optional[float] = None
    deposit_fee_percent: Optional[float] = None
    referral_bonus_percent: Optional[float] = None
    maintenance_mode: Optional[bool] = None
    max_daily_withdrawal: Optional[float] = None
    withdrawal_cooldown_hours: Optional[int] = None
    auto_approve_withdrawals: Optional[bool] = None
    auto_approve_max_amount: Optional[float] = None


@router.get("/settings", response_model=PlatformSettingsResponse)
async def get_platform_settings(
    admin: User = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """الحصول على إعدادات المنصة"""
    result = await db.execute(select(PlatformStats).limit(1))
    stats = result.scalar_one_or_none()
    
    if not stats:
        # إنشاء إعدادات افتراضية
        stats = PlatformStats()
        db.add(stats)
        await db.commit()
        await db.refresh(stats)
    
    return PlatformSettingsResponse(
        min_deposit=getattr(stats, 'min_deposit', 100.0),
        min_withdrawal=getattr(stats, 'min_withdrawal', 50.0),
        withdrawal_fee_percent=getattr(stats, 'withdrawal_fee_percent', 0.5),
        deposit_fee_percent=getattr(stats, 'deposit_fee_percent', 2.0),
        referral_bonus_percent=getattr(stats, 'referral_bonus_percent', 5.0),
        emergency_mode=stats.emergency_mode or "off",
        maintenance_mode=getattr(stats, 'maintenance_mode', False),
        max_daily_withdrawal=getattr(stats, 'max_daily_withdrawal', 10000.0),
        withdrawal_cooldown_hours=getattr(stats, 'withdrawal_cooldown_hours', 24),
        auto_approve_withdrawals=getattr(stats, 'auto_approve_withdrawals', False),
        auto_approve_max_amount=getattr(stats, 'auto_approve_max_amount', 500.0)
    )


@router.put("/settings")
async def update_platform_settings(
    data: UpdatePlatformSettingsRequest,
    admin: User = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """تحديث إعدادات المنصة"""
    result = await db.execute(select(PlatformStats).limit(1))
    stats = result.scalar_one_or_none()
    
    if not stats:
        stats = PlatformStats()
        db.add(stats)
    
    # تحديث الإعدادات المرسلة فقط
    update_fields = data.dict(exclude_unset=True)
    for field, value in update_fields.items():
        if hasattr(stats, field):
            setattr(stats, field, value)
    
    await db.commit()
    
    return {"message": "تم تحديث الإعدادات بنجاح", "updated_fields": list(update_fields.keys())}


# ============ Security Settings ============

class SecuritySettingsResponse(BaseModel):
    two_factor_required: bool
    session_timeout_minutes: int
    max_login_attempts: int
    lockout_duration_minutes: int
    ip_whitelist_enabled: bool
    ip_whitelist: list[str]
    admin_notification_email: Optional[str]
    suspicious_activity_alerts: bool
    withdrawal_confirmation_required: bool
    large_withdrawal_threshold: float


class UpdateSecuritySettingsRequest(BaseModel):
    two_factor_required: Optional[bool] = None
    session_timeout_minutes: Optional[int] = None
    max_login_attempts: Optional[int] = None
    lockout_duration_minutes: Optional[int] = None
    ip_whitelist_enabled: Optional[bool] = None
    ip_whitelist: Optional[list[str]] = None
    admin_notification_email: Optional[str] = None
    suspicious_activity_alerts: Optional[bool] = None
    withdrawal_confirmation_required: Optional[bool] = None
    large_withdrawal_threshold: Optional[float] = None


@router.get("/security", response_model=SecuritySettingsResponse)
async def get_security_settings(
    admin: User = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """الحصول على إعدادات الأمان"""
    result = await db.execute(select(PlatformStats).limit(1))
    stats = result.scalar_one_or_none()
    
    return SecuritySettingsResponse(
        two_factor_required=getattr(stats, 'two_factor_required', False) if stats else False,
        session_timeout_minutes=getattr(stats, 'session_timeout_minutes', 60) if stats else 60,
        max_login_attempts=getattr(stats, 'max_login_attempts', 5) if stats else 5,
        lockout_duration_minutes=getattr(stats, 'lockout_duration_minutes', 30) if stats else 30,
        ip_whitelist_enabled=getattr(stats, 'ip_whitelist_enabled', False) if stats else False,
        ip_whitelist=getattr(stats, 'ip_whitelist', []) if stats else [],
        admin_notification_email=getattr(stats, 'admin_notification_email', None) if stats else None,
        suspicious_activity_alerts=getattr(stats, 'suspicious_activity_alerts', True) if stats else True,
        withdrawal_confirmation_required=getattr(stats, 'withdrawal_confirmation_required', True) if stats else True,
        large_withdrawal_threshold=getattr(stats, 'large_withdrawal_threshold', 5000.0) if stats else 5000.0
    )


@router.put("/security")
async def update_security_settings(
    data: UpdateSecuritySettingsRequest,
    admin: User = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """تحديث إعدادات الأمان"""
    result = await db.execute(select(PlatformStats).limit(1))
    stats = result.scalar_one_or_none()
    
    if not stats:
        stats = PlatformStats()
        db.add(stats)
    
    # تحديث الإعدادات المرسلة فقط
    update_fields = data.dict(exclude_unset=True)
    for field, value in update_fields.items():
        if hasattr(stats, field):
            setattr(stats, field, value)
    
    await db.commit()
    
    return {"message": "تم تحديث إعدادات الأمان بنجاح", "updated_fields": list(update_fields.keys())}


# ============ Activity Logs ============

class ActivityLogResponse(BaseModel):
    id: int
    admin_id: int
    admin_email: str
    action: str
    target_type: str
    target_id: Optional[int]
    details: Optional[dict]
    ip_address: Optional[str]
    created_at: datetime


@router.get("/activity-logs")
async def get_activity_logs(
    limit: int = 100,
    admin: User = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """الحصول على سجل نشاط الأدمن"""
    # هذا placeholder - يمكن إضافة جدول ActivityLog لاحقاً
    return {
        "message": "سجل النشاط",
        "logs": [],
        "note": "سيتم تفعيل هذه الميزة قريباً"
    }


# ============ System Health ============

class SystemHealthResponse(BaseModel):
    status: str
    database: str
    redis: str
    binance_api: str
    nowpayments_api: str
    bot_status: str
    last_nav_update: Optional[datetime]
    last_trade: Optional[datetime]
    uptime_hours: float


@router.get("/system-health", response_model=SystemHealthResponse)
async def get_system_health(
    admin: User = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """فحص صحة النظام"""
    # فحص قاعدة البيانات
    try:
        await db.execute(select(func.count(User.id)))
        db_status = "healthy"
    except Exception:
        db_status = "error"
    
    # فحص آخر تحديث NAV
    result = await db.execute(
        select(NAVHistory)
        .order_by(NAVHistory.recorded_at.desc())
        .limit(1)
    )
    last_nav = result.scalar_one_or_none()
    
    # فحص آخر صفقة
    result = await db.execute(
        select(TradingHistory)
        .order_by(TradingHistory.executed_at.desc())
        .limit(1)
    )
    last_trade = result.scalar_one_or_none()
    
    # فحص حالة البوت
    result = await db.execute(select(PlatformStats).limit(1))
    stats = result.scalar_one_or_none()
    
    return SystemHealthResponse(
        status="healthy" if db_status == "healthy" else "degraded",
        database=db_status,
        redis="healthy",  # يمكن إضافة فحص حقيقي
        binance_api="healthy",  # يمكن إضافة فحص حقيقي
        nowpayments_api="healthy",  # يمكن إضافة فحص حقيقي
        bot_status=stats.bot_status if stats and hasattr(stats, 'bot_status') else "unknown",
        last_nav_update=last_nav.recorded_at if last_nav else None,
        last_trade=last_trade.executed_at if last_trade else None,
        uptime_hours=0  # يمكن حسابها من وقت بدء التشغيل
    )


# ============ Bulk Operations ============

class BulkNotificationRequest(BaseModel):
    title: str
    message: str
    user_ids: Optional[list[int]] = None  # إذا كان None، يتم الإرسال لجميع المستخدمين


@router.post("/notifications/bulk")
async def send_bulk_notification(
    data: BulkNotificationRequest,
    admin: User = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """إرسال إشعار جماعي للمستخدمين"""
    if data.user_ids:
        # إرسال لمستخدمين محددين
        result = await db.execute(
            select(User).where(User.id.in_(data.user_ids))
        )
        users = result.scalars().all()
    else:
        # إرسال لجميع المستخدمين النشطين
        result = await db.execute(
            select(User).where(User.status == "active")
        )
        users = result.scalars().all()
    
    notifications_created = 0
    for user in users:
        notification = Notification(
            user_id=user.id,
            type=NotificationType.SYSTEM,
            title=data.title,
            message=data.message,
            data={"bulk": True, "sent_by_admin": admin.id}
        )
        db.add(notification)
        notifications_created += 1
    
    await db.commit()
    
    return {
        "message": f"تم إرسال {notifications_created} إشعار بنجاح",
        "recipients_count": notifications_created
    }



# ============ Referrals Management ============

class ReferralResponse(BaseModel):
    id: int
    referrer_id: int
    referrer_email: str
    referrer_name: Optional[str] = None
    referred_id: int
    referred_email: str
    referred_name: Optional[str] = None
    referred_at: datetime
    reward_given: bool
    reward_amount: float

    class Config:
        from_attributes = True


class ReferralStatsResponse(BaseModel):
    total_referrals: int
    total_rewards_given: float
    pending_rewards: int


class GiveRewardRequest(BaseModel):
    referrer_id: int
    amount: float


@router.get("/referrals", response_model=list[ReferralResponse])
async def get_all_referrals(
    admin: User = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """الحصول على جميع الإحالات مع معلومات من أحال من"""
    # جلب جميع المستخدمين الذين تمت إحالتهم (لديهم referred_by)
    result = await db.execute(
        select(User)
        .where(User.referred_by.isnot(None))
        .order_by(User.created_at.desc())
    )
    referred_users = result.scalars().all()
    
    referrals = []
    for referred in referred_users:
        # جلب معلومات المُحيل
        result = await db.execute(
            select(User).where(User.id == referred.referred_by)
        )
        referrer = result.scalar_one_or_none()
        
        if referrer:
            # التحقق مما إذا تم إعطاء مكافأة (نبحث في transactions)
            result = await db.execute(
                select(Transaction)
                .where(Transaction.user_id == referrer.id)
                .where(Transaction.type == "referral_reward")
                .where(Transaction.notes.contains(str(referred.id)))
            )
            reward_transaction = result.scalar_one_or_none()
            
            referrals.append(ReferralResponse(
                id=referred.id,
                referrer_id=referrer.id,
                referrer_email=referrer.email,
                referrer_name=referrer.full_name,
                referred_id=referred.id,
                referred_email=referred.email,
                referred_name=referred.full_name,
                referred_at=referred.created_at,
                reward_given=reward_transaction is not None,
                reward_amount=reward_transaction.amount_usd if reward_transaction else 0
            ))
    
    return referrals


@router.get("/referrals/stats", response_model=ReferralStatsResponse)
async def get_referral_stats(
    admin: User = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """الحصول على إحصائيات الإحالات"""
    # إجمالي الإحالات
    result = await db.execute(
        select(func.count(User.id))
        .where(User.referred_by.isnot(None))
    )
    total_referrals = result.scalar() or 0
    
    # إجمالي المكافآت المدفوعة
    result = await db.execute(
        select(func.sum(Transaction.amount_usd))
        .where(Transaction.type == "referral_reward")
        .where(Transaction.status == "completed")
    )
    total_rewards_given = result.scalar() or 0
    
    # المكافآت المعلقة (الإحالات بدون مكافأة)
    # نحسب عدد الإحالات - عدد المكافآت المدفوعة
    result = await db.execute(
        select(func.count(Transaction.id))
        .where(Transaction.type == "referral_reward")
        .where(Transaction.status == "completed")
    )
    rewards_given_count = result.scalar() or 0
    pending_rewards = total_referrals - rewards_given_count
    
    return ReferralStatsResponse(
        total_referrals=total_referrals,
        total_rewards_given=total_rewards_given,
        pending_rewards=max(0, pending_rewards)
    )


@router.post("/referrals/reward")
async def give_referral_reward(
    data: GiveRewardRequest,
    admin: User = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """إضافة مكافأة للمُحيل"""
    # التحقق من وجود المُحيل
    result = await db.execute(
        select(User).where(User.id == data.referrer_id)
    )
    referrer = result.scalar_one_or_none()
    
    if not referrer:
        raise HTTPException(status_code=404, detail="المُحيل غير موجود")
    
    # جلب أو إنشاء رصيد المُحيل
    result = await db.execute(
        select(Balance).where(Balance.user_id == referrer.id)
    )
    balance = result.scalar_one_or_none()
    
    if not balance:
        balance = Balance(
            user_id=referrer.id,
            units=0,
            balance_usd=0,
            total_deposited=0,
            total_withdrawn=0
        )
        db.add(balance)
    
    # الحصول على NAV الحالي
    current_nav = await nav_service.get_current_nav(db)
    units_to_add = data.amount / current_nav
    
    # إضافة الرصيد
    balance.balance_usd = (balance.balance_usd or 0) + data.amount
    balance.units = (balance.units or 0) + units_to_add
    
    # تحديث رصيد المستخدم أيضاً
    referrer.balance = (referrer.balance or 0) + data.amount
    referrer.units = (referrer.units or 0) + units_to_add
    
    # إنشاء معاملة للمكافأة
    transaction = Transaction(
        user_id=referrer.id,
        type="referral_reward",
        amount_usd=data.amount,
        units_transacted=units_to_add,
        nav_at_transaction=current_nav,
        status="completed",
        notes=f"مكافأة إحالة من الأدمن"
    )
    db.add(transaction)
    
    # إنشاء إشعار للمُحيل
    notification = Notification(
        user_id=referrer.id,
        type=NotificationType.SYSTEM,
        title="مكافأة إحالة",
        message=f"تم إضافة ${data.amount:.2f} كمكافأة إحالة إلى رصيدك!",
        data={"amount": data.amount, "type": "referral_reward"}
    )
    db.add(notification)
    
    await db.commit()
    
    return {
        "message": f"تم إضافة ${data.amount:.2f} كمكافأة للمُحيل {referrer.email}",
        "new_balance": balance.balance_usd,
        "new_units": balance.units
    }
