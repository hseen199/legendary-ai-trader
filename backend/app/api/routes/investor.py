"""
Investor API Routes
مسارات API للمستثمرين
مدمج من نسخة المستخدم (crowdfund)
"""

from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from typing import Optional, List
from datetime import datetime, timedelta
from decimal import Decimal
from pydantic import BaseModel, Field

from app.core.database import get_db
from app.core.security import get_current_user
from app.models import User, Investor, Deposit, Withdrawal, UnitRecord, DepositStatus, WithdrawalStatus
from app.services.binance_subaccount_service import get_binance_service
from app.services.nav_service import nav_service
from app.services.email_service import email_service

router = APIRouter(prefix="/investor", tags=["Investor"])


# ==================== Schemas ====================

class InvestorProfileResponse(BaseModel):
    """استجابة ملف المستثمر"""
    id: int
    user_id: int
    binance_sub_email: Optional[str]
    deposit_address: Optional[str]
    deposit_network: str
    total_units: float
    total_deposited: float
    total_withdrawn: float
    fees_paid: float
    current_value: float
    profit_loss: float
    profit_loss_percent: float
    status: str
    created_at: datetime
    
    class Config:
        from_attributes = True


class DepositAddressResponse(BaseModel):
    """استجابة عنوان الإيداع"""
    address: str
    network: str
    coin: str
    qr_code_url: Optional[str] = None


class DepositHistoryItem(BaseModel):
    """عنصر في سجل الإيداعات"""
    id: int
    tx_hash: Optional[str]
    amount: float
    coin: str
    units_credited: Optional[float]
    nav_at_deposit: Optional[float]
    status: str
    lock_until: Optional[datetime]
    created_at: datetime
    confirmed_at: Optional[datetime]


class WithdrawalRequest(BaseModel):
    """طلب سحب جديد"""
    units: float = Field(..., gt=0, description="عدد الوحدات للسحب")
    to_address: str = Field(..., min_length=20, description="عنوان المحفظة")
    network: str = Field(default="TRX", description="الشبكة")


class WithdrawalHistoryItem(BaseModel):
    """عنصر في سجل السحوبات"""
    id: int
    units_redeemed: float
    gross_value: Optional[float]
    performance_fee: Optional[float]
    net_value: Optional[float]
    to_address: str
    status: str
    created_at: datetime
    completed_at: Optional[datetime]


class DashboardResponse(BaseModel):
    """استجابة لوحة التحكم"""
    total_units: float
    current_value: float
    total_deposited: float
    total_withdrawn: float
    profit_loss: float
    profit_loss_percent: float
    current_nav: float
    locked_units: float
    available_units: float
    pending_withdrawals: int
    recent_deposits: List[DepositHistoryItem]
    recent_withdrawals: List[WithdrawalHistoryItem]


# ==================== Endpoints ====================

@router.get("/profile", response_model=InvestorProfileResponse)
async def get_investor_profile(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    جلب ملف المستثمر
    """
    # جلب بيانات المستثمر
    result = await db.execute(
        select(Investor).where(Investor.user_id == current_user.id)
    )
    investor = result.scalar_one_or_none()
    
    if not investor:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="لم يتم العثور على ملف المستثمر"
        )
    
    # حساب القيمة الحالية
    current_nav = await nav_service.get_current_nav(db)
    total_units = float(investor.total_units or 0)
    current_value = total_units * current_nav
    total_deposited = float(investor.total_deposited or 0)
    
    # حساب الربح/الخسارة
    profit_loss = current_value - total_deposited + float(investor.total_withdrawn or 0)
    profit_loss_percent = (profit_loss / total_deposited * 100) if total_deposited > 0 else 0
    
    return InvestorProfileResponse(
        id=investor.id,
        user_id=investor.user_id,
        binance_sub_email=investor.binance_sub_email,
        deposit_address=investor.deposit_address_usdc,
        deposit_network=investor.deposit_network or "TRX",
        total_units=total_units,
        total_deposited=total_deposited,
        total_withdrawn=float(investor.total_withdrawn or 0),
        fees_paid=float(investor.fees_paid or 0),
        current_value=current_value,
        profit_loss=profit_loss,
        profit_loss_percent=profit_loss_percent,
        status=investor.status or "active",
        created_at=investor.created_at
    )


@router.post("/register")
async def register_as_investor(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    تسجيل المستخدم كمستثمر وإنشاء حساب فرعي في Binance
    """
    # التحقق من عدم وجود ملف مستثمر مسبقاً
    result = await db.execute(
        select(Investor).where(Investor.user_id == current_user.id)
    )
    existing = result.scalar_one_or_none()
    
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="لديك حساب استثماري بالفعل"
        )
    
    try:
        # إنشاء حساب فرعي في Binance
        binance_service = get_binance_service()
        await binance_service.initialize()
        
        sub_account = await binance_service.create_sub_account(str(current_user.id))
        
        # جلب عنوان الإيداع
        deposit_address = await binance_service.get_deposit_address(
            sub_account.email,
            coin="USDC",
            network="TRX"
        )
        
        # إنشاء سجل المستثمر
        investor = Investor(
            user_id=current_user.id,
            binance_sub_email=sub_account.email,
            binance_sub_id=sub_account.sub_account_id,
            deposit_address_usdc=deposit_address.address,
            deposit_network="TRX",
            status="active"
        )
        
        db.add(investor)
        await db.commit()
        await db.refresh(investor)
        
        return {
            "message": "تم إنشاء حسابك الاستثماري بنجاح",
            "investor_id": investor.id,
            "deposit_address": deposit_address.address,
            "deposit_network": "TRX",
            "coin": "USDC"
        }
        
    except Exception as e:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"فشل في إنشاء الحساب الاستثماري: {str(e)}"
        )


@router.get("/deposit-address", response_model=DepositAddressResponse)
async def get_deposit_address(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    جلب عنوان الإيداع للمستثمر
    """
    result = await db.execute(
        select(Investor).where(Investor.user_id == current_user.id)
    )
    investor = result.scalar_one_or_none()
    
    if not investor:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="لم يتم العثور على ملف المستثمر. يرجى التسجيل أولاً."
        )
    
    if not investor.deposit_address_usdc:
        # محاولة جلب العنوان من Binance
        try:
            binance_service = get_binance_service()
            await binance_service.initialize()
            
            deposit_address = await binance_service.get_deposit_address(
                investor.binance_sub_email,
                coin="USDC",
                network="TRX"
            )
            
            investor.deposit_address_usdc = deposit_address.address
            await db.commit()
            
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"فشل في جلب عنوان الإيداع: {str(e)}"
            )
    
    return DepositAddressResponse(
        address=investor.deposit_address_usdc,
        network=investor.deposit_network or "TRX",
        coin="USDC"
    )


@router.get("/deposits", response_model=List[DepositHistoryItem])
async def get_deposit_history(
    limit: int = 50,
    offset: int = 0,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    جلب سجل الإيداعات
    """
    result = await db.execute(
        select(Investor).where(Investor.user_id == current_user.id)
    )
    investor = result.scalar_one_or_none()
    
    if not investor:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="لم يتم العثور على ملف المستثمر"
        )
    
    result = await db.execute(
        select(Deposit)
        .where(Deposit.investor_id == investor.id)
        .order_by(Deposit.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    deposits = result.scalars().all()
    
    return [
        DepositHistoryItem(
            id=d.id,
            tx_hash=d.tx_hash,
            amount=float(d.amount),
            coin=d.coin,
            units_credited=float(d.units_credited) if d.units_credited else None,
            nav_at_deposit=float(d.nav_at_deposit) if d.nav_at_deposit else None,
            status=d.status.value if d.status else "pending",
            lock_until=d.lock_until,
            created_at=d.created_at,
            confirmed_at=d.confirmed_at
        )
        for d in deposits
    ]


@router.post("/withdraw")
async def request_withdrawal(
    request: WithdrawalRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    طلب سحب جديد
    """
    result = await db.execute(
        select(Investor).where(Investor.user_id == current_user.id)
    )
    investor = result.scalar_one_or_none()
    
    if not investor:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="لم يتم العثور على ملف المستثمر"
        )
    
    # حساب الوحدات المتاحة (غير المقفلة)
    result = await db.execute(
        select(func.sum(UnitRecord.units))
        .where(
            UnitRecord.investor_id == investor.id,
            UnitRecord.is_active == True,
            (UnitRecord.lock_until <= datetime.utcnow()) | (UnitRecord.lock_until == None)
        )
    )
    available_units = result.scalar() or 0
    
    if request.units > float(available_units):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"الوحدات المتاحة للسحب: {available_units:.4f}"
        )
    
    # حساب قيمة السحب
    current_nav = await nav_service.get_current_nav(db)
    gross_value = request.units * current_nav
    
    # حساب تكلفة الأساس
    result = await db.execute(
        select(UnitRecord)
        .where(
            UnitRecord.investor_id == investor.id,
            UnitRecord.is_active == True,
            (UnitRecord.lock_until <= datetime.utcnow()) | (UnitRecord.lock_until == None)
        )
        .order_by(UnitRecord.created_at)
    )
    unit_records = result.scalars().all()
    
    cost_basis = Decimal('0')
    remaining = Decimal(str(request.units))
    for record in unit_records:
        if remaining <= 0:
            break
        units_from_this = min(Decimal(str(record.units)), remaining)
        cost_per_unit = Decimal(str(record.cost_basis)) / Decimal(str(record.units))
        cost_basis += units_from_this * cost_per_unit
        remaining -= units_from_this
    
    # حساب الربح ورسوم الأداء
    profit = max(Decimal('0'), Decimal(str(gross_value)) - cost_basis)
    performance_fee = profit * Decimal('0.20')  # 20% رسوم أداء
    net_value = Decimal(str(gross_value)) - performance_fee
    
    # توليد توكن التأكيد
    email_token = email_service.generate_confirmation_token()
    
    # إنشاء طلب السحب
    withdrawal = Withdrawal(
        investor_id=investor.id,
        units_redeemed=Decimal(str(request.units)),
        nav_at_withdrawal=Decimal(str(current_nav)),
        gross_value=Decimal(str(gross_value)),
        cost_basis=cost_basis,
        profit=profit,
        performance_fee=performance_fee,
        net_value=net_value,
        to_address=request.to_address,
        to_network=request.network,
        email_token=email_token,
        email_token_expires=datetime.utcnow() + timedelta(hours=24),
        status=WithdrawalStatus.PENDING
    )
    
    db.add(withdrawal)
    await db.commit()
    await db.refresh(withdrawal)
    
    # إرسال بريد التأكيد في الخلفية
    background_tasks.add_task(
        email_service.send_withdrawal_confirmation,
        current_user.email,
        current_user.full_name or "مستثمر",
        float(net_value),
        email_token,
        withdrawal.id
    )
    
    return {
        "message": "تم إنشاء طلب السحب. يرجى تأكيده من بريدك الإلكتروني.",
        "withdrawal_id": withdrawal.id,
        "units_redeemed": float(request.units),
        "gross_value": float(gross_value),
        "performance_fee": float(performance_fee),
        "net_value": float(net_value),
        "status": "pending"
    }


@router.get("/withdrawals", response_model=List[WithdrawalHistoryItem])
async def get_withdrawal_history(
    limit: int = 50,
    offset: int = 0,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    جلب سجل السحوبات
    """
    result = await db.execute(
        select(Investor).where(Investor.user_id == current_user.id)
    )
    investor = result.scalar_one_or_none()
    
    if not investor:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="لم يتم العثور على ملف المستثمر"
        )
    
    result = await db.execute(
        select(Withdrawal)
        .where(Withdrawal.investor_id == investor.id)
        .order_by(Withdrawal.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    withdrawals = result.scalars().all()
    
    return [
        WithdrawalHistoryItem(
            id=w.id,
            units_redeemed=float(w.units_redeemed),
            gross_value=float(w.gross_value) if w.gross_value else None,
            performance_fee=float(w.performance_fee) if w.performance_fee else None,
            net_value=float(w.net_value) if w.net_value else None,
            to_address=w.to_address,
            status=w.status.value if w.status else "pending",
            created_at=w.created_at,
            completed_at=w.completed_at
        )
        for w in withdrawals
    ]


@router.get("/dashboard", response_model=DashboardResponse)
async def get_investor_dashboard(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    جلب لوحة تحكم المستثمر
    """
    result = await db.execute(
        select(Investor).where(Investor.user_id == current_user.id)
    )
    investor = result.scalar_one_or_none()
    
    if not investor:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="لم يتم العثور على ملف المستثمر"
        )
    
    # حساب القيم
    current_nav = await nav_service.get_current_nav(db)
    total_units = float(investor.total_units or 0)
    current_value = total_units * current_nav
    total_deposited = float(investor.total_deposited or 0)
    total_withdrawn = float(investor.total_withdrawn or 0)
    
    profit_loss = current_value - total_deposited + total_withdrawn
    profit_loss_percent = (profit_loss / total_deposited * 100) if total_deposited > 0 else 0
    
    # حساب الوحدات المقفلة والمتاحة
    result = await db.execute(
        select(func.sum(UnitRecord.units))
        .where(
            UnitRecord.investor_id == investor.id,
            UnitRecord.is_active == True,
            UnitRecord.lock_until > datetime.utcnow()
        )
    )
    locked_units = float(result.scalar() or 0)
    available_units = total_units - locked_units
    
    # عدد السحوبات المعلقة
    result = await db.execute(
        select(func.count(Withdrawal.id))
        .where(
            Withdrawal.investor_id == investor.id,
            Withdrawal.status.in_([
                WithdrawalStatus.PENDING,
                WithdrawalStatus.EMAIL_SENT,
                WithdrawalStatus.CONFIRMED,
                WithdrawalStatus.APPROVED,
                WithdrawalStatus.PROCESSING
            ])
        )
    )
    pending_withdrawals = result.scalar() or 0
    
    # آخر الإيداعات
    result = await db.execute(
        select(Deposit)
        .where(Deposit.investor_id == investor.id)
        .order_by(Deposit.created_at.desc())
        .limit(5)
    )
    recent_deposits = result.scalars().all()
    
    # آخر السحوبات
    result = await db.execute(
        select(Withdrawal)
        .where(Withdrawal.investor_id == investor.id)
        .order_by(Withdrawal.created_at.desc())
        .limit(5)
    )
    recent_withdrawals = result.scalars().all()
    
    return DashboardResponse(
        total_units=total_units,
        current_value=current_value,
        total_deposited=total_deposited,
        total_withdrawn=total_withdrawn,
        profit_loss=profit_loss,
        profit_loss_percent=profit_loss_percent,
        current_nav=current_nav,
        locked_units=locked_units,
        available_units=available_units,
        pending_withdrawals=pending_withdrawals,
        recent_deposits=[
            DepositHistoryItem(
                id=d.id,
                tx_hash=d.tx_hash,
                amount=float(d.amount),
                coin=d.coin,
                units_credited=float(d.units_credited) if d.units_credited else None,
                nav_at_deposit=float(d.nav_at_deposit) if d.nav_at_deposit else None,
                status=d.status.value if d.status else "pending",
                lock_until=d.lock_until,
                created_at=d.created_at,
                confirmed_at=d.confirmed_at
            )
            for d in recent_deposits
        ],
        recent_withdrawals=[
            WithdrawalHistoryItem(
                id=w.id,
                units_redeemed=float(w.units_redeemed),
                gross_value=float(w.gross_value) if w.gross_value else None,
                performance_fee=float(w.performance_fee) if w.performance_fee else None,
                net_value=float(w.net_value) if w.net_value else None,
                to_address=w.to_address,
                status=w.status.value if w.status else "pending",
                created_at=w.created_at,
                completed_at=w.completed_at
            )
            for w in recent_withdrawals
        ]
    )
