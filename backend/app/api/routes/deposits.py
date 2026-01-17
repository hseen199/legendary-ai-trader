"""
Deposits API - واجهة برمجة الإيداعات
"""

from fastapi import APIRouter, Depends, HTTPException, Request, Header
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
from typing import Optional
from datetime import datetime
from pydantic import BaseModel
import json

from app.core.database import get_db
from app.core.security import get_current_user
from app.core.config import settings
from app.models.user import User, Balance
from app.models.transaction import Transaction, NAVHistory
from app.services.nowpayments_service import nowpayments_service
from app.services.marketing_service import ReferralService

router = APIRouter()


# ============ Schemas ============

class CreateDepositRequest(BaseModel):
    """طلب إنشاء إيداع"""
    amount: float  # المبلغ بالدولار
    currency: str = "usdcbsc"  # العملة (usdcbsc, usdcsol)


class CreateDepositResponse(BaseModel):
    """استجابة إنشاء الإيداع"""
    payment_id: int
    payment_status: str
    pay_address: str
    pay_amount: float
    pay_currency: str
    price_amount: float
    price_currency: str
    order_id: str
    created_at: str
    expiration_estimate_date: Optional[str] = None


class DepositStatusResponse(BaseModel):
    """حالة الإيداع"""
    id: int
    status: str
    amount: float
    currency: str
    created_at: str
    confirmed_at: Optional[str] = None


class IPNPayload(BaseModel):
    """بيانات IPN من NOWPayments"""
    payment_id: int
    payment_status: str
    pay_address: str
    price_amount: float
    price_currency: str
    pay_amount: float
    pay_currency: str
    order_id: str
    order_description: Optional[str] = None
    actually_paid: Optional[float] = None
    outcome_amount: Optional[float] = None
    outcome_currency: Optional[str] = None


# ============ Endpoints ============

@router.get("/currencies")
async def get_available_currencies():
    """الحصول على العملات المتاحة للإيداع"""
    try:
        # العملات المدعومة (Solana و BEP20 فقط - TRC20 محظور في أوروبا)
        stable_currencies = [
            {"code": "usdcbsc", "name": "USDC (BNB Smart Chain)", "network": "BEP20"},
            {"code": "usdcsol", "name": "USDC (Solana)", "network": "Solana"},
        ]
        
        return {
            "currencies": stable_currencies,
            "default": "usdcbsc"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/minimum/{currency}")
async def get_minimum_deposit(currency: str):
    """الحصول على الحد الأدنى للإيداع"""
    try:
        min_amount = await nowpayments_service.get_minimum_amount(currency)
        return {
            "currency": currency,
            "minimum_amount": max(min_amount, settings.MIN_DEPOSIT),
            "platform_minimum": settings.MIN_DEPOSIT
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/create", response_model=CreateDepositResponse)
async def create_deposit(
    request: CreateDepositRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    إنشاء طلب إيداع جديد
    
    - يُنشئ فاتورة في NOWPayments
    - يُرجع عنوان الدفع للمستخدم
    """
    # التحقق من الحد الأدنى
    if request.amount < settings.MIN_DEPOSIT:
        raise HTTPException(
            status_code=400, 
            detail=f"الحد الأدنى للإيداع هو {settings.MIN_DEPOSIT} دولار"
        )
    
    # إنشاء معرف فريد للطلب
    order_id = f"DEP_{current_user.id}_{int(datetime.utcnow().timestamp())}"
    
    try:
        # إنشاء الدفعة في NOWPayments
        payment = await nowpayments_service.create_payment(
            price_amount=request.amount,
            price_currency="usd",
            pay_currency=request.currency,
            order_id=order_id,
            order_description=f"Deposit for user {current_user.id}",
            ipn_callback_url=f"{settings.BACKEND_URL}/api/v1/deposits/webhook",
            success_url=f"{settings.FRONTEND_URL}/wallet?status=success",
            cancel_url=f"{settings.FRONTEND_URL}/wallet?status=cancelled",
        )
        
        if "error" in payment:
            raise HTTPException(status_code=400, detail=payment.get("message", "خطأ في إنشاء الدفعة"))
        
        # حفظ المعاملة في قاعدة البيانات
        transaction = Transaction(
            user_id=current_user.id,
            type="deposit",
            amount_usd=request.amount,
            currency=request.currency,
            coin="USDC",
            status="pending",
            external_id=str(payment.get("payment_id")),
            payment_address=payment.get("pay_address"),
            metadata={
                "nowpayments_id": payment.get("payment_id"),
                "order_id": order_id,
                "pay_amount": payment.get("pay_amount"),
                "pay_currency": payment.get("pay_currency"),
            }
        )
        db.add(transaction)
        await db.commit()
        
        return CreateDepositResponse(
            payment_id=payment.get("payment_id"),
            payment_status=payment.get("payment_status", "waiting"),
            pay_address=payment.get("pay_address"),
            pay_amount=payment.get("pay_amount"),
            pay_currency=payment.get("pay_currency"),
            price_amount=payment.get("price_amount"),
            price_currency=payment.get("price_currency"),
            order_id=order_id,
            created_at=payment.get("created_at", datetime.utcnow().isoformat()),
            expiration_estimate_date=payment.get("expiration_estimate_date"),
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"خطأ في إنشاء الإيداع: {str(e)}")


@router.get("/status/{payment_id}")
async def get_deposit_status(
    payment_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """الحصول على حالة الإيداع"""
    try:
        # البحث في قاعدة البيانات
        result = await db.execute(
            select(Transaction).where(
                Transaction.external_id == str(payment_id),
                Transaction.user_id == current_user.id,
                Transaction.type == "deposit"
            )
        )
        transaction = result.scalar_one_or_none()
        
        if not transaction:
            raise HTTPException(status_code=404, detail="الإيداع غير موجود")
        
        # الحصول على الحالة من NOWPayments
        payment_status = await nowpayments_service.get_payment_status(payment_id)
        
        return {
            "id": transaction.id,
            "payment_id": payment_id,
            "status": payment_status.get("payment_status", transaction.status),
            "amount": transaction.amount_usd,
            "currency": transaction.currency,
            "pay_address": transaction.payment_address,
            "actually_paid": payment_status.get("actually_paid"),
            "created_at": transaction.created_at.isoformat(),
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history")
async def get_deposit_history(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    limit: int = 20,
    offset: int = 0
):
    """الحصول على سجل الإيداعات"""
    result = await db.execute(
        select(Transaction)
        .where(
            Transaction.user_id == current_user.id,
            Transaction.type == "deposit"
        )
        .order_by(Transaction.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    transactions = result.scalars().all()
    
    return {
        "deposits": [
            {
                "id": t.id,
                "amount": t.amount_usd,
                "currency": t.currency,
                "status": t.status,
                "payment_address": t.payment_address,
                "created_at": t.created_at.isoformat(),
                "confirmed_at": t.confirmed_at.isoformat() if t.confirmed_at else None,
            }
            for t in transactions
        ],
        "total": len(transactions)
    }


async def get_current_nav(db: AsyncSession) -> float:
    """الحصول على قيمة NAV الحالية"""
    result = await db.execute(
        select(NAVHistory)
        .order_by(NAVHistory.timestamp.desc())
        .limit(1)
    )
    nav_record = result.scalar_one_or_none()
    return nav_record.nav_value if nav_record else settings.INITIAL_NAV


@router.post("/webhook")
async def nowpayments_webhook(
    request: Request,
    db: AsyncSession = Depends(get_db),
    x_nowpayments_sig: Optional[str] = Header(None)
):
    """
    Webhook لاستقبال إشعارات NOWPayments
    
    يتم استدعاؤه تلقائياً عند تغيير حالة الدفع
    """
    try:
        # قراءة البيانات
        payload = await request.json()
        
        # التحقق من التوقيع
        if x_nowpayments_sig:
            if not nowpayments_service.verify_ipn_signature(payload, x_nowpayments_sig):
                raise HTTPException(status_code=401, detail="Invalid signature")
        
        payment_id = payload.get("payment_id")
        payment_status = payload.get("payment_status")
        order_id = payload.get("order_id")
        actually_paid = payload.get("actually_paid", 0)
        
        # البحث عن المعاملة
        result = await db.execute(
            select(Transaction).where(
                Transaction.external_id == str(payment_id)
            )
        )
        transaction = result.scalar_one_or_none()
        
        if not transaction:
            # قد تكون معاملة قديمة أو من نظام آخر
            return {"status": "ignored", "reason": "transaction not found"}
        
        # تحديث الحالة
        internal_status = nowpayments_service.parse_ipn_status(payment_status)
        
        # تحديث metadata
        current_metadata = transaction.metadata or {}
        current_metadata["last_ipn_status"] = payment_status
        current_metadata["actually_paid"] = actually_paid
        
        transaction.status = internal_status
        transaction.metadata = current_metadata
        
        # إذا اكتمل الدفع
        if payment_status in ["finished", "confirmed"]:
            transaction.status = "completed"
            transaction.confirmed_at = datetime.utcnow()
            transaction.completed_at = datetime.utcnow()
            
            # الحصول على NAV الحالي
            current_nav = await get_current_nav(db)
            units_to_add = transaction.amount_usd / current_nav
            
            transaction.units_transacted = units_to_add
            transaction.nav_at_transaction = current_nav
            
            # تحديث رصيد المستخدم في جدول User
            await db.execute(
                update(User)
                .where(User.id == transaction.user_id)
                .values(
                    balance=User.balance + transaction.amount_usd,
                    units=User.units + units_to_add,
                    total_deposited=User.total_deposited + transaction.amount_usd
                )
            )
            
            # تحديث رصيد المستخدم في جدول Balance
            balance_result = await db.execute(
                select(Balance).where(Balance.user_id == transaction.user_id)
            )
            balance = balance_result.scalar_one_or_none()
            
            if balance:
                balance.units += units_to_add
                balance.balance_usd += transaction.amount_usd
                balance.total_deposited += transaction.amount_usd
                balance.last_deposit_at = datetime.utcnow()
            else:
                # إنشاء سجل رصيد جديد إذا لم يكن موجوداً
                new_balance = Balance(
                    user_id=transaction.user_id,
                    units=units_to_add,
                    balance_usd=transaction.amount_usd,
                    total_deposited=transaction.amount_usd,
                    last_deposit_at=datetime.utcnow()
                )
                db.add(new_balance)

            # معالجة مكافأة الإحالة (إذا كان أول إيداع)
            try:
                referral_service = ReferralService(db)
                bonus = await referral_service.process_referral_bonus(transaction.user_id, transaction.amount_usd)
                if bonus:
                    print(f"Referral bonus ${bonus} processed for user {transaction.user_id}")
            except Exception as ref_error:
                print(f"Referral bonus error: {str(ref_error)}")
        
        await db.commit()
        
        return {"status": "ok", "payment_status": internal_status}
        
    except HTTPException:
        raise
    except Exception as e:
        # Log the error but return OK to prevent retries
        print(f"Webhook error: {str(e)}")
        return {"status": "error", "message": str(e)}
