"""
نظام إدارة المستخدمين الشامل للأدمن
يُضاف إلى /opt/asinax/backend/app/api/routes/admin_users.py
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, or_, and_
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime
from decimal import Decimal

from app.core.database import get_db
from app.core.security import get_current_admin, create_access_token
from app.models import User, Balance
from app.models.notification import Notification
from app.services.email_service import email_service

router = APIRouter(prefix="/admin/users", tags=["Admin Users"])


# ============ Schemas ============

class UserListResponse(BaseModel):
    id: int
    user_id: str  # معرف فريد ASX-XXXXX
    email: str
    full_name: Optional[str]
    phone_number: Optional[str]
    status: str
    vip_level: str
    balance: float
    total_deposited: float
    is_verified: bool
    two_factor_enabled: bool
    created_at: datetime
    last_login: Optional[datetime]


class UserDetailResponse(BaseModel):
    id: int
    user_id: str
    email: str
    full_name: Optional[str]
    phone_number: Optional[str]
    phone_verified: bool
    status: str
    vip_level: str
    balance: float
    total_deposited: float
    is_verified: bool
    two_factor_enabled: bool
    referral_code: Optional[str]
    avatar_url: Optional[str]
    created_at: datetime
    last_login: Optional[datetime]
    total_trades: int
    total_pnl: float


class UpdateBalanceRequest(BaseModel):
    amount: float
    operation: str  # add, subtract, set
    reason: str
    notify_user: bool = True


class UpdateStatusRequest(BaseModel):
    status: str  # active, suspended, banned
    reason: Optional[str] = None
    notify_user: bool = True


class UpdateVIPRequest(BaseModel):
    vip_level: str  # bronze, silver, gold, platinum, diamond
    reason: Optional[str] = None
    notify_user: bool = True


class AdminActionLog(BaseModel):
    action: str
    target_user_id: int
    admin_id: int
    details: dict
    timestamp: datetime


# ============ Helper Functions ============

def generate_user_id(user_id: int) -> str:
    """توليد معرف فريد للمستخدم"""
    return f"ASX-{user_id:05d}"


async def get_user_balance(db: AsyncSession, user_id: int) -> float:
    """جلب رصيد المستخدم"""
    result = await db.execute(
        select(Balance).where(Balance.user_id == user_id)
    )
    balance = result.scalar_one_or_none()
    return float(balance.units or 0) if balance else 0.0


async def update_user_balance(db: AsyncSession, user_id: int, new_balance: float):
    """تحديث رصيد المستخدم"""
    result = await db.execute(
        select(Balance).where(Balance.user_id == user_id)
    )
    balance = result.scalar_one_or_none()
    
    if balance:
        balance.units = Decimal(str(new_balance))
    else:
        balance = Balance(user_id=user_id, units=Decimal(str(new_balance)))
        db.add(balance)
    
    await db.commit()


async def send_admin_notification(
    db: AsyncSession,
    user: User,
    title: str,
    message: str,
    background_tasks: BackgroundTasks,
    send_email: bool = True
):
    """إرسال إشعار من الأدمن للمستخدم"""
    # إنشاء إشعار داخلي
    notification = Notification(
        user_id=user.id,
        title=title,
        message=message,
        type="admin",
        read=False
    )
    db.add(notification)
    
    # إرسال بريد إلكتروني
    if send_email:
        background_tasks.add_task(
            email_service.send_admin_notification,
            user.email,
            user.full_name or user.email,
            title,
            message
        )


# ============ Endpoints ============

@router.get("/list")
async def get_users_list(
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    search: Optional[str] = None,
    status: Optional[str] = None,
    vip_level: Optional[str] = None,
    sort_by: str = "created_at",
    sort_order: str = "desc",
    current_admin: User = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """قائمة المستخدمين مع البحث والفلترة"""
    
    query = select(User).where(User.is_admin == False)
    
    # البحث
    if search:
        query = query.where(
            or_(
                User.email.ilike(f"%{search}%"),
                User.full_name.ilike(f"%{search}%"),
                User.phone_number.ilike(f"%{search}%")
            )
        )
    
    # الفلترة حسب الحالة
    if status:
        query = query.where(User.status == status)
    
    # الفلترة حسب VIP
    if vip_level:
        query = query.where(User.vip_level == vip_level)
    
    # الترتيب
    if sort_order == "desc":
        query = query.order_by(getattr(User, sort_by).desc())
    else:
        query = query.order_by(getattr(User, sort_by).asc())
    
    # العد الكلي
    count_query = select(func.count(User.id)).where(User.is_admin == False)
    if search:
        count_query = count_query.where(
            or_(
                User.email.ilike(f"%{search}%"),
                User.full_name.ilike(f"%{search}%")
            )
        )
    if status:
        count_query = count_query.where(User.status == status)
    if vip_level:
        count_query = count_query.where(User.vip_level == vip_level)
    
    total_result = await db.execute(count_query)
    total = total_result.scalar() or 0
    
    # التصفح
    offset = (page - 1) * limit
    query = query.offset(offset).limit(limit)
    
    result = await db.execute(query)
    users = result.scalars().all()
    
    users_list = []
    for user in users:
        balance = await get_user_balance(db, user.id)
        users_list.append(UserListResponse(
            id=user.id,
            user_id=generate_user_id(user.id),
            email=user.email,
            full_name=user.full_name,
            phone_number=user.phone_number,
            status=user.status or "active",
            vip_level=user.vip_level or "bronze",
            balance=balance,
            total_deposited=float(user.total_deposited or 0),
            is_verified=user.is_verified,
            two_factor_enabled=user.two_factor_enabled or False,
            created_at=user.created_at,
            last_login=user.last_login
        ))
    
    return {
        "users": users_list,
        "total": total,
        "page": page,
        "limit": limit,
        "total_pages": (total + limit - 1) // limit
    }


@router.get("/{user_id}")
async def get_user_detail(
    user_id: int,
    current_admin: User = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """تفاصيل مستخدم محدد"""
    
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(status_code=404, detail="المستخدم غير موجود")
    
    balance = await get_user_balance(db, user.id)
    
    # إحصائيات الصفقات
    from app.models.trading import TradingHistory
    trades_result = await db.execute(
        select(func.count(TradingHistory.id), func.sum(TradingHistory.pnl))
        .where(TradingHistory.user_id == user.id)
    )
    trades_data = trades_result.one()
    total_trades = trades_data[0] or 0
    total_pnl = float(trades_data[1] or 0)
    
    return UserDetailResponse(
        id=user.id,
        user_id=generate_user_id(user.id),
        email=user.email,
        full_name=user.full_name,
        phone_number=user.phone_number,
        phone_verified=user.phone_verified or False,
        status=user.status or "active",
        vip_level=user.vip_level or "bronze",
        balance=balance,
        total_deposited=float(user.total_deposited or 0),
        is_verified=user.is_verified,
        two_factor_enabled=user.two_factor_enabled or False,
        referral_code=user.referral_code,
        avatar_url=user.avatar_url,
        created_at=user.created_at,
        last_login=user.last_login,
        total_trades=total_trades,
        total_pnl=total_pnl
    )


@router.post("/{user_id}/balance")
async def update_user_balance_admin(
    user_id: int,
    request: UpdateBalanceRequest,
    background_tasks: BackgroundTasks,
    current_admin: User = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """تحديث رصيد المستخدم (إضافة/خصم/تعيين)"""
    
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(status_code=404, detail="المستخدم غير موجود")
    
    current_balance = await get_user_balance(db, user_id)
    
    if request.operation == "add":
        new_balance = current_balance + request.amount
        action_text = f"إضافة {request.amount}$"
    elif request.operation == "subtract":
        if current_balance < request.amount:
            raise HTTPException(status_code=400, detail="الرصيد غير كافي للخصم")
        new_balance = current_balance - request.amount
        action_text = f"خصم {request.amount}$"
    elif request.operation == "set":
        new_balance = request.amount
        action_text = f"تعيين الرصيد إلى {request.amount}$"
    else:
        raise HTTPException(status_code=400, detail="عملية غير صالحة")
    
    await update_user_balance(db, user_id, new_balance)
    
    # إرسال إشعار للمستخدم
    if request.notify_user:
        await send_admin_notification(
            db, user,
            "تحديث الرصيد",
            f"تم {action_text} في حسابك. السبب: {request.reason}. الرصيد الجديد: {new_balance}$",
            background_tasks
        )
    
    await db.commit()
    
    return {
        "success": True,
        "message": f"تم {action_text} بنجاح",
        "previous_balance": current_balance,
        "new_balance": new_balance,
        "user_id": generate_user_id(user_id)
    }


@router.post("/{user_id}/status")
async def update_user_status(
    user_id: int,
    request: UpdateStatusRequest,
    background_tasks: BackgroundTasks,
    current_admin: User = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """تحديث حالة المستخدم (تفعيل/إيقاف/حظر)"""
    
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(status_code=404, detail="المستخدم غير موجود")
    
    valid_statuses = ["active", "suspended", "banned"]
    if request.status not in valid_statuses:
        raise HTTPException(status_code=400, detail="حالة غير صالحة")
    
    old_status = user.status
    user.status = request.status
    
    # إرسال إشعار للمستخدم
    if request.notify_user:
        status_messages = {
            "active": "تم تفعيل حسابك",
            "suspended": "تم إيقاف حسابك مؤقتاً",
            "banned": "تم حظر حسابك"
        }
        message = status_messages.get(request.status, "تم تحديث حالة حسابك")
        if request.reason:
            message += f". السبب: {request.reason}"
        
        await send_admin_notification(
            db, user,
            "تحديث حالة الحساب",
            message,
            background_tasks
        )
    
    await db.commit()
    
    return {
        "success": True,
        "message": f"تم تحديث حالة المستخدم إلى {request.status}",
        "previous_status": old_status,
        "new_status": request.status,
        "user_id": generate_user_id(user_id)
    }


@router.post("/{user_id}/vip")
async def update_user_vip(
    user_id: int,
    request: UpdateVIPRequest,
    background_tasks: BackgroundTasks,
    current_admin: User = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """ترقية/تخفيض مستوى VIP للمستخدم"""
    
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(status_code=404, detail="المستخدم غير موجود")
    
    valid_levels = ["bronze", "silver", "gold", "platinum", "diamond"]
    if request.vip_level not in valid_levels:
        raise HTTPException(status_code=400, detail="مستوى VIP غير صالح")
    
    old_level = user.vip_level
    user.vip_level = request.vip_level
    
    # إرسال إشعار للمستخدم
    if request.notify_user:
        level_names = {
            "bronze": "برونزي",
            "silver": "فضي",
            "gold": "ذهبي",
            "platinum": "بلاتيني",
            "diamond": "ماسي"
        }
        message = f"تم تحديث مستوى عضويتك إلى {level_names.get(request.vip_level, request.vip_level)}"
        if request.reason:
            message += f". السبب: {request.reason}"
        
        await send_admin_notification(
            db, user,
            "ترقية العضوية",
            message,
            background_tasks
        )
    
    await db.commit()
    
    return {
        "success": True,
        "message": f"تم تحديث مستوى VIP إلى {request.vip_level}",
        "previous_level": old_level,
        "new_level": request.vip_level,
        "user_id": generate_user_id(user_id)
    }


@router.post("/{user_id}/login-as")
async def login_as_user(
    user_id: int,
    current_admin: User = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """الدخول بحساب المستخدم (للأدمن فقط)"""
    
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(status_code=404, detail="المستخدم غير موجود")
    
    # إنشاء توكن للمستخدم
    access_token = create_access_token(
        data={
            "sub": str(user.id),
            "email": user.email,
            "is_admin": False,
            "impersonated_by": current_admin.id
        }
    )
    
    return {
        "success": True,
        "access_token": access_token,
        "token_type": "bearer",
        "user_id": generate_user_id(user_id),
        "user_email": user.email,
        "message": "يمكنك الآن الدخول بحساب المستخدم"
    }


@router.post("/{user_id}/send-message")
async def send_message_to_user(
    user_id: int,
    title: str,
    message: str,
    background_tasks: BackgroundTasks,
    send_email: bool = True,
    current_admin: User = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """إرسال رسالة مباشرة للمستخدم"""
    
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(status_code=404, detail="المستخدم غير موجود")
    
    await send_admin_notification(db, user, title, message, background_tasks, send_email)
    await db.commit()
    
    return {
        "success": True,
        "message": f"تم إرسال الرسالة إلى {user.email}",
        "user_id": generate_user_id(user_id)
    }


@router.post("/{user_id}/require-support")
async def require_user_support(
    user_id: int,
    reason: str,
    background_tasks: BackgroundTasks,
    current_admin: User = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """طلب من المستخدم التواصل مع الدعم"""
    
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(status_code=404, detail="المستخدم غير موجود")
    
    message = f"يرجى التواصل مع فريق الدعم بخصوص: {reason}"
    
    await send_admin_notification(
        db, user,
        "مطلوب التواصل مع الدعم",
        message,
        background_tasks
    )
    await db.commit()
    
    return {
        "success": True,
        "message": f"تم إرسال طلب التواصل مع الدعم إلى {user.email}",
        "user_id": generate_user_id(user_id)
    }


@router.get("/stats/overview")
async def get_users_stats(
    current_admin: User = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """إحصائيات عامة عن المستخدمين"""
    
    # إجمالي المستخدمين
    total_result = await db.execute(
        select(func.count(User.id)).where(User.is_admin == False)
    )
    total_users = total_result.scalar() or 0
    
    # المستخدمين النشطين
    active_result = await db.execute(
        select(func.count(User.id))
        .where(User.is_admin == False)
        .where(User.status == "active")
    )
    active_users = active_result.scalar() or 0
    
    # المستخدمين الموقوفين
    suspended_result = await db.execute(
        select(func.count(User.id))
        .where(User.is_admin == False)
        .where(User.status == "suspended")
    )
    suspended_users = suspended_result.scalar() or 0
    
    # المستخدمين المحظورين
    banned_result = await db.execute(
        select(func.count(User.id))
        .where(User.is_admin == False)
        .where(User.status == "banned")
    )
    banned_users = banned_result.scalar() or 0
    
    # توزيع VIP
    vip_distribution = {}
    for level in ["bronze", "silver", "gold", "platinum", "diamond"]:
        level_result = await db.execute(
            select(func.count(User.id))
            .where(User.is_admin == False)
            .where(User.vip_level == level)
        )
        vip_distribution[level] = level_result.scalar() or 0
    
    # المستخدمين مع 2FA
    two_fa_result = await db.execute(
        select(func.count(User.id))
        .where(User.is_admin == False)
        .where(User.two_factor_enabled == True)
    )
    two_fa_users = two_fa_result.scalar() or 0
    
    return {
        "total_users": total_users,
        "active_users": active_users,
        "suspended_users": suspended_users,
        "banned_users": banned_users,
        "vip_distribution": vip_distribution,
        "two_fa_enabled": two_fa_users,
        "two_fa_percentage": round((two_fa_users / total_users * 100) if total_users > 0 else 0, 1)
    }
