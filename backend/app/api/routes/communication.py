"""
Communication Routes - مسارات API للتواصل مع المشتركين
يُضاف إلى /opt/asinax/backend/app/api/routes/communication.py
"""
from datetime import datetime, timedelta
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, or_
from pydantic import BaseModel

from app.core.database import get_db
from app.core.auth import get_current_user, get_current_admin
from app.models import User, Notification

router = APIRouter(prefix="/communication", tags=["Communication"])


# ============ Schemas ============

class BroadcastRequest(BaseModel):
    title: str
    message: str
    message_type: str = "announcement"  # announcement, update, alert, promotion, maintenance
    target_audience: str = "all"  # all, investors, vip, vip_gold_plus, inactive, new_users
    vip_levels: Optional[List[str]] = None
    send_email: bool = True
    send_notification: bool = True


class BroadcastResponse(BaseModel):
    success: bool
    message: str
    sent_count: int
    failed_count: int = 0
    total_target: int = 0


class MaintenanceNotifyRequest(BaseModel):
    start_time: datetime
    end_time: datetime
    description: Optional[str] = None


class NewFeatureRequest(BaseModel):
    feature_name: str
    description: str
    available_for: str = "all"


class PersonalizedAlertRequest(BaseModel):
    user_id: int
    alert_type: str
    title: str
    message: str
    action_url: Optional[str] = None
    priority: str = "normal"  # low, normal, high


class CommunicationStatsResponse(BaseModel):
    total_notifications: int
    unread_notifications: int
    read_rate: float
    vip_distribution: dict
    recent_broadcasts: List[dict]


class UserNotificationPreferences(BaseModel):
    email_trade_alerts: bool = True
    email_daily_summary: bool = True
    email_weekly_report: bool = True
    email_promotions: bool = True
    email_system_updates: bool = True
    push_trade_alerts: bool = True
    push_price_alerts: bool = True


# ============ Admin Endpoints ============

@router.post("/broadcast", response_model=BroadcastResponse)
async def send_broadcast_message(
    request: BroadcastRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """
    إرسال رسالة جماعية للمشتركين (للأدمن فقط)
    """
    from app.services.subscriber_communication_service import SubscriberCommunicationService
    
    service = SubscriberCommunicationService(db)
    result = await service.send_broadcast_message(
        title=request.title,
        message=request.message,
        message_type=request.message_type,
        target_audience=request.target_audience,
        vip_levels=request.vip_levels,
        send_email=request.send_email,
        send_notification=request.send_notification
    )
    
    return BroadcastResponse(**result)


@router.post("/maintenance", response_model=BroadcastResponse)
async def notify_maintenance(
    request: MaintenanceNotifyRequest,
    current_user: User = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """
    إشعار بصيانة مجدولة (للأدمن فقط)
    """
    from app.services.subscriber_communication_service import SubscriberCommunicationService
    
    service = SubscriberCommunicationService(db)
    result = await service.notify_maintenance(
        start_time=request.start_time,
        end_time=request.end_time,
        description=request.description
    )
    
    return BroadcastResponse(**result)


@router.post("/new-feature", response_model=BroadcastResponse)
async def notify_new_feature(
    request: NewFeatureRequest,
    current_user: User = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """
    إشعار بميزة جديدة (للأدمن فقط)
    """
    from app.services.subscriber_communication_service import SubscriberCommunicationService
    
    service = SubscriberCommunicationService(db)
    result = await service.notify_new_feature(
        feature_name=request.feature_name,
        description=request.description,
        available_for=request.available_for
    )
    
    return BroadcastResponse(**result)


@router.post("/personalized-alert")
async def send_personalized_alert(
    request: PersonalizedAlertRequest,
    current_user: User = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """
    إرسال تنبيه مخصص لمستخدم معين (للأدمن فقط)
    """
    from app.services.subscriber_communication_service import SubscriberCommunicationService
    
    service = SubscriberCommunicationService(db)
    result = await service.send_personalized_alert(
        user_id=request.user_id,
        alert_type=request.alert_type,
        title=request.title,
        message=request.message,
        action_url=request.action_url,
        priority=request.priority
    )
    
    return result


@router.post("/market-update")
async def send_market_update(
    update_type: str = Query(..., enum=["bullish", "bearish", "volatile", "stable"]),
    summary: str = Query(...),
    details: Optional[str] = None,
    current_user: User = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """
    إرسال تحديث السوق (للأدمن فقط)
    """
    from app.services.subscriber_communication_service import SubscriberCommunicationService
    
    service = SubscriberCommunicationService(db)
    result = await service.notify_market_update(
        update_type=update_type,
        summary=summary,
        details=details
    )
    
    return result


@router.post("/inactivity-reminder")
async def send_inactivity_reminders(
    days_inactive: int = Query(30, ge=7, le=365),
    current_user: User = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """
    إرسال تذكيرات للمستخدمين غير النشطين (للأدمن فقط)
    """
    from app.services.subscriber_communication_service import SubscriberCommunicationService
    
    service = SubscriberCommunicationService(db)
    result = await service.send_inactivity_reminder(days_inactive=days_inactive)
    
    return result


@router.post("/deposit-reminder")
async def send_deposit_reminders(
    min_balance: float = Query(100, ge=0),
    current_user: User = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """
    إرسال تذكيرات للمستخدمين برصيد منخفض (للأدمن فقط)
    """
    from app.services.subscriber_communication_service import SubscriberCommunicationService
    
    service = SubscriberCommunicationService(db)
    result = await service.send_deposit_reminder(min_balance=min_balance)
    
    return result


@router.get("/stats", response_model=CommunicationStatsResponse)
async def get_communication_stats(
    current_user: User = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """
    إحصائيات التواصل (للأدمن فقط)
    """
    # إجمالي الإشعارات
    total_result = await db.execute(select(func.count(Notification.id)))
    total_notifications = total_result.scalar() or 0
    
    # الإشعارات غير المقروءة
    unread_result = await db.execute(
        select(func.count(Notification.id))
        .where(Notification.is_read == False)
    )
    unread_notifications = unread_result.scalar() or 0
    
    # نسبة القراءة
    read_rate = ((total_notifications - unread_notifications) / total_notifications * 100) if total_notifications > 0 else 0
    
    # توزيع المستخدمين حسب VIP
    vip_result = await db.execute(
        select(User.vip_level, func.count(User.id))
        .where(User.is_active == True)
        .group_by(User.vip_level)
    )
    vip_distribution = {}
    for row in vip_result.all():
        level = row[0] or "bronze"
        vip_distribution[level] = row[1]
    
    # آخر الرسائل الجماعية
    recent_result = await db.execute(
        select(Notification)
        .where(Notification.type.in_(["announcement", "update", "maintenance"]))
        .order_by(Notification.created_at.desc())
        .limit(10)
    )
    recent_broadcasts = [
        {
            "id": n.id,
            "type": n.type,
            "title": n.title,
            "created_at": n.created_at.isoformat()
        }
        for n in recent_result.scalars().all()
    ]
    
    return CommunicationStatsResponse(
        total_notifications=total_notifications,
        unread_notifications=unread_notifications,
        read_rate=read_rate,
        vip_distribution=vip_distribution,
        recent_broadcasts=recent_broadcasts
    )


@router.get("/audience-count")
async def get_audience_count(
    target_audience: str = Query("all", enum=["all", "investors", "vip", "vip_gold_plus", "inactive", "new_users"]),
    vip_levels: Optional[List[str]] = Query(None),
    current_user: User = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """
    الحصول على عدد المستخدمين المستهدفين (للأدمن فقط)
    """
    query = select(func.count(User.id)).where(User.is_active == True)
    
    if target_audience == "investors":
        query = query.where(User.total_deposited > 0)
    elif target_audience == "vip":
        if vip_levels:
            query = query.where(User.vip_level.in_(vip_levels))
        else:
            query = query.where(User.vip_level.in_(["silver", "gold", "platinum", "diamond"]))
    elif target_audience == "vip_gold_plus":
        query = query.where(User.vip_level.in_(["gold", "platinum", "diamond"]))
    elif target_audience == "inactive":
        cutoff = datetime.utcnow() - timedelta(days=30)
        query = query.where(User.last_login < cutoff)
    elif target_audience == "new_users":
        week_ago = datetime.utcnow() - timedelta(days=7)
        query = query.where(User.created_at >= week_ago)
    
    result = await db.execute(query)
    count = result.scalar() or 0
    
    return {
        "target_audience": target_audience,
        "vip_levels": vip_levels,
        "count": count
    }


# ============ User Endpoints ============

@router.get("/preferences")
async def get_notification_preferences(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    الحصول على تفضيلات الإشعارات للمستخدم
    """
    # TODO: جلب التفضيلات من قاعدة البيانات
    return UserNotificationPreferences(
        email_trade_alerts=True,
        email_daily_summary=True,
        email_weekly_report=True,
        email_promotions=True,
        email_system_updates=True,
        push_trade_alerts=True,
        push_price_alerts=True
    )


@router.put("/preferences")
async def update_notification_preferences(
    preferences: UserNotificationPreferences,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    تحديث تفضيلات الإشعارات للمستخدم
    """
    # TODO: حفظ التفضيلات في قاعدة البيانات
    return {
        "success": True,
        "message": "تم تحديث تفضيلات الإشعارات"
    }


@router.get("/unread-count")
async def get_unread_notifications_count(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    الحصول على عدد الإشعارات غير المقروءة
    """
    result = await db.execute(
        select(func.count(Notification.id))
        .where(
            and_(
                Notification.user_id == current_user.id,
                Notification.is_read == False
            )
        )
    )
    count = result.scalar() or 0
    
    return {"unread_count": count}


@router.post("/mark-all-read")
async def mark_all_notifications_read(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    تحديد جميع الإشعارات كمقروءة
    """
    from sqlalchemy import update
    
    await db.execute(
        update(Notification)
        .where(
            and_(
                Notification.user_id == current_user.id,
                Notification.is_read == False
            )
        )
        .values(is_read=True)
    )
    await db.commit()
    
    return {
        "success": True,
        "message": "تم تحديد جميع الإشعارات كمقروءة"
    }
