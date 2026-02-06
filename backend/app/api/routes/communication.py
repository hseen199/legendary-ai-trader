"""
نظام التواصل والرسائل المُحسّن
يُستبدل في /opt/asinax/backend/app/api/routes/communication.py
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, or_, and_
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime

from app.core.database import get_db
from app.core.security import get_current_user, get_current_admin
from app.models import User
from app.models.notification import Notification

router = APIRouter(prefix="/communication", tags=["Communication"])


# ============ Schemas ============

class UserNotificationPreferences(BaseModel):
    email_notifications: bool = True
    push_notifications: bool = True
    sms_notifications: bool = False
    marketing_emails: bool = False
    
    class Config:
        from_attributes = True


class BroadcastRequest(BaseModel):
    title: str
    message: str
    message_type: str = "announcement"  # announcement, alert, promotion, update
    target_audience: str = "all"  # all, active, vip, specific
    target_user_ids: Optional[List[int]] = None  # للإرسال لمستخدمين محددين
    target_vip_levels: Optional[List[str]] = None  # للإرسال لمستويات VIP محددة
    send_email: bool = True
    send_notification: bool = True


class DirectMessageRequest(BaseModel):
    user_id: int
    title: str
    message: str
    message_type: str = "direct"
    send_email: bool = True


class CommunicationStatsResponse(BaseModel):
    total_notifications: int
    unread_notifications: int
    read_rate: float
    total_users: int
    active_users: int
    vip_users: int


# ============ User Endpoints ============

@router.get("/preferences", response_model=UserNotificationPreferences)
async def get_notification_preferences(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """الحصول على تفضيلات الإشعارات للمستخدم"""
    # TODO: Load from user preferences table
    return UserNotificationPreferences(
        email_notifications=True,
        push_notifications=True,
        sms_notifications=False,
        marketing_emails=False
    )


@router.put("/preferences", response_model=UserNotificationPreferences)
async def update_notification_preferences(
    preferences: UserNotificationPreferences,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """تحديث تفضيلات الإشعارات للمستخدم"""
    # TODO: Save to user preferences table
    return preferences


# ============ Admin Endpoints ============

@router.get("/stats")
async def get_communication_stats(
    current_user: User = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """إحصائيات التواصل - للأدمن فقط"""
    
    # إجمالي المستخدمين
    total_users_result = await db.execute(
        select(func.count(User.id)).where(User.is_admin == False)
    )
    total_users = total_users_result.scalar() or 0
    
    # المستخدمين النشطين
    active_users_result = await db.execute(
        select(func.count(User.id))
        .where(User.is_admin == False)
        .where(User.status == "active")
    )
    active_users = active_users_result.scalar() or 0
    
    # مستخدمي VIP (غير برونزي)
    vip_users_result = await db.execute(
        select(func.count(User.id))
        .where(User.is_admin == False)
        .where(User.vip_level != "bronze")
        .where(User.vip_level != None)
    )
    vip_users = vip_users_result.scalar() or 0
    
    # إحصائيات الإشعارات
    total_notifications = 0
    unread_notifications = 0
    
    try:
        total_notif_result = await db.execute(select(func.count(Notification.id)))
        total_notifications = total_notif_result.scalar() or 0
        
        unread_notif_result = await db.execute(
            select(func.count(Notification.id)).where(Notification.read == False)
        )
        unread_notifications = unread_notif_result.scalar() or 0
    except:
        pass
    
    read_rate = 0.0
    if total_notifications > 0:
        read_rate = ((total_notifications - unread_notifications) / total_notifications) * 100
    
    # توزيع VIP
    vip_distribution = {}
    for level in ["bronze", "silver", "gold", "platinum", "diamond"]:
        level_result = await db.execute(
            select(func.count(User.id))
            .where(User.is_admin == False)
            .where(User.vip_level == level)
        )
        vip_distribution[level] = level_result.scalar() or 0
    
    return {
        "total_notifications": total_notifications,
        "unread_notifications": unread_notifications,
        "read_rate": round(read_rate, 1),
        "total_users": total_users,
        "active_users": active_users,
        "vip_users": vip_users,
        "vip_distribution": vip_distribution
    }


@router.get("/audience-count")
async def get_audience_count(
    target_audience: str = "all",
    vip_levels: Optional[str] = None,
    current_user: User = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """عدد المستخدمين المستهدفين"""
    
    query = select(func.count(User.id)).where(User.is_admin == False)
    
    if target_audience == "all":
        pass
    elif target_audience == "active":
        query = query.where(User.status == "active")
    elif target_audience == "vip":
        query = query.where(User.vip_level != "bronze").where(User.vip_level != None)
    elif target_audience == "specific_vip" and vip_levels:
        levels = vip_levels.split(",")
        query = query.where(User.vip_level.in_(levels))
    
    result = await db.execute(query)
    count = result.scalar() or 0
    
    return {"count": count, "target_audience": target_audience}


@router.get("/users-list")
async def get_users_for_messaging(
    search: Optional[str] = None,
    vip_level: Optional[str] = None,
    limit: int = 50,
    current_user: User = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """قائمة المستخدمين للإرسال المباشر"""
    
    query = select(User).where(User.is_admin == False).where(User.status == "active")
    
    if search:
        query = query.where(
            or_(
                User.email.ilike(f"%{search}%"),
                User.full_name.ilike(f"%{search}%")
            )
        )
    
    if vip_level:
        query = query.where(User.vip_level == vip_level)
    
    query = query.order_by(User.created_at.desc()).limit(limit)
    
    result = await db.execute(query)
    users = result.scalars().all()
    
    return [
        {
            "id": user.id,
            "user_id": f"ASX-{user.id:05d}",
            "email": user.email,
            "full_name": user.full_name,
            "vip_level": user.vip_level or "bronze"
        }
        for user in users
    ]


@router.post("/broadcast")
async def send_broadcast(
    request: BroadcastRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """إرسال رسالة جماعية"""
    
    # بناء query للمستخدمين المستهدفين
    query = select(User).where(User.is_admin == False)
    
    if request.target_audience == "active":
        query = query.where(User.status == "active")
    elif request.target_audience == "vip":
        query = query.where(User.vip_level != "bronze").where(User.vip_level != None)
    elif request.target_audience == "specific":
        if request.target_user_ids:
            query = query.where(User.id.in_(request.target_user_ids))
        elif request.target_vip_levels:
            query = query.where(User.vip_level.in_(request.target_vip_levels))
    
    result = await db.execute(query)
    users = result.scalars().all()
    
    if not users:
        raise HTTPException(status_code=400, detail="لا يوجد مستخدمين مستهدفين")
    
    sent_count = 0
    failed_count = 0
    
    for user in users:
        try:
            # إنشاء إشعار
            if request.send_notification:
                notification = Notification(
                    user_id=user.id,
                    title=request.title,
                    message=request.message,
                    type=request.message_type,
                    read=False
                )
                db.add(notification)
            
            # إرسال بريد إلكتروني
            if request.send_email:
                try:
                    from app.services.email_service import email_service
                    background_tasks.add_task(
                        email_service.send_broadcast_email,
                        user.email,
                        user.full_name or user.email,
                        request.title,
                        request.message,
                        request.message_type
                    )
                except Exception as e:
                    print(f"Failed to queue email for {user.email}: {e}")
            
            sent_count += 1
        except Exception as e:
            failed_count += 1
            print(f"Failed to send to user {user.id}: {e}")
    
    await db.commit()
    
    return {
        "success": True,
        "message": f"تم إرسال الرسالة إلى {sent_count} مستخدم",
        "sent_count": sent_count,
        "failed_count": failed_count,
        "total_targeted": len(users)
    }


@router.post("/direct-message")
async def send_direct_message(
    request: DirectMessageRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """إرسال رسالة مباشرة لمستخدم محدد"""
    
    # التحقق من وجود المستخدم
    user_result = await db.execute(select(User).where(User.id == request.user_id))
    user = user_result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(status_code=404, detail="المستخدم غير موجود")
    
    # إنشاء إشعار
    notification = Notification(
        user_id=user.id,
        title=request.title,
        message=request.message,
        type=request.message_type,
        read=False
    )
    db.add(notification)
    
    # إرسال بريد إلكتروني
    if request.send_email:
        try:
            from app.services.email_service import email_service
            background_tasks.add_task(
                email_service.send_direct_message_email,
                user.email,
                user.full_name or user.email,
                request.title,
                request.message
            )
        except Exception as e:
            print(f"Failed to send email: {e}")
    
    await db.commit()
    
    return {
        "success": True,
        "message": f"تم إرسال الرسالة إلى {user.email}",
        "user_id": user.id,
        "user_email": user.email
    }


@router.get("/history")
async def get_broadcast_history(
    limit: int = 50,
    current_user: User = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """سجل الرسائل المرسلة"""
    
    # جلب الإشعارات الأخيرة المرسلة من الأدمن
    result = await db.execute(
        select(Notification)
        .where(Notification.type.in_(["announcement", "alert", "promotion", "direct"]))
        .order_by(Notification.created_at.desc())
        .limit(limit)
    )
    notifications = result.scalars().all()
    
    # تجميع حسب العنوان والوقت
    history = {}
    for notif in notifications:
        key = f"{notif.title}_{notif.created_at.strftime('%Y%m%d%H%M')}"
        if key not in history:
            history[key] = {
                "title": notif.title,
                "message": notif.message[:100] + "..." if len(notif.message) > 100 else notif.message,
                "type": notif.type,
                "sent_at": notif.created_at,
                "recipients_count": 0,
                "read_count": 0
            }
        history[key]["recipients_count"] += 1
        if notif.read:
            history[key]["read_count"] += 1
    
    return list(history.values())[:20]
