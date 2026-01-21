# API routes للإشعارات
# /opt/asinax/backend/app/api/routes/notifications.py

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, update
from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime

from app.core.database import get_db
from app.core.auth import get_current_user
from app.models.user import User
from app.models.notification import Notification, NotificationType

router = APIRouter(prefix="/notifications", tags=["notifications"])

# ============ Schemas ============

class NotificationResponse(BaseModel):
    id: int
    type: str
    title: str
    message: str
    data: Optional[dict] = None
    is_read: bool
    read_at: Optional[datetime] = None
    created_at: datetime
    
    class Config:
        from_attributes = True

class NotificationCountResponse(BaseModel):
    count: int
    total: int

class UnreadCountResponse(BaseModel):
    count: int

class MarkReadRequest(BaseModel):
    notification_ids: List[int]

# ============ Endpoints ============

@router.get("", response_model=List[NotificationResponse])
async def get_notifications(
    limit: int = 50,
    unread_only: bool = False,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    الحصول على إشعارات المستخدم
    """
    query = select(Notification).where(
        Notification.user_id == current_user.id
    ).order_by(Notification.created_at.desc()).limit(limit)
    
    if unread_only:
        query = query.where(Notification.is_read == False)
    
    result = await db.execute(query)
    notifications = result.scalars().all()
    
    return notifications


@router.get("/unread-count", response_model=UnreadCountResponse)
async def get_unread_count(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    الحصول على عدد الإشعارات غير المقروءة
    """
    query = select(func.count(Notification.id)).where(
        Notification.user_id == current_user.id,
        Notification.is_read == False
    )
    result = await db.execute(query)
    count = result.scalar() or 0
    
    return UnreadCountResponse(count=count)


@router.get("/count", response_model=NotificationCountResponse)
async def get_notification_count(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    الحصول على عدد الإشعارات (المقروءة وغير المقروءة)
    """
    # عدد غير المقروءة
    unread_query = select(func.count(Notification.id)).where(
        Notification.user_id == current_user.id,
        Notification.is_read == False
    )
    unread_result = await db.execute(unread_query)
    unread_count = unread_result.scalar() or 0
    
    # العدد الإجمالي
    total_query = select(func.count(Notification.id)).where(
        Notification.user_id == current_user.id
    )
    total_result = await db.execute(total_query)
    total_count = total_result.scalar() or 0
    
    return NotificationCountResponse(count=unread_count, total=total_count)


@router.post("/{notification_id}/read")
async def mark_notification_as_read(
    notification_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    تحديد إشعار كمقروء
    """
    query = select(Notification).where(
        Notification.id == notification_id,
        Notification.user_id == current_user.id
    )
    result = await db.execute(query)
    notification = result.scalar_one_or_none()
    
    if not notification:
        raise HTTPException(status_code=404, detail="Notification not found")
    
    notification.is_read = True
    notification.read_at = datetime.utcnow()
    await db.commit()
    
    return {"message": "Notification marked as read"}


@router.post("/read-all")
async def mark_all_notifications_as_read(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    تحديد جميع الإشعارات كمقروءة
    """
    stmt = update(Notification).where(
        Notification.user_id == current_user.id,
        Notification.is_read == False
    ).values(is_read=True, read_at=datetime.utcnow())
    
    result = await db.execute(stmt)
    await db.commit()
    
    return {"message": f"Marked {result.rowcount} notifications as read"}


@router.post("/read-multiple")
async def mark_multiple_as_read(
    request: MarkReadRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    تحديد عدة إشعارات كمقروءة
    """
    stmt = update(Notification).where(
        Notification.id.in_(request.notification_ids),
        Notification.user_id == current_user.id
    ).values(is_read=True, read_at=datetime.utcnow())
    
    result = await db.execute(stmt)
    await db.commit()
    
    return {"message": f"Marked {result.rowcount} notifications as read"}


@router.delete("/{notification_id}")
async def delete_notification(
    notification_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    حذف إشعار
    """
    query = select(Notification).where(
        Notification.id == notification_id,
        Notification.user_id == current_user.id
    )
    result = await db.execute(query)
    notification = result.scalar_one_or_none()
    
    if not notification:
        raise HTTPException(status_code=404, detail="Notification not found")
    
    await db.delete(notification)
    await db.commit()
    
    return {"message": "Notification deleted"}
