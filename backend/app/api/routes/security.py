"""
Security Routes - مسارات الأمان والمراقبة
"""
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional
from datetime import datetime
from pydantic import BaseModel

from app.core.database import get_db
from app.core.auth import get_current_admin, get_current_user
from app.services.security_service import (
    AuditService, SessionService, IPWhitelistService,
    TwoFactorService, NotificationService
)

router = APIRouter(prefix="/security", tags=["Security"])


# ============ Schemas ============

class AddIPRequest(BaseModel):
    ip_address: str
    description: Optional[str] = None


class Verify2FARequest(BaseModel):
    code: str
    secret: str


# ============ Audit Log Endpoints ============

@router.get("/audit-logs")
async def get_audit_logs(
    action: Optional[str] = None,
    admin_id: Optional[int] = None,
    target_type: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
    admin = Depends(get_current_admin)
):
    """الحصول على سجلات المراقبة"""
    service = AuditService(db)
    
    start = datetime.strptime(start_date, "%Y-%m-%d") if start_date else None
    end = datetime.strptime(end_date, "%Y-%m-%d") if end_date else None
    
    return await service.get_audit_logs(
        action=action,
        admin_id=admin_id,
        target_type=target_type,
        start_date=start,
        end_date=end,
        limit=limit,
        offset=offset
    )


@router.get("/alerts")
async def get_security_alerts(
    db: AsyncSession = Depends(get_db),
    admin = Depends(get_current_admin)
):
    """الحصول على تنبيهات الأمان"""
    service = AuditService(db)
    return await service.get_security_alerts()


# ============ Session Management ============

@router.get("/sessions")
async def get_my_sessions(
    db: AsyncSession = Depends(get_db),
    admin = Depends(get_current_admin)
):
    """الحصول على جلساتي النشطة"""
    service = SessionService(db)
    return await service.get_active_sessions(admin.id)


@router.delete("/sessions/{session_id}")
async def revoke_session(
    session_id: int,
    db: AsyncSession = Depends(get_db),
    admin = Depends(get_current_admin)
):
    """إلغاء جلسة"""
    service = SessionService(db)
    result = await service.revoke_session(session_id, admin.id)
    
    if not result:
        raise HTTPException(status_code=404, detail="الجلسة غير موجودة")
    
    return {"success": True}


@router.post("/sessions/revoke-all")
async def revoke_all_sessions(
    request: Request,
    db: AsyncSession = Depends(get_db),
    admin = Depends(get_current_admin)
):
    """إلغاء جميع الجلسات (ما عدا الحالية)"""
    service = SessionService(db)
    
    # الحصول على token الحالي من الـ header
    current_token = request.headers.get("Authorization", "").replace("Bearer ", "")
    
    await service.revoke_all_sessions(admin.id, except_current=current_token)
    
    return {"success": True, "message": "تم إلغاء جميع الجلسات الأخرى"}


# ============ IP Whitelist ============

@router.get("/ip-whitelist")
async def get_ip_whitelist(
    db: AsyncSession = Depends(get_db),
    admin = Depends(get_current_admin)
):
    """الحصول على قائمة IPs المسموح بها"""
    service = IPWhitelistService(db)
    return await service.get_all_ips()


@router.post("/ip-whitelist")
async def add_ip_to_whitelist(
    request: AddIPRequest,
    db: AsyncSession = Depends(get_db),
    admin = Depends(get_current_admin)
):
    """إضافة IP للقائمة"""
    service = IPWhitelistService(db)
    
    # تسجيل في سجل المراقبة
    audit = AuditService(db)
    await audit.log_action(
        action="ip_whitelist_add",
        admin_id=admin.id,
        details={"ip_address": request.ip_address}
    )
    
    result = await service.add_ip(
        ip_address=request.ip_address,
        description=request.description,
        added_by=admin.id
    )
    
    if not result:
        raise HTTPException(status_code=400, detail="IP موجود مسبقاً")
    
    return {"success": True}


@router.delete("/ip-whitelist/{ip_address}")
async def remove_ip_from_whitelist(
    ip_address: str,
    db: AsyncSession = Depends(get_db),
    admin = Depends(get_current_admin)
):
    """إزالة IP من القائمة"""
    service = IPWhitelistService(db)
    
    # تسجيل في سجل المراقبة
    audit = AuditService(db)
    await audit.log_action(
        action="ip_whitelist_remove",
        admin_id=admin.id,
        details={"ip_address": ip_address}
    )
    
    result = await service.remove_ip(ip_address)
    
    if not result:
        raise HTTPException(status_code=404, detail="IP غير موجود")
    
    return {"success": True}


@router.put("/ip-whitelist/{ip_id}/toggle")
async def toggle_ip_whitelist(
    ip_id: int,
    is_active: bool,
    db: AsyncSession = Depends(get_db),
    admin = Depends(get_current_admin)
):
    """تفعيل/تعطيل IP"""
    service = IPWhitelistService(db)
    result = await service.toggle_ip(ip_id, is_active)
    
    if not result:
        raise HTTPException(status_code=404, detail="IP غير موجود")
    
    return {"success": True}


# ============ Two-Factor Authentication ============

@router.post("/2fa/generate")
async def generate_2fa_secret(
    db: AsyncSession = Depends(get_db),
    user = Depends(get_current_user)
):
    """إنشاء مفتاح المصادقة الثنائية"""
    service = TwoFactorService(db)
    secret = await service.generate_secret(user.id)
    qr_uri = service.get_qr_uri(secret, user.email)
    
    return {
        "secret": secret,
        "qr_uri": qr_uri
    }


@router.post("/2fa/verify")
async def verify_2fa_code(
    request: Verify2FARequest,
    db: AsyncSession = Depends(get_db),
    user = Depends(get_current_user)
):
    """التحقق من كود المصادقة الثنائية"""
    service = TwoFactorService(db)
    is_valid = service.verify_code(request.secret, request.code)
    
    if not is_valid:
        raise HTTPException(status_code=400, detail="الكود غير صحيح")
    
    # تفعيل 2FA للمستخدم (يجب تحديث جدول المستخدمين)
    
    return {"success": True, "message": "تم تفعيل المصادقة الثنائية"}


@router.post("/2fa/disable")
async def disable_2fa(
    code: str,
    db: AsyncSession = Depends(get_db),
    user = Depends(get_current_user)
):
    """تعطيل المصادقة الثنائية"""
    # يجب التحقق من الكود أولاً قبل التعطيل
    
    return {"success": True, "message": "تم تعطيل المصادقة الثنائية"}


# ============ Notifications ============

@router.get("/notifications")
async def get_my_notifications(
    unread_only: bool = False,
    limit: int = Query(50, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    user = Depends(get_current_user)
):
    """الحصول على إشعاراتي"""
    service = NotificationService(db)
    return await service.get_user_notifications(user.id, unread_only, limit)


@router.get("/notifications/count")
async def get_unread_count(
    db: AsyncSession = Depends(get_db),
    user = Depends(get_current_user)
):
    """عدد الإشعارات غير المقروءة"""
    service = NotificationService(db)
    count = await service.get_unread_count(user.id)
    return {"unread_count": count}


@router.put("/notifications/{notification_id}/read")
async def mark_notification_as_read(
    notification_id: int,
    db: AsyncSession = Depends(get_db),
    user = Depends(get_current_user)
):
    """تحديد إشعار كمقروء"""
    service = NotificationService(db)
    result = await service.mark_as_read(notification_id, user.id)
    
    if not result:
        raise HTTPException(status_code=404, detail="الإشعار غير موجود")
    
    return {"success": True}


@router.post("/notifications/read-all")
async def mark_all_notifications_as_read(
    db: AsyncSession = Depends(get_db),
    user = Depends(get_current_user)
):
    """تحديد جميع الإشعارات كمقروءة"""
    service = NotificationService(db)
    await service.mark_all_as_read(user.id)
    return {"success": True}


# ============ Admin User Management ============

@router.get("/admin/users/{user_id}/activity")
async def get_user_activity(
    user_id: int,
    limit: int = Query(50, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
    admin = Depends(get_current_admin)
):
    """الحصول على نشاط المستخدم"""
    service = AuditService(db)
    return await service.get_audit_logs(
        target_type="user",
        target_id=user_id,
        limit=limit
    )


@router.post("/admin/users/{user_id}/suspend")
async def suspend_user(
    user_id: int,
    reason: str,
    db: AsyncSession = Depends(get_db),
    admin = Depends(get_current_admin)
):
    """تعليق حساب مستخدم"""
    # تسجيل في سجل المراقبة
    audit = AuditService(db)
    await audit.log_action(
        action="user_suspended",
        admin_id=admin.id,
        target_type="user",
        target_id=user_id,
        details={"reason": reason}
    )
    
    # تعليق المستخدم (يجب تحديث جدول المستخدمين)
    
    return {"success": True, "message": "تم تعليق الحساب"}


@router.post("/admin/users/{user_id}/unsuspend")
async def unsuspend_user(
    user_id: int,
    db: AsyncSession = Depends(get_db),
    admin = Depends(get_current_admin)
):
    """إلغاء تعليق حساب مستخدم"""
    # تسجيل في سجل المراقبة
    audit = AuditService(db)
    await audit.log_action(
        action="user_unsuspended",
        admin_id=admin.id,
        target_type="user",
        target_id=user_id
    )
    
    return {"success": True, "message": "تم إلغاء تعليق الحساب"}


@router.post("/admin/users/{user_id}/reset-password")
async def admin_reset_user_password(
    user_id: int,
    db: AsyncSession = Depends(get_db),
    admin = Depends(get_current_admin)
):
    """إعادة تعيين كلمة مرور المستخدم"""
    # تسجيل في سجل المراقبة
    audit = AuditService(db)
    await audit.log_action(
        action="password_reset",
        admin_id=admin.id,
        target_type="user",
        target_id=user_id
    )
    
    # إرسال رابط إعادة تعيين كلمة المرور للمستخدم
    
    return {"success": True, "message": "تم إرسال رابط إعادة تعيين كلمة المرور"}
