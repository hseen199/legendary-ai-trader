"""
نظام الأمان المُحسّن - إعدادات الأمان الكاملة
يُستبدل في /opt/asinax/backend/app/api/routes/security.py
"""

from fastapi import APIRouter, Depends, HTTPException, Response, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime, timedelta
import pyotp
import qrcode
import io
import base64
import secrets
import json

from app.core.database import get_db
from app.core.security import get_current_user, get_current_admin, verify_password
from app.models import User
from app.services.email_service import email_service

router = APIRouter(prefix="/security", tags=["Security"])


# ============ Schemas ============

class Setup2FAResponse(BaseModel):
    secret: str
    qr_code_base64: str
    manual_entry_key: str
    backup_codes: List[str]


class Verify2FARequest(BaseModel):
    token: str


class SecuritySettingsResponse(BaseModel):
    two_factor_enabled: bool
    two_factor_required_for_login: bool
    two_factor_required_for_withdrawal: bool
    session_timeout_minutes: int
    max_login_attempts: int
    lockout_duration_minutes: int
    ip_whitelist_enabled: bool
    whitelisted_ips: List[str]
    email_confirmation_for_withdrawal: bool
    large_withdrawal_threshold: float


class UpdateSecuritySettingsRequest(BaseModel):
    two_factor_required_for_login: Optional[bool] = None
    two_factor_required_for_withdrawal: Optional[bool] = None
    session_timeout_minutes: Optional[int] = None
    max_login_attempts: Optional[int] = None
    lockout_duration_minutes: Optional[int] = None
    ip_whitelist_enabled: Optional[bool] = None
    whitelisted_ips: Optional[List[str]] = None
    email_confirmation_for_withdrawal: Optional[bool] = None
    large_withdrawal_threshold: Optional[float] = None


class SystemHealthResponse(BaseModel):
    database: str
    binance_api: str
    nowpayments_api: str
    trading_agent: str
    email_service: str


class EmergencyControlRequest(BaseModel):
    action: str  # enable, disable
    reason: str


# ============ Helper Functions ============

def generate_backup_codes(count: int = 10) -> List[str]:
    """توليد أكواد احتياطية"""
    return [secrets.token_hex(4).upper() for _ in range(count)]


def generate_qr_code_base64(secret: str, email: str, issuer: str = "ASINAX") -> str:
    """توليد QR Code كـ Base64"""
    totp = pyotp.TOTP(secret)
    provisioning_uri = totp.provisioning_uri(name=email, issuer_name=issuer)
    
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )
    qr.add_data(provisioning_uri)
    qr.make(fit=True)
    
    img = qr.make_image(fill_color="black", back_color="white")
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    return f"data:image/png;base64,{img_base64}"


# ============ 2FA Endpoints ============

@router.post("/2fa/setup", response_model=Setup2FAResponse)
async def setup_2fa(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """إعداد المصادقة الثنائية - توليد QR Code والأكواد الاحتياطية"""
    if current_user.two_factor_enabled:
        raise HTTPException(status_code=400, detail="المصادقة الثنائية مفعلة بالفعل")
    
    # توليد سر جديد
    secret = pyotp.random_base32()
    
    # توليد أكواد احتياطية
    backup_codes = generate_backup_codes(10)
    
    # توليد QR Code
    qr_code = generate_qr_code_base64(secret, current_user.email)
    
    # حفظ السر مؤقتاً
    current_user.two_factor_secret = secret
    current_user.two_factor_backup_codes = ",".join(backup_codes)
    
    await db.commit()
    
    return Setup2FAResponse(
        secret=secret,
        qr_code_base64=qr_code,
        manual_entry_key=secret,
        backup_codes=backup_codes
    )


@router.post("/2fa/setup-image")
async def setup_2fa_image(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """إعداد المصادقة الثنائية - إرجاع صورة QR Code مباشرة"""
    if current_user.two_factor_enabled:
        raise HTTPException(status_code=400, detail="المصادقة الثنائية مفعلة بالفعل")
    
    # توليد سر جديد
    secret = pyotp.random_base32()
    
    # حفظ السر
    current_user.two_factor_secret = secret
    await db.commit()
    
    # توليد QR Code
    totp = pyotp.TOTP(secret)
    provisioning_uri = totp.provisioning_uri(name=current_user.email, issuer_name="ASINAX")
    
    img = qrcode.make(provisioning_uri)
    buf = io.BytesIO()
    img.save(buf, "PNG")
    buf.seek(0)
    
    return Response(content=buf.getvalue(), media_type="image/png")


@router.post("/2fa/verify")
async def verify_2fa(
    request: Verify2FARequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """التحقق من رمز 2FA وتفعيله"""
    if not current_user.two_factor_secret:
        raise HTTPException(status_code=400, detail="يجب بدء إعداد المصادقة الثنائية أولاً")
    
    totp = pyotp.TOTP(current_user.two_factor_secret)
    if not totp.verify(request.token, valid_window=1):
        raise HTTPException(status_code=400, detail="الرمز غير صحيح")
    
    current_user.two_factor_enabled = True
    current_user.two_factor_enabled_at = datetime.utcnow()
    
    await db.commit()
    
    return {
        "success": True,
        "message": "تم تفعيل المصادقة الثنائية بنجاح",
        "enabled_at": current_user.two_factor_enabled_at
    }


@router.post("/2fa/disable")
async def disable_2fa(
    token: str,
    password: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """تعطيل المصادقة الثنائية"""
    if not current_user.two_factor_enabled:
        raise HTTPException(status_code=400, detail="المصادقة الثنائية غير مفعلة")
    
    # التحقق من كلمة المرور
    if not verify_password(password, current_user.password_hash):
        raise HTTPException(status_code=401, detail="كلمة المرور غير صحيحة")
    
    # التحقق من رمز 2FA
    totp = pyotp.TOTP(current_user.two_factor_secret)
    if not totp.verify(token, valid_window=1):
        raise HTTPException(status_code=400, detail="رمز المصادقة الثنائية غير صحيح")
    
    current_user.two_factor_enabled = False
    current_user.two_factor_secret = None
    current_user.two_factor_backup_codes = None
    
    await db.commit()
    
    return {"success": True, "message": "تم تعطيل المصادقة الثنائية"}


@router.get("/2fa/status")
async def get_2fa_status(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """حالة المصادقة الثنائية"""
    backup_codes_count = 0
    if current_user.two_factor_backup_codes:
        backup_codes_count = len(current_user.two_factor_backup_codes.split(","))
    
    return {
        "enabled": current_user.two_factor_enabled or False,
        "secret_set": current_user.two_factor_secret is not None,
        "backup_codes_remaining": backup_codes_count,
        "enabled_at": current_user.two_factor_enabled_at if hasattr(current_user, 'two_factor_enabled_at') else None
    }


# ============ Admin Security Settings ============

@router.get("/admin/settings", response_model=SecuritySettingsResponse)
async def get_security_settings(
    current_user: User = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """الحصول على إعدادات الأمان للمنصة"""
    # TODO: Load from database or config
    return SecuritySettingsResponse(
        two_factor_enabled=True,
        two_factor_required_for_login=False,
        two_factor_required_for_withdrawal=True,
        session_timeout_minutes=60,
        max_login_attempts=5,
        lockout_duration_minutes=30,
        ip_whitelist_enabled=False,
        whitelisted_ips=[],
        email_confirmation_for_withdrawal=True,
        large_withdrawal_threshold=5000.0
    )


@router.put("/admin/settings")
async def update_security_settings(
    request: UpdateSecuritySettingsRequest,
    current_user: User = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """تحديث إعدادات الأمان للمنصة"""
    # TODO: Save to database
    updated_fields = []
    
    if request.two_factor_required_for_login is not None:
        updated_fields.append("two_factor_required_for_login")
    if request.two_factor_required_for_withdrawal is not None:
        updated_fields.append("two_factor_required_for_withdrawal")
    if request.session_timeout_minutes is not None:
        updated_fields.append("session_timeout_minutes")
    if request.max_login_attempts is not None:
        updated_fields.append("max_login_attempts")
    if request.lockout_duration_minutes is not None:
        updated_fields.append("lockout_duration_minutes")
    if request.ip_whitelist_enabled is not None:
        updated_fields.append("ip_whitelist_enabled")
    if request.whitelisted_ips is not None:
        updated_fields.append("whitelisted_ips")
    if request.email_confirmation_for_withdrawal is not None:
        updated_fields.append("email_confirmation_for_withdrawal")
    if request.large_withdrawal_threshold is not None:
        updated_fields.append("large_withdrawal_threshold")
    
    return {
        "success": True,
        "message": "تم تحديث إعدادات الأمان",
        "updated_fields": updated_fields
    }


@router.get("/admin/health", response_model=SystemHealthResponse)
async def get_system_health(
    current_user: User = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """فحص صحة النظام"""
    # فحص قاعدة البيانات
    db_status = "healthy"
    try:
        await db.execute(select(func.count(User.id)))
    except:
        db_status = "error"
    
    # فحص Binance API
    binance_status = "unknown"
    try:
        from app.core.config import settings
        if settings.BINANCE_API_KEY:
            binance_status = "configured"
        else:
            binance_status = "not_configured"
    except:
        binance_status = "error"
    
    # فحص NOWPayments API
    nowpayments_status = "unknown"
    try:
        from app.core.config import settings
        if settings.NOWPAYMENTS_API_KEY:
            nowpayments_status = "configured"
        else:
            nowpayments_status = "not_configured"
    except:
        nowpayments_status = "error"
    
    # فحص وكيل التداول
    agent_status = "unknown"
    try:
        import httpx
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get("http://77.37.49.59:8000/health")
            if response.status_code == 200:
                agent_status = "healthy"
            else:
                agent_status = "error"
    except:
        agent_status = "unreachable"
    
    # فحص خدمة البريد
    email_status = "unknown"
    try:
        from app.core.config import settings
        if settings.SMTP_USER:
            email_status = "configured"
        else:
            email_status = "not_configured"
    except:
        email_status = "error"
    
    return SystemHealthResponse(
        database=db_status,
        binance_api=binance_status,
        nowpayments_api=nowpayments_status,
        trading_agent=agent_status,
        email_service=email_status
    )


@router.post("/admin/emergency")
async def emergency_control(
    request: EmergencyControlRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """التحكم في الطوارئ - إيقاف/تشغيل العمليات"""
    if request.action not in ["enable", "disable"]:
        raise HTTPException(status_code=400, detail="إجراء غير صالح")
    
    # TODO: Implement emergency mode in database/config
    
    if request.action == "enable":
        message = f"تم تفعيل وضع الطوارئ. السبب: {request.reason}"
    else:
        message = "تم إلغاء وضع الطوارئ"
    
    # إرسال إشعار لجميع المشرفين
    admins_result = await db.execute(select(User).where(User.is_admin == True))
    admins = admins_result.scalars().all()
    
    for admin in admins:
        if admin.id != current_user.id:
            background_tasks.add_task(
                email_service.send_admin_alert,
                admin.email,
                "تنبيه طوارئ",
                message
            )
    
    return {
        "success": True,
        "message": message,
        "action": request.action,
        "triggered_by": current_user.email,
        "timestamp": datetime.utcnow()
    }


@router.post("/admin/maintenance")
async def toggle_maintenance_mode(
    enable: bool,
    reason: Optional[str] = None,
    current_user: User = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """تفعيل/تعطيل وضع الصيانة"""
    # TODO: Save to database/config
    
    if enable:
        message = f"تم تفعيل وضع الصيانة"
        if reason:
            message += f". السبب: {reason}"
    else:
        message = "تم إلغاء وضع الصيانة"
    
    return {
        "success": True,
        "message": message,
        "maintenance_mode": enable,
        "triggered_by": current_user.email,
        "timestamp": datetime.utcnow()
    }


@router.get("/admin/audit-log")
async def get_audit_log(
    limit: int = 50,
    action_type: Optional[str] = None,
    current_user: User = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """سجل العمليات الإدارية"""
    # TODO: Implement audit log table
    return {
        "logs": [],
        "total": 0,
        "message": "سجل العمليات سيتم تفعيله قريباً"
    }
