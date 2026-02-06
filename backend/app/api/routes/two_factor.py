"""
نظام المصادقة الثنائية (2FA) الكامل
يُضاف إلى /opt/asinax/backend/app/api/routes/two_factor.py
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel
from typing import Optional
import pyotp
import qrcode
import io
import base64
import secrets
from datetime import datetime

from app.core.database import get_db
from app.core.security import get_current_user
from app.models.user import User

router = APIRouter(prefix="/2fa", tags=["Two Factor Authentication"])


# ============ Schemas ============

class Setup2FAResponse(BaseModel):
    """استجابة إعداد المصادقة الثنائية"""
    secret: str
    qr_code_base64: str
    manual_entry_key: str
    backup_codes: list[str]


class Verify2FARequest(BaseModel):
    """طلب التحقق من رمز 2FA"""
    code: str


class Disable2FARequest(BaseModel):
    """طلب تعطيل 2FA"""
    code: str
    password: str


class Status2FAResponse(BaseModel):
    """حالة المصادقة الثنائية"""
    enabled: bool
    pending_setup: bool
    enabled_at: Optional[datetime] = None


# ============ Helper Functions ============

def generate_backup_codes(count: int = 10) -> list[str]:
    """توليد أكواد احتياطية"""
    return [secrets.token_hex(4).upper() for _ in range(count)]


def generate_qr_code(secret: str, email: str, issuer: str = "ASINAX") -> str:
    """توليد QR Code كـ Base64"""
    totp = pyotp.TOTP(secret)
    provisioning_uri = totp.provisioning_uri(name=email, issuer_name=issuer)
    
    # إنشاء QR Code
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )
    qr.add_data(provisioning_uri)
    qr.make(fit=True)
    
    # تحويل إلى صورة
    img = qr.make_image(fill_color="black", back_color="white")
    
    # تحويل إلى Base64
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    return f"data:image/png;base64,{img_base64}"


def verify_totp_code(secret: str, code: str) -> bool:
    """التحقق من رمز TOTP"""
    totp = pyotp.TOTP(secret)
    return totp.verify(code, valid_window=1)  # يسمح بنافذة ±30 ثانية


# ============ Endpoints ============

@router.get("/status", response_model=Status2FAResponse)
async def get_2fa_status(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """الحصول على حالة المصادقة الثنائية للمستخدم"""
    return Status2FAResponse(
        enabled=current_user.two_factor_enabled or False,
        pending_setup=current_user.two_factor_pending if hasattr(current_user, 'two_factor_pending') else False,
        enabled_at=current_user.two_factor_enabled_at if hasattr(current_user, 'two_factor_enabled_at') else None
    )


@router.post("/setup", response_model=Setup2FAResponse)
async def setup_2fa(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    بدء إعداد المصادقة الثنائية
    يُنشئ سر جديد ويُرجع QR Code للمسح
    """
    if current_user.two_factor_enabled:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="المصادقة الثنائية مفعلة بالفعل"
        )
    
    # توليد سر جديد
    secret = pyotp.random_base32()
    
    # توليد أكواد احتياطية
    backup_codes = generate_backup_codes(10)
    
    # توليد QR Code
    qr_code = generate_qr_code(secret, current_user.email)
    
    # حفظ السر مؤقتاً (pending)
    current_user.two_factor_secret = secret
    current_user.two_factor_pending = True
    current_user.two_factor_backup_codes = ",".join(backup_codes)
    
    await db.commit()
    
    return Setup2FAResponse(
        secret=secret,
        qr_code_base64=qr_code,
        manual_entry_key=secret,
        backup_codes=backup_codes
    )


@router.post("/verify-setup")
async def verify_2fa_setup(
    request: Verify2FARequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    التحقق من إعداد 2FA وتفعيله
    يجب إدخال رمز صحيح من تطبيق المصادقة
    """
    if current_user.two_factor_enabled:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="المصادقة الثنائية مفعلة بالفعل"
        )
    
    if not current_user.two_factor_secret:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="يجب بدء إعداد المصادقة الثنائية أولاً"
        )
    
    # التحقق من الرمز
    if not verify_totp_code(current_user.two_factor_secret, request.code):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="الرمز غير صحيح. تأكد من إدخال الرمز الصحيح من تطبيق المصادقة"
        )
    
    # تفعيل 2FA
    current_user.two_factor_enabled = True
    current_user.two_factor_pending = False
    current_user.two_factor_enabled_at = datetime.utcnow()
    
    await db.commit()
    
    return {
        "success": True,
        "message": "تم تفعيل المصادقة الثنائية بنجاح",
        "enabled_at": current_user.two_factor_enabled_at
    }


@router.post("/verify")
async def verify_2fa_code(
    request: Verify2FARequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    التحقق من رمز 2FA (للاستخدام في تسجيل الدخول والسحب)
    """
    if not current_user.two_factor_enabled:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="المصادقة الثنائية غير مفعلة"
        )
    
    if not current_user.two_factor_secret:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="خطأ في إعدادات المصادقة الثنائية"
        )
    
    # التحقق من الرمز العادي
    if verify_totp_code(current_user.two_factor_secret, request.code):
        return {"success": True, "message": "تم التحقق بنجاح"}
    
    # التحقق من الأكواد الاحتياطية
    if current_user.two_factor_backup_codes:
        backup_codes = current_user.two_factor_backup_codes.split(",")
        if request.code.upper() in backup_codes:
            # إزالة الكود المستخدم
            backup_codes.remove(request.code.upper())
            current_user.two_factor_backup_codes = ",".join(backup_codes)
            await db.commit()
            return {
                "success": True,
                "message": "تم التحقق باستخدام كود احتياطي",
                "remaining_backup_codes": len(backup_codes)
            }
    
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="الرمز غير صحيح"
    )


@router.post("/disable")
async def disable_2fa(
    request: Disable2FARequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    تعطيل المصادقة الثنائية
    يتطلب رمز 2FA صحيح وكلمة المرور
    """
    from app.core.security import verify_password
    
    if not current_user.two_factor_enabled:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="المصادقة الثنائية غير مفعلة"
        )
    
    # التحقق من كلمة المرور
    if not verify_password(request.password, current_user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="كلمة المرور غير صحيحة"
        )
    
    # التحقق من رمز 2FA
    if not verify_totp_code(current_user.two_factor_secret, request.code):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="رمز المصادقة الثنائية غير صحيح"
        )
    
    # تعطيل 2FA
    current_user.two_factor_enabled = False
    current_user.two_factor_secret = None
    current_user.two_factor_pending = False
    current_user.two_factor_backup_codes = None
    current_user.two_factor_enabled_at = None
    
    await db.commit()
    
    return {
        "success": True,
        "message": "تم تعطيل المصادقة الثنائية بنجاح"
    }


@router.post("/regenerate-backup-codes")
async def regenerate_backup_codes(
    request: Verify2FARequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    إعادة توليد الأكواد الاحتياطية
    يتطلب رمز 2FA صحيح
    """
    if not current_user.two_factor_enabled:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="المصادقة الثنائية غير مفعلة"
        )
    
    # التحقق من رمز 2FA
    if not verify_totp_code(current_user.two_factor_secret, request.code):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="رمز المصادقة الثنائية غير صحيح"
        )
    
    # توليد أكواد جديدة
    new_backup_codes = generate_backup_codes(10)
    current_user.two_factor_backup_codes = ",".join(new_backup_codes)
    
    await db.commit()
    
    return {
        "success": True,
        "message": "تم توليد أكواد احتياطية جديدة",
        "backup_codes": new_backup_codes
    }


@router.get("/backup-codes")
async def get_backup_codes(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    عرض الأكواد الاحتياطية المتبقية
    """
    if not current_user.two_factor_enabled:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="المصادقة الثنائية غير مفعلة"
        )
    
    backup_codes = []
    if current_user.two_factor_backup_codes:
        backup_codes = current_user.two_factor_backup_codes.split(",")
    
    return {
        "backup_codes_count": len(backup_codes),
        "backup_codes": backup_codes
    }
