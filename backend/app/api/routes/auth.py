import httpx
from pydantic import BaseModel
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks, Request
from fastapi.responses import RedirectResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from datetime import datetime, timedelta
from app.core.database import get_db
from app.core.security import (
    create_access_token, 
    get_password_hash, 
    verify_password, 
    get_current_user
)
from app.core.config import settings
from app.models import User, Balance, OTP
from app.schemas import UserCreate, UserLogin, Token, OTPSendRequest, OTPVerifyRequest
from app.services.marketing_service import ReferralService
from app.services import email_service
from app.utils.helpers import generate_otp, get_client_info

router = APIRouter(prefix="/auth", tags=["Authentication"])

# Google OAuth settings
GOOGLE_CLIENT_ID = settings.GOOGLE_CLIENT_ID
GOOGLE_CLIENT_SECRET = settings.GOOGLE_CLIENT_SECRET
GOOGLE_REDIRECT_URI = settings.GOOGLE_REDIRECT_URI
GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_URL = "https://www.googleapis.com/oauth2/v1/userinfo"

@router.post("/register", status_code=status.HTTP_201_CREATED)
async def register(
    request: UserCreate,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
):
    """Register a new user - Step 1: Create account and send OTP"""
    email = request.email.lower()
    
    # Check if user already exists
    result = await db.execute(select(User).where(User.email == email))
    existing_user = result.scalar_one_or_none()
    
    if existing_user:
        if existing_user.is_verified:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="البريد الإلكتروني مسجل مسبقاً"
            )
        else:
            # User exists but not verified - resend OTP
            otp_code = generate_otp()
            otp_expiry = datetime.utcnow() + timedelta(minutes=10)  # OTP valid for 10 minutes
            otp = OTP(user_id=existing_user.id, code=otp_code, purpose="email_verification", expires_at=otp_expiry)
            db.add(otp)
            await db.commit()
            background_tasks.add_task(
                email_service.send_otp,
                email,
                otp_code
            )
            return {
                "success": True,
                "message": "تم إرسال رمز التحقق إلى بريدك الإلكتروني",
                "requires_otp": True,
                "email": email
            }
    
    # Create new user (unverified)
    hashed_password = get_password_hash(request.password)
    user = User(
        email=email,
        full_name=request.full_name,
        password_hash=hashed_password,
        is_verified=False,
        status="active"
    )
    db.add(user)
    await db.flush()
    
    # Create balance
    balance = Balance(user_id=user.id, units=0.0)
    db.add(balance)
    
    # Handle referral
    if hasattr(request, 'referral_code') and request.referral_code:
        referral_service = ReferralService(db)
        await referral_service.apply_referral_code(user.id, request.referral_code)
    
    await db.commit()
    await db.refresh(user)
    
    # Send verification OTP
    otp_code = generate_otp()
    otp_expiry = datetime.utcnow() + timedelta(minutes=10)  # OTP valid for 10 minutes
    otp = OTP(user_id=user.id, code=otp_code, purpose="email_verification", expires_at=otp_expiry)
    db.add(otp)
    await db.commit()
    
    background_tasks.add_task(
        email_service.send_otp,
        email,
        otp_code
    )
    
    return {
        "success": True,
        "message": "تم إنشاء الحساب. يرجى إدخال رمز التحقق المرسل إلى بريدك الإلكتروني",
        "requires_otp": True,
        "email": email
    }


# Pydantic model for register verify
class RegisterVerifyRequest(BaseModel):
    email: str
    otp: str


@router.post("/register/verify")
async def register_verify(
    request: RegisterVerifyRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
):
    """Register - Step 2: Verify OTP and activate account"""
    email = request.email.lower()
    otp = request.otp.strip()
    
    # Get user
    result = await db.execute(select(User).where(User.email == email))
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(
            status_code=404,
            detail="المستخدم غير موجود"
        )
    
    # Verify OTP
    result = await db.execute(
        select(OTP)
        .where(OTP.user_id == user.id)
        .where(OTP.code == otp)
        .where(OTP.expires_at > datetime.utcnow())
        .order_by(OTP.created_at.desc())
    )
    otp_record = result.scalars().first()
    
    if not otp_record:
        raise HTTPException(
            status_code=400,
            detail="رمز التحقق غير صحيح أو منتهي الصلاحية"
        )
    
    # Activate account
    user.is_verified = True
    await db.delete(otp_record)
    await db.commit()
    
    # Send welcome email
    background_tasks.add_task(
        email_service.send_welcome_email,
        email,
        user.full_name or "مستخدم"
    )
    
    # Create JWT token
    access_token = create_access_token(
        data={"sub": str(user.id), "email": user.email, "is_admin": user.is_admin}
    )
    
    return {
        "success": True,
        "message": "تم تفعيل حسابك بنجاح",
        "access_token": access_token,
        "token_type": "bearer"
    }

@router.post("/login", response_model=Token)
async def login(
    request: UserLogin,
    background_tasks: BackgroundTasks,
    http_request: Request,
    db: AsyncSession = Depends(get_db)
):
    """Login user"""
    result = await db.execute(select(User).where(User.email == request.email))
    user = result.scalar_one_or_none()

    if not user or not verify_password(request.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.is_verified:
        raise HTTPException(status_code=400, detail="Email not verified")

    user.last_login = datetime.utcnow()
    await db.commit()

    # Send login notification
    client_info = get_client_info(http_request)
    background_tasks.add_task(
        email_service.send_login_notification,
        user.email,
        user.full_name or user.email,
        client_info.get("ip", "Unknown"),
        client_info.get("user_agent", "Unknown")
    )

    access_token = create_access_token(
        data={"sub": str(user.id), "email": user.email, "is_admin": user.is_admin}
    )
    return {"access_token": access_token, "token_type": "bearer"}

# ============== OTP Endpoints ==============
@router.post("/otp/send")
async def send_otp(
    request: OTPSendRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
):
    """Generate and send OTP to user email"""
    result = await db.execute(select(User).where(User.email == request.email))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    otp_code = generate_otp()
    otp_expiry = datetime.utcnow() + timedelta(minutes=10)  # OTP valid for 10 minutes

    otp = OTP(user_id=user.id, code=otp_code, purpose="email_verification", expires_at=otp_expiry)
    db.add(otp)
    await db.commit()

    background_tasks.add_task(
        email_service.send_otp_email, 
        user.email, 
        otp_code
    )

    return {"message": "OTP sent to your email"}

@router.post("/otp/verify")
async def verify_otp(
    request: OTPVerifyRequest,
    db: AsyncSession = Depends(get_db)
):
    """Verify OTP and user email"""
    result = await db.execute(select(User).where(User.email == request.email))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    result = await db.execute(
        select(OTP)
        .where(OTP.user_id == user.id)
        .where(OTP.code == request.otp)
        .where(OTP.expires_at > datetime.utcnow())
        .order_by(OTP.created_at.desc())
    )
    otp = result.scalars().first()

    if not otp:
        raise HTTPException(status_code=400, detail="Invalid or expired OTP")

    user.is_verified = True
    await db.delete(otp) # OTP is single-use
    await db.commit()

    return {"message": "Email verified successfully"}

@router.post("/otp/resend")
async def resend_otp(
    request: OTPSendRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
):
    """Resend OTP to email"""
    return await send_otp(request, background_tasks, db)

# ============== Google OAuth Endpoints ==============
@router.get("/google/login")
async def google_login():
    """Redirect to Google OAuth login page"""
    params = {
        "client_id": GOOGLE_CLIENT_ID,
        "redirect_uri": GOOGLE_REDIRECT_URI,
        "response_type": "code",
        "scope": "openid email profile",
        "access_type": "offline",
        "prompt": "consent"
    }
    
    query_string = "&".join([f"{k}={v}" for k, v in params.items()])
    auth_url = f"{GOOGLE_AUTH_URL}?{query_string}"
    
    return RedirectResponse(url=auth_url)

@router.get("/google/callback")
async def google_callback(
    code: str,
    request: Request,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
):
    """Handle Google OAuth callback"""
    try:
        async with httpx.AsyncClient() as client:
            token_response = await client.post(
                GOOGLE_TOKEN_URL,
                data={
                    "client_id": GOOGLE_CLIENT_ID,
                    "client_secret": GOOGLE_CLIENT_SECRET,
                    "code": code,
                    "grant_type": "authorization_code",
                    "redirect_uri": GOOGLE_REDIRECT_URI
                }
            )
            
            if token_response.status_code != 200:
                return RedirectResponse(url="https://asinax.cloud/login?error=token_failed")
            
            tokens = token_response.json()
            access_token = tokens.get("access_token")
            
            userinfo_response = await client.get(
                GOOGLE_USERINFO_URL,
                headers={"Authorization": f"Bearer {access_token}"}
            )
            
            if userinfo_response.status_code != 200:
                return RedirectResponse(url="https://asinax.cloud/login?error=userinfo_failed")
            
            userinfo = userinfo_response.json()
        
        email = userinfo.get("email")
        name = userinfo.get("name", "")
        google_id = userinfo.get("sub")
        picture = userinfo.get("picture", "")
        
        if not email:
            return RedirectResponse(url="https://asinax.cloud/login?error=no_email")
        
        # Check if user exists
        result = await db.execute(select(User).where(User.email == email))
        user = result.scalar_one_or_none()
        
        is_new_user = False
        if user:
            # Update Google info
            user.google_id = google_id
            user.avatar_url = picture
            user.is_verified = True
            user.last_login = datetime.utcnow()
            await db.commit()
        else:
            # Create new user
            is_new_user = True
            user = User(
                email=email,
                full_name=name,
                google_id=google_id,
                avatar_url=picture,
                is_verified=True,
                password_hash="",  # No password for Google users
                status="active"
            )

            # Handle referral from cookie or session if available
            referral_code = request.cookies.get("referral_code")
            if referral_code:
                referral_service = ReferralService(db)
                await referral_service.apply_referral_code(user.id, referral_code)
            db.add(user)
            await db.flush()
            
            # Create balance
            balance = Balance(user_id=user.id, units=0.0)
            db.add(balance)
            
            await db.commit()
            await db.refresh(user)
        
        # Send login notification
        client_info = get_client_info(request)
        background_tasks.add_task(
            email_service.send_login_notification,
            email,
            name,
            client_info.get("ip", "Unknown"),
            client_info.get("user_agent", "Unknown")
        )
        
        # Send welcome email for new users
        if is_new_user:
            background_tasks.add_task(
                email_service.send_welcome_email,
                email,
                name or "user"
            )
        
        # Create JWT token
        jwt_token = create_access_token(
            data={"sub": str(user.id), "email": user.email, "is_admin": user.is_admin}
        )
        
        # Redirect to frontend with token
        return RedirectResponse(url=f"https://asinax.cloud/auth/callback?token={jwt_token}")
        
    except Exception as e:
        import logging
        logging.error(f"Google OAuth error: {str(e)}")
        return RedirectResponse(url=f"https://asinax.cloud/login?error=google_auth_failed")

@router.post("/refresh-token")
async def refresh_token(
    current_user: User = Depends(get_current_user)
):
    """Refresh access token with updated user data"""
    # Create new token with current user data
    access_token = create_access_token(
        data={"sub": str(current_user.id), "email": current_user.email, "is_admin": current_user.is_admin}
    )
    return {"access_token": access_token, "token_type": "bearer"}

# نقطة النهاية للتسجيل المرحلي (للتوافق مع الواجهة الأمامية)
class LoginStep1Request(BaseModel):
    email: str
    password: str

class LoginStep1Response(BaseModel):
    requires_otp: bool = False
    requires_verification: bool = False
    access_token: Optional[str] = None
    token_type: str = "bearer"

@router.post("/login/step1", response_model=LoginStep1Response)
async def login_step1(
    request: LoginStep1Request,
    background_tasks: BackgroundTasks,
    req: Request,
    db: AsyncSession = Depends(get_db)
):
    """
    الخطوة الأولى من تسجيل الدخول - التحقق من البريد وكلمة المرور
    """
    # البحث عن المستخدم
    result = await db.execute(select(User).where(User.email == request.email.lower()))
    user = result.scalar_one_or_none()
    
    if not user or not verify_password(request.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="البريد الإلكتروني أو كلمة المرور غير صحيح"
        )
    
    # التحقق من تفعيل الحساب
    if not user.is_verified:
        return LoginStep1Response(requires_verification=True)
    
    # إنشاء التوكن مباشرة (بدون OTP)
    access_token = create_access_token(data={"sub": str(user.id)})
    
    # إرسال إشعار تسجيل الدخول
    client_info = get_client_info(req)
    background_tasks.add_task(
        email_service.send_login_notification,
        user.email,
        user.full_name or user.email,
        client_info.get("ip", "Unknown"),
        client_info.get("user_agent", "Unknown")
    )
    
    return LoginStep1Response(
        requires_otp=False,
        access_token=access_token,
        token_type="bearer"
    )

# Get current user info
@router.get("/me")
async def get_current_user_info(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get current authenticated user information"""
    return {
        "id": current_user.id,
        "email": current_user.email,
        "full_name": current_user.full_name,
        "avatar_url": current_user.avatar_url,
        "is_admin": current_user.is_admin,
        "is_verified": current_user.is_verified,
        "phone_number": current_user.phone_number,
        "phone_verified": current_user.phone_verified,
        "two_factor_enabled": current_user.two_factor_enabled,
        "status": current_user.status,
        "created_at": current_user.created_at.isoformat() if current_user.created_at else None
    }
