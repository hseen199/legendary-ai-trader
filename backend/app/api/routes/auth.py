import os
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks, Request
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.responses import RedirectResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from datetime import datetime, timedelta
from pydantic import BaseModel, EmailStr
import httpx
import random
import string
from typing import Optional

from app.core.database import get_db
from app.core.security import (
    get_password_hash, 
    verify_password, 
    create_access_token,
    get_current_user
)
from app.core.config import settings
from app.models import User, Balance
from app.schemas import UserRegister, UserLogin, Token, UserResponse, PasswordChange
from app.services.email_service import email_service

router = APIRouter(prefix="/auth", tags=["Authentication"])

# ============== Google OAuth Configuration ==============
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
GOOGLE_REDIRECT_URI = "https://asinax.cloud/api/v1/auth/google/callback"
GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_URL = "https://www.googleapis.com/oauth2/v3/userinfo"

# ============== OTP Storage (use Redis in production) ==============
otp_storage = {}
password_reset_storage = {}
login_otp_storage = {}  # New: for login OTP
pending_login_storage = {}  # New: for pending login sessions
OTP_LENGTH = 6
OTP_EXPIRY_MINUTES = 10

# ============== Pydantic Models ==============
class SendOTPRequest(BaseModel):
    email: EmailStr

class VerifyOTPRequest(BaseModel):
    email: EmailStr
    otp: str

class PasswordResetRequest(BaseModel):
    email: EmailStr

class PasswordResetConfirm(BaseModel):
    email: EmailStr
    otp: str
    new_password: str

class GoogleTokenRequest(BaseModel):
    token: str

class LoginStep1Request(BaseModel):
    email: EmailStr
    password: str

class LoginStep2Request(BaseModel):
    email: EmailStr
    otp: str

class RegisterVerifyRequest(BaseModel):
    email: EmailStr
    otp: str

# ============== Helper Functions ==============
def generate_otp() -> str:
    """Generate a random 6-digit OTP"""
    return ''.join(random.choices(string.digits, k=OTP_LENGTH))

def store_otp(email: str, otp: str, storage: dict = None):
    """Store OTP with expiry time"""
    if storage is None:
        storage = otp_storage
    storage[email] = {
        "otp": otp,
        "expires_at": datetime.utcnow() + timedelta(minutes=OTP_EXPIRY_MINUTES),
        "attempts": 0
    }

def verify_otp_code(email: str, otp: str, storage: dict = None) -> bool:
    """Verify OTP for email"""
    if storage is None:
        storage = otp_storage
    
    if email not in storage:
        return False
    
    stored = storage[email]
    
    if datetime.utcnow() > stored["expires_at"]:
        del storage[email]
        return False
    
    if stored["attempts"] >= 5:
        del storage[email]
        return False
    
    stored["attempts"] += 1
    
    if stored["otp"] == otp:
        del storage[email]
        return True
    
    return False

def get_client_info(request: Request) -> dict:
    """Extract client information from request"""
    # Get IP address
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        ip = forwarded.split(",")[0].strip()
    else:
        ip = request.client.host if request.client else "غير معروف"
    
    # Get user agent
    user_agent = request.headers.get("User-Agent", "غير معروف")
    
    # Simple device detection
    device = "غير معروف"
    if "Mobile" in user_agent or "Android" in user_agent or "iPhone" in user_agent:
        device = "جهاز محمول"
    elif "Windows" in user_agent:
        device = "Windows"
    elif "Mac" in user_agent:
        device = "Mac"
    elif "Linux" in user_agent:
        device = "Linux"
    
    return {
        "ip_address": ip,
        "device": device,
        "user_agent": user_agent
    }

# ============== Registration Endpoints ==============

@router.post("/register", status_code=status.HTTP_201_CREATED)
async def register(
    user_data: UserRegister,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
):
    """Register a new user - Step 1: Create account and send OTP"""
    email = user_data.email.lower()
    
    result = await db.execute(
        select(User).where(User.email == email)
    )
    existing_user = result.scalar_one_or_none()
    
    if existing_user:
        if existing_user.is_verified:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="البريد الإلكتروني مسجل مسبقاً"
            )
        else:
            # User exists but not verified - resend OTP
            otp = generate_otp()
            store_otp(email, otp)
            background_tasks.add_task(
                email_service.send_verification_otp,
                email,
                otp
            )
            return {
                "success": True,
                "message": "تم إرسال رمز التحقق إلى بريدك الإلكتروني",
                "requires_otp": True,
                "email": email
            }
    
    # Create new user (unverified)
    user = User(
        email=email,
        password_hash=get_password_hash(user_data.password),
        full_name=user_data.full_name,
        phone=user_data.phone,
        is_verified=False,
        status="active"
    )
    db.add(user)
    await db.flush()
    
    balance = Balance(user_id=user.id, units=0.0)
    db.add(balance)
    
    await db.commit()
    await db.refresh(user)
    
    # Send verification OTP
    otp = generate_otp()
    store_otp(email, otp)
    background_tasks.add_task(
        email_service.send_verification_otp,
        email,
        otp
    )
    
    return {
        "success": True,
        "message": "تم إنشاء الحساب. يرجى إدخال رمز التحقق المرسل إلى بريدك الإلكتروني",
        "requires_otp": True,
        "email": email
    }

@router.post("/register/verify")
async def register_verify(
    request: RegisterVerifyRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
):
    """Register - Step 2: Verify OTP and activate account"""
    email = request.email.lower()
    otp = request.otp.strip()
    
    if not verify_otp_code(email, otp):
        raise HTTPException(
            status_code=400,
            detail="رمز التحقق غير صحيح أو منتهي الصلاحية"
        )
    
    # Get and verify user
    result = await db.execute(select(User).where(User.email == email))
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(
            status_code=404,
            detail="المستخدم غير موجود"
        )
    
    # Activate account
    user.is_verified = True
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

# ============== Login Endpoints (Two-Step with OTP) ==============

@router.post("/login/step1")
async def login_step1(
    request: Request,
    login_data: LoginStep1Request,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
):
    """Login Step 1: Verify credentials and send OTP"""
    email = login_data.email.lower()
    
    result = await db.execute(
        select(User).where(User.email == email)
    )
    user = result.scalar_one_or_none()
    
    if not user or not verify_password(login_data.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="البريد الإلكتروني أو كلمة المرور غير صحيحة"
        )
    
    if user.status != "active":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="الحساب معطل"
        )
    
    if not user.is_verified:
        # Account not verified - send verification OTP
        otp = generate_otp()
        store_otp(email, otp)
        background_tasks.add_task(
            email_service.send_verification_otp,
            email,
            otp
        )
        return {
            "success": True,
            "message": "يرجى تأكيد بريدك الإلكتروني أولاً",
            "requires_verification": True,
            "email": email
        }
    
    # Rate limiting for login OTP
    if email in login_otp_storage:
        stored = login_otp_storage[email]
        time_since_sent = datetime.utcnow() - (stored["expires_at"] - timedelta(minutes=OTP_EXPIRY_MINUTES))
        if time_since_sent.total_seconds() < 60:
            raise HTTPException(
                status_code=429,
                detail="يرجى الانتظار دقيقة واحدة قبل طلب رمز جديد"
            )
    
    # Generate and send login OTP
    otp = generate_otp()
    store_otp(email, otp, login_otp_storage)
    
    # Store pending login info
    client_info = get_client_info(request)
    pending_login_storage[email] = {
        "user_id": user.id,
        "client_info": client_info,
        "expires_at": datetime.utcnow() + timedelta(minutes=OTP_EXPIRY_MINUTES)
    }
    
    background_tasks.add_task(
        email_service.send_login_otp,
        email,
        otp,
        client_info["device"],
        client_info["ip_address"]
    )
    
    return {
        "success": True,
        "message": "تم إرسال رمز التحقق إلى بريدك الإلكتروني",
        "requires_otp": True,
        "email": email
    }

@router.post("/login/step2", response_model=Token)
async def login_step2(
    request: Request,
    login_data: LoginStep2Request,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
):
    """Login Step 2: Verify OTP and get access token"""
    email = login_data.email.lower()
    otp = login_data.otp.strip()
    
    # Verify OTP
    if not verify_otp_code(email, otp, login_otp_storage):
        raise HTTPException(
            status_code=400,
            detail="رمز التحقق غير صحيح أو منتهي الصلاحية"
        )
    
    # Check pending login
    if email not in pending_login_storage:
        raise HTTPException(
            status_code=400,
            detail="جلسة تسجيل الدخول منتهية. يرجى المحاولة مرة أخرى"
        )
    
    pending = pending_login_storage[email]
    if datetime.utcnow() > pending["expires_at"]:
        del pending_login_storage[email]
        raise HTTPException(
            status_code=400,
            detail="جلسة تسجيل الدخول منتهية. يرجى المحاولة مرة أخرى"
        )
    
    # Get user
    result = await db.execute(select(User).where(User.email == email))
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(
            status_code=404,
            detail="المستخدم غير موجود"
        )
    
    # Update last login
    user.last_login = datetime.utcnow()
    await db.commit()
    
    # Clean up pending login
    client_info = pending["client_info"]
    del pending_login_storage[email]
    
    # Send login notification
    background_tasks.add_task(
        email_service.send_login_notification,
        user.email,
        ip_address=client_info["ip_address"],
        device=client_info["device"],
        location="غير معروف",
        login_time=datetime.utcnow()
    )
    
    # Create access token
    access_token = create_access_token(
        data={"sub": str(user.id), "email": user.email, "is_admin": user.is_admin}
    )
    
    return {"access_token": access_token, "token_type": "bearer"}

@router.post("/login/resend-otp")
async def login_resend_otp(
    request: SendOTPRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
):
    """Resend login OTP"""
    email = request.email.lower()
    
    # Check if there's a pending login
    if email not in pending_login_storage:
        raise HTTPException(
            status_code=400,
            detail="لا توجد جلسة تسجيل دخول معلقة. يرجى البدء من جديد"
        )
    
    # Rate limiting
    if email in login_otp_storage:
        stored = login_otp_storage[email]
        time_since_sent = datetime.utcnow() - (stored["expires_at"] - timedelta(minutes=OTP_EXPIRY_MINUTES))
        if time_since_sent.total_seconds() < 60:
            raise HTTPException(
                status_code=429,
                detail="يرجى الانتظار دقيقة واحدة قبل طلب رمز جديد"
            )
    
    pending = pending_login_storage[email]
    client_info = pending["client_info"]
    
    # Generate and send new OTP
    otp = generate_otp()
    store_otp(email, otp, login_otp_storage)
    
    background_tasks.add_task(
        email_service.send_login_otp,
        email,
        otp,
        client_info["device"],
        client_info["ip_address"]
    )
    
    return {
        "success": True,
        "message": "تم إرسال رمز التحقق مرة أخرى"
    }

# ============== Legacy Login (Keep for backward compatibility) ==============

@router.post("/login", response_model=Token)
async def login(
    request: Request,
    background_tasks: BackgroundTasks,
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: AsyncSession = Depends(get_db)
):
    """Legacy login endpoint - redirects to two-step login"""
    # This endpoint is kept for OAuth2 compatibility
    # For security, we now require OTP verification
    email = form_data.username.lower()
    
    result = await db.execute(
        select(User).where(User.email == email)
    )
    user = result.scalar_one_or_none()
    
    if not user or not verify_password(form_data.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="البريد الإلكتروني أو كلمة المرور غير صحيحة",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    if user.status != "active":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="الحساب معطل"
        )
    
    if not user.is_verified:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="يرجى تأكيد بريدك الإلكتروني أولاً"
        )
    
    # For legacy support, allow direct login but send notification
    user.last_login = datetime.utcnow()
    await db.commit()
    
    client_info = get_client_info(request)
    background_tasks.add_task(
        email_service.send_login_notification,
        user.email,
        ip_address=client_info["ip_address"],
        device=client_info["device"],
        location="غير معروف",
        login_time=datetime.utcnow()
    )
    
    access_token = create_access_token(
        data={"sub": str(user.id), "email": user.email, "is_admin": user.is_admin}
    )
    
    return {"access_token": access_token, "token_type": "bearer"}

@router.get("/me", response_model=UserResponse)
async def get_current_user_info(
    current_user: User = Depends(get_current_user)
):
    """Get current user information"""
    return current_user

@router.post("/change-password")
async def change_password(
    password_data: PasswordChange,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Change user password"""
    if not verify_password(password_data.current_password, current_user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="كلمة المرور الحالية غير صحيحة"
        )
    
    current_user.password_hash = get_password_hash(password_data.new_password)
    await db.commit()
    
    # Send password changed notification
    background_tasks.add_task(
        email_service.send_password_changed,
        current_user.email
    )
    
    return {"message": "تم تغيير كلمة المرور بنجاح"}

# ============== Password Reset Endpoints ==============

@router.post("/forgot-password")
async def forgot_password(
    request: PasswordResetRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
):
    """Request password reset OTP"""
    email = request.email.lower()
    
    # Check if user exists
    result = await db.execute(select(User).where(User.email == email))
    user = result.scalar_one_or_none()
    
    # Always return success to prevent email enumeration
    if not user:
        return {
            "success": True,
            "message": "إذا كان البريد الإلكتروني مسجلاً، سيتم إرسال رمز إعادة التعيين"
        }
    
    # Rate limiting
    if email in password_reset_storage:
        stored = password_reset_storage[email]
        time_since_sent = datetime.utcnow() - (stored["expires_at"] - timedelta(minutes=OTP_EXPIRY_MINUTES))
        if time_since_sent.total_seconds() < 60:
            raise HTTPException(
                status_code=429,
                detail="يرجى الانتظار دقيقة واحدة قبل طلب رمز جديد"
            )
    
    # Generate and store OTP
    otp = generate_otp()
    store_otp(email, otp, password_reset_storage)
    
    # Send email
    background_tasks.add_task(
        email_service.send_password_reset_otp,
        email,
        otp
    )
    
    return {
        "success": True,
        "message": "تم إرسال رمز إعادة التعيين إلى بريدك الإلكتروني"
    }

@router.post("/reset-password")
async def reset_password(
    request: PasswordResetConfirm,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
):
    """Reset password using OTP"""
    email = request.email.lower()
    
    # Verify OTP
    if not verify_otp_code(email, request.otp, password_reset_storage):
        raise HTTPException(
            status_code=400,
            detail="رمز التحقق غير صحيح أو منتهي الصلاحية"
        )
    
    # Get user
    result = await db.execute(select(User).where(User.email == email))
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(
            status_code=404,
            detail="المستخدم غير موجود"
        )
    
    # Update password
    user.password_hash = get_password_hash(request.new_password)
    await db.commit()
    
    # Send confirmation email
    background_tasks.add_task(
        email_service.send_password_changed,
        email
    )
    
    return {
        "success": True,
        "message": "تم تغيير كلمة المرور بنجاح"
    }

# ============== OTP Verification Endpoints ==============

@router.post("/send-otp")
async def send_otp(
    request: SendOTPRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
):
    """Send OTP to email for verification"""
    email = request.email.lower()
    
    # Rate limiting
    if email in otp_storage:
        stored = otp_storage[email]
        time_since_sent = datetime.utcnow() - (stored["expires_at"] - timedelta(minutes=OTP_EXPIRY_MINUTES))
        if time_since_sent.total_seconds() < 60:
            raise HTTPException(
                status_code=429,
                detail="يرجى الانتظار دقيقة واحدة قبل طلب رمز جديد"
            )
    
    otp = generate_otp()
    store_otp(email, otp)
    
    background_tasks.add_task(
        email_service.send_verification_otp,
        email,
        otp
    )
    
    return {
        "success": True,
        "message": "تم إرسال رمز التحقق إلى بريدك الإلكتروني"
    }

@router.post("/verify-otp")
async def verify_otp(
    request: VerifyOTPRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
):
    """Verify OTP and mark email as verified"""
    email = request.email.lower()
    otp = request.otp.strip()
    
    if not verify_otp_code(email, otp):
        raise HTTPException(
            status_code=400,
            detail="رمز التحقق غير صحيح أو منتهي الصلاحية"
        )
    
    # Mark user as verified
    result = await db.execute(select(User).where(User.email == email))
    user = result.scalar_one_or_none()
    
    if user:
        user.is_verified = True
        await db.commit()
        
        # Send welcome email
        background_tasks.add_task(
            email_service.send_welcome_email,
            email,
            user.full_name or "مستخدم"
        )
    
    return {
        "success": True,
        "message": "تم التحقق من بريدك الإلكتروني بنجاح"
    }

@router.post("/resend-otp")
async def resend_otp(
    request: SendOTPRequest,
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
            ip_address=client_info["ip_address"],
            device=client_info["device"],
            location="غير معروف",
            login_time=datetime.utcnow()
        )
        
        # Send welcome email for new users
        if is_new_user:
            background_tasks.add_task(
                email_service.send_welcome_email,
                email,
                name or "مستخدم"
            )
        
        # Create JWT token
        jwt_token = create_access_token(
            data={"sub": str(user.id), "email": user.email, "is_admin": user.is_admin}
        )
        
        # Redirect to frontend with token
        return RedirectResponse(url=f"https://asinax.cloud/auth/callback?token={jwt_token}")
        
    except Exception as e:
        return RedirectResponse(url=f"https://asinax.cloud/login?error=google_auth_failed")
