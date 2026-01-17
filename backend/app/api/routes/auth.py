from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
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
from app.services.email_service import EmailService

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
OTP_LENGTH = 6
OTP_EXPIRY_MINUTES = 10


# ============== Pydantic Models ==============
class SendOTPRequest(BaseModel):
    email: EmailStr


class VerifyOTPRequest(BaseModel):
    email: EmailStr
    otp: str


class GoogleTokenRequest(BaseModel):
    token: str


# ============== Helper Functions ==============
def generate_otp() -> str:
    """Generate a random 6-digit OTP"""
    return ''.join(random.choices(string.digits, k=OTP_LENGTH))


def store_otp(email: str, otp: str):
    """Store OTP with expiry time"""
    otp_storage[email] = {
        "otp": otp,
        "expires_at": datetime.utcnow() + timedelta(minutes=OTP_EXPIRY_MINUTES),
        "attempts": 0
    }


def verify_otp_code(email: str, otp: str) -> bool:
    """Verify OTP for email"""
    if email not in otp_storage:
        return False
    
    stored = otp_storage[email]
    
    if datetime.utcnow() > stored["expires_at"]:
        del otp_storage[email]
        return False
    
    if stored["attempts"] >= 5:
        del otp_storage[email]
        return False
    
    stored["attempts"] += 1
    
    if stored["otp"] == otp:
        del otp_storage[email]
        return True
    
    return False


async def send_otp_email(email: str, otp: str):
    """Send OTP via email"""
    email_service = EmailService()
    
    html_content = f"""
    <!DOCTYPE html>
    <html dir="rtl" lang="ar">
    <head>
        <meta charset="UTF-8">
        <style>
            body {{ font-family: 'Segoe UI', sans-serif; background-color: #0a0a0a; color: #fff; padding: 20px; direction: rtl; }}
            .container {{ max-width: 600px; margin: 0 auto; background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); border-radius: 16px; padding: 40px; border: 1px solid #3b82f6; }}
            .logo {{ text-align: center; margin-bottom: 30px; }}
            .logo h1 {{ color: #3b82f6; font-size: 32px; margin: 0; }}
            .otp-box {{ background: rgba(59, 130, 246, 0.1); border: 2px solid #3b82f6; border-radius: 12px; padding: 30px; text-align: center; margin: 30px 0; }}
            .otp-code {{ font-size: 48px; font-weight: bold; color: #3b82f6; letter-spacing: 10px; margin: 0; }}
            .message {{ color: #9ca3af; font-size: 16px; line-height: 1.8; text-align: center; }}
            .warning {{ background: rgba(245, 158, 11, 0.1); border: 1px solid #f59e0b; border-radius: 8px; padding: 15px; margin-top: 20px; color: #f59e0b; font-size: 14px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="logo"><h1>🚀 ASINAX</h1></div>
            <p class="message">مرحباً! 👋<br>رمز التحقق الخاص بك:</p>
            <div class="otp-box"><p class="otp-code">{otp}</p></div>
            <p class="message">صالح لمدة <strong>10 دقائق</strong></p>
            <div class="warning">⚠️ لا تشارك هذا الرمز مع أي شخص</div>
        </div>
    </body>
    </html>
    """
    
    await email_service.send_email(
        to_email=email,
        subject="ASINAX - رمز التحقق",
        html_content=html_content
    )


# ============== Original Auth Endpoints ==============
@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(
    user_data: UserRegister,
    db: AsyncSession = Depends(get_db)
):
    """Register a new user"""
    result = await db.execute(
        select(User).where(User.email == user_data.email)
    )
    if result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    user = User(
        email=user_data.email,
        password_hash=get_password_hash(user_data.password),
        full_name=user_data.full_name,
        phone=user_data.phone,
        is_verified=False  # Require email verification
    )
    db.add(user)
    await db.flush()
    
    balance = Balance(user_id=user.id, units=0.0)
    db.add(balance)
    
    await db.commit()
    await db.refresh(user)
    
    return user


@router.post("/login", response_model=Token)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: AsyncSession = Depends(get_db)
):
    """Login and get access token"""
    result = await db.execute(
        select(User).where(User.email == form_data.username)
    )
    user = result.scalar_one_or_none()
    
    if not user or not verify_password(form_data.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    if user.status != "active":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is disabled"
        )
    
    user.last_login = datetime.utcnow()
    await db.commit()
    
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
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Change user password"""
    if not verify_password(password_data.current_password, current_user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Current password is incorrect"
        )
    
    current_user.password_hash = get_password_hash(password_data.new_password)
    await db.commit()
    
    return {"message": "Password changed successfully"}


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
async def google_callback(code: str, db: AsyncSession = Depends(get_db)):
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
        
        if user:
            # Update Google info
            user.google_id = google_id
            user.avatar_url = picture
            user.is_verified = True
            user.last_login = datetime.utcnow()
            await db.commit()
        else:
            # Create new user
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
        
        # Create JWT token
        jwt_token = create_access_token(
            data={"sub": str(user.id), "email": user.email, "is_admin": user.is_admin}
        )
        
        # Redirect to frontend with token
        return RedirectResponse(url=f"https://asinax.cloud/auth/callback?token={jwt_token}")
        
    except Exception as e:
        return RedirectResponse(url=f"https://asinax.cloud/login?error=google_auth_failed")


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
    
    background_tasks.add_task(send_otp_email, email, otp)
    
    return {
        "success": True,
        "message": "تم إرسال رمز التحقق إلى بريدك الإلكتروني"
    }


@router.post("/verify-otp")
async def verify_otp(
    request: VerifyOTPRequest,
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
