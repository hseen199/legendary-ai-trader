from pydantic import BaseModel, EmailStr
from typing import Optional
from datetime import datetime


# ============ Auth Schemas ============

class UserRegister(BaseModel):
    email: EmailStr
    password: str
    full_name: Optional[str] = None
    phone: Optional[str] = None


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class TokenData(BaseModel):
    user_id: Optional[int] = None


# ============ User Schemas ============

class UserBase(BaseModel):
    email: EmailStr
    full_name: Optional[str] = None
    phone: Optional[str] = None


class UserResponse(UserBase):
    id: int
    status: str
    is_admin: bool
    is_verified: bool
    two_factor_enabled: bool
    created_at: datetime
    last_login: Optional[datetime] = None
    
    class Config:
        from_attributes = True


class UserUpdate(BaseModel):
    full_name: Optional[str] = None
    phone: Optional[str] = None


class PasswordChange(BaseModel):
    current_password: str
    new_password: str


# ============ Balance Schemas ============

class BalanceResponse(BaseModel):
    units: float
    current_value_usd: float
    nav: float
    profit_loss: float
    profit_loss_percent: float
    last_deposit_at: Optional[datetime] = None
    can_withdraw: bool  # بناءً على فترة القفل
    # حقول إضافية للعرض في صفحة المحفظة
    balance_usd: Optional[float] = None  # الرصيد الحالي (نفس current_value_usd)
    total_deposited: Optional[float] = None  # إجمالي الإيداعات
    total_withdrawn: Optional[float] = None  # إجمالي السحوبات
    
    class Config:
        from_attributes = True


# ============ Trusted Address Schemas ============

class TrustedAddressCreate(BaseModel):
    address: str
    network: str
    label: Optional[str] = None


class TrustedAddressResponse(BaseModel):
    id: int
    address: str
    network: str
    label: Optional[str] = None
    is_active: bool
    activated_at: Optional[datetime] = None
    created_at: datetime
    
    class Config:
        from_attributes = True
