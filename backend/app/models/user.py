from sqlalchemy import Column, Integer, String, Boolean, DateTime, Float, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.core.database import Base
import enum


class UserStatus(str, enum.Enum):
    PENDING = "pending"
    ACTIVE = "active"
    LOCKED = "locked"
    SUSPENDED = "suspended"


class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    
    # Binance Sub-Account
    sub_account_email = Column(String(255), unique=True, nullable=True)
    
    # User Info
    full_name = Column(String(255), nullable=True)
    phone = Column(String(50), nullable=True)
    
    # Google OAuth
    google_id = Column(String(255), unique=True, nullable=True)
    avatar_url = Column(String(500), nullable=True)
    
    # Status and Roles
    status = Column(String(50), default=UserStatus.ACTIVE)
    is_admin = Column(Boolean, default=False)
    is_verified = Column(Boolean, default=False)
    is_active = Column(Boolean, default=True)
    
    # Security
    two_factor_enabled = Column(Boolean, default=False)
    two_factor_secret = Column(String(255), nullable=True)
    
    # Balance fields (for quick access - also tracked in Balance table)
    balance = Column(Float, default=0.0)  # الرصيد الحالي بالدولار
    units = Column(Float, default=0.0)  # الوحدات
    total_deposited = Column(Float, default=0.0)  # إجمالي الإيداعات
    total_withdrawn = Column(Float, default=0.0)  # إجمالي السحوبات
    
    # Referral
    referral_code = Column(String(20), unique=True, nullable=True, index=True)
    referred_by = Column(Integer, ForeignKey("users.id"), nullable=True)
    
    # VIP Level
    vip_level = Column(String(20), default="bronze")  # bronze, silver, gold, platinum
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    last_login = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    balance_record = relationship("Balance", back_populates="user", uselist=False)
    transactions = relationship("Transaction", back_populates="user")
    withdrawal_requests = relationship("WithdrawalRequest", back_populates="user", foreign_keys="WithdrawalRequest.user_id")
    trusted_addresses = relationship("TrustedAddress", back_populates="user")
    investor = relationship("Investor", back_populates="user", uselist=False)
    referrals = relationship("User", backref="referrer", remote_side=[id])


class Balance(Base):
    __tablename__ = "balances"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), unique=True, nullable=False)
    
    # Units (الوحدات)
    units = Column(Float, default=0.0)
    
    # Balance in USD
    balance_usd = Column(Float, default=0.0)
    
    # Totals
    total_deposited = Column(Float, default=0.0)
    total_withdrawn = Column(Float, default=0.0)
    
    # Lock period tracking
    last_deposit_at = Column(DateTime(timezone=True), nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    user = relationship("User", back_populates="balance_record")


class TrustedAddress(Base):
    __tablename__ = "trusted_addresses"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    address = Column(String(255), nullable=False)
    network = Column(String(50), nullable=False)  # Solana, BEP20, ERC20
    label = Column(String(100), nullable=True)
    
    # 24-hour delay for new addresses
    is_active = Column(Boolean, default=False)
    activated_at = Column(DateTime(timezone=True), nullable=True)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    user = relationship("User", back_populates="trusted_addresses")
