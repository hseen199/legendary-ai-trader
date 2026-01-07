from sqlalchemy import Column, Integer, String, Boolean, DateTime, Float, ForeignKey, Enum
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
    
    # Status and Roles
    status = Column(String(50), default=UserStatus.ACTIVE)
    is_admin = Column(Boolean, default=False)
    is_verified = Column(Boolean, default=False)
    
    # Security
    two_factor_enabled = Column(Boolean, default=False)
    two_factor_secret = Column(String(255), nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    last_login = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    balance = relationship("Balance", back_populates="user", uselist=False)
    transactions = relationship("Transaction", back_populates="user")
    withdrawal_requests = relationship("WithdrawalRequest", back_populates="user")
    trusted_addresses = relationship("TrustedAddress", back_populates="user")


class Balance(Base):
    __tablename__ = "balances"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), unique=True, nullable=False)
    
    # Units (الوحدات)
    units = Column(Float, default=0.0)
    
    # Lock period tracking
    last_deposit_at = Column(DateTime(timezone=True), nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    user = relationship("User", back_populates="balance")


class TrustedAddress(Base):
    __tablename__ = "trusted_addresses"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    address = Column(String(255), nullable=False)
    network = Column(String(50), nullable=False)  # TRC20, ERC20, BEP20
    label = Column(String(100), nullable=True)
    
    # 24-hour delay for new addresses
    is_active = Column(Boolean, default=False)
    activated_at = Column(DateTime(timezone=True), nullable=True)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    user = relationship("User", back_populates="trusted_addresses")
