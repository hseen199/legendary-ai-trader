"""
Investor Models
نماذج المستثمرين والوحدات والإيداعات
مدمج من نسخة المستخدم (crowdfund)
"""

from datetime import datetime
from decimal import Decimal
from typing import Optional
from sqlalchemy import (
    Column, Integer, String, Float, DateTime, 
    Boolean, Text, ForeignKey, Numeric, Enum as SQLEnum
)
from sqlalchemy.orm import relationship
import enum

from app.core.database import Base


class DepositStatus(enum.Enum):
    """حالات الإيداع"""
    PENDING = "pending"           # في انتظار التأكيد
    CONFIRMED = "confirmed"       # تم التأكيد من الشبكة
    CREDITED = "credited"         # تم إضافة الوحدات
    FAILED = "failed"             # فشل


class WithdrawalStatus(enum.Enum):
    """حالات السحب"""
    PENDING = "pending"           # في انتظار المراجعة
    EMAIL_SENT = "email_sent"     # تم إرسال بريد التأكيد
    CONFIRMED = "confirmed"       # تم تأكيد البريد
    APPROVED = "approved"         # تمت الموافقة من الأدمن
    PROCESSING = "processing"     # جاري التنفيذ
    COMPLETED = "completed"       # تم بنجاح
    REJECTED = "rejected"         # مرفوض


class Investor(Base):
    """
    نموذج المستثمر
    يحتوي على بيانات الحساب الفرعي في Binance والوحدات
    """
    __tablename__ = 'investors'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), unique=True, nullable=False)
    
    # بيانات Binance Sub-Account
    binance_sub_email = Column(String(255), unique=True, index=True)
    binance_sub_id = Column(String(100))
    deposit_address_usdc = Column(String(255))
    deposit_network = Column(String(50), default='TRX')
    
    # إحصائيات المستثمر
    total_units = Column(Numeric(20, 8), default=0)
    total_deposited = Column(Numeric(20, 8), default=0)
    total_withdrawn = Column(Numeric(20, 8), default=0)
    fees_paid = Column(Numeric(20, 8), default=0)
    
    # الحالة
    status = Column(String(50), default='active')
    kyc_verified = Column(Boolean, default=False)
    
    # التواريخ
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # العلاقات
    user = relationship("User", back_populates="investor")
    deposits = relationship("Deposit", back_populates="investor", lazy="dynamic")
    withdrawals = relationship("Withdrawal", back_populates="investor", lazy="dynamic")
    unit_records = relationship("UnitRecord", back_populates="investor", lazy="dynamic")
    
    def __repr__(self):
        return f"<Investor(id={self.id}, user_id={self.user_id}, units={self.total_units})>"
        
    @property
    def current_value(self) -> Decimal:
        """حساب القيمة الحالية (يحتاج NAV)"""
        # يتم حسابها في الخدمة
        return Decimal('0')


class Deposit(Base):
    """
    نموذج الإيداع
    يسجل كل إيداع من المستثمر
    """
    __tablename__ = 'deposits'
    
    id = Column(Integer, primary_key=True)
    investor_id = Column(Integer, ForeignKey('investors.id'), nullable=False, index=True)
    
    # بيانات المعاملة
    tx_hash = Column(String(255), unique=True, index=True)
    amount = Column(Numeric(20, 8), nullable=False)
    coin = Column(String(20), default='USDC')
    network = Column(String(50))
    
    # بيانات الوحدات
    units_credited = Column(Numeric(20, 8))
    nav_at_deposit = Column(Numeric(20, 8))
    
    # الحالة والقفل
    status = Column(SQLEnum(DepositStatus), default=DepositStatus.PENDING, index=True)
    lock_until = Column(DateTime)
    
    # التواريخ
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    confirmed_at = Column(DateTime)
    
    # العلاقات
    investor = relationship("Investor", back_populates="deposits")
    
    def __repr__(self):
        return f"<Deposit(id={self.id}, amount={self.amount}, status={self.status})>"
        
    @property
    def is_locked(self) -> bool:
        """هل الإيداع مقفل؟"""
        if not self.lock_until:
            return False
        return datetime.utcnow() < self.lock_until


class Withdrawal(Base):
    """
    نموذج السحب
    يسجل طلبات السحب من المستثمرين
    """
    __tablename__ = 'withdrawals'
    
    id = Column(Integer, primary_key=True)
    investor_id = Column(Integer, ForeignKey('investors.id'), nullable=False, index=True)
    
    # بيانات السحب
    units_redeemed = Column(Numeric(20, 8), nullable=False)
    nav_at_withdrawal = Column(Numeric(20, 8))
    gross_value = Column(Numeric(20, 8))
    cost_basis = Column(Numeric(20, 8))
    profit = Column(Numeric(20, 8))
    performance_fee = Column(Numeric(20, 8))
    net_value = Column(Numeric(20, 8))
    
    # بيانات التحويل
    to_address = Column(String(255))
    to_network = Column(String(50), default='TRX')
    tx_hash = Column(String(255))
    
    # الحالة
    status = Column(SQLEnum(WithdrawalStatus), default=WithdrawalStatus.PENDING, index=True)
    
    # تأكيد البريد
    email_token = Column(String(255), index=True)
    email_token_expires = Column(DateTime)
    email_confirmed_at = Column(DateTime)
    
    # موافقة الأدمن
    admin_approved_by = Column(Integer, ForeignKey('users.id'))
    admin_approved_at = Column(DateTime)
    rejection_reason = Column(Text)
    
    # التواريخ
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    completed_at = Column(DateTime)
    
    # العلاقات
    investor = relationship("Investor", back_populates="withdrawals")
    
    def __repr__(self):
        return f"<Withdrawal(id={self.id}, units={self.units_redeemed}, status={self.status})>"


class UnitRecord(Base):
    """
    نموذج سجل الوحدات
    يتتبع كل دفعة من الوحدات لحساب تكلفة الأساس بدقة
    """
    __tablename__ = 'unit_records'
    
    id = Column(Integer, primary_key=True)
    investor_id = Column(Integer, ForeignKey('investors.id'), nullable=False, index=True)
    deposit_id = Column(Integer, ForeignKey('deposits.id'))
    
    # بيانات الوحدات
    units = Column(Numeric(20, 8), nullable=False)
    cost_basis = Column(Numeric(20, 8), nullable=False)
    nav_at_purchase = Column(Numeric(20, 8))
    
    # القفل والحالة
    lock_until = Column(DateTime)
    is_active = Column(Boolean, default=True, index=True)
    
    # التواريخ
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    
    # العلاقات
    investor = relationship("Investor", back_populates="unit_records")
    
    def __repr__(self):
        return f"<UnitRecord(id={self.id}, units={self.units}, active={self.is_active})>"
        
    @property
    def is_locked(self) -> bool:
        """هل الوحدات مقفلة؟"""
        if not self.lock_until:
            return False
        return datetime.utcnow() < self.lock_until


class Trade(Base):
    """
    نموذج الصفقات
    يسجل صفقات البوت
    """
    __tablename__ = 'trades'
    
    id = Column(Integer, primary_key=True)
    
    # بيانات الصفقة
    symbol = Column(String(20), nullable=False, index=True)
    side = Column(String(10), nullable=False)  # BUY / SELL
    entry_price = Column(Numeric(20, 8))
    exit_price = Column(Numeric(20, 8))
    quantity = Column(Numeric(20, 8))
    
    # الربح/الخسارة
    pnl = Column(Numeric(20, 8))
    pnl_percent = Column(Numeric(10, 4))
    
    # الحالة
    status = Column(String(20), default='open', index=True)  # open, closed, cancelled
    
    # بيانات الذكاء الاصطناعي
    agent_decision = Column(Text)
    risk_score = Column(Float)
    confidence = Column(Float)
    
    # التواريخ
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    closed_at = Column(DateTime)
    
    def __repr__(self):
        return f"<Trade(id={self.id}, symbol={self.symbol}, pnl={self.pnl})>"


class AgentState(Base):
    """
    نموذج حالة الوكلاء
    يتتبع حالة كل وكيل في البوت
    """
    __tablename__ = 'agent_states'
    
    id = Column(Integer, primary_key=True)
    
    # بيانات الوكيل
    agent_name = Column(String(100), nullable=False, index=True)
    agent_type = Column(String(50))
    
    # الحالة والإشارات
    status = Column(String(50))  # active, paused, error
    last_signal = Column(String(50))  # buy, sell, hold
    signal_strength = Column(Float)
    
    # التحليل
    analysis_summary = Column(Text)
    
    # التواريخ
    last_update = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<AgentState(name={self.agent_name}, signal={self.last_signal})>"
