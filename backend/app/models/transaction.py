from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Text, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.core.database import Base
import enum


class TransactionType(str, enum.Enum):
    DEPOSIT = "deposit"
    WITHDRAWAL = "withdrawal"
    PERFORMANCE_FEE = "performance_fee"
    INTERNAL_TRANSFER = "internal_transfer"


class TransactionStatus(str, enum.Enum):
    PENDING = "pending"
    CONFIRMING = "confirming"
    CONFIRMED = "confirmed"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"
    PARTIAL = "partial"
    REFUNDED = "refunded"


class Transaction(Base):
    __tablename__ = "transactions"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    # Transaction Details
    type = Column(String(50), nullable=False)
    amount_usd = Column(Float, nullable=False)  # المبلغ بالدولار (amount alias)
    units_transacted = Column(Float, nullable=True)  # الوحدات المضافة/المخصومة
    nav_at_transaction = Column(Float, nullable=True)  # قيمة الوحدة وقت العملية
    
    # Crypto Details
    coin = Column(String(20), default="USDC")
    currency = Column(String(20), nullable=True)  # العملة المستخدمة للدفع (usdcbsc, usdcsol, etc.)
    network = Column(String(50), nullable=True)
    tx_hash = Column(String(255), nullable=True)  # معرف العملية على البلوكتشين
    
    # NOWPayments Integration
    external_id = Column(String(100), nullable=True, index=True)  # payment_id من NOWPayments
    payment_address = Column(String(255), nullable=True)  # عنوان الدفع
    metadata = Column(JSON, nullable=True)  # بيانات إضافية من NOWPayments
    
    # Status
    status = Column(String(50), default=TransactionStatus.PENDING)
    
    # Addresses (for withdrawals)
    from_address = Column(String(255), nullable=True)
    to_address = Column(String(255), nullable=True)
    
    # Notes
    notes = Column(Text, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    confirmed_at = Column(DateTime(timezone=True), nullable=True)  # وقت تأكيد الدفع
    completed_at = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    user = relationship("User", back_populates="transactions")
    
    # Property alias for amount
    @property
    def amount(self):
        return self.amount_usd
    
    @amount.setter
    def amount(self, value):
        self.amount_usd = value


class WithdrawalRequest(Base):
    __tablename__ = "withdrawal_requests"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    # Request Details
    amount = Column(Float, nullable=False)
    units_to_withdraw = Column(Float, nullable=False)
    to_address = Column(String(255), nullable=False)
    network = Column(String(50), nullable=False)
    coin = Column(String(20), default="USDC")
    
    # Status: pending_approval, approved, rejected, processing, completed, failed
    status = Column(String(50), default="pending_approval")
    
    # Email Confirmation
    confirmation_token = Column(String(255), nullable=True)
    email_confirmed = Column(DateTime(timezone=True), nullable=True)
    
    # Admin Review
    reviewed_by = Column(Integer, ForeignKey("users.id"), nullable=True)
    reviewed_at = Column(DateTime(timezone=True), nullable=True)
    rejection_reason = Column(Text, nullable=True)
    
    # Timestamps
    requested_at = Column(DateTime(timezone=True), server_default=func.now())
    completed_at = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    user = relationship("User", back_populates="withdrawal_requests", foreign_keys=[user_id])
    reviewer = relationship("User", foreign_keys=[reviewed_by])


class NAVHistory(Base):
    """تاريخ قيمة الوحدة"""
    __tablename__ = "nav_history"
    
    id = Column(Integer, primary_key=True, index=True)
    nav_value = Column(Float, nullable=False)
    total_assets_usd = Column(Float, nullable=False)
    total_units = Column(Float, nullable=False)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())


class TradingHistory(Base):
    """تاريخ صفقات البوت"""
    __tablename__ = "trading_history"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Trade Details
    symbol = Column(String(20), nullable=False)  # e.g., BTCUSDC
    side = Column(String(10), nullable=False)  # BUY or SELL
    order_type = Column(String(20), nullable=False)  # MARKET, LIMIT
    
    price = Column(Float, nullable=False)
    quantity = Column(Float, nullable=False)
    total_value = Column(Float, nullable=False)
    
    # Binance Order Info
    order_id = Column(String(100), nullable=True)
    
    # Profit/Loss
    pnl = Column(Float, nullable=True)
    pnl_percent = Column(Float, nullable=True)
    
    # Timestamps
    executed_at = Column(DateTime(timezone=True), server_default=func.now())


class PlatformStats(Base):
    """إحصائيات المنصة"""
    __tablename__ = "platform_stats"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # High Water Mark for performance fee
    high_water_mark = Column(Float, default=1.0)
    last_fee_calculation = Column(DateTime(timezone=True), nullable=True)
    
    # Total fees collected
    total_fees_collected = Column(Float, default=0.0)
    
    # Emergency mode
    emergency_mode = Column(String(10), default="off")  # on/off
    
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
