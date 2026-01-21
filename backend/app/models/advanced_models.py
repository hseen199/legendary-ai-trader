"""
Advanced Models for Admin Dashboard
نماذج متقدمة للوحة تحكم الأدمن
"""
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, Text, ForeignKey, Enum, JSON
from sqlalchemy.orm import relationship
from datetime import datetime
import enum
from app.core.database import Base

# استيراد NotificationType من notification.py لتجنب التكرار
from app.models.notification import NotificationType


# ============ Enums ============

class VIPLevel(str, enum.Enum):
    """مستويات VIP"""
    BRONZE = "bronze"
    SILVER = "silver"
    GOLD = "gold"
    PLATINUM = "platinum"
    DIAMOND = "diamond"


class TicketStatus(str, enum.Enum):
    """حالات التذاكر"""
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    WAITING_USER = "waiting_user"
    RESOLVED = "resolved"
    CLOSED = "closed"


class TicketPriority(str, enum.Enum):
    """أولوية التذاكر"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


class AuditAction(str, enum.Enum):
    """أنواع إجراءات السجل"""
    LOGIN = "login"
    LOGOUT = "logout"
    LOGIN_FAILED = "login_failed"
    WITHDRAWAL_APPROVED = "withdrawal_approved"
    WITHDRAWAL_REJECTED = "withdrawal_rejected"
    USER_SUSPENDED = "user_suspended"
    USER_ACTIVATED = "user_activated"
    BOT_STARTED = "bot_started"
    BOT_STOPPED = "bot_stopped"
    BOT_SETTINGS_CHANGED = "bot_settings_changed"
    SETTINGS_CHANGED = "settings_changed"
    COLD_WALLET_TRANSFER = "cold_wallet_transfer"
    ADMIN_CREATED = "admin_created"
    ADMIN_DELETED = "admin_deleted"


class CouponType(str, enum.Enum):
    """أنواع الكوبونات"""
    PERCENTAGE = "percentage"
    FIXED = "fixed"


# تم نقل NotificationType إلى notification.py لتجنب التكرار
# يتم استيراده من هناك الآن


# ============ VIP System ============

class VIPTier(Base):
    """مستويات VIP وامتيازاتها"""
    __tablename__ = "vip_tiers"
    
    id = Column(Integer, primary_key=True, index=True)
    level = Column(Enum(VIPLevel), unique=True, nullable=False)
    name_ar = Column(String(50), nullable=False)  # الاسم بالعربي
    name_en = Column(String(50), nullable=False)  # الاسم بالإنجليزي
    min_deposit = Column(Float, default=0)  # الحد الأدنى للإيداع
    fee_discount = Column(Float, default=0)  # نسبة الخصم على الرسوم (0-100)
    withdrawal_limit_daily = Column(Float, default=10000)  # حد السحب اليومي
    priority_support = Column(Boolean, default=False)  # دعم أولوية
    early_access = Column(Boolean, default=False)  # وصول مبكر للميزات
    custom_referral_bonus = Column(Float, default=0)  # مكافأة إحالة مخصصة
    created_at = Column(DateTime, default=datetime.utcnow)
    
    users = relationship("UserVIP", back_populates="tier")


class UserVIP(Base):
    """ربط المستخدمين بمستويات VIP"""
    __tablename__ = "user_vip"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    tier_id = Column(Integer, ForeignKey("vip_tiers.id"), nullable=False)
    upgraded_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=True)  # null = لا ينتهي
    is_manual = Column(Boolean, default=False)  # ترقية يدوية من الأدمن
    
    tier = relationship("VIPTier", back_populates="users")


# ============ Referral System ============

class ReferralCode(Base):
    """أكواد الإحالة"""
    __tablename__ = "referral_codes"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    code = Column(String(20), unique=True, nullable=False, index=True)
    bonus_percent = Column(Float, default=5)  # نسبة المكافأة
    max_uses = Column(Integer, nullable=True)  # الحد الأقصى للاستخدام (null = غير محدود)
    times_used = Column(Integer, default=0)
    total_earned = Column(Float, default=0)  # إجمالي المكافآت المكتسبة
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    referrals = relationship("Referral", back_populates="code")


class Referral(Base):
    """سجل الإحالات"""
    __tablename__ = "referrals"
    
    id = Column(Integer, primary_key=True, index=True)
    referrer_id = Column(Integer, ForeignKey("users.id"), nullable=False)  # المُحيل
    referred_id = Column(Integer, ForeignKey("users.id"), nullable=False)  # المُحال
    code_id = Column(Integer, ForeignKey("referral_codes.id"), nullable=False)
    bonus_amount = Column(Float, default=0)  # مبلغ المكافأة
    bonus_paid = Column(Boolean, default=False)  # هل تم دفع المكافأة
    referred_deposit = Column(Float, default=0)  # إيداع المُحال
    created_at = Column(DateTime, default=datetime.utcnow)
    
    code = relationship("ReferralCode", back_populates="referrals")


# ============ Coupon System ============

class Coupon(Base):
    """الكوبونات والخصومات"""
    __tablename__ = "coupons"
    
    id = Column(Integer, primary_key=True, index=True)
    code = Column(String(30), unique=True, nullable=False, index=True)
    type = Column(Enum(CouponType), nullable=False)
    value = Column(Float, nullable=False)  # النسبة أو المبلغ
    description = Column(String(200), nullable=True)
    min_deposit = Column(Float, default=0)  # الحد الأدنى للإيداع
    max_discount = Column(Float, nullable=True)  # الحد الأقصى للخصم
    max_uses = Column(Integer, nullable=True)  # الحد الأقصى للاستخدام
    times_used = Column(Integer, default=0)
    valid_from = Column(DateTime, default=datetime.utcnow)
    valid_until = Column(DateTime, nullable=True)
    is_active = Column(Boolean, default=True)
    created_by = Column(Integer, ForeignKey("users.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    usages = relationship("CouponUsage", back_populates="coupon")


class CouponUsage(Base):
    """سجل استخدام الكوبونات"""
    __tablename__ = "coupon_usages"
    
    id = Column(Integer, primary_key=True, index=True)
    coupon_id = Column(Integer, ForeignKey("coupons.id"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    discount_amount = Column(Float, nullable=False)
    used_at = Column(DateTime, default=datetime.utcnow)
    
    coupon = relationship("Coupon", back_populates="usages")


# ============ Support Tickets ============

class SupportTicket(Base):
    """تذاكر الدعم الفني"""
    __tablename__ = "support_tickets"
    
    id = Column(Integer, primary_key=True, index=True)
    ticket_number = Column(String(20), unique=True, nullable=False, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    subject = Column(String(200), nullable=False)
    category = Column(String(50), nullable=False)  # deposit, withdrawal, technical, other
    status = Column(Enum(TicketStatus), default=TicketStatus.OPEN)
    priority = Column(Enum(TicketPriority), default=TicketPriority.MEDIUM)
    assigned_to = Column(Integer, ForeignKey("users.id"), nullable=True)  # الأدمن المسؤول
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    resolved_at = Column(DateTime, nullable=True)
    
    messages = relationship("TicketMessage", back_populates="ticket")


class TicketMessage(Base):
    """رسائل التذاكر"""
    __tablename__ = "ticket_messages"
    
    id = Column(Integer, primary_key=True, index=True)
    ticket_id = Column(Integer, ForeignKey("support_tickets.id"), nullable=False)
    sender_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    message = Column(Text, nullable=False)
    is_admin = Column(Boolean, default=False)
    attachments = Column(JSON, nullable=True)  # قائمة روابط المرفقات
    created_at = Column(DateTime, default=datetime.utcnow)
    
    ticket = relationship("SupportTicket", back_populates="messages")


# ============ Audit Log ============

class AuditLog(Base):
    """سجل الأمان والمراقبة"""
    __tablename__ = "audit_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    admin_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    action = Column(Enum(AuditAction), nullable=False)
    target_type = Column(String(50), nullable=True)  # user, withdrawal, bot, settings
    target_id = Column(Integer, nullable=True)
    details = Column(JSON, nullable=True)  # تفاصيل إضافية
    ip_address = Column(String(50), nullable=True)
    user_agent = Column(String(500), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


# ============ Admin Sessions ============

class AdminSession(Base):
    """جلسات الأدمن النشطة"""
    __tablename__ = "admin_sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    admin_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    session_token = Column(String(255), unique=True, nullable=False)
    ip_address = Column(String(50), nullable=False)
    user_agent = Column(String(500), nullable=True)
    device_info = Column(String(200), nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_activity = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=False)


# ============ IP Whitelist ============

class IPWhitelist(Base):
    """قائمة IPs المسموح بها للأدمن"""
    __tablename__ = "ip_whitelist"
    
    id = Column(Integer, primary_key=True, index=True)
    ip_address = Column(String(50), unique=True, nullable=False)
    description = Column(String(200), nullable=True)
    added_by = Column(Integer, ForeignKey("users.id"), nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)


# ============ Notifications Settings ============
# تم حذف class Notification المكرر - يستخدم الآن من notification.py

class NotificationSetting(Base):
    """إعدادات الإشعارات للأدمن"""
    __tablename__ = "notification_settings"
    
    id = Column(Integer, primary_key=True, index=True)
    admin_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    notification_type = Column(Enum(NotificationType), nullable=False)
    email_enabled = Column(Boolean, default=True)
    min_amount = Column(Float, nullable=True)  # الحد الأدنى للمبلغ للإشعار
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


# ============ Bot Settings ============

class BotSettings(Base):
    """إعدادات البوت المتقدمة"""
    __tablename__ = "bot_settings"
    
    id = Column(Integer, primary_key=True, index=True)
    # إدارة المخاطر
    max_daily_loss_percent = Column(Float, default=5.0)  # الحد الأقصى للخسارة اليومية
    max_position_size_percent = Column(Float, default=2.0)  # حجم الصفقة الواحدة
    default_stop_loss_percent = Column(Float, default=3.0)
    default_take_profit_percent = Column(Float, default=6.0)
    max_open_positions = Column(Integer, default=5)
    
    # العملات المفعّلة
    enabled_pairs = Column(JSON, default=["BTCUSDC", "ETHUSDC", "SOLUSDC"])
    
    # جدول التداول
    trading_24_7 = Column(Boolean, default=True)
    trading_start_hour = Column(Integer, nullable=True)
    trading_end_hour = Column(Integer, nullable=True)
    
    # وضع الطوارئ
    emergency_mode = Column(Boolean, default=False)
    emergency_activated_at = Column(DateTime, nullable=True)
    emergency_activated_by = Column(Integer, ForeignKey("users.id"), nullable=True)
    
    # إعدادات أخرى
    auto_compound = Column(Boolean, default=True)  # إعادة استثمار الأرباح
    min_trade_amount = Column(Float, default=10.0)
    
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    updated_by = Column(Integer, ForeignKey("users.id"), nullable=True)


# ============ Platform Settings ============

class PlatformSettings(Base):
    """إعدادات المنصة"""
    __tablename__ = "platform_settings"
    
    id = Column(Integer, primary_key=True, index=True)
    key = Column(String(100), unique=True, nullable=False)
    value = Column(Text, nullable=True)
    value_type = Column(String(20), default="string")  # string, number, boolean, json
    description = Column(String(200), nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    updated_by = Column(Integer, ForeignKey("users.id"), nullable=True)


# ============ Cold Wallet ============

class ColdWalletTransfer(Base):
    """تحويلات المحفظة الباردة"""
    __tablename__ = "cold_wallet_transfers"
    
    id = Column(Integer, primary_key=True, index=True)
    direction = Column(String(10), nullable=False)  # to_cold, from_cold
    amount = Column(Float, nullable=False)
    asset = Column(String(20), default="USDC")
    wallet_address = Column(String(100), nullable=False)
    tx_hash = Column(String(100), nullable=True)
    status = Column(String(20), default="pending")  # pending, completed, failed
    initiated_by = Column(Integer, ForeignKey("users.id"), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)


# ============ Admin Notes ============

class UserNote(Base):
    """ملاحظات الأدمن على المستخدمين"""
    __tablename__ = "user_notes"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    admin_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    note = Column(Text, nullable=False)
    is_important = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)


# ============ Backup Log ============

class BackupLog(Base):
    """سجل النسخ الاحتياطية"""
    __tablename__ = "backup_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(200), nullable=False)
    size_mb = Column(Float, nullable=False)
    type = Column(String(20), default="auto")  # auto, manual
    status = Column(String(20), default="completed")  # completed, failed
    initiated_by = Column(Integer, ForeignKey("users.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


# ============ Analytics Cache ============

class AnalyticsCache(Base):
    """كاش التحليلات"""
    __tablename__ = "analytics_cache"
    
    id = Column(Integer, primary_key=True, index=True)
    key = Column(String(100), unique=True, nullable=False)
    data = Column(JSON, nullable=False)
    expires_at = Column(DateTime, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
