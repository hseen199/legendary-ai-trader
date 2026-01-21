# نموذج الإشعارات
# يُضاف إلى /opt/asinax/backend/app/models/notification.py
from sqlalchemy import Column, Integer, String, Boolean, DateTime, ForeignKey, Text, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.core.database import Base
import enum


class NotificationType(str, enum.Enum):
    DEPOSIT = "deposit"
    WITHDRAWAL = "withdrawal"
    BALANCE = "balance"
    REFERRAL = "referral"
    SYSTEM = "system"
    TRADE = "trade"
    SECURITY = "security"
    MARKETING = "marketing"  # تمت إضافته للتوافق مع advanced_models


class Notification(Base):
    """نموذج الإشعارات"""
    __tablename__ = "notifications"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    
    # نوع الإشعار
    type = Column(String(50), nullable=False, index=True)
    
    # محتوى الإشعار
    title = Column(String(255), nullable=False)
    message = Column(Text, nullable=False)
    
    # بيانات إضافية (JSON)
    data = Column(JSON, nullable=True)
    
    # حالة القراءة
    is_read = Column(Boolean, default=False, index=True)
    read_at = Column(DateTime(timezone=True), nullable=True)
    
    # الطوابع الزمنية
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # العلاقات
    user = relationship("User", backref="notifications")
    
    def to_dict(self):
        return {
            "id": self.id,
            "type": self.type,
            "title": self.title,
            "message": self.message,
            "data": self.data,
            "is_read": self.is_read,
            "read_at": self.read_at.isoformat() if self.read_at else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
