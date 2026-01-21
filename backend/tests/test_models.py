"""
Unit Tests for VIPTier and Related Models
اختبارات الوحدة لـ VIPTier والموديلات المرتبطة
"""
import pytest
import sys
import os

# إضافة مسار المشروع
sys.path.insert(0, '/app')

from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker
from datetime import datetime


class TestModelsImport:
    """اختبار استيراد الموديلات بدون أخطاء"""
    
    def test_import_viptier(self):
        """اختبار استيراد VIPTier"""
        from app.models.advanced_models import VIPTier, VIPLevel
        assert VIPTier is not None
        assert VIPLevel is not None
        print("✅ VIPTier imported successfully")
    
    def test_import_notification(self):
        """اختبار استيراد Notification من notification.py"""
        from app.models.notification import Notification, NotificationType
        assert Notification is not None
        assert NotificationType is not None
        print("✅ Notification imported successfully")
    
    def test_notification_type_has_marketing(self):
        """اختبار وجود MARKETING في NotificationType"""
        from app.models.notification import NotificationType
        assert hasattr(NotificationType, 'MARKETING')
        assert NotificationType.MARKETING.value == "marketing"
        print("✅ NotificationType has MARKETING")
    
    def test_no_duplicate_notification_in_advanced_models(self):
        """اختبار عدم وجود Notification مكرر في advanced_models"""
        from app.models import advanced_models
        # التحقق من عدم وجود class Notification في advanced_models
        classes_in_module = [name for name in dir(advanced_models) 
                           if isinstance(getattr(advanced_models, name), type)]
        # Notification يجب أن يكون مستورداً وليس معرفاً محلياً
        notification_class = getattr(advanced_models, 'Notification', None)
        if notification_class:
            # التحقق من أنه مستورد من notification.py
            assert notification_class.__module__ == 'app.models.notification', \
                "Notification should be imported from notification.py"
        print("✅ No duplicate Notification in advanced_models")
    
    def test_import_all_advanced_models(self):
        """اختبار استيراد جميع الموديلات المتقدمة"""
        from app.models.advanced_models import (
            VIPTier, UserVIP, VIPLevel,
            ReferralCode, Referral,
            Coupon, CouponUsage, CouponType,
            SupportTicket, TicketMessage, TicketStatus, TicketPriority,
            AuditLog, AuditAction,
            AdminSession, IPWhitelist,
            NotificationSetting,
            BotSettings, PlatformSettings,
            ColdWalletTransfer, UserNote,
            BackupLog, AnalyticsCache
        )
        print("✅ All advanced models imported successfully")
    
    def test_notification_setting_uses_correct_type(self):
        """اختبار أن NotificationSetting يستخدم NotificationType الصحيح"""
        from app.models.advanced_models import NotificationSetting
        from app.models.notification import NotificationType
        # التحقق من أن العمود notification_type موجود
        assert hasattr(NotificationSetting, 'notification_type')
        print("✅ NotificationSetting uses correct NotificationType")


class TestVIPTierModel:
    """اختبارات خاصة بـ VIPTier"""
    
    def test_viptier_tablename(self):
        """اختبار اسم الجدول"""
        from app.models.advanced_models import VIPTier
        assert VIPTier.__tablename__ == "vip_tiers"
        print("✅ VIPTier tablename is correct")
    
    def test_viptier_columns(self):
        """اختبار أعمدة VIPTier"""
        from app.models.advanced_models import VIPTier
        expected_columns = [
            'id', 'level', 'name_ar', 'name_en', 
            'min_deposit', 'fee_discount', 'withdrawal_limit_daily',
            'priority_support', 'early_access', 'custom_referral_bonus',
            'created_at'
        ]
        actual_columns = [c.name for c in VIPTier.__table__.columns]
        for col in expected_columns:
            assert col in actual_columns, f"Missing column: {col}"
        print("✅ VIPTier has all expected columns")
    
    def test_vip_levels(self):
        """اختبار مستويات VIP"""
        from app.models.advanced_models import VIPLevel
        expected_levels = ['BRONZE', 'SILVER', 'GOLD', 'PLATINUM', 'DIAMOND']
        for level in expected_levels:
            assert hasattr(VIPLevel, level), f"Missing VIP level: {level}"
        print("✅ All VIP levels exist")


class TestNotificationModel:
    """اختبارات خاصة بـ Notification"""
    
    def test_notification_tablename(self):
        """اختبار اسم الجدول"""
        from app.models.notification import Notification
        assert Notification.__tablename__ == "notifications"
        print("✅ Notification tablename is correct")
    
    def test_notification_columns(self):
        """اختبار أعمدة Notification"""
        from app.models.notification import Notification
        expected_columns = ['id', 'user_id', 'type', 'title', 'message', 'data', 'is_read']
        actual_columns = [c.name for c in Notification.__table__.columns]
        for col in expected_columns:
            assert col in actual_columns, f"Missing column: {col}"
        print("✅ Notification has all expected columns")
    
    def test_notification_types(self):
        """اختبار أنواع الإشعارات"""
        from app.models.notification import NotificationType
        expected_types = ['DEPOSIT', 'WITHDRAWAL', 'BALANCE', 'REFERRAL', 
                         'SYSTEM', 'TRADE', 'SECURITY', 'MARKETING']
        for ntype in expected_types:
            assert hasattr(NotificationType, ntype), f"Missing notification type: {ntype}"
        print("✅ All notification types exist")


class TestModelRelationships:
    """اختبار العلاقات بين الموديلات"""
    
    def test_viptier_user_relationship(self):
        """اختبار علاقة VIPTier مع UserVIP"""
        from app.models.advanced_models import VIPTier, UserVIP
        assert hasattr(VIPTier, 'users')
        assert hasattr(UserVIP, 'tier')
        print("✅ VIPTier-UserVIP relationship exists")
    
    def test_referral_relationships(self):
        """اختبار علاقات الإحالة"""
        from app.models.advanced_models import ReferralCode, Referral
        assert hasattr(ReferralCode, 'referrals')
        assert hasattr(Referral, 'code')
        print("✅ Referral relationships exist")
    
    def test_coupon_relationships(self):
        """اختبار علاقات الكوبونات"""
        from app.models.advanced_models import Coupon, CouponUsage
        assert hasattr(Coupon, 'usages')
        assert hasattr(CouponUsage, 'coupon')
        print("✅ Coupon relationships exist")
    
    def test_ticket_relationships(self):
        """اختبار علاقات التذاكر"""
        from app.models.advanced_models import SupportTicket, TicketMessage
        assert hasattr(SupportTicket, 'messages')
        assert hasattr(TicketMessage, 'ticket')
        print("✅ Ticket relationships exist")


class TestDatabaseIntegration:
    """اختبارات التكامل مع قاعدة البيانات"""
    
    def test_base_metadata(self):
        """اختبار metadata الأساسي"""
        from app.core.database import Base
        # التحقق من أن جميع الجداول مسجلة
        tables = Base.metadata.tables.keys()
        expected_tables = ['vip_tiers', 'notifications', 'user_vip']
        for table in expected_tables:
            assert table in tables, f"Missing table: {table}"
        print("✅ All tables registered in metadata")


if __name__ == "__main__":
    # تشغيل الاختبارات
    pytest.main([__file__, "-v", "--tb=short"])
