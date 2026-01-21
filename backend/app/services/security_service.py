"""
Security Service - خدمة الأمان والمراقبة
سجل الأمان، إدارة الجلسات، IP Whitelist
"""
import secrets
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, desc, delete
import logging

logger = logging.getLogger(__name__)


class AuditService:
    """خدمة سجل المراقبة"""
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def log_action(
        self,
        action: str,
        admin_id: Optional[int] = None,
        target_type: Optional[str] = None,
        target_id: Optional[int] = None,
        details: Optional[Dict] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ):
        """تسجيل إجراء في سجل المراقبة"""
        from app.models.advanced_models import AuditLog, AuditAction
        
        try:
            log = AuditLog(
                admin_id=admin_id,
                action=AuditAction(action),
                target_type=target_type,
                target_id=target_id,
                details=details,
                ip_address=ip_address,
                user_agent=user_agent
            )
            
            self.db.add(log)
            await self.db.commit()
            
            logger.info(f"Audit log: {action} by admin {admin_id}")
        except Exception as e:
            logger.error(f"Failed to log audit action: {e}")
    
    async def get_audit_logs(
        self,
        action: Optional[str] = None,
        admin_id: Optional[int] = None,
        target_type: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100,
        offset: int = 0
    ) -> Dict:
        """الحصول على سجلات المراقبة"""
        from app.models.advanced_models import AuditLog, AuditAction
        
        query = select(AuditLog)
        
        if action:
            query = query.where(AuditLog.action == AuditAction(action))
        if admin_id:
            query = query.where(AuditLog.admin_id == admin_id)
        if target_type:
            query = query.where(AuditLog.target_type == target_type)
        if start_date:
            query = query.where(AuditLog.created_at >= start_date)
        if end_date:
            query = query.where(AuditLog.created_at <= end_date)
        
        # العدد الإجمالي
        count_query = select(func.count()).select_from(query.subquery())
        result = await self.db.execute(count_query)
        total = result.scalar() or 0
        
        # جلب السجلات
        query = query.order_by(desc(AuditLog.created_at)).limit(limit).offset(offset)
        
        result = await self.db.execute(query)
        logs = result.scalars().all()
        
        return {
            "total": total,
            "logs": [
                {
                    "id": log.id,
                    "admin_id": log.admin_id,
                    "action": log.action.value,
                    "target_type": log.target_type,
                    "target_id": log.target_id,
                    "details": log.details,
                    "ip_address": log.ip_address,
                    "created_at": log.created_at.isoformat()
                }
                for log in logs
            ]
        }
    
    async def get_security_alerts(self) -> List[Dict]:
        """الحصول على تنبيهات الأمان"""
        from app.models.advanced_models import AuditLog, AuditAction
        
        # آخر 24 ساعة
        since = datetime.utcnow() - timedelta(hours=24)
        
        alerts = []
        
        # محاولات تسجيل دخول فاشلة
        result = await self.db.execute(
            select(func.count(AuditLog.id))
            .where(AuditLog.action == AuditAction.LOGIN_FAILED)
            .where(AuditLog.created_at >= since)
        )
        failed_logins = result.scalar() or 0
        
        if failed_logins > 10:
            alerts.append({
                "type": "warning",
                "title": "محاولات تسجيل دخول فاشلة",
                "message": f"{failed_logins} محاولة فاشلة في آخر 24 ساعة",
                "count": failed_logins
            })
        
        # سحوبات كبيرة
        result = await self.db.execute(
            select(AuditLog)
            .where(AuditLog.action == AuditAction.WITHDRAWAL_APPROVED)
            .where(AuditLog.created_at >= since)
        )
        withdrawals = result.scalars().all()
        
        large_withdrawals = [
            w for w in withdrawals
            if w.details and w.details.get("amount", 0) > 10000
        ]
        
        if large_withdrawals:
            alerts.append({
                "type": "info",
                "title": "سحوبات كبيرة",
                "message": f"{len(large_withdrawals)} سحب كبير (> $10,000) في آخر 24 ساعة",
                "count": len(large_withdrawals)
            })
        
        # تغييرات في إعدادات البوت
        result = await self.db.execute(
            select(func.count(AuditLog.id))
            .where(AuditLog.action == AuditAction.BOT_SETTINGS_CHANGED)
            .where(AuditLog.created_at >= since)
        )
        bot_changes = result.scalar() or 0
        
        if bot_changes > 0:
            alerts.append({
                "type": "info",
                "title": "تغييرات في إعدادات البوت",
                "message": f"{bot_changes} تغيير في آخر 24 ساعة",
                "count": bot_changes
            })
        
        return alerts


class SessionService:
    """خدمة إدارة جلسات الأدمن"""
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def create_session(
        self,
        admin_id: int,
        ip_address: str,
        user_agent: Optional[str] = None,
        device_info: Optional[str] = None,
        expires_hours: int = 24
    ) -> str:
        """إنشاء جلسة جديدة"""
        from app.models.advanced_models import AdminSession
        
        session_token = secrets.token_urlsafe(32)
        expires_at = datetime.utcnow() + timedelta(hours=expires_hours)
        
        session = AdminSession(
            admin_id=admin_id,
            session_token=session_token,
            ip_address=ip_address,
            user_agent=user_agent,
            device_info=device_info,
            expires_at=expires_at
        )
        
        self.db.add(session)
        await self.db.commit()
        
        logger.info(f"Created session for admin {admin_id}")
        return session_token
    
    async def validate_session(self, session_token: str) -> Optional[int]:
        """التحقق من صلاحية الجلسة"""
        from app.models.advanced_models import AdminSession
        
        result = await self.db.execute(
            select(AdminSession)
            .where(AdminSession.session_token == session_token)
            .where(AdminSession.is_active == True)
            .where(AdminSession.expires_at > datetime.utcnow())
        )
        session = result.scalar_one_or_none()
        
        if session:
            # تحديث آخر نشاط
            session.last_activity = datetime.utcnow()
            await self.db.commit()
            return session.admin_id
        
        return None
    
    async def get_active_sessions(self, admin_id: int) -> List[Dict]:
        """الحصول على الجلسات النشطة"""
        from app.models.advanced_models import AdminSession
        
        result = await self.db.execute(
            select(AdminSession)
            .where(AdminSession.admin_id == admin_id)
            .where(AdminSession.is_active == True)
            .where(AdminSession.expires_at > datetime.utcnow())
            .order_by(desc(AdminSession.last_activity))
        )
        sessions = result.scalars().all()
        
        return [
            {
                "id": s.id,
                "ip_address": s.ip_address,
                "device_info": s.device_info,
                "created_at": s.created_at.isoformat(),
                "last_activity": s.last_activity.isoformat(),
                "expires_at": s.expires_at.isoformat()
            }
            for s in sessions
        ]
    
    async def revoke_session(self, session_id: int, admin_id: int) -> bool:
        """إلغاء جلسة"""
        from app.models.advanced_models import AdminSession
        
        result = await self.db.execute(
            select(AdminSession)
            .where(AdminSession.id == session_id)
            .where(AdminSession.admin_id == admin_id)
        )
        session = result.scalar_one_or_none()
        
        if session:
            session.is_active = False
            await self.db.commit()
            logger.info(f"Revoked session {session_id}")
            return True
        
        return False
    
    async def revoke_all_sessions(self, admin_id: int, except_current: Optional[str] = None):
        """إلغاء جميع الجلسات"""
        from app.models.advanced_models import AdminSession
        
        query = select(AdminSession).where(AdminSession.admin_id == admin_id)
        
        if except_current:
            query = query.where(AdminSession.session_token != except_current)
        
        result = await self.db.execute(query)
        sessions = result.scalars().all()
        
        for session in sessions:
            session.is_active = False
        
        await self.db.commit()
        logger.info(f"Revoked all sessions for admin {admin_id}")
    
    async def cleanup_expired_sessions(self):
        """تنظيف الجلسات المنتهية"""
        from app.models.advanced_models import AdminSession
        
        await self.db.execute(
            delete(AdminSession)
            .where(AdminSession.expires_at < datetime.utcnow())
        )
        await self.db.commit()


class IPWhitelistService:
    """خدمة قائمة IPs المسموح بها"""
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def add_ip(
        self,
        ip_address: str,
        description: Optional[str] = None,
        added_by: int = None
    ) -> bool:
        """إضافة IP للقائمة"""
        from app.models.advanced_models import IPWhitelist
        
        # التحقق من عدم وجوده مسبقاً
        result = await self.db.execute(
            select(IPWhitelist)
            .where(IPWhitelist.ip_address == ip_address)
        )
        existing = result.scalar_one_or_none()
        
        if existing:
            return False
        
        ip_entry = IPWhitelist(
            ip_address=ip_address,
            description=description,
            added_by=added_by,
            is_active=True
        )
        
        self.db.add(ip_entry)
        await self.db.commit()
        
        logger.info(f"Added IP {ip_address} to whitelist")
        return True
    
    async def remove_ip(self, ip_address: str) -> bool:
        """إزالة IP من القائمة"""
        from app.models.advanced_models import IPWhitelist
        
        result = await self.db.execute(
            select(IPWhitelist)
            .where(IPWhitelist.ip_address == ip_address)
        )
        ip_entry = result.scalar_one_or_none()
        
        if ip_entry:
            await self.db.delete(ip_entry)
            await self.db.commit()
            logger.info(f"Removed IP {ip_address} from whitelist")
            return True
        
        return False
    
    async def is_whitelisted(self, ip_address: str) -> bool:
        """التحقق من وجود IP في القائمة"""
        from app.models.advanced_models import IPWhitelist
        
        result = await self.db.execute(
            select(IPWhitelist)
            .where(IPWhitelist.ip_address == ip_address)
            .where(IPWhitelist.is_active == True)
        )
        return result.scalar_one_or_none() is not None
    
    async def get_all_ips(self) -> List[Dict]:
        """الحصول على جميع IPs"""
        from app.models.advanced_models import IPWhitelist
        
        result = await self.db.execute(
            select(IPWhitelist)
            .order_by(desc(IPWhitelist.created_at))
        )
        ips = result.scalars().all()
        
        return [
            {
                "id": ip.id,
                "ip_address": ip.ip_address,
                "description": ip.description,
                "added_by": ip.added_by,
                "is_active": ip.is_active,
                "created_at": ip.created_at.isoformat()
            }
            for ip in ips
        ]
    
    async def toggle_ip(self, ip_id: int, is_active: bool) -> bool:
        """تفعيل/تعطيل IP"""
        from app.models.advanced_models import IPWhitelist
        
        result = await self.db.execute(
            select(IPWhitelist)
            .where(IPWhitelist.id == ip_id)
        )
        ip_entry = result.scalar_one_or_none()
        
        if ip_entry:
            ip_entry.is_active = is_active
            await self.db.commit()
            return True
        
        return False


class TwoFactorService:
    """خدمة المصادقة الثنائية"""
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def generate_secret(self, user_id: int) -> str:
        """إنشاء مفتاح سري للمصادقة الثنائية"""
        import pyotp
        
        secret = pyotp.random_base32()
        
        # تخزين المفتاح (يجب إضافة حقل في جدول المستخدمين)
        # هنا نعيد المفتاح فقط للتبسيط
        
        return secret
    
    def verify_code(self, secret: str, code: str) -> bool:
        """التحقق من كود المصادقة"""
        import pyotp
        
        totp = pyotp.TOTP(secret)
        return totp.verify(code)
    
    def get_qr_uri(self, secret: str, email: str, issuer: str = "Legendary AI Trader") -> str:
        """الحصول على رابط QR"""
        import pyotp
        
        totp = pyotp.TOTP(secret)
        return totp.provisioning_uri(name=email, issuer_name=issuer)


class NotificationService:
    """خدمة الإشعارات"""
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def create_notification(
        self,
        notification_type: str,
        title: str,
        message: str,
        user_id: Optional[int] = None,
        data: Optional[Dict] = None
    ):
        """إنشاء إشعار"""
        from app.models.notification import Notification, NotificationType
        
        notification = Notification(
            user_id=user_id,
            type=NotificationType(notification_type),
            title=title,
            message=message,
            data=data
        )
        
        self.db.add(notification)
        await self.db.commit()
    
    async def get_user_notifications(
        self,
        user_id: int,
        unread_only: bool = False,
        limit: int = 50
    ) -> List[Dict]:
        """الحصول على إشعارات المستخدم"""
        from app.models.notification import Notification
        
        query = select(Notification).where(
            or_(
                Notification.user_id == user_id,
                Notification.user_id == None  # إشعارات عامة
            )
        )
        
        if unread_only:
            query = query.where(Notification.is_read == False)
        
        query = query.order_by(desc(Notification.created_at)).limit(limit)
        
        result = await self.db.execute(query)
        notifications = result.scalars().all()
        
        return [
            {
                "id": n.id,
                "type": n.type.value,
                "title": n.title,
                "message": n.message,
                "data": n.data,
                "is_read": n.is_read,
                "created_at": n.created_at.isoformat()
            }
            for n in notifications
        ]
    
    async def mark_as_read(self, notification_id: int, user_id: int) -> bool:
        """تحديد الإشعار كمقروء"""
        from app.models.notification import Notification
        
        result = await self.db.execute(
            select(Notification)
            .where(Notification.id == notification_id)
            .where(or_(
                Notification.user_id == user_id,
                Notification.user_id == None
            ))
        )
        notification = result.scalar_one_or_none()
        
        if notification:
            notification.is_read = True
            await self.db.commit()
            return True
        
        return False
    
    async def mark_all_as_read(self, user_id: int):
        """تحديد جميع الإشعارات كمقروءة"""
        from app.models.notification import Notification
        
        result = await self.db.execute(
            select(Notification)
            .where(or_(
                Notification.user_id == user_id,
                Notification.user_id == None
            ))
            .where(Notification.is_read == False)
        )
        notifications = result.scalars().all()
        
        for n in notifications:
            n.is_read = True
        
        await self.db.commit()
    
    async def get_unread_count(self, user_id: int) -> int:
        """الحصول على عدد الإشعارات غير المقروءة"""
        from app.models.notification import Notification
        
        result = await self.db.execute(
            select(func.count(Notification.id))
            .where(or_(
                Notification.user_id == user_id,
                Notification.user_id == None
            ))
            .where(Notification.is_read == False)
        )
        return result.scalar() or 0
