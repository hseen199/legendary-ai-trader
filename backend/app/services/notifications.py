# خدمة الإشعارات للمستخدمين
# /opt/asinax/backend/app/services/notifications.py

from datetime import datetime
from typing import Optional, List
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
from sqlalchemy.orm import selectinload

from app.models.user import User
from app.models.notification import Notification, NotificationType
from app.services.email_service import EmailService

email_service = EmailService()


class NotificationService:
    """خدمة إدارة الإشعارات للمستخدمين"""
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def create_notification(
        self,
        user_id: int,
        type: NotificationType,
        title: str,
        message: str,
        data: Optional[dict] = None,
        send_email: bool = True
    ) -> Notification:
        """إنشاء إشعار جديد"""
        notification = Notification(
            user_id=user_id,
            type=type,
            title=title,
            message=message,
            data=data or {},
            is_read=False,
            created_at=datetime.utcnow()
        )
        self.db.add(notification)
        await self.db.commit()
        await self.db.refresh(notification)
        
        # إرسال بريد إلكتروني إذا كان مطلوباً
        if send_email:
            user = await self.get_user(user_id)
            if user and user.email:
                await self._send_email_notification(user.email, title, message, type)
        
        return notification
    
    async def get_user(self, user_id: int) -> Optional[User]:
        """الحصول على بيانات المستخدم"""
        result = await self.db.execute(
            select(User).where(User.id == user_id)
        )
        return result.scalar_one_or_none()
    
    async def get_user_notifications(
        self,
        user_id: int,
        limit: int = 50,
        unread_only: bool = False
    ) -> List[Notification]:
        """الحصول على إشعارات المستخدم"""
        query = select(Notification).where(Notification.user_id == user_id)
        
        if unread_only:
            query = query.where(Notification.is_read == False)
        
        query = query.order_by(Notification.created_at.desc()).limit(limit)
        
        result = await self.db.execute(query)
        return result.scalars().all()
    
    async def mark_as_read(self, notification_id: int, user_id: int) -> bool:
        """تحديد الإشعار كمقروء"""
        result = await self.db.execute(
            update(Notification)
            .where(Notification.id == notification_id)
            .where(Notification.user_id == user_id)
            .values(is_read=True, read_at=datetime.utcnow())
        )
        await self.db.commit()
        return result.rowcount > 0
    
    async def mark_all_as_read(self, user_id: int) -> int:
        """تحديد جميع الإشعارات كمقروءة"""
        result = await self.db.execute(
            update(Notification)
            .where(Notification.user_id == user_id)
            .where(Notification.is_read == False)
            .values(is_read=True, read_at=datetime.utcnow())
        )
        await self.db.commit()
        return result.rowcount
    
    async def get_unread_count(self, user_id: int) -> int:
        """الحصول على عدد الإشعارات غير المقروءة"""
        result = await self.db.execute(
            select(Notification)
            .where(Notification.user_id == user_id)
            .where(Notification.is_read == False)
        )
        return len(result.scalars().all())
    
    # ============ إشعارات الإيداع ============
    
    async def notify_deposit_pending(self, user_id: int, amount: float, payment_id: str):
        """إشعار بإنشاء طلب إيداع"""
        await self.create_notification(
            user_id=user_id,
            type=NotificationType.DEPOSIT,
            title="طلب إيداع جديد",
            message=f"تم إنشاء طلب إيداع بمبلغ ${amount:.2f} USDC. رقم العملية: {payment_id}",
            data={"amount": amount, "payment_id": payment_id, "status": "pending"},
            send_email=True
        )
    
    async def notify_deposit_confirmed(self, user_id: int, amount: float, units: float):
        """إشعار بتأكيد الإيداع"""
        await self.create_notification(
            user_id=user_id,
            type=NotificationType.DEPOSIT,
            title="تم تأكيد الإيداع ✅",
            message=f"تم تأكيد إيداعك بمبلغ ${amount:.2f} USDC وإضافة {units:.4f} وحدة إلى رصيدك.",
            data={"amount": amount, "units": units, "status": "completed"},
            send_email=True
        )
    
    async def notify_deposit_failed(self, user_id: int, amount: float, reason: str = ""):
        """إشعار بفشل الإيداع"""
        await self.create_notification(
            user_id=user_id,
            type=NotificationType.DEPOSIT,
            title="فشل الإيداع ❌",
            message=f"فشل إيداعك بمبلغ ${amount:.2f} USDC. {reason}",
            data={"amount": amount, "status": "failed", "reason": reason},
            send_email=True
        )
    
    # ============ إشعارات السحب ============
    
    async def notify_withdrawal_pending(self, user_id: int, amount: float, withdrawal_id: int):
        """إشعار بإنشاء طلب سحب"""
        await self.create_notification(
            user_id=user_id,
            type=NotificationType.WITHDRAWAL,
            title="طلب سحب جديد",
            message=f"تم إرسال طلب سحب بمبلغ ${amount:.2f} USDC للمراجعة. رقم الطلب: #{withdrawal_id}",
            data={"amount": amount, "withdrawal_id": withdrawal_id, "status": "pending_approval"},
            send_email=True
        )
    
    async def notify_withdrawal_approved(self, user_id: int, amount: float, withdrawal_id: int):
        """إشعار بالموافقة على السحب"""
        await self.create_notification(
            user_id=user_id,
            type=NotificationType.WITHDRAWAL,
            title="تمت الموافقة على السحب ✅",
            message=f"تمت الموافقة على طلب السحب #{withdrawal_id} بمبلغ ${amount:.2f} USDC. جاري المعالجة...",
            data={"amount": amount, "withdrawal_id": withdrawal_id, "status": "approved"},
            send_email=True
        )
    
    async def notify_withdrawal_rejected(self, user_id: int, amount: float, withdrawal_id: int, reason: str):
        """إشعار برفض السحب"""
        await self.create_notification(
            user_id=user_id,
            type=NotificationType.WITHDRAWAL,
            title="تم رفض طلب السحب ❌",
            message=f"تم رفض طلب السحب #{withdrawal_id} بمبلغ ${amount:.2f} USDC. السبب: {reason}",
            data={"amount": amount, "withdrawal_id": withdrawal_id, "status": "rejected", "reason": reason},
            send_email=True
        )
    
    async def notify_withdrawal_completed(self, user_id: int, amount: float, withdrawal_id: int, tx_hash: str):
        """إشعار بإكمال السحب"""
        await self.create_notification(
            user_id=user_id,
            type=NotificationType.WITHDRAWAL,
            title="تم إكمال السحب ✅",
            message=f"تم إرسال ${amount:.2f} USDC إلى محفظتك بنجاح. TX: {tx_hash[:20]}...",
            data={"amount": amount, "withdrawal_id": withdrawal_id, "status": "completed", "tx_hash": tx_hash},
            send_email=True
        )
    
    # ============ إشعارات عامة ============
    
    async def notify_balance_update(self, user_id: int, new_balance: float, change: float, reason: str):
        """إشعار بتحديث الرصيد"""
        change_text = f"+${change:.2f}" if change > 0 else f"-${abs(change):.2f}"
        await self.create_notification(
            user_id=user_id,
            type=NotificationType.BALANCE,
            title="تحديث الرصيد",
            message=f"تم تحديث رصيدك ({change_text}). الرصيد الجديد: ${new_balance:.2f}. السبب: {reason}",
            data={"new_balance": new_balance, "change": change, "reason": reason},
            send_email=False  # لا نرسل إيميل لكل تحديث رصيد
        )
    
    async def notify_referral_bonus(self, user_id: int, bonus: float, referred_user: str):
        """إشعار بمكافأة الإحالة"""
        await self.create_notification(
            user_id=user_id,
            type=NotificationType.REFERRAL,
            title="مكافأة إحالة 🎉",
            message=f"حصلت على مكافأة إحالة بقيمة ${bonus:.2f} من {referred_user}!",
            data={"bonus": bonus, "referred_user": referred_user},
            send_email=True
        )
    
    async def _send_email_notification(
        self,
        email: str,
        title: str,
        message: str,
        type: NotificationType
    ):
        """إرسال إشعار بالبريد الإلكتروني"""
        try:
            subject_prefix = {
                NotificationType.DEPOSIT: "💰 إيداع",
                NotificationType.WITHDRAWAL: "💸 سحب",
                NotificationType.BALANCE: "📊 رصيد",
                NotificationType.REFERRAL: "🎁 إحالة",
                NotificationType.SYSTEM: "🔔 نظام",
            }.get(type, "🔔")
            
            # استخدام دالة إرسال البريد الموجودة
            await email_service.send_email(
                to_email=email,
                subject=f"ASINAX - {subject_prefix} - {title}",
                html_content=f"""
                <div style="font-family: Arial, sans-serif; direction: rtl; text-align: right;">
                    <h2>{title}</h2>
                    <p>{message}</p>
                    <hr>
                    <p style="color: #666; font-size: 12px;">
                        هذا إشعار تلقائي من منصة ASINAX
                    </p>
                </div>
                """
            )
        except Exception as e:
            print(f"Failed to send email notification: {e}")
