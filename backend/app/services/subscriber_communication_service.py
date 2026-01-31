"""
Subscriber Communication Service - خدمة التواصل مع المشتركين
يُضاف إلى /opt/asinax/backend/app/services/subscriber_communication_service.py
"""
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, or_
import logging

logger = logging.getLogger(__name__)


class MessageType:
    """أنواع الرسائل"""
    ANNOUNCEMENT = "announcement"  # إعلان عام
    UPDATE = "update"  # تحديث
    ALERT = "alert"  # تنبيه
    PROMOTION = "promotion"  # عرض ترويجي
    MAINTENANCE = "maintenance"  # صيانة
    NEWSLETTER = "newsletter"  # نشرة إخبارية


class TargetAudience:
    """الجمهور المستهدف"""
    ALL = "all"  # جميع المستخدمين
    INVESTORS = "investors"  # المستثمرين فقط
    VIP = "vip"  # مستخدمي VIP
    VIP_GOLD_PLUS = "vip_gold_plus"  # ذهبي وأعلى
    INACTIVE = "inactive"  # غير نشطين
    NEW_USERS = "new_users"  # مستخدمين جدد


class SubscriberCommunicationService:
    """
    خدمة التواصل مع المشتركين
    تدعم الرسائل الجماعية والتنبيهات المخصصة
    """
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    # ============ الرسائل الجماعية ============
    
    async def send_broadcast_message(
        self,
        title: str,
        message: str,
        message_type: str = MessageType.ANNOUNCEMENT,
        target_audience: str = TargetAudience.ALL,
        vip_levels: List[str] = None,
        send_email: bool = True,
        send_notification: bool = True,
        scheduled_at: datetime = None,
        data: Dict = None
    ) -> Dict[str, Any]:
        """
        إرسال رسالة جماعية للمشتركين
        """
        try:
            # الحصول على المستخدمين المستهدفين
            users = await self._get_target_users(target_audience, vip_levels)
            
            if not users:
                return {
                    "success": False,
                    "message": "لا يوجد مستخدمين مستهدفين",
                    "sent_count": 0
                }
            
            sent_count = 0
            failed_count = 0
            
            for user in users:
                try:
                    # إنشاء إشعار داخلي
                    if send_notification:
                        await self._create_notification(
                            user_id=user.id,
                            type=message_type,
                            title=title,
                            message=message,
                            data=data or {}
                        )
                    
                    # إرسال بريد إلكتروني
                    if send_email and user.email:
                        await self._send_email(
                            to_email=user.email,
                            subject=title,
                            content=message,
                            message_type=message_type
                        )
                    
                    sent_count += 1
                    
                except Exception as e:
                    logger.error(f"Failed to send message to user {user.id}: {e}")
                    failed_count += 1
            
            logger.info(f"Broadcast sent: {sent_count} success, {failed_count} failed")
            
            return {
                "success": True,
                "message": f"تم إرسال الرسالة إلى {sent_count} مستخدم",
                "sent_count": sent_count,
                "failed_count": failed_count,
                "total_target": len(users)
            }
            
        except Exception as e:
            logger.error(f"Error sending broadcast: {e}")
            return {
                "success": False,
                "message": str(e),
                "sent_count": 0
            }
    
    async def send_vip_exclusive_message(
        self,
        title: str,
        message: str,
        min_vip_level: str = "gold"
    ) -> Dict[str, Any]:
        """
        إرسال رسالة حصرية لمستخدمي VIP
        """
        vip_order = ["bronze", "silver", "gold", "platinum", "diamond"]
        min_index = vip_order.index(min_vip_level) if min_vip_level in vip_order else 0
        target_levels = vip_order[min_index:]
        
        return await self.send_broadcast_message(
            title=f"🌟 حصري VIP: {title}",
            message=message,
            message_type=MessageType.PROMOTION,
            target_audience=TargetAudience.VIP,
            vip_levels=target_levels,
            send_email=True,
            send_notification=True
        )
    
    # ============ التنبيهات المخصصة ============
    
    async def send_personalized_alert(
        self,
        user_id: int,
        alert_type: str,
        title: str,
        message: str,
        action_url: str = None,
        priority: str = "normal"
    ):
        """
        إرسال تنبيه مخصص لمستخدم محدد
        """
        try:
            data = {
                "alert_type": alert_type,
                "priority": priority,
                "action_url": action_url
            }
            
            await self._create_notification(
                user_id=user_id,
                type="alert",
                title=title,
                message=message,
                data=data
            )
            
            # إرسال بريد إلكتروني للتنبيهات ذات الأولوية العالية
            if priority == "high":
                from app.models import User
                result = await self.db.execute(
                    select(User).where(User.id == user_id)
                )
                user = result.scalar_one_or_none()
                
                if user and user.email:
                    await self._send_email(
                        to_email=user.email,
                        subject=f"⚠️ تنبيه هام: {title}",
                        content=message,
                        message_type="alert"
                    )
            
            return {"success": True}
            
        except Exception as e:
            logger.error(f"Error sending personalized alert: {e}")
            return {"success": False, "error": str(e)}
    
    # ============ تنبيهات النظام ============
    
    async def notify_maintenance(
        self,
        start_time: datetime,
        end_time: datetime,
        description: str = None
    ):
        """
        إشعار بصيانة مجدولة
        """
        duration = end_time - start_time
        hours = duration.total_seconds() / 3600
        
        title = "🔧 صيانة مجدولة"
        message = f"""
سيتم إجراء صيانة مجدولة للمنصة:

⏰ وقت البدء: {start_time.strftime('%Y-%m-%d %H:%M')} UTC
⏱️ المدة المتوقعة: {hours:.1f} ساعة

{description or 'قد تكون بعض الخدمات غير متاحة خلال هذه الفترة.'}

نعتذر عن أي إزعاج.
"""
        
        return await self.send_broadcast_message(
            title=title,
            message=message,
            message_type=MessageType.MAINTENANCE,
            target_audience=TargetAudience.ALL,
            send_email=True,
            send_notification=True
        )
    
    async def notify_new_feature(
        self,
        feature_name: str,
        description: str,
        available_for: str = TargetAudience.ALL
    ):
        """
        إشعار بميزة جديدة
        """
        title = f"🚀 ميزة جديدة: {feature_name}"
        message = f"""
يسعدنا الإعلان عن ميزة جديدة في ASINAX!

✨ {feature_name}

{description}

جرّب الميزة الجديدة الآن من لوحة التحكم!
"""
        
        return await self.send_broadcast_message(
            title=title,
            message=message,
            message_type=MessageType.UPDATE,
            target_audience=available_for,
            send_email=True,
            send_notification=True
        )
    
    async def notify_market_update(
        self,
        update_type: str,
        summary: str,
        details: str = None
    ):
        """
        إشعار بتحديث السوق
        """
        icons = {
            "bullish": "📈",
            "bearish": "📉",
            "volatile": "⚡",
            "stable": "➡️"
        }
        
        icon = icons.get(update_type, "📊")
        
        title = f"{icon} تحديث السوق"
        message = f"""
{summary}

{details or ''}

الوكيل الذكي يراقب السوق ويعدل الاستراتيجيات وفقاً لذلك.
"""
        
        return await self.send_broadcast_message(
            title=title,
            message=message,
            message_type=MessageType.UPDATE,
            target_audience=TargetAudience.INVESTORS,
            send_email=False,  # لا نرسل بريد لتحديثات السوق
            send_notification=True
        )
    
    # ============ تذكيرات تلقائية ============
    
    async def send_inactivity_reminder(self, days_inactive: int = 30):
        """
        إرسال تذكير للمستخدمين غير النشطين
        """
        from app.models import User
        
        cutoff_date = datetime.utcnow() - timedelta(days=days_inactive)
        
        result = await self.db.execute(
            select(User).where(
                and_(
                    User.last_login < cutoff_date,
                    User.is_active == True
                )
            )
        )
        inactive_users = result.scalars().all()
        
        title = "🔔 نفتقدك في ASINAX!"
        message = f"""
مرحباً!

لاحظنا أنك لم تزر ASINAX منذ فترة.

الوكيل الذكي يعمل باستمرار لتحقيق أفضل العوائد.
تفقد محفظتك الآن لمعرفة آخر التطورات!

🔗 سجل الدخول الآن: https://asinax.cloud

نتطلع لرؤيتك قريباً!
"""
        
        sent_count = 0
        for user in inactive_users:
            try:
                await self._create_notification(
                    user_id=user.id,
                    type="system",
                    title=title,
                    message=message,
                    data={"type": "inactivity_reminder"}
                )
                
                if user.email:
                    await self._send_email(
                        to_email=user.email,
                        subject=title,
                        content=message,
                        message_type="reminder"
                    )
                
                sent_count += 1
            except Exception as e:
                logger.error(f"Failed to send reminder to user {user.id}: {e}")
        
        return {
            "success": True,
            "sent_count": sent_count,
            "total_inactive": len(inactive_users)
        }
    
    async def send_deposit_reminder(self, min_balance: float = 100):
        """
        تذكير للمستخدمين برصيد منخفض
        """
        from app.models import User
        
        result = await self.db.execute(
            select(User).where(
                and_(
                    User.balance < min_balance,
                    User.is_active == True
                )
            )
        )
        low_balance_users = result.scalars().all()
        
        title = "💰 زد استثمارك!"
        message = f"""
مرحباً!

رصيدك الحالي منخفض. زد استثمارك للاستفادة من:

✅ عوائد أعلى
✅ ترقية مستوى VIP
✅ رسوم أداء مخفضة

أودع الآن واستفد من قوة الذكاء الاصطناعي في التداول!
"""
        
        sent_count = 0
        for user in low_balance_users:
            try:
                await self._create_notification(
                    user_id=user.id,
                    type="system",
                    title=title,
                    message=message,
                    data={"type": "deposit_reminder", "current_balance": user.balance}
                )
                sent_count += 1
            except Exception as e:
                logger.error(f"Failed to send deposit reminder to user {user.id}: {e}")
        
        return {
            "success": True,
            "sent_count": sent_count
        }
    
    # ============ دوال مساعدة ============
    
    async def _get_target_users(self, target_audience: str, vip_levels: List[str] = None) -> List:
        """الحصول على المستخدمين المستهدفين"""
        from app.models import User, Investor
        
        query = select(User).where(User.is_active == True)
        
        if target_audience == TargetAudience.INVESTORS:
            # المستثمرين فقط (لديهم إيداعات)
            query = query.where(User.total_deposited > 0)
        
        elif target_audience == TargetAudience.VIP:
            if vip_levels:
                query = query.where(User.vip_level.in_(vip_levels))
            else:
                query = query.where(User.vip_level.in_(["silver", "gold", "platinum", "diamond"]))
        
        elif target_audience == TargetAudience.VIP_GOLD_PLUS:
            query = query.where(User.vip_level.in_(["gold", "platinum", "diamond"]))
        
        elif target_audience == TargetAudience.NEW_USERS:
            week_ago = datetime.utcnow() - timedelta(days=7)
            query = query.where(User.created_at >= week_ago)
        
        result = await self.db.execute(query)
        return result.scalars().all()
    
    async def _create_notification(
        self,
        user_id: int,
        type: str,
        title: str,
        message: str,
        data: Dict = None
    ):
        """إنشاء إشعار"""
        from app.models import Notification
        
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
    
    async def _send_email(
        self,
        to_email: str,
        subject: str,
        content: str,
        message_type: str
    ):
        """إرسال بريد إلكتروني"""
        try:
            from app.services.email_service import EmailService
            
            # تحديد الأيقونة حسب نوع الرسالة
            icons = {
                MessageType.ANNOUNCEMENT: "📢",
                MessageType.UPDATE: "🔄",
                MessageType.ALERT: "⚠️",
                MessageType.PROMOTION: "🎁",
                MessageType.MAINTENANCE: "🔧",
                MessageType.NEWSLETTER: "📰"
            }
            icon = icons.get(message_type, "📧")
            
            email_service = EmailService()
            await email_service.send_email(
                to_email=to_email,
                subject=f"ASINAX {icon} {subject}",
                html_content=f"""
                <div style="font-family: Arial, sans-serif; direction: rtl; text-align: right; background: #0a0a0a; color: #fff; padding: 30px;">
                    <div style="max-width: 600px; margin: 0 auto;">
                        <h2 style="color: #10b981; margin-bottom: 20px;">{subject}</h2>
                        <div style="white-space: pre-line; line-height: 1.8; color: #e0e0e0;">
                            {content}
                        </div>
                        <hr style="border-color: #333; margin: 30px 0;">
                        <p style="color: #666; font-size: 12px; text-align: center;">
                            ASINAX - منصة التداول الذكي
                            <br>
                            <a href="https://asinax.cloud" style="color: #10b981;">asinax.cloud</a>
                        </p>
                    </div>
                </div>
                """
            )
        except Exception as e:
            logger.error(f"Failed to send email to {to_email}: {e}")


# ============ API Endpoints للإضافة ============

"""
# يُضاف إلى /opt/asinax/backend/app/api/routes/admin.py

@router.post("/communication/broadcast")
async def send_broadcast(
    title: str,
    message: str,
    message_type: str = "announcement",
    target_audience: str = "all",
    send_email: bool = True,
    current_user: User = Depends(get_admin_user),
    db: AsyncSession = Depends(get_db)
):
    '''إرسال رسالة جماعية (للأدمن فقط)'''
    service = SubscriberCommunicationService(db)
    return await service.send_broadcast_message(
        title=title,
        message=message,
        message_type=message_type,
        target_audience=target_audience,
        send_email=send_email
    )

@router.post("/communication/maintenance")
async def notify_maintenance(
    start_time: datetime,
    end_time: datetime,
    description: str = None,
    current_user: User = Depends(get_admin_user),
    db: AsyncSession = Depends(get_db)
):
    '''إشعار بصيانة مجدولة (للأدمن فقط)'''
    service = SubscriberCommunicationService(db)
    return await service.notify_maintenance(start_time, end_time, description)
"""
