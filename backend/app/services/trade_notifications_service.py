"""
Trade Notifications Service - خدمة إشعارات الصفقات للمشتركين
يُضاف إلى /opt/asinax/backend/app/services/trade_notifications_service.py
"""
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Optional, List, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_
import logging

logger = logging.getLogger(__name__)

# سيتم استيراد هذه من الملفات الموجودة
# from app.models import User, Investor, Trade, Notification, NotificationType
# from app.services.email_service import EmailService
# from app.services.notifications import NotificationService


class TradeNotificationService:
    """
    خدمة إشعارات الصفقات للمشتركين
    ترسل إشعارات فورية عند فتح/إغلاق الصفقات
    """
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    # ============ إشعارات فتح الصفقات ============
    
    async def notify_trade_opened(
        self,
        trade_id: int,
        symbol: str,
        side: str,
        entry_price: float,
        confidence: float = None,
        risk_score: float = None
    ):
        """
        إشعار بفتح صفقة جديدة
        يُرسل لجميع المشتركين النشطين
        """
        try:
            # الحصول على جميع المستثمرين النشطين
            investors = await self._get_active_investors()
            
            side_ar = "شراء 📈" if side.upper() == "BUY" else "بيع 📉"
            
            title = f"صفقة جديدة: {symbol}"
            message = f"""
تم فتح صفقة {side_ar} على {symbol}
سعر الدخول: ${entry_price:,.4f}
"""
            if confidence:
                message += f"مستوى الثقة: {confidence:.1f}%\n"
            if risk_score:
                message += f"درجة المخاطرة: {risk_score:.1f}/10\n"
            
            data = {
                "trade_id": trade_id,
                "symbol": symbol,
                "side": side,
                "entry_price": entry_price,
                "confidence": confidence,
                "risk_score": risk_score,
                "type": "trade_opened"
            }
            
            # إرسال إشعار لكل مستثمر
            for investor in investors:
                await self._create_notification(
                    user_id=investor.user_id,
                    type="trade",
                    title=title,
                    message=message.strip(),
                    data=data,
                    send_email=await self._should_send_email(investor.user_id, "trade_opened")
                )
            
            logger.info(f"Trade opened notification sent to {len(investors)} investors")
            
        except Exception as e:
            logger.error(f"Error sending trade opened notification: {e}")
    
    async def notify_trade_closed(
        self,
        trade_id: int,
        symbol: str,
        side: str,
        entry_price: float,
        exit_price: float,
        pnl: float,
        pnl_percent: float,
        duration_hours: float = None
    ):
        """
        إشعار بإغلاق صفقة
        يُرسل لجميع المشتركين النشطين
        """
        try:
            investors = await self._get_active_investors()
            
            # تحديد الأيقونة والحالة
            if pnl > 0:
                status_icon = "✅"
                status_text = "ربح"
                pnl_text = f"+${pnl:,.2f} (+{pnl_percent:.2f}%)"
            else:
                status_icon = "❌"
                status_text = "خسارة"
                pnl_text = f"-${abs(pnl):,.2f} ({pnl_percent:.2f}%)"
            
            title = f"صفقة مغلقة {status_icon}: {symbol}"
            message = f"""
تم إغلاق صفقة {symbol}
سعر الدخول: ${entry_price:,.4f}
سعر الخروج: ${exit_price:,.4f}
النتيجة: {pnl_text}
"""
            if duration_hours:
                if duration_hours < 1:
                    duration_text = f"{int(duration_hours * 60)} دقيقة"
                elif duration_hours < 24:
                    duration_text = f"{duration_hours:.1f} ساعة"
                else:
                    duration_text = f"{duration_hours / 24:.1f} يوم"
                message += f"مدة الصفقة: {duration_text}\n"
            
            data = {
                "trade_id": trade_id,
                "symbol": symbol,
                "side": side,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "pnl": pnl,
                "pnl_percent": pnl_percent,
                "is_profitable": pnl > 0,
                "type": "trade_closed"
            }
            
            for investor in investors:
                await self._create_notification(
                    user_id=investor.user_id,
                    type="trade",
                    title=title,
                    message=message.strip(),
                    data=data,
                    send_email=await self._should_send_email(investor.user_id, "trade_closed")
                )
            
            logger.info(f"Trade closed notification sent to {len(investors)} investors")
            
        except Exception as e:
            logger.error(f"Error sending trade closed notification: {e}")
    
    # ============ إشعارات الأداء ============
    
    async def notify_profit_milestone(
        self,
        user_id: int,
        milestone_type: str,
        amount: float,
        percentage: float
    ):
        """
        إشعار بتحقيق هدف ربح معين
        milestone_type: daily_profit, weekly_profit, monthly_profit, total_profit
        """
        try:
            milestone_names = {
                "daily_profit": "ربح يومي",
                "weekly_profit": "ربح أسبوعي",
                "monthly_profit": "ربح شهري",
                "total_profit": "إجمالي الربح"
            }
            
            milestone_name = milestone_names.get(milestone_type, "ربح")
            
            title = f"🎉 تهانينا! {milestone_name} جديد"
            message = f"""
مبروك! حققت {milestone_name} بقيمة ${amount:,.2f}
نسبة العائد: +{percentage:.2f}%

استمر في الاستثمار لتحقيق المزيد من الأرباح!
"""
            
            data = {
                "milestone_type": milestone_type,
                "amount": amount,
                "percentage": percentage,
                "type": "profit_milestone"
            }
            
            await self._create_notification(
                user_id=user_id,
                type="balance",
                title=title,
                message=message.strip(),
                data=data,
                send_email=True
            )
            
        except Exception as e:
            logger.error(f"Error sending profit milestone notification: {e}")
    
    async def notify_loss_alert(
        self,
        user_id: int,
        loss_amount: float,
        loss_percentage: float,
        period: str = "daily"
    ):
        """
        تنبيه بخسارة معينة
        """
        try:
            period_names = {
                "daily": "اليوم",
                "weekly": "هذا الأسبوع",
                "monthly": "هذا الشهر"
            }
            
            period_name = period_names.get(period, "")
            
            title = f"⚠️ تنبيه: انخفاض في المحفظة"
            message = f"""
انخفضت قيمة محفظتك {period_name} بنسبة {abs(loss_percentage):.2f}%
قيمة الانخفاض: ${abs(loss_amount):,.2f}

الوكيل الذكي يعمل على تحسين الأداء وتقليل المخاطر.
"""
            
            data = {
                "loss_amount": loss_amount,
                "loss_percentage": loss_percentage,
                "period": period,
                "type": "loss_alert"
            }
            
            await self._create_notification(
                user_id=user_id,
                type="balance",
                title=title,
                message=message.strip(),
                data=data,
                send_email=True
            )
            
        except Exception as e:
            logger.error(f"Error sending loss alert notification: {e}")
    
    # ============ التقارير التلقائية ============
    
    async def send_daily_summary(self, user_id: int, summary_data: Dict[str, Any]):
        """
        إرسال ملخص يومي للمستثمر
        """
        try:
            trades_count = summary_data.get("trades_count", 0)
            total_pnl = summary_data.get("total_pnl", 0)
            win_rate = summary_data.get("win_rate", 0)
            portfolio_value = summary_data.get("portfolio_value", 0)
            
            pnl_icon = "📈" if total_pnl >= 0 else "📉"
            pnl_text = f"+${total_pnl:,.2f}" if total_pnl >= 0 else f"-${abs(total_pnl):,.2f}"
            
            title = f"📊 ملخصك اليومي - {datetime.now().strftime('%Y-%m-%d')}"
            message = f"""
ملخص أداء محفظتك اليوم:

{pnl_icon} الربح/الخسارة: {pnl_text}
📈 عدد الصفقات: {trades_count}
🎯 نسبة النجاح: {win_rate:.1f}%
💰 قيمة المحفظة: ${portfolio_value:,.2f}

شكراً لثقتك في ASINAX!
"""
            
            data = {
                "trades_count": trades_count,
                "total_pnl": total_pnl,
                "win_rate": win_rate,
                "portfolio_value": portfolio_value,
                "date": datetime.now().isoformat(),
                "type": "daily_summary"
            }
            
            await self._create_notification(
                user_id=user_id,
                type="system",
                title=title,
                message=message.strip(),
                data=data,
                send_email=await self._should_send_email(user_id, "daily_summary")
            )
            
        except Exception as e:
            logger.error(f"Error sending daily summary: {e}")
    
    async def send_weekly_report_notification(self, user_id: int, report_data: Dict[str, Any]):
        """
        إشعار بإصدار التقرير الأسبوعي
        """
        try:
            total_pnl = report_data.get("total_pnl", 0)
            pnl_percent = report_data.get("pnl_percent", 0)
            
            pnl_icon = "🟢" if total_pnl >= 0 else "🔴"
            
            title = "📑 تقريرك الأسبوعي جاهز!"
            message = f"""
تم إصدار تقريرك الأسبوعي.

{pnl_icon} أداء الأسبوع: {'+' if total_pnl >= 0 else ''}{pnl_percent:.2f}%
💵 الربح/الخسارة: ${total_pnl:,.2f}

يمكنك تحميل التقرير الكامل من لوحة التحكم.
"""
            
            data = {
                "report_type": "weekly",
                "total_pnl": total_pnl,
                "pnl_percent": pnl_percent,
                "type": "weekly_report"
            }
            
            await self._create_notification(
                user_id=user_id,
                type="system",
                title=title,
                message=message.strip(),
                data=data,
                send_email=True
            )
            
        except Exception as e:
            logger.error(f"Error sending weekly report notification: {e}")
    
    # ============ دوال مساعدة ============
    
    async def _get_active_investors(self) -> List:
        """الحصول على جميع المستثمرين النشطين"""
        from app.models import Investor
        
        result = await self.db.execute(
            select(Investor).where(Investor.status == 'active')
        )
        return result.scalars().all()
    
    async def _create_notification(
        self,
        user_id: int,
        type: str,
        title: str,
        message: str,
        data: Dict = None,
        send_email: bool = False
    ):
        """إنشاء إشعار جديد"""
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
        
        if send_email:
            await self._send_email_notification(user_id, title, message, type)
        
        return notification
    
    async def _should_send_email(self, user_id: int, notification_type: str) -> bool:
        """
        التحقق من تفضيلات البريد الإلكتروني للمستخدم
        يمكن توسيعها لاحقاً لدعم تفضيلات مخصصة
        """
        # افتراضياً: إرسال بريد للصفقات المغلقة والتقارير فقط
        email_enabled_types = ["trade_closed", "daily_summary", "weekly_report", "profit_milestone"]
        return notification_type in email_enabled_types
    
    async def _send_email_notification(
        self,
        user_id: int,
        title: str,
        message: str,
        type: str
    ):
        """إرسال إشعار بالبريد الإلكتروني"""
        try:
            from app.models import User
            from app.services.email_service import EmailService
            
            result = await self.db.execute(
                select(User).where(User.id == user_id)
            )
            user = result.scalar_one_or_none()
            
            if user and user.email:
                email_service = EmailService()
                await email_service.send_email(
                    to_email=user.email,
                    subject=f"ASINAX - {title}",
                    html_content=f"""
                    <div style="font-family: Arial, sans-serif; direction: rtl; text-align: right; background: #0a0a0a; color: #fff; padding: 20px;">
                        <h2 style="color: #10b981;">{title}</h2>
                        <div style="white-space: pre-line; line-height: 1.8;">{message}</div>
                        <hr style="border-color: #333; margin: 20px 0;">
                        <p style="color: #666; font-size: 12px;">
                            هذا إشعار تلقائي من منصة ASINAX
                        </p>
                    </div>
                    """
                )
        except Exception as e:
            logger.error(f"Failed to send email notification: {e}")


# ============ دوال مساعدة للتكامل مع الوكيل ============

async def on_trade_opened(db: AsyncSession, trade_data: Dict[str, Any]):
    """
    يُستدعى عند فتح صفقة جديدة من الوكيل
    """
    service = TradeNotificationService(db)
    await service.notify_trade_opened(
        trade_id=trade_data.get("id"),
        symbol=trade_data.get("symbol"),
        side=trade_data.get("side"),
        entry_price=trade_data.get("entry_price"),
        confidence=trade_data.get("confidence"),
        risk_score=trade_data.get("risk_score")
    )


async def on_trade_closed(db: AsyncSession, trade_data: Dict[str, Any]):
    """
    يُستدعى عند إغلاق صفقة من الوكيل
    """
    service = TradeNotificationService(db)
    await service.notify_trade_closed(
        trade_id=trade_data.get("id"),
        symbol=trade_data.get("symbol"),
        side=trade_data.get("side"),
        entry_price=trade_data.get("entry_price"),
        exit_price=trade_data.get("exit_price"),
        pnl=trade_data.get("pnl"),
        pnl_percent=trade_data.get("pnl_percent"),
        duration_hours=trade_data.get("duration_hours")
    )
