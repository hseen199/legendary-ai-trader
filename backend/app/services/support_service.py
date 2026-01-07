"""
Support Service - خدمة الدعم الفني
نظام التذاكر والدردشة مع المستخدمين
"""
import secrets
from datetime import datetime
from typing import Dict, List, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, or_, desc
import logging

logger = logging.getLogger(__name__)


class SupportService:
    """خدمة الدعم الفني"""
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    def _generate_ticket_number(self) -> str:
        """إنشاء رقم تذكرة فريد"""
        timestamp = datetime.utcnow().strftime("%Y%m%d")
        random_part = secrets.token_hex(3).upper()
        return f"TKT-{timestamp}-{random_part}"
    
    # ============ Ticket Management ============
    
    async def create_ticket(
        self,
        user_id: int,
        subject: str,
        category: str,
        message: str,
        priority: str = "medium"
    ) -> Dict:
        """إنشاء تذكرة جديدة"""
        from app.models.advanced_models import SupportTicket, TicketMessage, TicketStatus, TicketPriority
        
        ticket_number = self._generate_ticket_number()
        
        # إنشاء التذكرة
        ticket = SupportTicket(
            ticket_number=ticket_number,
            user_id=user_id,
            subject=subject,
            category=category,
            status=TicketStatus.OPEN,
            priority=TicketPriority(priority)
        )
        
        self.db.add(ticket)
        await self.db.flush()  # للحصول على ID
        
        # إضافة الرسالة الأولى
        first_message = TicketMessage(
            ticket_id=ticket.id,
            sender_id=user_id,
            message=message,
            is_admin=False
        )
        
        self.db.add(first_message)
        await self.db.commit()
        
        logger.info(f"Created ticket {ticket_number} for user {user_id}")
        
        return {
            "ticket_number": ticket_number,
            "id": ticket.id,
            "status": "open",
            "created_at": ticket.created_at.isoformat()
        }
    
    async def get_user_tickets(
        self,
        user_id: int,
        status: Optional[str] = None
    ) -> List[Dict]:
        """الحصول على تذاكر المستخدم"""
        from app.models.advanced_models import SupportTicket, TicketStatus
        
        query = select(SupportTicket).where(SupportTicket.user_id == user_id)
        
        if status:
            query = query.where(SupportTicket.status == TicketStatus(status))
        
        query = query.order_by(desc(SupportTicket.updated_at))
        
        result = await self.db.execute(query)
        tickets = result.scalars().all()
        
        return [
            {
                "id": t.id,
                "ticket_number": t.ticket_number,
                "subject": t.subject,
                "category": t.category,
                "status": t.status.value,
                "priority": t.priority.value,
                "created_at": t.created_at.isoformat(),
                "updated_at": t.updated_at.isoformat()
            }
            for t in tickets
        ]
    
    async def get_ticket_details(
        self,
        ticket_id: int,
        user_id: Optional[int] = None
    ) -> Optional[Dict]:
        """الحصول على تفاصيل التذكرة"""
        from app.models.advanced_models import SupportTicket, TicketMessage
        
        query = select(SupportTicket).where(SupportTicket.id == ticket_id)
        
        if user_id:
            query = query.where(SupportTicket.user_id == user_id)
        
        result = await self.db.execute(query)
        ticket = result.scalar_one_or_none()
        
        if not ticket:
            return None
        
        # جلب الرسائل
        result = await self.db.execute(
            select(TicketMessage)
            .where(TicketMessage.ticket_id == ticket_id)
            .order_by(TicketMessage.created_at)
        )
        messages = result.scalars().all()
        
        return {
            "id": ticket.id,
            "ticket_number": ticket.ticket_number,
            "subject": ticket.subject,
            "category": ticket.category,
            "status": ticket.status.value,
            "priority": ticket.priority.value,
            "assigned_to": ticket.assigned_to,
            "created_at": ticket.created_at.isoformat(),
            "updated_at": ticket.updated_at.isoformat(),
            "resolved_at": ticket.resolved_at.isoformat() if ticket.resolved_at else None,
            "messages": [
                {
                    "id": m.id,
                    "sender_id": m.sender_id,
                    "message": m.message,
                    "is_admin": m.is_admin,
                    "attachments": m.attachments,
                    "created_at": m.created_at.isoformat()
                }
                for m in messages
            ]
        }
    
    async def add_message(
        self,
        ticket_id: int,
        sender_id: int,
        message: str,
        is_admin: bool = False,
        attachments: Optional[List[str]] = None
    ) -> Dict:
        """إضافة رسالة للتذكرة"""
        from app.models.advanced_models import SupportTicket, TicketMessage, TicketStatus
        
        # التحقق من التذكرة
        result = await self.db.execute(
            select(SupportTicket)
            .where(SupportTicket.id == ticket_id)
        )
        ticket = result.scalar_one_or_none()
        
        if not ticket:
            return {"error": "التذكرة غير موجودة"}
        
        # إضافة الرسالة
        new_message = TicketMessage(
            ticket_id=ticket_id,
            sender_id=sender_id,
            message=message,
            is_admin=is_admin,
            attachments=attachments
        )
        
        self.db.add(new_message)
        
        # تحديث حالة التذكرة
        if is_admin:
            ticket.status = TicketStatus.WAITING_USER
        else:
            ticket.status = TicketStatus.IN_PROGRESS
        
        ticket.updated_at = datetime.utcnow()
        
        await self.db.commit()
        
        logger.info(f"Added message to ticket {ticket_id}")
        
        return {
            "id": new_message.id,
            "created_at": new_message.created_at.isoformat()
        }
    
    async def update_ticket_status(
        self,
        ticket_id: int,
        status: str,
        admin_id: Optional[int] = None
    ) -> bool:
        """تحديث حالة التذكرة"""
        from app.models.advanced_models import SupportTicket, TicketStatus
        
        result = await self.db.execute(
            select(SupportTicket)
            .where(SupportTicket.id == ticket_id)
        )
        ticket = result.scalar_one_or_none()
        
        if not ticket:
            return False
        
        ticket.status = TicketStatus(status)
        ticket.updated_at = datetime.utcnow()
        
        if status == "resolved":
            ticket.resolved_at = datetime.utcnow()
        
        if admin_id and not ticket.assigned_to:
            ticket.assigned_to = admin_id
        
        await self.db.commit()
        
        logger.info(f"Updated ticket {ticket_id} status to {status}")
        return True
    
    async def assign_ticket(
        self,
        ticket_id: int,
        admin_id: int
    ) -> bool:
        """تعيين التذكرة لأدمن"""
        from app.models.advanced_models import SupportTicket, TicketStatus
        
        result = await self.db.execute(
            select(SupportTicket)
            .where(SupportTicket.id == ticket_id)
        )
        ticket = result.scalar_one_or_none()
        
        if not ticket:
            return False
        
        ticket.assigned_to = admin_id
        ticket.status = TicketStatus.IN_PROGRESS
        ticket.updated_at = datetime.utcnow()
        
        await self.db.commit()
        
        logger.info(f"Assigned ticket {ticket_id} to admin {admin_id}")
        return True
    
    # ============ Admin Functions ============
    
    async def get_all_tickets(
        self,
        status: Optional[str] = None,
        priority: Optional[str] = None,
        category: Optional[str] = None,
        assigned_to: Optional[int] = None,
        limit: int = 50,
        offset: int = 0
    ) -> Dict:
        """الحصول على جميع التذاكر (للأدمن)"""
        from app.models.advanced_models import SupportTicket, TicketStatus, TicketPriority
        
        query = select(SupportTicket)
        
        if status:
            query = query.where(SupportTicket.status == TicketStatus(status))
        if priority:
            query = query.where(SupportTicket.priority == TicketPriority(priority))
        if category:
            query = query.where(SupportTicket.category == category)
        if assigned_to:
            query = query.where(SupportTicket.assigned_to == assigned_to)
        
        # العدد الإجمالي
        count_query = select(func.count()).select_from(query.subquery())
        result = await self.db.execute(count_query)
        total = result.scalar() or 0
        
        # جلب التذاكر
        query = query.order_by(
            desc(SupportTicket.priority),
            desc(SupportTicket.updated_at)
        ).limit(limit).offset(offset)
        
        result = await self.db.execute(query)
        tickets = result.scalars().all()
        
        return {
            "total": total,
            "tickets": [
                {
                    "id": t.id,
                    "ticket_number": t.ticket_number,
                    "user_id": t.user_id,
                    "subject": t.subject,
                    "category": t.category,
                    "status": t.status.value,
                    "priority": t.priority.value,
                    "assigned_to": t.assigned_to,
                    "created_at": t.created_at.isoformat(),
                    "updated_at": t.updated_at.isoformat()
                }
                for t in tickets
            ]
        }
    
    async def get_support_stats(self) -> Dict:
        """الحصول على إحصائيات الدعم"""
        from app.models.advanced_models import SupportTicket, TicketStatus
        
        # إجمالي التذاكر
        result = await self.db.execute(
            select(func.count(SupportTicket.id))
        )
        total = result.scalar() or 0
        
        # التذاكر المفتوحة
        result = await self.db.execute(
            select(func.count(SupportTicket.id))
            .where(SupportTicket.status == TicketStatus.OPEN)
        )
        open_tickets = result.scalar() or 0
        
        # التذاكر قيد المعالجة
        result = await self.db.execute(
            select(func.count(SupportTicket.id))
            .where(SupportTicket.status == TicketStatus.IN_PROGRESS)
        )
        in_progress = result.scalar() or 0
        
        # التذاكر المحلولة اليوم
        today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        result = await self.db.execute(
            select(func.count(SupportTicket.id))
            .where(SupportTicket.status == TicketStatus.RESOLVED)
            .where(SupportTicket.resolved_at >= today)
        )
        resolved_today = result.scalar() or 0
        
        # متوسط وقت الحل
        result = await self.db.execute(
            select(SupportTicket)
            .where(SupportTicket.resolved_at != None)
            .limit(100)
        )
        resolved_tickets = result.scalars().all()
        
        if resolved_tickets:
            total_time = sum(
                (t.resolved_at - t.created_at).total_seconds()
                for t in resolved_tickets
            )
            avg_resolution_time = total_time / len(resolved_tickets) / 3600  # بالساعات
        else:
            avg_resolution_time = 0
        
        # التذاكر حسب الفئة
        result = await self.db.execute(
            select(
                SupportTicket.category,
                func.count(SupportTicket.id)
            )
            .group_by(SupportTicket.category)
        )
        by_category = {row[0]: row[1] for row in result.all()}
        
        # التذاكر حسب الأولوية
        result = await self.db.execute(
            select(
                SupportTicket.priority,
                func.count(SupportTicket.id)
            )
            .where(SupportTicket.status.in_([TicketStatus.OPEN, TicketStatus.IN_PROGRESS]))
            .group_by(SupportTicket.priority)
        )
        by_priority = {row[0].value: row[1] for row in result.all()}
        
        return {
            "total": total,
            "open": open_tickets,
            "in_progress": in_progress,
            "resolved_today": resolved_today,
            "avg_resolution_hours": round(avg_resolution_time, 1),
            "by_category": by_category,
            "by_priority": by_priority
        }
    
    async def get_unassigned_tickets(self) -> List[Dict]:
        """الحصول على التذاكر غير المعينة"""
        from app.models.advanced_models import SupportTicket, TicketStatus
        
        result = await self.db.execute(
            select(SupportTicket)
            .where(SupportTicket.assigned_to == None)
            .where(SupportTicket.status.in_([TicketStatus.OPEN, TicketStatus.IN_PROGRESS]))
            .order_by(desc(SupportTicket.priority), SupportTicket.created_at)
        )
        tickets = result.scalars().all()
        
        return [
            {
                "id": t.id,
                "ticket_number": t.ticket_number,
                "user_id": t.user_id,
                "subject": t.subject,
                "category": t.category,
                "priority": t.priority.value,
                "created_at": t.created_at.isoformat()
            }
            for t in tickets
        ]


class FAQService:
    """خدمة الأسئلة الشائعة"""
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def get_faqs(self, category: Optional[str] = None) -> List[Dict]:
        """الحصول على الأسئلة الشائعة"""
        # يمكن تخزينها في قاعدة البيانات أو ملف JSON
        faqs = [
            {
                "category": "deposit",
                "question": "كيف أقوم بالإيداع؟",
                "answer": "يمكنك الإيداع عن طريق إرسال USDC إلى عنوان محفظتك الخاص الموجود في صفحة الإيداع."
            },
            {
                "category": "deposit",
                "question": "ما هو الحد الأدنى للإيداع؟",
                "answer": "الحد الأدنى للإيداع هو 100 USDC."
            },
            {
                "category": "withdrawal",
                "question": "كم يستغرق السحب؟",
                "answer": "تتم معالجة طلبات السحب خلال 24 ساعة عمل بعد الموافقة."
            },
            {
                "category": "withdrawal",
                "question": "ما هي رسوم السحب؟",
                "answer": "رسوم السحب هي 1% من المبلغ المسحوب بحد أدنى 1 USDC."
            },
            {
                "category": "trading",
                "question": "كيف يعمل البوت؟",
                "answer": "يستخدم البوت الذكاء الاصطناعي لتحليل السوق واتخاذ قرارات التداول تلقائياً."
            },
            {
                "category": "trading",
                "question": "هل يمكنني خسارة أموالي؟",
                "answer": "نعم، التداول ينطوي على مخاطر. نحن نستخدم استراتيجيات إدارة المخاطر لتقليل الخسائر."
            },
            {
                "category": "account",
                "question": "كيف أغير كلمة المرور؟",
                "answer": "يمكنك تغيير كلمة المرور من صفحة الإعدادات في حسابك."
            },
            {
                "category": "account",
                "question": "كيف أفعّل المصادقة الثنائية؟",
                "answer": "اذهب إلى الإعدادات > الأمان > تفعيل المصادقة الثنائية."
            }
        ]
        
        if category:
            faqs = [f for f in faqs if f["category"] == category]
        
        return faqs
