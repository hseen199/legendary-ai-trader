"""
Support Routes - مسارات الدعم الفني
"""
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional, List
from pydantic import BaseModel

from app.core.database import get_db
from app.core.auth import get_current_admin, get_current_user
from app.services.support_service import SupportService, FAQService

router = APIRouter(prefix="/support", tags=["Support"])


# ============ Schemas ============

class CreateTicketRequest(BaseModel):
    subject: str
    category: str  # deposit, withdrawal, trading, account, other
    message: str
    priority: str = "medium"  # low, medium, high, urgent


class AddMessageRequest(BaseModel):
    message: str
    attachments: Optional[List[str]] = None


class UpdateTicketStatusRequest(BaseModel):
    status: str  # open, in_progress, waiting_user, resolved, closed


# ============ User Endpoints ============

@router.post("/tickets")
async def create_ticket(
    request: CreateTicketRequest,
    db: AsyncSession = Depends(get_db),
    user = Depends(get_current_user)
):
    """إنشاء تذكرة دعم جديدة"""
    service = SupportService(db)
    return await service.create_ticket(
        user_id=user.id,
        subject=request.subject,
        category=request.category,
        message=request.message,
        priority=request.priority
    )


@router.get("/tickets/my")
async def get_my_tickets(
    status: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
    user = Depends(get_current_user)
):
    """الحصول على تذاكري"""
    service = SupportService(db)
    return await service.get_user_tickets(user.id, status)


@router.get("/tickets/{ticket_id}")
async def get_ticket_details(
    ticket_id: int,
    db: AsyncSession = Depends(get_db),
    user = Depends(get_current_user)
):
    """الحصول على تفاصيل التذكرة"""
    service = SupportService(db)
    ticket = await service.get_ticket_details(ticket_id, user.id)
    
    if not ticket:
        raise HTTPException(status_code=404, detail="التذكرة غير موجودة")
    
    return ticket


@router.post("/tickets/{ticket_id}/messages")
async def add_message_to_ticket(
    ticket_id: int,
    request: AddMessageRequest,
    db: AsyncSession = Depends(get_db),
    user = Depends(get_current_user)
):
    """إضافة رسالة للتذكرة"""
    service = SupportService(db)
    result = await service.add_message(
        ticket_id=ticket_id,
        sender_id=user.id,
        message=request.message,
        is_admin=False,
        attachments=request.attachments
    )
    
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    
    return result


@router.get("/faq")
async def get_faqs(
    category: Optional[str] = None,
    db: AsyncSession = Depends(get_db)
):
    """الحصول على الأسئلة الشائعة"""
    service = FAQService(db)
    return await service.get_faqs(category)


# ============ Admin Endpoints ============

@router.get("/admin/tickets")
async def get_all_tickets(
    status: Optional[str] = None,
    priority: Optional[str] = None,
    category: Optional[str] = None,
    assigned_to: Optional[int] = None,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
    admin = Depends(get_current_admin)
):
    """الحصول على جميع التذاكر (للأدمن)"""
    service = SupportService(db)
    return await service.get_all_tickets(
        status=status,
        priority=priority,
        category=category,
        assigned_to=assigned_to,
        limit=limit,
        offset=offset
    )


@router.get("/admin/tickets/{ticket_id}")
async def admin_get_ticket_details(
    ticket_id: int,
    db: AsyncSession = Depends(get_db),
    admin = Depends(get_current_admin)
):
    """الحصول على تفاصيل التذكرة (للأدمن)"""
    service = SupportService(db)
    ticket = await service.get_ticket_details(ticket_id)
    
    if not ticket:
        raise HTTPException(status_code=404, detail="التذكرة غير موجودة")
    
    return ticket


@router.post("/admin/tickets/{ticket_id}/messages")
async def admin_add_message(
    ticket_id: int,
    request: AddMessageRequest,
    db: AsyncSession = Depends(get_db),
    admin = Depends(get_current_admin)
):
    """إضافة رد من الأدمن"""
    service = SupportService(db)
    result = await service.add_message(
        ticket_id=ticket_id,
        sender_id=admin.id,
        message=request.message,
        is_admin=True,
        attachments=request.attachments
    )
    
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    
    return result


@router.put("/admin/tickets/{ticket_id}/status")
async def update_ticket_status(
    ticket_id: int,
    request: UpdateTicketStatusRequest,
    db: AsyncSession = Depends(get_db),
    admin = Depends(get_current_admin)
):
    """تحديث حالة التذكرة"""
    service = SupportService(db)
    result = await service.update_ticket_status(
        ticket_id=ticket_id,
        status=request.status,
        admin_id=admin.id
    )
    
    if not result:
        raise HTTPException(status_code=404, detail="التذكرة غير موجودة")
    
    return {"success": True}


@router.post("/admin/tickets/{ticket_id}/assign")
async def assign_ticket(
    ticket_id: int,
    db: AsyncSession = Depends(get_db),
    admin = Depends(get_current_admin)
):
    """تعيين التذكرة للأدمن الحالي"""
    service = SupportService(db)
    result = await service.assign_ticket(ticket_id, admin.id)
    
    if not result:
        raise HTTPException(status_code=404, detail="التذكرة غير موجودة")
    
    return {"success": True}


@router.get("/admin/tickets/unassigned")
async def get_unassigned_tickets(
    db: AsyncSession = Depends(get_db),
    admin = Depends(get_current_admin)
):
    """الحصول على التذاكر غير المعينة"""
    service = SupportService(db)
    return await service.get_unassigned_tickets()


@router.get("/admin/stats")
async def get_support_stats(
    db: AsyncSession = Depends(get_db),
    admin = Depends(get_current_admin)
):
    """إحصائيات الدعم الفني"""
    service = SupportService(db)
    return await service.get_support_stats()
