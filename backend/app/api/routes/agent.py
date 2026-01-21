"""
═══════════════════════════════════════════════════════════════════════════════
                    🔗 AGENT ROUTES
                    مسارات API للوكيل
═══════════════════════════════════════════════════════════════════════════════
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional
from datetime import datetime

from app.core.database import get_db
from app.services.agent_sync_service import get_agent_sync_service
from app.core.security import get_current_user, get_current_admin
from app.models.user import User

router = APIRouter()


@router.get("/agent/status")
async def get_agent_status(
    current_user: User = Depends(get_current_user)
):
    """الحصول على حالة الوكيل"""
    agent_service = get_agent_sync_service()
    status = await agent_service.get_agent_status()
    return status


@router.get("/agent/portfolio")
async def get_agent_portfolio(
    current_user: User = Depends(get_current_user)
):
    """الحصول على بيانات محفظة الوكيل"""
    agent_service = get_agent_sync_service()
    data = await agent_service.fetch_agent_data()
    
    if not data:
        raise HTTPException(status_code=503, detail="Agent unavailable")
    
    return data


@router.get("/agent/trades")
async def get_agent_trades(
    limit: int = 50,
    current_user: User = Depends(get_current_user)
):
    """الحصول على صفقات الوكيل"""
    agent_service = get_agent_sync_service()
    trades = await agent_service.fetch_agent_trades(limit=limit)
    
    if trades is None:
        raise HTTPException(status_code=503, detail="Agent unavailable")
    
    return {"trades": trades, "total": len(trades)}


@router.get("/agent/performance")
async def get_agent_performance(
    current_user: User = Depends(get_current_user)
):
    """الحصول على أداء الوكيل"""
    agent_service = get_agent_sync_service()
    performance = await agent_service.fetch_agent_performance()
    
    if not performance:
        raise HTTPException(status_code=503, detail="Agent unavailable")
    
    return performance


@router.post("/admin/sync-nav")
async def sync_nav_now(
    db: AsyncSession = Depends(get_db),
    current_admin: User = Depends(get_current_admin)
):
    """مزامنة NAV يدوياً (للأدمن فقط)"""
    agent_service = get_agent_sync_service()
    nav = await agent_service.sync_nav(db)
    
    return {
        "success": True,
        "nav": float(nav),
        "synced_at": datetime.utcnow().isoformat()
    }


@router.get("/nav/current")
async def get_current_nav(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """الحصول على NAV الحالي"""
    agent_service = get_agent_sync_service()
    nav = await agent_service.get_current_nav(db)
    
    return {
        "nav": float(nav),
        "timestamp": datetime.utcnow().isoformat()
    }


@router.get("/nav/history")
async def get_nav_history(
    limit: int = 100,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """الحصول على تاريخ NAV"""
    from sqlalchemy import select
    from app.models.transaction import NAVHistory
    
    result = await db.execute(
        select(NAVHistory)
        .order_by(NAVHistory.timestamp.desc())
        .limit(limit)
    )
    records = result.scalars().all()
    
    return {
        "history": [
            {
                "nav": r.nav_value,
                "total_value": r.total_assets_usd,
                "total_units": r.total_units,
                "timestamp": r.timestamp.isoformat()
            }
            for r in records
        ],
        "total": len(records)
    }
