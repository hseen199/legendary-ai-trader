"""
═══════════════════════════════════════════════════════════════════════════════
                    🔔 WEBHOOK ROUTES
                    مسارات Webhook لاستقبال بيانات الوكيل
═══════════════════════════════════════════════════════════════════════════════
"""
from fastapi import APIRouter, Depends, HTTPException, Header
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime
import os
import logging

from app.core.database import get_db
from app.models.transaction import TradingHistory, NAVHistory
from app.models.user import User
from sqlalchemy import func, select

logger = logging.getLogger(__name__)

router = APIRouter()

# مفتاح API للوكيل
AGENT_API_KEY = os.getenv("AGENT_API_KEY", "asinax_platform_secret_key_2024")


class TradeWebhook(BaseModel):
    """نموذج استقبال الصفقة من الوكيل"""
    symbol: str
    side: str  # BUY or SELL
    order_type: str  # MARKET, LIMIT
    price: float
    quantity: float
    total_value: float
    order_id: Optional[str] = None
    pnl: Optional[float] = None
    pnl_percent: Optional[float] = None
    executed_at: Optional[datetime] = None


class NAVWebhook(BaseModel):
    """نموذج تحديث NAV من الوكيل"""
    nav_value: float
    total_assets_usd: float
    total_units: float


class BatchTradesWebhook(BaseModel):
    """نموذج استقبال مجموعة صفقات"""
    trades: List[TradeWebhook]


def verify_agent_key(x_api_key: str = Header(...)):
    """التحقق من مفتاح API الوكيل"""
    if x_api_key != AGENT_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return True


@router.post("/webhook/trade")
async def receive_trade(
    trade: TradeWebhook,
    db: AsyncSession = Depends(get_db),
    _: bool = Depends(verify_agent_key)
):
    """استقبال صفقة جديدة من الوكيل"""
    try:
        new_trade = TradingHistory(
            symbol=trade.symbol,
            side=trade.side,
            order_type=trade.order_type,
            price=trade.price,
            quantity=trade.quantity,
            total_value=trade.total_value,
            order_id=trade.order_id,
            pnl=trade.pnl,
            pnl_percent=trade.pnl_percent,
            executed_at=trade.executed_at or datetime.utcnow()
        )
        
        db.add(new_trade)
        await db.commit()
        await db.refresh(new_trade)
        
        logger.info(f"Trade received: {trade.symbol} {trade.side} @ {trade.price}")
        
        return {
            "success": True,
            "trade_id": new_trade.id,
            "message": "Trade recorded successfully"
        }
        
    except Exception as e:
        logger.error(f"Error recording trade: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/webhook/trades/batch")
async def receive_trades_batch(
    data: BatchTradesWebhook,
    db: AsyncSession = Depends(get_db),
    _: bool = Depends(verify_agent_key)
):
    """استقبال مجموعة صفقات من الوكيل"""
    try:
        trade_ids = []
        
        for trade in data.trades:
            new_trade = TradingHistory(
                symbol=trade.symbol,
                side=trade.side,
                order_type=trade.order_type,
                price=trade.price,
                quantity=trade.quantity,
                total_value=trade.total_value,
                order_id=trade.order_id,
                pnl=trade.pnl,
                pnl_percent=trade.pnl_percent,
                executed_at=trade.executed_at or datetime.utcnow()
            )
            db.add(new_trade)
            trade_ids.append(new_trade)
        
        await db.commit()
        
        logger.info(f"Batch trades received: {len(data.trades)} trades")
        
        return {
            "success": True,
            "trades_count": len(data.trades),
            "message": "Trades recorded successfully"
        }
        
    except Exception as e:
        logger.error(f"Error recording batch trades: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/webhook/nav")
async def receive_nav_update(
    nav_data: NAVWebhook,
    db: AsyncSession = Depends(get_db),
    _: bool = Depends(verify_agent_key)
):
    """استقبال تحديث NAV من الوكيل"""
    try:
        new_nav = NAVHistory(
            nav_value=nav_data.nav_value,
            total_assets_usd=nav_data.total_assets_usd,
            total_units=nav_data.total_units
        )
        
        db.add(new_nav)
        await db.commit()
        
        logger.info(f"NAV updated: {nav_data.nav_value}")
        
        return {
            "success": True,
            "nav": nav_data.nav_value,
            "message": "NAV updated successfully"
        }
        
    except Exception as e:
        logger.error(f"Error updating NAV: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/webhook/platform-stats")
async def get_platform_stats(
    db: AsyncSession = Depends(get_db),
    _: bool = Depends(verify_agent_key)
):
    """
    جلب إحصائيات المنصة للوكيل
    يُستخدم لحساب NAV بناءً على إجمالي الوحدات
    """
    try:
        # حساب إجمالي الوحدات من جميع المستخدمين النشطين
        total_units_result = await db.execute(
            select(func.coalesce(func.sum(User.units), 0.0))
            .where(User.is_active == True)
        )
        total_units = total_units_result.scalar() or 0.0
        
        # حساب عدد المستخدمين النشطين
        active_users_result = await db.execute(
            select(func.count(User.id))
            .where(User.is_active == True)
            .where(User.units > 0)
        )
        active_investors = active_users_result.scalar() or 0
        
        # حساب إجمالي الإيداعات
        total_deposits_result = await db.execute(
            select(func.coalesce(func.sum(User.total_deposited), 0.0))
        )
        total_deposits = total_deposits_result.scalar() or 0.0
        
        # حساب إجمالي السحوبات
        total_withdrawals_result = await db.execute(
            select(func.coalesce(func.sum(User.total_withdrawn), 0.0))
        )
        total_withdrawals = total_withdrawals_result.scalar() or 0.0
        
        # جلب آخر قيمة NAV
        latest_nav_result = await db.execute(
            select(NAVHistory)
            .order_by(NAVHistory.created_at.desc())
            .limit(1)
        )
        latest_nav = latest_nav_result.scalar_one_or_none()
        
        current_nav = latest_nav.nav_value if latest_nav else 1.0
        total_assets = latest_nav.total_assets_usd if latest_nav else 0.0
        
        logger.info(f"Platform stats requested: {total_units} units, {active_investors} investors")
        
        return {
            "success": True,
            "total_units": round(total_units, 6),
            "active_investors": active_investors,
            "total_deposits": round(total_deposits, 2),
            "total_withdrawals": round(total_withdrawals, 2),
            "current_nav": round(current_nav, 6),
            "total_assets_usd": round(total_assets, 2),
            "net_deposits": round(total_deposits - total_withdrawals, 2),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting platform stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))
