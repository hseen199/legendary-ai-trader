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
