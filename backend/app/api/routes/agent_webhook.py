"""
═══════════════════════════════════════════════════════════════════════════════
                    🔗 AGENT WEBHOOK ROUTES
                    مسارات استقبال تحديثات الوكيل
═══════════════════════════════════════════════════════════════════════════════
"""
from fastapi import APIRouter, HTTPException, Header, Depends
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime
import os
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

# مفتاح API للوكيل
AGENT_API_KEY = os.getenv("AGENT_API_KEY", "asinax_platform_secret_key_2024")

# ═══════════════════════════════════════════════════════════════════════════════
# MODELS
# ═══════════════════════════════════════════════════════════════════════════════

class PositionUpdate(BaseModel):
    """تحديث صفقة"""
    symbol: str
    asset: str
    quantity: float
    value_usdc: float
    pnl_percent: float
    current_price: float

class PortfolioUpdate(BaseModel):
    """تحديث المحفظة"""
    portfolio_value: float
    usdc_balance: float
    positions_count: int
    positions: List[PositionUpdate]
    timestamp: str

class TradeNotification(BaseModel):
    """إشعار صفقة"""
    trade_id: str
    symbol: str
    side: str  # BUY or SELL
    price: float
    quantity: float
    value_usdc: float
    pnl: Optional[float] = None
    pnl_percent: Optional[float] = None
    reason: Optional[str] = None
    timestamp: str

# ═══════════════════════════════════════════════════════════════════════════════
# AUTHENTICATION
# ═══════════════════════════════════════════════════════════════════════════════

async def verify_agent_key(x_api_key: Optional[str] = Header(None)):
    """التحقق من مفتاح API الوكيل"""
    if x_api_key != AGENT_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return True

# ═══════════════════════════════════════════════════════════════════════════════
# STORAGE (في الذاكرة - يمكن استبداله بقاعدة بيانات)
# ═══════════════════════════════════════════════════════════════════════════════

_latest_portfolio: Optional[PortfolioUpdate] = None
_recent_trades: List[TradeNotification] = []
_last_update_time: Optional[datetime] = None

# ═══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS - استقبال من الوكيل
# ═══════════════════════════════════════════════════════════════════════════════

@router.post("/agent/portfolio-update")
async def receive_portfolio_update(
    update: PortfolioUpdate,
    _: bool = Depends(verify_agent_key)
):
    """
    استقبال تحديث المحفظة من الوكيل
    
    يُستدعى دورياً من نظام المزامنة على سيرفر الوكيل
    """
    global _latest_portfolio, _last_update_time
    
    try:
        _latest_portfolio = update
        _last_update_time = datetime.now()
        
        logger.info(f"📥 Portfolio update received: ${update.portfolio_value:.2f} | {update.positions_count} positions")
        
        return {
            "success": True,
            "message": "Portfolio update received",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error processing portfolio update: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/agent/trade-notification")
async def receive_trade_notification(
    trade: TradeNotification,
    _: bool = Depends(verify_agent_key)
):
    """
    استقبال إشعار صفقة من الوكيل
    
    يُستدعى عند فتح أو إغلاق صفقة
    """
    global _recent_trades
    
    try:
        # إضافة للقائمة (الاحتفاظ بآخر 100 صفقة)
        _recent_trades.insert(0, trade)
        if len(_recent_trades) > 100:
            _recent_trades = _recent_trades[:100]
        
        logger.info(f"📥 Trade notification: {trade.side} {trade.symbol} @ ${trade.price:.4f}")
        
        # يمكن إضافة إشعارات للمستخدمين هنا
        # await notify_users_about_trade(trade)
        
        return {
            "success": True,
            "message": "Trade notification received",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error processing trade notification: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ═══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS - قراءة من المنصة
# ═══════════════════════════════════════════════════════════════════════════════

@router.get("/agent/latest-portfolio")
async def get_latest_portfolio():
    """
    الحصول على آخر تحديث للمحفظة
    
    يُستخدم من واجهة المنصة لعرض البيانات
    """
    if _latest_portfolio:
        return {
            "success": True,
            "data": _latest_portfolio.dict(),
            "last_update": _last_update_time.isoformat() if _last_update_time else None
        }
    
    return {
        "success": False,
        "message": "No portfolio data available",
        "data": None
    }

@router.get("/agent/recent-trades")
async def get_recent_trades(limit: int = 20):
    """
    الحصول على آخر الصفقات
    
    يُستخدم من واجهة المنصة لعرض الصفقات
    """
    trades = _recent_trades[:limit]
    
    return {
        "success": True,
        "data": [t.dict() for t in trades],
        "total": len(trades)
    }

@router.get("/agent/sync-status")
async def get_sync_status():
    """
    حالة المزامنة مع الوكيل
    """
    return {
        "has_portfolio_data": _latest_portfolio is not None,
        "last_update": _last_update_time.isoformat() if _last_update_time else None,
        "recent_trades_count": len(_recent_trades),
        "portfolio_value": _latest_portfolio.portfolio_value if _latest_portfolio else None,
        "positions_count": _latest_portfolio.positions_count if _latest_portfolio else 0
    }
