"""
═══════════════════════════════════════════════════════════════════════════════
                    🤖 AGENT WEBHOOK ROUTES - Smart Transparency System
                    مسارات استقبال بيانات الوكيل مع نظام الشفافية الذكية
═══════════════════════════════════════════════════════════════════════════════
"""
from fastapi import APIRouter, Depends, HTTPException, Header
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime
import os
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

# مفتاح API للوكيل
AGENT_API_KEY = os.getenv("AGENT_API_KEY", "asinax_platform_secret_key_2024")

# ═══════════════════════════════════════════════════════════════════════════════
# SCHEMAS
# ═══════════════════════════════════════════════════════════════════════════════
class PositionUpdate(BaseModel):
    """تحديث صفقة"""
    symbol: str
    asset: str
    quantity: float
    value_usdc: float
    pnl_percent: float
    current_price: float
    entry_price: float = 0

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

# نماذج مفلترة للعرض العام
class FilteredPositionUpdate(BaseModel):
    """صفقة مفلترة - بدون معلومات حساسة"""
    symbol: str
    asset: str
    pnl_percent: float
    # لا نعرض: quantity, value_usdc, current_price

class FilteredPortfolioSummary(BaseModel):
    """ملخص المحفظة المفلتر - بدون معلومات حساسة"""
    positions_count: int
    is_active: bool
    last_update: Optional[str]
    # لا نعرض: portfolio_value, usdc_balance, positions details

class FilteredTradeNotification(BaseModel):
    """إشعار صفقة مفلتر"""
    symbol: str
    side: str
    pnl_percent: Optional[float] = None
    timestamp: str
    is_profitable: bool
    # لا نعرض: price, quantity, value_usdc, pnl

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
# ENDPOINTS - استقبال من الوكيل (محمية بمفتاح API)
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
        
        return {
            "success": True,
            "message": "Trade notification received",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error processing trade notification: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ═══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS - قراءة عامة (مفلترة - بدون معلومات حساسة)
# ═══════════════════════════════════════════════════════════════════════════════
@router.get("/agent/portfolio-summary", response_model=FilteredPortfolioSummary)
async def get_portfolio_summary():
    """
    ملخص المحفظة المفلتر - متاح للجميع
    يُظهر فقط أن الوكيل يعمل بدون كشف حجم المحفظة
    """
    if _latest_portfolio:
        return FilteredPortfolioSummary(
            positions_count=_latest_portfolio.positions_count,
            is_active=True,
            last_update=_last_update_time.isoformat() if _last_update_time else None
        )
    
    return FilteredPortfolioSummary(
        positions_count=0,
        is_active=False,
        last_update=None
    )

@router.get("/agent/recent-trades-filtered", response_model=List[FilteredTradeNotification])
async def get_recent_trades_filtered(limit: int = 20):
    """
    آخر الصفقات المفلترة - متاح للجميع
    يُظهر الصفقات بدون قيم حساسة
    """
    trades = _recent_trades[:limit]
    
    return [
        FilteredTradeNotification(
            symbol=t.symbol,
            side=t.side,
            pnl_percent=t.pnl_percent,
            timestamp=t.timestamp,
            is_profitable=(t.pnl or 0) > 0
        )
        for t in trades
    ]

@router.get("/agent/sync-status")
async def get_sync_status():
    """
    حالة المزامنة مع الوكيل - معلومات عامة فقط
    """
    return {
        "has_portfolio_data": _latest_portfolio is not None,
        "last_update": _last_update_time.isoformat() if _last_update_time else None,
        "recent_trades_count": len(_recent_trades),
        # لا نعرض: portfolio_value
        "positions_count": _latest_portfolio.positions_count if _latest_portfolio else 0,
        "is_active": _latest_portfolio is not None and _last_update_time is not None
    }

# ═══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS - للأدمن فقط (كامل البيانات)
# ═══════════════════════════════════════════════════════════════════════════════
@router.get("/agent/latest-portfolio")
async def get_latest_portfolio(x_admin_key: Optional[str] = Header(None)):
    """
    الحصول على آخر تحديث للمحفظة - للأدمن فقط
    
    يتطلب مفتاح أدمن في الـ Header
    """
    # التحقق من مفتاح الأدمن
    admin_key = os.getenv("ADMIN_API_KEY", "asinax_admin_key_2024")
    if x_admin_key != admin_key:
        raise HTTPException(
            status_code=403, 
            detail="Admin access required. This endpoint contains sensitive portfolio information."
        )
    
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
async def get_recent_trades(limit: int = 20, x_admin_key: Optional[str] = Header(None)):
    """
    الحصول على آخر الصفقات الكاملة - للأدمن فقط
    
    يتطلب مفتاح أدمن في الـ Header
    """
    # التحقق من مفتاح الأدمن
    admin_key = os.getenv("ADMIN_API_KEY", "asinax_admin_key_2024")
    if x_admin_key != admin_key:
        raise HTTPException(
            status_code=403, 
            detail="Admin access required. This endpoint contains sensitive trade information."
        )
    
    trades = _recent_trades[:limit]
    
    return {
        "success": True,
        "data": [t.dict() for t in trades],
        "total": len(trades)
    }
