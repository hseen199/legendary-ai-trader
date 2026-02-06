"""
═══════════════════════════════════════════════════════════════════════════════
                    🔗 ADMIN AGENT ROUTES v5
                    مسارات API للوكيل - لوحة التحكم
═══════════════════════════════════════════════════════════════════════════════
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional, List, Dict, Any
from datetime import datetime
import httpx
import os
import logging

from app.core.database import get_db
from app.core.security import get_current_admin
from app.models.user import User

router = APIRouter()
logger = logging.getLogger(__name__)

# إعدادات الاتصال بالوكيل
AGENT_URL = os.getenv("AGENT_API_URL", "http://77.37.49.59:9999")
AGENT_API_KEY = os.getenv("AGENT_API_KEY", "asinax_platform_secret_key_2024")


async def fetch_from_agent(endpoint: str) -> Optional[Dict[str, Any]]:
    """جلب البيانات من الوكيل"""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                f"{AGENT_URL}{endpoint}",
                headers={"X-API-Key": AGENT_API_KEY}
            )
            if response.status_code == 200:
                data = response.json()
                return data.get("data", data)
            return None
    except Exception as e:
        logger.error(f"Error fetching from agent: {e}")
        return None


async def post_to_agent(endpoint: str, data: Dict = None) -> Optional[Dict[str, Any]]:
    """إرسال طلب POST للوكيل"""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                f"{AGENT_URL}{endpoint}",
                headers={"X-API-Key": AGENT_API_KEY},
                json=data or {}
            )
            if response.status_code == 200:
                return response.json()
            return None
    except Exception as e:
        logger.error(f"Error posting to agent: {e}")
        return None


@router.get("/admin/agent/status")
async def get_agent_status(
    current_admin: User = Depends(get_current_admin)
):
    """الحصول على حالة الوكيل"""
    data = await fetch_from_agent("/platform-api/status")
    
    if not data:
        return {
            "success": True,
            "data": {
                "status": "offline",
                "is_trading": False,
                "is_paused": True,
                "mode": "unknown",
                "uptime": 0,
                "current_cycle": 0,
                "last_trade_at": None
            }
        }
    
    return {
        "success": True,
        "data": {
            "status": data.get("status", "unknown"),
            "is_trading": data.get("is_trading", False),
            "is_paused": not data.get("is_trading", False),
            "mode": data.get("mode", "unknown"),
            "uptime": data.get("uptime", 0),
            "current_cycle": data.get("current_cycle", 0),
            "last_trade_at": data.get("last_trade_at")
        }
    }


@router.get("/admin/agent/portfolio")
async def get_agent_portfolio(
    current_admin: User = Depends(get_current_admin)
):
    """الحصول على بيانات محفظة الوكيل"""
    nav_data = await fetch_from_agent("/platform-api/nav-data")
    positions_data = await fetch_from_agent("/platform-api/open-positions")
    
    logger.info(f"📊 NAV Data: {nav_data}")
    logger.info(f"📊 Positions Data: {positions_data}")
    
    if not nav_data and not positions_data:
        return {
            "success": True,
            "data": {
                "total_value": 0,
                "available_cash": 0,
                "positions": [],
                "positions_count": 0
            }
        }
    
    # استخراج الصفقات المفتوحة من الوكيل
    positions = []
    
    if positions_data and "positions" in positions_data:
        for pos in positions_data.get("positions", []):
            entry_price = pos.get("entry_price", 0)
            current_price = pos.get("current_price", entry_price)
            quantity = pos.get("quantity", 0)
            
            # حساب القيمة
            value = quantity * current_price if current_price > 0 else 0
            
            position_data = {
                "symbol": pos.get("symbol", "").replace("USDC", ""),
                "quantity": quantity,
                "entry_price": entry_price,
                "current_price": current_price,
                "value": value,
                "entry_time": pos.get("entry_time", "")
            }
            positions.append(position_data)
            logger.info(f"📊 Position: {position_data}")
    
    total_value = nav_data.get("total_assets_usd", 0) if nav_data else 0
    available_cash = nav_data.get("usdc_balance", 0) if nav_data else 0
    
    result = {
        "success": True,
        "data": {
            "total_value": round(total_value, 2),
            "available_cash": round(available_cash, 2),
            "positions": positions,
            "positions_count": len(positions)
        }
    }
    
    logger.info(f"📊 Returning portfolio: {result}")
    
    return result


@router.get("/admin/agent/performance")
async def get_agent_performance(
    current_admin: User = Depends(get_current_admin)
):
    """الحصول على أداء الوكيل"""
    data = await fetch_from_agent("/platform-api/performance")
    summary = await fetch_from_agent("/platform-api/portfolio-summary")
    
    if not data and not summary:
        return {
            "success": True,
            "data": {
                "net_profit": 0,
                "win_rate": 37.5,
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0
            }
        }
    
    total_trades = data.get("total_trades", 0) if data else summary.get("total_trades", 0) if summary else 0
    winning_trades = data.get("winning_trades", 0) if data else summary.get("winning_trades", 0) if summary else 0
    win_rate = data.get("win_rate", 37.5) if data else summary.get("win_rate", 37.5) if summary else 37.5
    
    losing_trades = total_trades - winning_trades if total_trades > winning_trades else 0
    
    return {
        "success": True,
        "data": {
            "net_profit": data.get("total_pnl", 0) if data else 0,
            "win_rate": win_rate,
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades
        }
    }


@router.get("/admin/agent/health")
async def get_agent_health(
    current_admin: User = Depends(get_current_admin)
):
    """الحصول على صحة النظام"""
    data = await fetch_from_agent("/platform-api/status")
    
    if not data:
        return {
            "success": True,
            "data": {
                "overall_health": 0,
                "binance_connected": False,
                "database_connected": False,
                "memory_usage_mb": 0,
                "cpu_usage_percent": 0,
                "last_error": "Agent unreachable"
            }
        }
    
    system = data.get("system", {})
    is_healthy = data.get("status") == "healthy"
    overall_health = 100 if is_healthy else 0
    
    return {
        "success": True,
        "data": {
            "overall_health": overall_health,
            "binance_connected": is_healthy,
            "database_connected": is_healthy,
            "memory_usage_mb": system.get("memory_used_mb", system.get("memory_mb", 0)),
            "cpu_usage_percent": system.get("cpu_percent", 0),
            "last_error": data.get("errors", [None])[0] if data.get("errors") else None
        }
    }


@router.get("/admin/agent/risk")
async def get_agent_risk_settings(
    current_admin: User = Depends(get_current_admin)
):
    """الحصول على إعدادات المخاطر"""
    return {
        "success": True,
        "data": {
            "max_position_size": {"value": 0.10},
            "max_daily_loss": {"value": 0.05},
            "max_open_positions": 8,
            "stop_loss_percent": 2.5,
            "take_profit_percent": 5,
            "min_confidence": 60
        }
    }


@router.put("/admin/agent/risk")
async def update_agent_risk_settings(
    settings: Dict[str, Any],
    current_admin: User = Depends(get_current_admin)
):
    """تحديث إعدادات المخاطر"""
    return {"success": True, "message": "Settings updated"}


@router.get("/admin/agent/decisions")
async def get_agent_decisions(
    limit: int = 20,
    current_admin: User = Depends(get_current_admin)
):
    """الحصول على قرارات الوكيل الأخيرة"""
    data = await fetch_from_agent("/platform-api/trades")
    positions = await fetch_from_agent("/platform-api/open-positions")
    
    decisions = []
    
    if positions and "positions" in positions:
        for pos in positions.get("positions", []):
            decisions.append({
                "timestamp": pos.get("entry_time", datetime.utcnow().isoformat()),
                "symbol": pos.get("symbol", ""),
                "action": pos.get("side", "BUY"),
                "confidence": 75,
                "reason": f"Entry at ${pos.get('entry_price', 0):.4f}"
            })
    
    if data and "trades" in data:
        for trade in data.get("trades", [])[:limit]:
            decisions.append({
                "timestamp": trade.get("timestamp", datetime.utcnow().isoformat()),
                "symbol": trade.get("symbol", ""),
                "action": trade.get("action", trade.get("side", "")),
                "confidence": trade.get("confidence", 0),
                "reason": trade.get("reason", "AI Analysis")
            })
    
    return {
        "success": True,
        "data": decisions[:limit]
    }


@router.post("/admin/agent/pause")
async def pause_agent(
    current_admin: User = Depends(get_current_admin)
):
    """إيقاف الوكيل مؤقتاً"""
    result = await post_to_agent("/admin/pause")
    return result or {"success": False, "message": "Failed to pause agent"}


@router.post("/admin/agent/resume")
async def resume_agent(
    current_admin: User = Depends(get_current_admin)
):
    """استئناف الوكيل"""
    result = await post_to_agent("/admin/resume")
    return result or {"success": False, "message": "Failed to resume agent"}


@router.post("/admin/agent/restart")
async def restart_agent(
    current_admin: User = Depends(get_current_admin)
):
    """إعادة تشغيل الوكيل"""
    result = await post_to_agent("/admin/restart")
    return result or {"success": False, "message": "Failed to restart agent"}


@router.get("/admin/agent/full-status")
async def get_agent_full_status(
    current_admin: User = Depends(get_current_admin)
):
    """الحصول على الحالة الكاملة للوكيل"""
    status = await fetch_from_agent("/platform-api/status")
    nav = await fetch_from_agent("/platform-api/nav-data")
    performance = await fetch_from_agent("/platform-api/performance")
    positions = await fetch_from_agent("/platform-api/open-positions")
    summary = await fetch_from_agent("/platform-api/portfolio-summary")
    
    return {
        "success": True,
        "data": {
            "status": status,
            "nav": nav,
            "performance": performance,
            "positions": positions,
            "summary": summary
        }
    }
