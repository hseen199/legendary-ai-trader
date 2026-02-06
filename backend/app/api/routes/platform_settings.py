"""
إعدادات المنصة الشاملة
يُضاف إلى /opt/asinax/backend/app/api/routes/platform_settings.py
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime
import json

from app.core.database import get_db
from app.core.security import get_current_admin

router = APIRouter(prefix="/platform-settings", tags=["Platform Settings"])


# ============ Schemas ============

class TradingSettings(BaseModel):
    auto_trading_enabled: bool = True
    max_daily_trades: int = 100
    max_position_size: float = 10000.0
    stop_loss_percentage: float = 5.0
    take_profit_percentage: float = 10.0
    allowed_symbols: List[str] = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]


class DepositSettings(BaseModel):
    min_deposit_amount: float = 10.0
    max_deposit_amount: float = 100000.0
    manual_deposit_enabled: bool = True
    crypto_deposit_enabled: bool = True
    supported_currencies: List[str] = ["USDT", "BTC", "ETH"]
    deposit_confirmation_required: bool = True


class WithdrawalSettings(BaseModel):
    min_withdrawal_amount: float = 10.0
    max_withdrawal_amount: float = 50000.0
    daily_withdrawal_limit: float = 10000.0
    withdrawal_fee_percentage: float = 1.0
    auto_approve_threshold: float = 500.0
    require_2fa_for_withdrawal: bool = True
    require_email_confirmation: bool = True


class VIPSettings(BaseModel):
    levels: Dict[str, Dict[str, Any]] = {
        "bronze": {
            "name_ar": "برونزي",
            "name_en": "Bronze",
            "min_deposit": 0,
            "max_deposit": 999,
            "profit_share": 70,
            "withdrawal_fee": 2.0,
            "priority_support": False
        },
        "silver": {
            "name_ar": "فضي",
            "name_en": "Silver",
            "min_deposit": 1000,
            "max_deposit": 4999,
            "profit_share": 75,
            "withdrawal_fee": 1.5,
            "priority_support": False
        },
        "gold": {
            "name_ar": "ذهبي",
            "name_en": "Gold",
            "min_deposit": 5000,
            "max_deposit": 19999,
            "profit_share": 80,
            "withdrawal_fee": 1.0,
            "priority_support": True
        },
        "platinum": {
            "name_ar": "بلاتيني",
            "name_en": "Platinum",
            "min_deposit": 20000,
            "max_deposit": 99999,
            "profit_share": 85,
            "withdrawal_fee": 0.5,
            "priority_support": True
        },
        "diamond": {
            "name_ar": "ماسي",
            "name_en": "Diamond",
            "min_deposit": 100000,
            "max_deposit": float('inf'),
            "profit_share": 90,
            "withdrawal_fee": 0,
            "priority_support": True
        }
    }


class NotificationSettings(BaseModel):
    email_notifications_enabled: bool = True
    admin_email: str = "admin@asinax.com"
    notify_on_deposit: bool = True
    notify_on_withdrawal: bool = True
    notify_on_trade: bool = True
    notify_on_new_user: bool = True


class MaintenanceSettings(BaseModel):
    maintenance_mode: bool = False
    maintenance_message: str = "المنصة تحت الصيانة حالياً"
    expected_end_time: Optional[datetime] = None


class AllPlatformSettings(BaseModel):
    trading: TradingSettings
    deposit: DepositSettings
    withdrawal: WithdrawalSettings
    vip: VIPSettings
    notification: NotificationSettings
    maintenance: MaintenanceSettings


# ============ In-Memory Settings Storage (Replace with DB) ============

_platform_settings = {
    "trading": TradingSettings().dict(),
    "deposit": DepositSettings().dict(),
    "withdrawal": WithdrawalSettings().dict(),
    "vip": VIPSettings().dict(),
    "notification": NotificationSettings().dict(),
    "maintenance": MaintenanceSettings().dict()
}


# ============ Endpoints ============

@router.get("/all")
async def get_all_settings(
    current_user = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """الحصول على جميع إعدادات المنصة"""
    return _platform_settings


@router.get("/trading")
async def get_trading_settings(
    current_user = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """الحصول على إعدادات التداول"""
    return _platform_settings["trading"]


@router.put("/trading")
async def update_trading_settings(
    settings: TradingSettings,
    current_user = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """تحديث إعدادات التداول"""
    _platform_settings["trading"] = settings.dict()
    return {"success": True, "message": "تم تحديث إعدادات التداول", "settings": settings}


@router.get("/deposit")
async def get_deposit_settings(
    current_user = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """الحصول على إعدادات الإيداع"""
    return _platform_settings["deposit"]


@router.put("/deposit")
async def update_deposit_settings(
    settings: DepositSettings,
    current_user = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """تحديث إعدادات الإيداع"""
    _platform_settings["deposit"] = settings.dict()
    return {"success": True, "message": "تم تحديث إعدادات الإيداع", "settings": settings}


@router.get("/withdrawal")
async def get_withdrawal_settings(
    current_user = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """الحصول على إعدادات السحب"""
    return _platform_settings["withdrawal"]


@router.put("/withdrawal")
async def update_withdrawal_settings(
    settings: WithdrawalSettings,
    current_user = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """تحديث إعدادات السحب"""
    _platform_settings["withdrawal"] = settings.dict()
    return {"success": True, "message": "تم تحديث إعدادات السحب", "settings": settings}


@router.get("/vip")
async def get_vip_settings(
    current_user = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """الحصول على إعدادات VIP"""
    return _platform_settings["vip"]


@router.put("/vip")
async def update_vip_settings(
    settings: VIPSettings,
    current_user = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """تحديث إعدادات VIP"""
    _platform_settings["vip"] = settings.dict()
    return {"success": True, "message": "تم تحديث إعدادات VIP", "settings": settings}


@router.get("/notification")
async def get_notification_settings(
    current_user = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """الحصول على إعدادات الإشعارات"""
    return _platform_settings["notification"]


@router.put("/notification")
async def update_notification_settings(
    settings: NotificationSettings,
    current_user = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """تحديث إعدادات الإشعارات"""
    _platform_settings["notification"] = settings.dict()
    return {"success": True, "message": "تم تحديث إعدادات الإشعارات", "settings": settings}


@router.get("/maintenance")
async def get_maintenance_settings(
    current_user = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """الحصول على إعدادات الصيانة"""
    return _platform_settings["maintenance"]


@router.put("/maintenance")
async def update_maintenance_settings(
    settings: MaintenanceSettings,
    current_user = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """تحديث إعدادات الصيانة"""
    _platform_settings["maintenance"] = settings.dict()
    return {"success": True, "message": "تم تحديث إعدادات الصيانة", "settings": settings}


@router.post("/maintenance/toggle")
async def toggle_maintenance(
    enable: bool,
    message: Optional[str] = None,
    current_user = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """تفعيل/تعطيل وضع الصيانة"""
    _platform_settings["maintenance"]["maintenance_mode"] = enable
    if message:
        _platform_settings["maintenance"]["maintenance_message"] = message
    
    status = "تفعيل" if enable else "تعطيل"
    return {
        "success": True,
        "message": f"تم {status} وضع الصيانة",
        "maintenance_mode": enable
    }


@router.post("/auto-withdrawal/toggle")
async def toggle_auto_withdrawal(
    enable: bool,
    threshold: Optional[float] = None,
    current_user = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """تفعيل/تعطيل الموافقة التلقائية على السحب"""
    if threshold is not None:
        _platform_settings["withdrawal"]["auto_approve_threshold"] = threshold
    
    status = "تفعيل" if enable else "تعطيل"
    return {
        "success": True,
        "message": f"تم {status} الموافقة التلقائية على السحب",
        "auto_approve_enabled": enable,
        "threshold": _platform_settings["withdrawal"]["auto_approve_threshold"]
    }


@router.post("/trading/toggle")
async def toggle_auto_trading(
    enable: bool,
    current_user = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """تفعيل/تعطيل التداول التلقائي"""
    _platform_settings["trading"]["auto_trading_enabled"] = enable
    
    status = "تفعيل" if enable else "تعطيل"
    return {
        "success": True,
        "message": f"تم {status} التداول التلقائي",
        "auto_trading_enabled": enable
    }
