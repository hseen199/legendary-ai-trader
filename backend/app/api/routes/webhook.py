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
from app.models.fund_ledger import FundLedger, LedgerEntryType
from app.services.ledger_service import LedgerService
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
            .order_by(NAVHistory.timestamp.desc())
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


# ═══════════════════════════════════════════════════════════════════════════════
#                    📊 TRADING PNL ENDPOINT
#                    استقبال أرباح/خسائر التداول من الوكيل
# ═══════════════════════════════════════════════════════════════════════════════

class TradingPnLWebhook(BaseModel):
    """نموذج استقبال أرباح/خسائر التداول من الوكيل"""
    pnl_amount: float  # مبلغ الربح (+) أو الخسارة (-)
    trade_id: Optional[str] = None  # معرف الصفقة (اختياري)
    symbol: Optional[str] = None  # رمز العملة (اختياري)
    description: Optional[str] = None  # وصف إضافي


class FeeDeductionWebhook(BaseModel):
    """نموذج خصم الرسوم"""
    fee_amount: float  # مبلغ الرسوم
    fee_type: str  # نوع الرسوم: performance, management, trading
    description: Optional[str] = None


@router.post("/webhook/trading-pnl")
async def receive_trading_pnl(
    data: TradingPnLWebhook,
    db: AsyncSession = Depends(get_db),
    _: bool = Depends(verify_agent_key)
):
    """
    استقبال أرباح/خسائر التداول من الوكيل
    
    هذا الـ endpoint يُستخدم لتسجيل الأرباح والخسائر في سجل المحاسبة
    بحيث يتم حساب NAV بدقة بناءً على الأداء الفعلي للتداول
    
    - pnl_amount موجب = ربح
    - pnl_amount سالب = خسارة
    """
    try:
        ledger = LedgerService(db)
        
        # تسجيل الربح أو الخسارة
        ledger_entry = await ledger.record_trading_pnl(
            pnl=data.pnl_amount,
            trade_id=data.trade_id,
            symbol=data.symbol,
            description=data.description or f"Trading PnL: {data.symbol or 'N/A'}"
        )
        
        # حساب NAV الجديد
        fund_summary = await ledger.get_fund_summary()
        
        pnl_type = "PROFIT" if data.pnl_amount >= 0 else "LOSS"
        logger.info(f"Trading PnL recorded: {pnl_type} ${abs(data.pnl_amount):.2f} | New NAV: ${fund_summary['current_nav']:.6f}")
        
        return {
            "success": True,
            "entry_id": ledger_entry.id,
            "pnl_type": pnl_type,
            "pnl_amount": data.pnl_amount,
            "new_nav": fund_summary["current_nav"],
            "total_capital": fund_summary["total_capital"],
            "total_pnl": fund_summary["total_pnl"],
            "message": f"Trading {pnl_type.lower()} recorded successfully"
        }
        
    except Exception as e:
        logger.error(f"Error recording trading PnL: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/webhook/fee-deduction")
async def receive_fee_deduction(
    data: FeeDeductionWebhook,
    db: AsyncSession = Depends(get_db),
    _: bool = Depends(verify_agent_key)
):
    """
    استقبال خصم الرسوم من الوكيل
    
    أنواع الرسوم:
    - performance: رسوم الأداء (نسبة من الأرباح)
    - management: رسوم الإدارة (نسبة سنوية)
    - trading: رسوم التداول (عمولات)
    """
    try:
        ledger = LedgerService(db)
        
        # تسجيل الرسوم
        ledger_entry = await ledger.record_fee(
            fee_amount=data.fee_amount,
            fee_type=data.fee_type,
            description=data.description or f"{data.fee_type.capitalize()} fee"
        )
        
        # حساب NAV الجديد
        fund_summary = await ledger.get_fund_summary()
        
        logger.info(f"Fee deduction recorded: ${data.fee_amount:.2f} ({data.fee_type}) | New NAV: ${fund_summary['current_nav']:.6f}")
        
        return {
            "success": True,
            "entry_id": ledger_entry.id,
            "fee_type": data.fee_type,
            "fee_amount": data.fee_amount,
            "new_nav": fund_summary["current_nav"],
            "total_fees": fund_summary["total_fees"],
            "message": "Fee deduction recorded successfully"
        }
        
    except Exception as e:
        logger.error(f"Error recording fee deduction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/webhook/fund-summary")
async def get_fund_summary(
    db: AsyncSession = Depends(get_db),
    _: bool = Depends(verify_agent_key)
):
    """
    جلب ملخص الصندوق الكامل من سجل المحاسبة
    
    يُستخدم من قبل الوكيل لمعرفة:
    - رأس المال المُدار
    - إجمالي الأرباح/الخسائر
    - NAV الحالي
    - إجمالي الوحدات
    """
    try:
        ledger = LedgerService(db)
        fund_summary = await ledger.get_fund_summary()
        
        logger.info(f"Fund summary requested: NAV=${fund_summary['current_nav']:.6f}, Capital=${fund_summary['total_capital']:.2f}")
        
        return {
            "success": True,
            **fund_summary,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting fund summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/webhook/ledger-history")
async def get_ledger_history(
    limit: int = 100,
    entry_type: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
    _: bool = Depends(verify_agent_key)
):
    """
    جلب سجل المحاسبة
    
    يمكن تصفية النتائج حسب نوع القيد:
    - INITIAL: رأس المال الأولي
    - DEPOSIT: إيداع
    - WITHDRAWAL: سحب
    - TRADE_PNL: ربح/خسارة تداول
    - FEE: رسوم
    - ADJUSTMENT: تعديل
    """
    try:
        ledger = LedgerService(db)
        entries = await ledger.get_ledger_entries(limit=limit, entry_type=entry_type)
        
        return {
            "success": True,
            "count": len(entries),
            "entries": [
                {
                    "id": e.id,
                    "type": e.entry_type.value,
                    "amount": e.amount,
                    "units_delta": e.units_delta,
                    "nav_at_entry": e.nav_at_entry,
                    "running_total_capital": e.running_total_capital,
                    "running_total_units": e.running_total_units,
                    "description": e.description,
                    "timestamp": e.timestamp.isoformat()
                }
                for e in entries
            ]
        }
        
    except Exception as e:
        logger.error(f"Error getting ledger history: {e}")
        raise HTTPException(status_code=500, detail=str(e))
