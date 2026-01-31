"""
═══════════════════════════════════════════════════════════════════════════════
                    📊 DASHBOARD ROUTES - Smart Transparency System
                    لوحة تحكم المستخدم مع نظام الشفافية الذكية
═══════════════════════════════════════════════════════════════════════════════
"""
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from datetime import datetime, timedelta
from typing import Optional
from pydantic import BaseModel
from app.core.database import get_db
from app.core.security import get_current_user
from app.core.config import settings
from app.models import User, Balance, Transaction, WithdrawalRequest, TradingHistory, NAVHistory
from app.schemas import (
    UserDashboard,
    TransactionResponse,
    WithdrawalRequestResponse,
    NAVResponse,
    NAVHistoryItem,
    TradeResponse
)
from app.services import nav_service

router = APIRouter(prefix="/dashboard", tags=["Dashboard"])

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS - ثوابت نظام الشفافية الذكية
# ═══════════════════════════════════════════════════════════════════════════════
TRADE_DELAY_HOURS = 6  # تأخير عرض الصفقات بالساعات
PERFORMANCE_INDEX_BASE = 100  # قيمة مؤشر الأداء الأساسية

# ═══════════════════════════════════════════════════════════════════════════════
# SCHEMAS - نماذج البيانات الجديدة
# ═══════════════════════════════════════════════════════════════════════════════
class PublicPerformanceIndex(BaseModel):
    """مؤشر الأداء العام (بدون كشف حجم المحفظة)"""
    performance_index: float  # يبدأ من 100
    change_24h: float
    change_7d: float
    change_30d: float
    last_updated: datetime

class ActivityPulse(BaseModel):
    """نبض النشاط - يُظهر أن الوكيل يعمل بدون كشف تفاصيل"""
    is_active: bool
    last_trade_time: Optional[datetime]
    trades_today: int
    trades_this_week: int
    win_rate_percent: float
    market_sentiment: str  # "bullish", "bearish", "neutral"

class FilteredTradeResponse(BaseModel):
    """صفقة مفلترة - بدون معلومات حساسة"""
    id: int
    symbol: str
    side: str
    order_type: str
    price: float
    # لا نعرض: quantity, total_value
    pnl_percent: Optional[float]  # نسبة فقط، ليس القيمة
    executed_at: datetime
    is_profitable: bool

class FilteredNAVResponse(BaseModel):
    """NAV مفلتر - بدون حجم المحفظة"""
    current_nav: float
    # لا نعرض: total_assets_usd, total_units
    change_24h: float
    change_7d: float
    change_30d: float

# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS - دوال مساعدة
# ═══════════════════════════════════════════════════════════════════════════════
async def is_user_investor(user: User, db: AsyncSession) -> bool:
    """التحقق إذا كان المستخدم مستثمراً (أودع أموال)"""
    result = await db.execute(
        select(func.sum(Transaction.amount_usd))
        .where(Transaction.user_id == user.id)
        .where(Transaction.type == "deposit")
        .where(Transaction.status == "completed")
    )
    total_deposited = result.scalar() or 0
    return total_deposited > 0

async def calculate_performance_index(db: AsyncSession) -> float:
    """حساب مؤشر الأداء النسبي (يبدأ من 100)"""
    # نحصل على أول NAV مسجل
    result = await db.execute(
        select(NAVHistory.nav_value)
        .order_by(NAVHistory.timestamp.asc())
        .limit(1)
    )
    first_nav = result.scalar() or 1.0
    
    # نحصل على NAV الحالي
    current_nav = await nav_service.get_current_nav(db)
    
    # نحسب المؤشر النسبي
    performance_index = (current_nav / first_nav) * PERFORMANCE_INDEX_BASE
    return round(performance_index, 2)

# ═══════════════════════════════════════════════════════════════════════════════
# PUBLIC ENDPOINTS - متاحة للجميع (بدون معلومات حساسة)
# ═══════════════════════════════════════════════════════════════════════════════
@router.get("/public/performance-index", response_model=PublicPerformanceIndex)
async def get_public_performance_index(
    db: AsyncSession = Depends(get_db)
):
    """
    مؤشر الأداء العام - متاح للجميع
    يُظهر أداء الصندوق كنسبة (يبدأ من 100) بدون كشف حجم المحفظة
    """
    performance_index = await calculate_performance_index(db)
    
    change_24h = await nav_service.get_nav_change(db, 1)
    change_7d = await nav_service.get_nav_change(db, 7)
    change_30d = await nav_service.get_nav_change(db, 30)
    
    return PublicPerformanceIndex(
        performance_index=performance_index,
        change_24h=change_24h,
        change_7d=change_7d,
        change_30d=change_30d,
        last_updated=datetime.utcnow()
    )

@router.get("/public/activity-pulse", response_model=ActivityPulse)
async def get_activity_pulse(
    db: AsyncSession = Depends(get_db)
):
    """
    نبض النشاط - متاح للجميع
    يُظهر أن الوكيل يعمل بدون كشف تفاصيل الصفقات
    """
    now = datetime.utcnow()
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    week_start = today_start - timedelta(days=7)
    
    # آخر صفقة
    result = await db.execute(
        select(TradingHistory.executed_at)
        .order_by(TradingHistory.executed_at.desc())
        .limit(1)
    )
    last_trade_time = result.scalar()
    
    # عدد صفقات اليوم
    result = await db.execute(
        select(func.count(TradingHistory.id))
        .where(TradingHistory.executed_at >= today_start)
    )
    trades_today = result.scalar() or 0
    
    # عدد صفقات الأسبوع
    result = await db.execute(
        select(func.count(TradingHistory.id))
        .where(TradingHistory.executed_at >= week_start)
    )
    trades_this_week = result.scalar() or 0
    
    # نسبة النجاح
    result = await db.execute(
        select(func.count(TradingHistory.id))
        .where(TradingHistory.pnl > 0)
    )
    winning_trades = result.scalar() or 0
    
    result = await db.execute(
        select(func.count(TradingHistory.id))
    )
    total_trades = result.scalar() or 1
    
    win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
    
    # تحديد حالة النشاط
    is_active = last_trade_time and (now - last_trade_time) < timedelta(hours=24)
    
    # تحديد اتجاه السوق بناءً على آخر الصفقات
    result = await db.execute(
        select(TradingHistory.side)
        .order_by(TradingHistory.executed_at.desc())
        .limit(10)
    )
    recent_sides = result.scalars().all()
    buy_count = sum(1 for s in recent_sides if s == "BUY")
    sell_count = len(recent_sides) - buy_count
    
    if buy_count > sell_count + 2:
        market_sentiment = "bullish"
    elif sell_count > buy_count + 2:
        market_sentiment = "bearish"
    else:
        market_sentiment = "neutral"
    
    return ActivityPulse(
        is_active=is_active,
        last_trade_time=last_trade_time,
        trades_today=trades_today,
        trades_this_week=trades_this_week,
        win_rate_percent=round(win_rate, 1),
        market_sentiment=market_sentiment
    )

# ═══════════════════════════════════════════════════════════════════════════════
# USER DASHBOARD - لوحة تحكم المستخدم (بياناته الشخصية فقط)
# ═══════════════════════════════════════════════════════════════════════════════
@router.get("/", response_model=UserDashboard)
async def get_user_dashboard(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get user dashboard with all relevant information"""
    # Get balance
    result = await db.execute(
        select(Balance).where(Balance.user_id == current_user.id)
    )
    balance = result.scalar_one_or_none()
    units = balance.units if balance else 0
    
    # Get current NAV
    current_nav = await nav_service.get_current_nav(db)
    current_value = units * current_nav
    
    # Calculate total deposited
    result = await db.execute(
        select(func.sum(Transaction.amount_usd))
        .where(Transaction.user_id == current_user.id)
        .where(Transaction.type == "deposit")
        .where(Transaction.status == "completed")
    )
    total_deposited = result.scalar() or 0
    
    # Calculate profit/loss
    profit_loss = current_value - total_deposited
    profit_loss_percent = (profit_loss / total_deposited * 100) if total_deposited > 0 else 0
    
    # Check withdrawal eligibility
    can_withdraw = True
    lock_period_ends = None
    if balance and balance.last_deposit_at:
        lock_end = balance.last_deposit_at + timedelta(days=settings.LOCK_PERIOD_DAYS)
        if datetime.utcnow() < lock_end:
            can_withdraw = False
            lock_period_ends = lock_end
    
    # Get recent transactions
    result = await db.execute(
        select(Transaction)
        .where(Transaction.user_id == current_user.id)
        .order_by(Transaction.created_at.desc())
        .limit(10)
    )
    recent_transactions = result.scalars().all()
    
    # Get pending withdrawals
    result = await db.execute(
        select(WithdrawalRequest)
        .where(WithdrawalRequest.user_id == current_user.id)
        .where(WithdrawalRequest.status.in_(["pending_approval", "approved", "processing"]))
    )
    pending_withdrawals = result.scalars().all()
    
    return UserDashboard(
        balance=current_value,
        units=units,
        current_nav=current_nav,
        total_deposited=total_deposited,
        current_value=current_value,
        profit_loss=profit_loss,
        profit_loss_percent=profit_loss_percent,
        can_withdraw=can_withdraw,
        lock_period_ends=lock_period_ends,
        recent_transactions=[
            TransactionResponse(
                id=t.id,
                type=t.type,
                amount_usd=t.amount_usd,
                units_transacted=t.units_transacted,
                nav_at_transaction=t.nav_at_transaction,
                coin=t.coin,
                status=t.status,
                tx_hash=t.tx_hash,
                created_at=t.created_at,
                completed_at=t.completed_at
            )
            for t in recent_transactions
        ],
        pending_withdrawals=[
            WithdrawalRequestResponse(
                id=w.id,
                amount=w.amount,
                units_to_withdraw=w.units_to_withdraw,
                to_address=w.to_address,
                network=w.network,
                coin=w.coin,
                status=w.status,
                requested_at=w.requested_at,
                reviewed_at=w.reviewed_at,
                rejection_reason=w.rejection_reason,
                completed_at=w.completed_at
            )
            for w in pending_withdrawals
        ]
    )

# ═══════════════════════════════════════════════════════════════════════════════
# NAV ENDPOINTS - معلومات NAV (مفلترة)
# ═══════════════════════════════════════════════════════════════════════════════
@router.get("/nav", response_model=FilteredNAVResponse)
async def get_nav_info(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get current NAV and changes
    ملاحظة: تم إخفاء total_assets_usd و total_units للحماية
    """
    current_nav = await nav_service.get_current_nav(db)
    
    # Get changes
    change_24h = await nav_service.get_nav_change(db, 1)
    change_7d = await nav_service.get_nav_change(db, 7)
    change_30d = await nav_service.get_nav_change(db, 30)
    
    # لا نُرجع total_assets_usd و total_units للمستخدمين العاديين
    return FilteredNAVResponse(
        current_nav=current_nav,
        change_24h=change_24h,
        change_7d=change_7d,
        change_30d=change_30d
    )

@router.get("/nav/history", response_model=list[NAVHistoryItem])
async def get_nav_history(
    days: int = 30,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get NAV history for chart"""
    history = await nav_service.get_nav_history(db, days)
    return [
        NAVHistoryItem(
            nav_value=h.nav_value,
            total_assets_usd=None,  # مخفي للحماية
            timestamp=h.timestamp
        )
        for h in history
    ]

# ═══════════════════════════════════════════════════════════════════════════════
# TRADES ENDPOINTS - الصفقات (مفلترة ومؤجلة)
# ═══════════════════════════════════════════════════════════════════════════════
@router.get("/trades", response_model=list[FilteredTradeResponse])
async def get_public_trades(
    limit: int = 50,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get recent bot trades (filtered for security)
    الصفقات مؤجلة 6 ساعات ومفلترة من المعلومات الحساسة
    """
    # تأخير 6 ساعات
    delay_cutoff = datetime.utcnow() - timedelta(hours=TRADE_DELAY_HOURS)
    
    result = await db.execute(
        select(TradingHistory)
        .where(TradingHistory.executed_at <= delay_cutoff)  # فقط الصفقات القديمة
        .order_by(TradingHistory.executed_at.desc())
        .limit(limit)
    )
    trades = result.scalars().all()
    
    return [
        FilteredTradeResponse(
            id=t.id,
            symbol=t.symbol,
            side=t.side,
            order_type=t.order_type,
            price=t.price,
            # لا نعرض: quantity, total_value
            pnl_percent=t.pnl_percent,
            executed_at=t.executed_at,
            is_profitable=(t.pnl or 0) > 0
        )
        for t in trades
    ]

# ═══════════════════════════════════════════════════════════════════════════════
# ADMIN ONLY - للأدمن فقط (كامل البيانات)
# ═══════════════════════════════════════════════════════════════════════════════
@router.get("/admin/nav-full", response_model=NAVResponse)
async def get_nav_info_full(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get full NAV info - ADMIN ONLY
    للأدمن فقط - جميع البيانات
    """
    # التحقق من صلاحيات الأدمن
    if not current_user.is_admin:
        from fastapi import HTTPException
        raise HTTPException(status_code=403, detail="Admin access required")
    
    current_nav = await nav_service.get_current_nav(db)
    total_units = await nav_service.get_total_units(db)
    total_assets = current_nav * total_units
    
    change_24h = await nav_service.get_nav_change(db, 1)
    change_7d = await nav_service.get_nav_change(db, 7)
    change_30d = await nav_service.get_nav_change(db, 30)
    
    return NAVResponse(
        current_nav=current_nav,
        total_assets_usd=total_assets,
        total_units=total_units,
        change_24h=change_24h,
        change_7d=change_7d,
        change_30d=change_30d
    )

@router.get("/admin/trades-full", response_model=list[TradeResponse])
async def get_trades_full(
    limit: int = 50,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get full trades info - ADMIN ONLY
    للأدمن فقط - جميع الصفقات بدون تأخير
    """
    # التحقق من صلاحيات الأدمن
    if not current_user.is_admin:
        from fastapi import HTTPException
        raise HTTPException(status_code=403, detail="Admin access required")
    
    result = await db.execute(
        select(TradingHistory)
        .order_by(TradingHistory.executed_at.desc())
        .limit(limit)
    )
    trades = result.scalars().all()
    
    return [
        TradeResponse(
            id=t.id,
            symbol=t.symbol,
            side=t.side,
            order_type=t.order_type,
            price=t.price,
            quantity=t.quantity,
            total_value=t.total_value,
            pnl=t.pnl,
            pnl_percent=t.pnl_percent,
            executed_at=t.executed_at
        )
        for t in trades
    ]
