"""
نظام التقارير المُحسّن - تقارير مخصصة لكل مستخدم
يُستبدل في /opt/asinax/backend/app/api/routes/reports.py
"""

from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime, timedelta
from io import BytesIO
import calendar

from app.core.database import get_db
from app.core.security import get_current_user, get_current_admin
from app.models import User, Balance
from app.models import TradingHistory
from app.services.email_service import email_service

router = APIRouter(prefix="/reports", tags=["Reports"])


# ============ Schemas ============

class ReportSummary(BaseModel):
    period: str
    start_date: str
    end_date: str
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    pnl_percent: float
    portfolio_value: float


class AvailableReport(BaseModel):
    type: str
    period: str
    year: int
    month: Optional[int] = None
    week: Optional[int] = None
    available: bool
    generated_at: Optional[datetime] = None


class UserReportStats(BaseModel):
    user_id: int
    user_email: str
    full_name: str
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    current_balance: float
    vip_level: str


class AdminReportStats(BaseModel):
    total_users: int
    active_users: int
    total_trades: int
    total_pnl: float
    winning_trades: int
    losing_trades: int
    overall_win_rate: float
    top_performers: List[dict]


# ============ Helper Functions ============

async def get_user_trades_for_period(db: AsyncSession, user_id: int, start_date: datetime, end_date: datetime):
    """جلب صفقات مستخدم معين لفترة محددة"""
    result = await db.execute(
        select(TradingHistory)
        .where(TradingHistory.user_id == user_id)
        .where(TradingHistory.executed_at >= start_date)
        .where(TradingHistory.executed_at < end_date)
        .order_by(TradingHistory.executed_at.desc())
    )
    return result.scalars().all()


async def get_all_trades_for_period(db: AsyncSession, start_date: datetime, end_date: datetime):
    """جلب جميع الصفقات لفترة محددة"""
    result = await db.execute(
        select(TradingHistory)
        .where(TradingHistory.executed_at >= start_date)
        .where(TradingHistory.executed_at < end_date)
        .order_by(TradingHistory.executed_at.desc())
    )
    return result.scalars().all()


def calculate_trade_stats(trades):
    """حساب إحصائيات الصفقات"""
    if not trades:
        return {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate": 0.0,
            "total_pnl": 0.0
        }
    
    total_trades = len(trades)
    winning_trades = sum(1 for t in trades if t.pnl and t.pnl > 0)
    losing_trades = sum(1 for t in trades if t.pnl and t.pnl < 0)
    total_pnl = sum(t.pnl or 0 for t in trades)
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0.0
    
    return {
        "total_trades": total_trades,
        "winning_trades": winning_trades,
        "losing_trades": losing_trades,
        "win_rate": round(win_rate, 2),
        "total_pnl": round(total_pnl, 2)
    }


async def get_user_portfolio_data(user: User, db: AsyncSession):
    """جلب بيانات محفظة المستخدم"""
    result = await db.execute(
        select(Balance).where(Balance.user_id == user.id)
    )
    balance = result.scalar_one_or_none()
    
    total_value = float(balance.units or 0) if balance else 0
    total_deposited = float(user.total_deposited or 0)
    
    profit = total_value - total_deposited
    profit_percent = (profit / total_deposited * 100) if total_deposited > 0 else 0
    
    return {
        "total_value": round(total_value, 2),
        "total_deposited": round(total_deposited, 2),
        "profit": round(profit, 2),
        "profit_percent": round(profit_percent, 2)
    }


def get_period_dates(period: str):
    """حساب تواريخ الفترة"""
    now = datetime.utcnow()
    
    if period == "daily":
        start_date = now.replace(hour=0, minute=0, second=0, microsecond=0)
        end_date = start_date + timedelta(days=1)
    elif period == "weekly":
        start_date = now - timedelta(days=now.weekday())
        start_date = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
        end_date = start_date + timedelta(days=7)
    elif period == "monthly":
        start_date = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        if now.month == 12:
            end_date = datetime(now.year + 1, 1, 1)
        else:
            end_date = datetime(now.year, now.month + 1, 1)
    elif period == "yearly":
        start_date = now.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
        end_date = datetime(now.year + 1, 1, 1)
    else:
        raise ValueError("Invalid period")
    
    return start_date, end_date


# ============ User Endpoints ============

@router.get("/my-summary/{period}")
async def get_my_report_summary(
    period: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """الحصول على ملخص تقرير المستخدم الحالي"""
    try:
        start_date, end_date = get_period_dates(period)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid period. Use: daily, weekly, monthly, yearly")
    
    trades = await get_user_trades_for_period(db, current_user.id, start_date, end_date)
    portfolio_data = await get_user_portfolio_data(current_user, db)
    trade_stats = calculate_trade_stats(trades)
    
    return ReportSummary(
        period=period,
        start_date=start_date.strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d"),
        total_trades=trade_stats["total_trades"],
        winning_trades=trade_stats["winning_trades"],
        losing_trades=trade_stats["losing_trades"],
        win_rate=trade_stats["win_rate"],
        total_pnl=trade_stats["total_pnl"],
        pnl_percent=portfolio_data["profit_percent"],
        portfolio_value=portfolio_data["total_value"]
    )


@router.get("/my-trades")
async def get_my_trades(
    period: str = "monthly",
    limit: int = 100,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """جلب صفقات المستخدم الحالي"""
    try:
        start_date, end_date = get_period_dates(period)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid period")
    
    trades = await get_user_trades_for_period(db, current_user.id, start_date, end_date)
    
    return [
        {
            "id": t.id,
            "symbol": t.symbol,
            "side": t.side,
            "quantity": t.quantity,
            "entry_price": t.entry_price,
            "exit_price": t.exit_price,
            "pnl": t.pnl,
            "pnl_percent": t.pnl_percent,
            "executed_at": t.executed_at,
            "closed_at": t.closed_at
        }
        for t in trades[:limit]
    ]


@router.post("/send-to-email")
async def send_report_to_my_email(
    period: str,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """إرسال التقرير لبريد المستخدم الحالي"""
    try:
        start_date, end_date = get_period_dates(period)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid period")
    
    trades = await get_user_trades_for_period(db, current_user.id, start_date, end_date)
    portfolio_data = await get_user_portfolio_data(current_user, db)
    trade_stats = calculate_trade_stats(trades)
    
    # إرسال البريد في الخلفية
    background_tasks.add_task(
        email_service.send_user_report,
        current_user.email,
        current_user.full_name or current_user.email,
        period,
        start_date.strftime("%Y-%m-%d"),
        end_date.strftime("%Y-%m-%d"),
        trade_stats,
        portfolio_data
    )
    
    return {
        "success": True,
        "message": f"سيتم إرسال التقرير إلى {current_user.email}"
    }


# ============ Admin Endpoints ============

@router.get("/admin/overview")
async def get_admin_overview(
    period: str = "monthly",
    current_user: User = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """نظرة عامة للأدمن على جميع التقارير"""
    try:
        start_date, end_date = get_period_dates(period)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid period")
    
    # إحصائيات عامة
    total_users_result = await db.execute(
        select(func.count(User.id)).where(User.is_admin == False)
    )
    total_users = total_users_result.scalar() or 0
    
    active_users_result = await db.execute(
        select(func.count(User.id))
        .where(User.is_admin == False)
        .where(User.status == "active")
    )
    active_users = active_users_result.scalar() or 0
    
    # جميع الصفقات
    all_trades = await get_all_trades_for_period(db, start_date, end_date)
    overall_stats = calculate_trade_stats(all_trades)
    
    # أفضل المستخدمين
    top_performers = []
    users_result = await db.execute(
        select(User).where(User.is_admin == False).limit(100)
    )
    users = users_result.scalars().all()
    
    for user in users:
        user_trades = [t for t in all_trades if t.user_id == user.id]
        if user_trades:
            user_stats = calculate_trade_stats(user_trades)
            top_performers.append({
                "user_id": user.id,
                "email": user.email,
                "full_name": user.full_name,
                "total_pnl": user_stats["total_pnl"],
                "win_rate": user_stats["win_rate"],
                "total_trades": user_stats["total_trades"]
            })
    
    # ترتيب حسب الربح
    top_performers.sort(key=lambda x: x["total_pnl"], reverse=True)
    
    return AdminReportStats(
        total_users=total_users,
        active_users=active_users,
        total_trades=overall_stats["total_trades"],
        total_pnl=overall_stats["total_pnl"],
        winning_trades=overall_stats["winning_trades"],
        losing_trades=overall_stats["losing_trades"],
        overall_win_rate=overall_stats["win_rate"],
        top_performers=top_performers[:10]
    )


@router.get("/admin/user/{user_id}")
async def get_user_report_admin(
    user_id: int,
    period: str = "monthly",
    current_user: User = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """تقرير مستخدم محدد للأدمن"""
    # التحقق من وجود المستخدم
    user_result = await db.execute(select(User).where(User.id == user_id))
    user = user_result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(status_code=404, detail="المستخدم غير موجود")
    
    try:
        start_date, end_date = get_period_dates(period)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid period")
    
    trades = await get_user_trades_for_period(db, user_id, start_date, end_date)
    portfolio_data = await get_user_portfolio_data(user, db)
    trade_stats = calculate_trade_stats(trades)
    
    return UserReportStats(
        user_id=user.id,
        user_email=user.email,
        full_name=user.full_name or "",
        total_trades=trade_stats["total_trades"],
        winning_trades=trade_stats["winning_trades"],
        losing_trades=trade_stats["losing_trades"],
        win_rate=trade_stats["win_rate"],
        total_pnl=trade_stats["total_pnl"],
        current_balance=portfolio_data["total_value"],
        vip_level=user.vip_level or "bronze"
    )


@router.post("/admin/send-to-user/{user_id}")
async def send_report_to_user(
    user_id: int,
    period: str,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """إرسال تقرير لمستخدم محدد"""
    # التحقق من وجود المستخدم
    user_result = await db.execute(select(User).where(User.id == user_id))
    user = user_result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(status_code=404, detail="المستخدم غير موجود")
    
    try:
        start_date, end_date = get_period_dates(period)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid period")
    
    trades = await get_user_trades_for_period(db, user_id, start_date, end_date)
    portfolio_data = await get_user_portfolio_data(user, db)
    trade_stats = calculate_trade_stats(trades)
    
    # إرسال البريد
    background_tasks.add_task(
        email_service.send_user_report,
        user.email,
        user.full_name or user.email,
        period,
        start_date.strftime("%Y-%m-%d"),
        end_date.strftime("%Y-%m-%d"),
        trade_stats,
        portfolio_data
    )
    
    return {
        "success": True,
        "message": f"سيتم إرسال التقرير إلى {user.email}"
    }


@router.post("/admin/send-to-all")
async def send_reports_to_all_users(
    period: str,
    vip_level: Optional[str] = None,
    background_tasks: BackgroundTasks = None,
    current_user: User = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """إرسال تقارير لجميع المستخدمين"""
    query = select(User).where(User.is_admin == False).where(User.status == "active")
    
    if vip_level:
        query = query.where(User.vip_level == vip_level)
    
    result = await db.execute(query)
    users = result.scalars().all()
    
    try:
        start_date, end_date = get_period_dates(period)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid period")
    
    sent_count = 0
    for user in users:
        try:
            trades = await get_user_trades_for_period(db, user.id, start_date, end_date)
            portfolio_data = await get_user_portfolio_data(user, db)
            trade_stats = calculate_trade_stats(trades)
            
            background_tasks.add_task(
                email_service.send_user_report,
                user.email,
                user.full_name or user.email,
                period,
                start_date.strftime("%Y-%m-%d"),
                end_date.strftime("%Y-%m-%d"),
                trade_stats,
                portfolio_data
            )
            sent_count += 1
        except Exception as e:
            print(f"Failed to queue report for {user.email}: {e}")
    
    return {
        "success": True,
        "message": f"تم جدولة إرسال التقارير لـ {sent_count} مستخدم",
        "sent_count": sent_count,
        "total_users": len(users)
    }
