"""
Reports Routes - مسارات API للتقارير
يُضاف إلى /opt/asinax/backend/app/api/routes/reports.py
"""
from datetime import datetime, timedelta
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_
from pydantic import BaseModel
from io import BytesIO
import calendar

from app.core.database import get_db
from app.core.auth import get_current_user
from app.models import User, TradingHistory, NAVHistory, Transaction

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
    month: Optional[int]
    week: Optional[int]
    available: bool
    generated_at: Optional[str]


# ============ Helper Functions ============

async def get_portfolio_data(user: User, db: AsyncSession) -> dict:
    """حساب بيانات المحفظة"""
    from app.services.nav_service import nav_service
    
    current_nav = await nav_service.get_current_nav(db)
    units = float(user.units or 0)
    total_value = units * current_nav
    total_deposited = float(user.total_deposited or 0)
    total_profit = total_value - total_deposited
    profit_percent = (total_profit / total_deposited * 100) if total_deposited > 0 else 0
    
    return {
        "total_value": total_value,
        "total_deposited": total_deposited,
        "total_profit": total_profit,
        "profit_percent": profit_percent,
        "units": units,
        "current_nav": current_nav
    }


async def get_trades_for_period(
    db: AsyncSession,
    start_date: datetime,
    end_date: datetime
) -> List[dict]:
    """جلب الصفقات لفترة معينة"""
    result = await db.execute(
        select(TradingHistory)
        .where(TradingHistory.executed_at >= start_date)
        .where(TradingHistory.executed_at < end_date)
        .order_by(TradingHistory.executed_at.desc())
    )
    
    trades = []
    for t in result.scalars().all():
        trades.append({
            "id": t.id,
            "symbol": t.symbol,
            "side": t.side,
            "entry_price": float(t.price or 0),
            "exit_price": float(t.exit_price or t.price or 0),
            "quantity": float(t.quantity or 0),
            "pnl": float(t.pnl or 0),
            "pnl_percent": float(t.pnl_percent or 0),
            "executed_at": t.executed_at,
            "closed_at": t.executed_at
        })
    
    return trades


async def get_nav_history_for_period(
    db: AsyncSession,
    start_date: datetime,
    end_date: datetime
) -> List[dict]:
    """جلب تاريخ NAV لفترة معينة"""
    result = await db.execute(
        select(NAVHistory)
        .where(NAVHistory.timestamp >= start_date)
        .where(NAVHistory.timestamp < end_date)
        .order_by(NAVHistory.timestamp.asc())
    )
    
    return [
        {"timestamp": n.timestamp, "nav_value": float(n.nav_value)}
        for n in result.scalars().all()
    ]


def calculate_trade_stats(trades: List[dict]) -> dict:
    """حساب إحصائيات الصفقات"""
    total_trades = len(trades)
    winning_trades = sum(1 for t in trades if t.get('pnl', 0) > 0)
    losing_trades = sum(1 for t in trades if t.get('pnl', 0) < 0)
    total_pnl = sum(t.get('pnl', 0) for t in trades)
    
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    
    avg_profit = 0
    avg_loss = 0
    if winning_trades > 0:
        avg_profit = sum(t.get('pnl', 0) for t in trades if t.get('pnl', 0) > 0) / winning_trades
    if losing_trades > 0:
        avg_loss = sum(t.get('pnl', 0) for t in trades if t.get('pnl', 0) < 0) / losing_trades
    
    return {
        "total_trades": total_trades,
        "winning_trades": winning_trades,
        "losing_trades": losing_trades,
        "win_rate": win_rate,
        "total_pnl": total_pnl,
        "avg_profit": avg_profit,
        "avg_loss": avg_loss
    }


# ============ Endpoints ============

@router.get("/monthly")
async def download_monthly_report(
    month: int = Query(None, ge=1, le=12),
    year: int = Query(None, ge=2020),
    language: str = Query("ar", enum=["ar", "en"]),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    تحميل التقرير الشهري PDF
    """
    from app.services.enhanced_report_service import EnhancedReportService
    
    # تحديد الفترة
    now = datetime.utcnow()
    if month is None:
        month = now.month
    if year is None:
        year = now.year
    
    start_date = datetime(year, month, 1)
    if month == 12:
        end_date = datetime(year + 1, 1, 1)
    else:
        end_date = datetime(year, month + 1, 1)
    
    # جلب البيانات
    trades = await get_trades_for_period(db, start_date, end_date)
    nav_history = await get_nav_history_for_period(db, start_date, end_date)
    portfolio_data = await get_portfolio_data(current_user, db)
    
    # إنشاء التقرير
    report_service = EnhancedReportService()
    pdf_bytes = await report_service.generate_detailed_performance_report(
        user_id=current_user.id,
        user_name=current_user.full_name or current_user.email.split('@')[0],
        user_email=current_user.email,
        vip_level=current_user.vip_level or "bronze",
        start_date=start_date,
        end_date=end_date,
        portfolio_data=portfolio_data,
        trades=trades,
        nav_history=nav_history,
        language=language
    )
    
    month_name = calendar.month_name[month]
    filename = f"ASINAX_Report_{year}_{month:02d}_{month_name}.pdf"
    
    return StreamingResponse(
        BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


@router.get("/weekly")
async def download_weekly_report(
    week_offset: int = Query(0, ge=0, le=52, description="0 = الأسبوع الحالي، 1 = الأسبوع الماضي"),
    language: str = Query("ar", enum=["ar", "en"]),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    تحميل التقرير الأسبوعي PDF
    """
    from app.services.enhanced_report_service import EnhancedReportService
    
    # تحديد الفترة
    now = datetime.utcnow()
    end_date = now - timedelta(days=now.weekday()) - timedelta(weeks=week_offset)
    end_date = end_date.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=7)
    start_date = end_date - timedelta(days=7)
    
    # جلب البيانات
    trades = await get_trades_for_period(db, start_date, end_date)
    nav_history = await get_nav_history_for_period(db, start_date, end_date)
    portfolio_data = await get_portfolio_data(current_user, db)
    
    # إنشاء التقرير
    report_service = EnhancedReportService()
    pdf_bytes = await report_service.generate_detailed_performance_report(
        user_id=current_user.id,
        user_name=current_user.full_name or current_user.email.split('@')[0],
        user_email=current_user.email,
        vip_level=current_user.vip_level or "bronze",
        start_date=start_date,
        end_date=end_date,
        portfolio_data=portfolio_data,
        trades=trades,
        nav_history=nav_history,
        language=language
    )
    
    filename = f"ASINAX_Weekly_Report_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.pdf"
    
    return StreamingResponse(
        BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


@router.get("/custom")
async def download_custom_report(
    start_date: str = Query(..., description="تاريخ البداية (YYYY-MM-DD)"),
    end_date: str = Query(..., description="تاريخ النهاية (YYYY-MM-DD)"),
    language: str = Query("ar", enum=["ar", "en"]),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    تحميل تقرير مخصص لفترة معينة
    """
    from app.services.enhanced_report_service import EnhancedReportService
    
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
    
    if end <= start:
        raise HTTPException(status_code=400, detail="End date must be after start date")
    
    if (end - start).days > 365:
        raise HTTPException(status_code=400, detail="Maximum report period is 1 year")
    
    # جلب البيانات
    trades = await get_trades_for_period(db, start, end)
    nav_history = await get_nav_history_for_period(db, start, end)
    portfolio_data = await get_portfolio_data(current_user, db)
    
    # إنشاء التقرير
    report_service = EnhancedReportService()
    pdf_bytes = await report_service.generate_detailed_performance_report(
        user_id=current_user.id,
        user_name=current_user.full_name or current_user.email.split('@')[0],
        user_email=current_user.email,
        vip_level=current_user.vip_level or "bronze",
        start_date=start,
        end_date=end,
        portfolio_data=portfolio_data,
        trades=trades,
        nav_history=nav_history,
        language=language
    )
    
    filename = f"ASINAX_Report_{start_date}_{end_date}.pdf"
    
    return StreamingResponse(
        BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


@router.get("/summary/{period}")
async def get_report_summary(
    period: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    الحصول على ملخص التقرير بدون تحميل PDF
    """
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
        raise HTTPException(status_code=400, detail="Invalid period. Use: daily, weekly, monthly, yearly")
    
    # جلب البيانات
    trades = await get_trades_for_period(db, start_date, end_date)
    portfolio_data = await get_portfolio_data(current_user, db)
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


@router.get("/available")
async def get_available_reports(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    الحصول على قائمة التقارير المتاحة
    """
    now = datetime.utcnow()
    available_reports = []
    
    # التقارير الشهرية (آخر 12 شهر)
    for i in range(12):
        date = now - timedelta(days=30 * i)
        year = date.year
        month = date.month
        
        start_date = datetime(year, month, 1)
        if month == 12:
            end_date = datetime(year + 1, 1, 1)
        else:
            end_date = datetime(year, month + 1, 1)
        
        # التحقق من وجود بيانات
        result = await db.execute(
            select(func.count(TradingHistory.id))
            .where(TradingHistory.executed_at >= start_date)
            .where(TradingHistory.executed_at < end_date)
        )
        has_data = (result.scalar() or 0) > 0
        
        available_reports.append(AvailableReport(
            type="monthly",
            period=f"{calendar.month_name[month]} {year}",
            year=year,
            month=month,
            week=None,
            available=has_data,
            generated_at=None
        ))
    
    # التقارير الأسبوعية (آخر 8 أسابيع)
    for i in range(8):
        end_date = now - timedelta(days=now.weekday()) - timedelta(weeks=i)
        end_date = end_date.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=7)
        start_date = end_date - timedelta(days=7)
        
        # التحقق من وجود بيانات
        result = await db.execute(
            select(func.count(TradingHistory.id))
            .where(TradingHistory.executed_at >= start_date)
            .where(TradingHistory.executed_at < end_date)
        )
        has_data = (result.scalar() or 0) > 0
        
        week_num = start_date.isocalendar()[1]
        
        available_reports.append(AvailableReport(
            type="weekly",
            period=f"Week {week_num}, {start_date.year}",
            year=start_date.year,
            month=None,
            week=week_num,
            available=has_data,
            generated_at=None
        ))
    
    return available_reports


@router.post("/send-email/{report_type}")
async def send_report_by_email(
    report_type: str,
    background_tasks: BackgroundTasks,
    month: int = Query(None, ge=1, le=12),
    year: int = Query(None, ge=2020),
    language: str = Query("ar", enum=["ar", "en"]),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    إرسال التقرير بالبريد الإلكتروني
    """
    if report_type not in ["monthly", "weekly"]:
        raise HTTPException(status_code=400, detail="Invalid report type")
    
    # TODO: إضافة مهمة خلفية لإنشاء وإرسال التقرير
    
    return {
        "success": True,
        "message": f"سيتم إرسال التقرير إلى {current_user.email}"
    }
