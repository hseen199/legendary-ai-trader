"""
Analytics Routes - مسارات التحليلات المتقدمة
"""
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional
from datetime import datetime, timedelta

from app.core.database import get_db
from app.core.auth import get_current_admin
from app.services.analytics_service import AnalyticsService

router = APIRouter(prefix="/analytics", tags=["Analytics"])


@router.get("/dashboard")
async def get_dashboard_stats(
    db: AsyncSession = Depends(get_db),
    admin = Depends(get_current_admin)
):
    """الحصول على إحصائيات لوحة التحكم الرئيسية"""
    service = AnalyticsService(db)
    return await service.get_dashboard_stats()


@router.get("/performance")
async def get_performance_data(
    days: int = Query(30, ge=1, le=365),
    db: AsyncSession = Depends(get_db),
    admin = Depends(get_current_admin)
):
    """الحصول على بيانات الأداء"""
    service = AnalyticsService(db)
    return await service.get_performance_chart(days)


@router.get("/users")
async def get_user_analytics(
    db: AsyncSession = Depends(get_db),
    admin = Depends(get_current_admin)
):
    """الحصول على تحليلات المستخدمين"""
    service = AnalyticsService(db)
    return await service.get_user_analytics()


@router.get("/trading")
async def get_trading_analytics(
    days: int = Query(30, ge=1, le=365),
    db: AsyncSession = Depends(get_db),
    admin = Depends(get_current_admin)
):
    """الحصول على تحليلات التداول"""
    service = AnalyticsService(db)
    return await service.get_trading_analytics(days)


@router.get("/financial")
async def get_financial_analytics(
    db: AsyncSession = Depends(get_db),
    admin = Depends(get_current_admin)
):
    """الحصول على التحليلات المالية"""
    service = AnalyticsService(db)
    return await service.get_financial_analytics()


@router.get("/agents")
async def get_agent_performance(
    db: AsyncSession = Depends(get_db),
    admin = Depends(get_current_admin)
):
    """الحصول على أداء الوكلاء"""
    service = AnalyticsService(db)
    return await service.get_agent_performance()


@router.get("/comparison")
async def get_market_comparison(
    days: int = Query(30, ge=1, le=365),
    db: AsyncSession = Depends(get_db),
    admin = Depends(get_current_admin)
):
    """مقارنة الأداء مع السوق"""
    service = AnalyticsService(db)
    return await service.compare_with_market(days)


@router.get("/reports/daily")
async def get_daily_report(
    date: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
    admin = Depends(get_current_admin)
):
    """الحصول على التقرير اليومي"""
    service = AnalyticsService(db)
    
    if date:
        report_date = datetime.strptime(date, "%Y-%m-%d")
    else:
        report_date = datetime.utcnow()
    
    return await service.generate_daily_report(report_date)


@router.get("/reports/weekly")
async def get_weekly_report(
    db: AsyncSession = Depends(get_db),
    admin = Depends(get_current_admin)
):
    """الحصول على التقرير الأسبوعي"""
    service = AnalyticsService(db)
    return await service.generate_weekly_report()


@router.get("/reports/monthly")
async def get_monthly_report(
    month: Optional[int] = None,
    year: Optional[int] = None,
    db: AsyncSession = Depends(get_db),
    admin = Depends(get_current_admin)
):
    """الحصول على التقرير الشهري"""
    service = AnalyticsService(db)
    
    now = datetime.utcnow()
    month = month or now.month
    year = year or now.year
    
    return await service.generate_monthly_report(month, year)


@router.get("/heatmap/deposits")
async def get_deposits_heatmap(
    db: AsyncSession = Depends(get_db),
    admin = Depends(get_current_admin)
):
    """خريطة حرارية للإيداعات"""
    service = AnalyticsService(db)
    return await service.get_deposits_heatmap()


@router.get("/heatmap/withdrawals")
async def get_withdrawals_heatmap(
    db: AsyncSession = Depends(get_db),
    admin = Depends(get_current_admin)
):
    """خريطة حرارية للسحوبات"""
    service = AnalyticsService(db)
    return await service.get_withdrawals_heatmap()


@router.get("/predictions")
async def get_predictions(
    db: AsyncSession = Depends(get_db),
    admin = Depends(get_current_admin)
):
    """الحصول على التنبؤات"""
    service = AnalyticsService(db)
    return await service.get_predictions()
