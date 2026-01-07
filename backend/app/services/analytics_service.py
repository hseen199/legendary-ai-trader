"""
Analytics Service - خدمة التحليلات المتقدمة
تحليلات وإحصائيات شاملة للوحة تحكم الأدمن
"""
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, or_, desc
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """مقاييس الأداء"""
    total_return: float
    daily_return: float
    weekly_return: float
    monthly_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    profitable_trades: int
    losing_trades: int


@dataclass
class UserAnalytics:
    """تحليلات المستخدمين"""
    total_users: int
    active_users: int
    new_users_today: int
    new_users_week: int
    new_users_month: int
    retention_rate: float
    avg_deposit: float
    total_deposited: float
    total_withdrawn: float
    churn_rate: float


@dataclass
class RevenueAnalytics:
    """تحليلات الإيرادات"""
    total_fees: float
    fees_today: float
    fees_week: float
    fees_month: float
    performance_fees: float
    withdrawal_fees: float
    avg_fee_per_user: float


class AnalyticsService:
    """خدمة التحليلات الشاملة"""
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    # ============ Performance Analytics ============
    
    async def get_performance_metrics(
        self,
        period_days: int = 30
    ) -> PerformanceMetrics:
        """الحصول على مقاييس أداء البوت"""
        from app.models import TradingHistory, NAVHistory
        
        start_date = datetime.utcnow() - timedelta(days=period_days)
        
        # جلب الصفقات
        result = await self.db.execute(
            select(TradingHistory)
            .where(TradingHistory.created_at >= start_date)
            .order_by(TradingHistory.created_at)
        )
        trades = result.scalars().all()
        
        # حساب الإحصائيات
        total_trades = len(trades)
        profitable_trades = len([t for t in trades if t.pnl and t.pnl > 0])
        losing_trades = len([t for t in trades if t.pnl and t.pnl < 0])
        
        win_rate = (profitable_trades / total_trades * 100) if total_trades > 0 else 0
        
        # جلب تاريخ NAV
        result = await self.db.execute(
            select(NAVHistory)
            .where(NAVHistory.created_at >= start_date)
            .order_by(NAVHistory.created_at)
        )
        nav_history = result.scalars().all()
        
        # حساب العوائد
        if len(nav_history) >= 2:
            first_nav = nav_history[0].nav_value
            last_nav = nav_history[-1].nav_value
            total_return = ((last_nav - first_nav) / first_nav * 100) if first_nav > 0 else 0
        else:
            total_return = 0
        
        # حساب العوائد اليومية والأسبوعية والشهرية
        daily_return = await self._calculate_period_return(1)
        weekly_return = await self._calculate_period_return(7)
        monthly_return = await self._calculate_period_return(30)
        
        # حساب Max Drawdown
        max_drawdown = await self._calculate_max_drawdown(nav_history)
        
        # حساب Sharpe Ratio (مبسط)
        sharpe_ratio = await self._calculate_sharpe_ratio(nav_history)
        
        return PerformanceMetrics(
            total_return=round(total_return, 2),
            daily_return=round(daily_return, 2),
            weekly_return=round(weekly_return, 2),
            monthly_return=round(monthly_return, 2),
            sharpe_ratio=round(sharpe_ratio, 2),
            max_drawdown=round(max_drawdown, 2),
            win_rate=round(win_rate, 2),
            total_trades=total_trades,
            profitable_trades=profitable_trades,
            losing_trades=losing_trades
        )
    
    async def _calculate_period_return(self, days: int) -> float:
        """حساب العائد لفترة محددة"""
        from app.models import NAVHistory
        
        start_date = datetime.utcnow() - timedelta(days=days)
        
        result = await self.db.execute(
            select(NAVHistory)
            .where(NAVHistory.created_at >= start_date)
            .order_by(NAVHistory.created_at)
        )
        nav_history = result.scalars().all()
        
        if len(nav_history) >= 2:
            first_nav = nav_history[0].nav_value
            last_nav = nav_history[-1].nav_value
            return ((last_nav - first_nav) / first_nav * 100) if first_nav > 0 else 0
        return 0
    
    async def _calculate_max_drawdown(self, nav_history: List) -> float:
        """حساب أقصى انخفاض"""
        if not nav_history:
            return 0
        
        peak = nav_history[0].nav_value
        max_dd = 0
        
        for nav in nav_history:
            if nav.nav_value > peak:
                peak = nav.nav_value
            dd = (peak - nav.nav_value) / peak * 100 if peak > 0 else 0
            max_dd = max(max_dd, dd)
        
        return max_dd
    
    async def _calculate_sharpe_ratio(self, nav_history: List) -> float:
        """حساب نسبة شارب (مبسط)"""
        if len(nav_history) < 2:
            return 0
        
        # حساب العوائد اليومية
        returns = []
        for i in range(1, len(nav_history)):
            prev_nav = nav_history[i-1].nav_value
            curr_nav = nav_history[i].nav_value
            if prev_nav > 0:
                daily_return = (curr_nav - prev_nav) / prev_nav
                returns.append(daily_return)
        
        if not returns:
            return 0
        
        # حساب المتوسط والانحراف المعياري
        avg_return = sum(returns) / len(returns)
        variance = sum((r - avg_return) ** 2 for r in returns) / len(returns)
        std_dev = variance ** 0.5
        
        # نسبة شارب (افتراض معدل خالي من المخاطر = 0)
        risk_free_rate = 0
        if std_dev > 0:
            sharpe = (avg_return - risk_free_rate) / std_dev * (365 ** 0.5)  # سنوياً
            return sharpe
        return 0
    
    # ============ User Analytics ============
    
    async def get_user_analytics(self) -> UserAnalytics:
        """الحصول على تحليلات المستخدمين"""
        from app.models import User, Transaction, Balance
        
        now = datetime.utcnow()
        today = now.replace(hour=0, minute=0, second=0, microsecond=0)
        week_ago = now - timedelta(days=7)
        month_ago = now - timedelta(days=30)
        
        # إجمالي المستخدمين
        result = await self.db.execute(select(func.count(User.id)))
        total_users = result.scalar() or 0
        
        # المستخدمين النشطين (آخر 30 يوم)
        result = await self.db.execute(
            select(func.count(User.id))
            .where(User.last_login >= month_ago)
        )
        active_users = result.scalar() or 0
        
        # المستخدمين الجدد اليوم
        result = await self.db.execute(
            select(func.count(User.id))
            .where(User.created_at >= today)
        )
        new_users_today = result.scalar() or 0
        
        # المستخدمين الجدد هذا الأسبوع
        result = await self.db.execute(
            select(func.count(User.id))
            .where(User.created_at >= week_ago)
        )
        new_users_week = result.scalar() or 0
        
        # المستخدمين الجدد هذا الشهر
        result = await self.db.execute(
            select(func.count(User.id))
            .where(User.created_at >= month_ago)
        )
        new_users_month = result.scalar() or 0
        
        # إجمالي الإيداعات
        result = await self.db.execute(
            select(func.sum(Transaction.amount_usd))
            .where(Transaction.type == "deposit")
            .where(Transaction.status == "completed")
        )
        total_deposited = result.scalar() or 0
        
        # إجمالي السحوبات
        result = await self.db.execute(
            select(func.sum(Transaction.amount_usd))
            .where(Transaction.type == "withdrawal")
            .where(Transaction.status == "completed")
        )
        total_withdrawn = result.scalar() or 0
        
        # متوسط الإيداع
        result = await self.db.execute(
            select(func.avg(Transaction.amount_usd))
            .where(Transaction.type == "deposit")
            .where(Transaction.status == "completed")
        )
        avg_deposit = result.scalar() or 0
        
        # معدل الاحتفاظ (المستخدمين الذين لديهم رصيد > 0)
        result = await self.db.execute(
            select(func.count(Balance.id))
            .where(Balance.units > 0)
        )
        users_with_balance = result.scalar() or 0
        retention_rate = (users_with_balance / total_users * 100) if total_users > 0 else 0
        
        # معدل المغادرة
        churn_rate = 100 - retention_rate
        
        return UserAnalytics(
            total_users=total_users,
            active_users=active_users,
            new_users_today=new_users_today,
            new_users_week=new_users_week,
            new_users_month=new_users_month,
            retention_rate=round(retention_rate, 2),
            avg_deposit=round(avg_deposit, 2),
            total_deposited=round(total_deposited, 2),
            total_withdrawn=round(total_withdrawn, 2),
            churn_rate=round(churn_rate, 2)
        )
    
    # ============ Revenue Analytics ============
    
    async def get_revenue_analytics(self) -> RevenueAnalytics:
        """الحصول على تحليلات الإيرادات"""
        from app.models import Transaction, PlatformStats
        
        now = datetime.utcnow()
        today = now.replace(hour=0, minute=0, second=0, microsecond=0)
        week_ago = now - timedelta(days=7)
        month_ago = now - timedelta(days=30)
        
        # إجمالي الرسوم
        result = await self.db.execute(
            select(func.sum(Transaction.fee_amount))
            .where(Transaction.status == "completed")
        )
        total_fees = result.scalar() or 0
        
        # رسوم اليوم
        result = await self.db.execute(
            select(func.sum(Transaction.fee_amount))
            .where(Transaction.status == "completed")
            .where(Transaction.created_at >= today)
        )
        fees_today = result.scalar() or 0
        
        # رسوم الأسبوع
        result = await self.db.execute(
            select(func.sum(Transaction.fee_amount))
            .where(Transaction.status == "completed")
            .where(Transaction.created_at >= week_ago)
        )
        fees_week = result.scalar() or 0
        
        # رسوم الشهر
        result = await self.db.execute(
            select(func.sum(Transaction.fee_amount))
            .where(Transaction.status == "completed")
            .where(Transaction.created_at >= month_ago)
        )
        fees_month = result.scalar() or 0
        
        # رسوم الأداء
        result = await self.db.execute(
            select(func.sum(Transaction.fee_amount))
            .where(Transaction.type == "performance_fee")
            .where(Transaction.status == "completed")
        )
        performance_fees = result.scalar() or 0
        
        # رسوم السحب
        result = await self.db.execute(
            select(func.sum(Transaction.fee_amount))
            .where(Transaction.type == "withdrawal")
            .where(Transaction.status == "completed")
        )
        withdrawal_fees = result.scalar() or 0
        
        # متوسط الرسوم لكل مستخدم
        from app.models import User
        result = await self.db.execute(select(func.count(User.id)))
        total_users = result.scalar() or 1
        avg_fee_per_user = total_fees / total_users
        
        return RevenueAnalytics(
            total_fees=round(total_fees, 2),
            fees_today=round(fees_today, 2),
            fees_week=round(fees_week, 2),
            fees_month=round(fees_month, 2),
            performance_fees=round(performance_fees, 2),
            withdrawal_fees=round(withdrawal_fees, 2),
            avg_fee_per_user=round(avg_fee_per_user, 2)
        )
    
    # ============ Chart Data ============
    
    async def get_nav_chart_data(self, days: int = 30) -> List[Dict]:
        """الحصول على بيانات رسم NAV"""
        from app.models import NAVHistory
        
        start_date = datetime.utcnow() - timedelta(days=days)
        
        result = await self.db.execute(
            select(NAVHistory)
            .where(NAVHistory.created_at >= start_date)
            .order_by(NAVHistory.created_at)
        )
        nav_history = result.scalars().all()
        
        return [
            {
                "date": nav.created_at.strftime("%Y-%m-%d"),
                "nav": round(nav.nav_value, 4),
                "total_assets": round(nav.total_assets, 2) if hasattr(nav, 'total_assets') else 0
            }
            for nav in nav_history
        ]
    
    async def get_users_chart_data(self, days: int = 30) -> List[Dict]:
        """الحصول على بيانات رسم نمو المستخدمين"""
        from app.models import User
        
        start_date = datetime.utcnow() - timedelta(days=days)
        data = []
        
        for i in range(days):
            date = start_date + timedelta(days=i)
            next_date = date + timedelta(days=1)
            
            result = await self.db.execute(
                select(func.count(User.id))
                .where(User.created_at < next_date)
            )
            total = result.scalar() or 0
            
            result = await self.db.execute(
                select(func.count(User.id))
                .where(and_(
                    User.created_at >= date,
                    User.created_at < next_date
                ))
            )
            new = result.scalar() or 0
            
            data.append({
                "date": date.strftime("%Y-%m-%d"),
                "total": total,
                "new": new
            })
        
        return data
    
    async def get_deposits_chart_data(self, days: int = 30) -> List[Dict]:
        """الحصول على بيانات رسم الإيداعات"""
        from app.models import Transaction
        
        start_date = datetime.utcnow() - timedelta(days=days)
        data = []
        
        for i in range(days):
            date = start_date + timedelta(days=i)
            next_date = date + timedelta(days=1)
            
            result = await self.db.execute(
                select(func.sum(Transaction.amount_usd))
                .where(Transaction.type == "deposit")
                .where(Transaction.status == "completed")
                .where(and_(
                    Transaction.created_at >= date,
                    Transaction.created_at < next_date
                ))
            )
            amount = result.scalar() or 0
            
            result = await self.db.execute(
                select(func.count(Transaction.id))
                .where(Transaction.type == "deposit")
                .where(Transaction.status == "completed")
                .where(and_(
                    Transaction.created_at >= date,
                    Transaction.created_at < next_date
                ))
            )
            count = result.scalar() or 0
            
            data.append({
                "date": date.strftime("%Y-%m-%d"),
                "amount": round(amount, 2),
                "count": count
            })
        
        return data
    
    async def get_trades_chart_data(self, days: int = 30) -> List[Dict]:
        """الحصول على بيانات رسم الصفقات"""
        from app.models import TradingHistory
        
        start_date = datetime.utcnow() - timedelta(days=days)
        data = []
        
        for i in range(days):
            date = start_date + timedelta(days=i)
            next_date = date + timedelta(days=1)
            
            result = await self.db.execute(
                select(TradingHistory)
                .where(and_(
                    TradingHistory.created_at >= date,
                    TradingHistory.created_at < next_date
                ))
            )
            trades = result.scalars().all()
            
            total_pnl = sum(t.pnl or 0 for t in trades)
            wins = len([t for t in trades if t.pnl and t.pnl > 0])
            losses = len([t for t in trades if t.pnl and t.pnl < 0])
            
            data.append({
                "date": date.strftime("%Y-%m-%d"),
                "total": len(trades),
                "wins": wins,
                "losses": losses,
                "pnl": round(total_pnl, 2)
            })
        
        return data
    
    # ============ Comparison with Market ============
    
    async def get_btc_comparison(self, days: int = 30) -> Dict:
        """مقارنة أداء البوت مع Bitcoin"""
        # هذا يحتاج API خارجي لجلب سعر BTC
        # للتبسيط، سنعيد بيانات وهمية
        bot_return = await self._calculate_period_return(days)
        
        return {
            "bot_return": round(bot_return, 2),
            "btc_return": 0,  # يحتاج API خارجي
            "outperformance": round(bot_return, 2)
        }
    
    # ============ Top Users ============
    
    async def get_top_depositors(self, limit: int = 10) -> List[Dict]:
        """الحصول على أكبر المودعين"""
        from app.models import User, Transaction
        
        result = await self.db.execute(
            select(
                User.id,
                User.email,
                User.full_name,
                func.sum(Transaction.amount_usd).label("total_deposited")
            )
            .join(Transaction, User.id == Transaction.user_id)
            .where(Transaction.type == "deposit")
            .where(Transaction.status == "completed")
            .group_by(User.id)
            .order_by(desc("total_deposited"))
            .limit(limit)
        )
        
        return [
            {
                "user_id": row.id,
                "email": row.email,
                "name": row.full_name or "بدون اسم",
                "total_deposited": round(row.total_deposited, 2)
            }
            for row in result.all()
        ]
    
    async def get_top_earners(self, limit: int = 10) -> List[Dict]:
        """الحصول على أكبر الرابحين"""
        from app.models import User, Balance
        
        result = await self.db.execute(
            select(
                User.id,
                User.email,
                User.full_name,
                Balance.units,
                Balance.total_deposited
            )
            .join(Balance, User.id == Balance.user_id)
            .where(Balance.units > 0)
            .order_by(desc(Balance.units))
            .limit(limit)
        )
        
        # جلب NAV الحالي
        from app.services import nav_service
        current_nav = await nav_service.get_current_nav(self.db)
        
        users = []
        for row in result.all():
            current_value = row.units * current_nav
            profit = current_value - (row.total_deposited or 0)
            profit_percent = (profit / row.total_deposited * 100) if row.total_deposited else 0
            
            users.append({
                "user_id": row.id,
                "email": row.email,
                "name": row.full_name or "بدون اسم",
                "current_value": round(current_value, 2),
                "profit": round(profit, 2),
                "profit_percent": round(profit_percent, 2)
            })
        
        return sorted(users, key=lambda x: x["profit"], reverse=True)
