from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from datetime import datetime, timedelta
from typing import Optional, Tuple
from app.models import Balance, NAVHistory, PlatformStats
from app.services.binance_service import binance_service
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)


class NAVService:
    """
    Service for calculating and managing NAV (Net Asset Value)
    NAV = Total Assets / Total Units
    """
    
    async def get_total_units(self, db: AsyncSession) -> float:
        """Get total units across all users"""
        result = await db.execute(
            select(func.sum(Balance.units))
        )
        total = result.scalar()
        return total if total else 0.0
    
    async def get_current_nav(self, db: AsyncSession) -> float:
        """
        Calculate current NAV
        NAV = Total Assets USD / Total Units
        """
        total_units = await self.get_total_units(db)
        
        if total_units == 0:
            return settings.INITIAL_NAV
        
        total_assets = await binance_service.get_total_assets_usd()
        
        if total_assets == 0:
            return settings.INITIAL_NAV
        
        return total_assets / total_units
    
    async def calculate_units_for_deposit(
        self, 
        db: AsyncSession, 
        amount_usd: float
    ) -> Tuple[float, float]:
        """
        Calculate how many units a deposit should receive
        Returns: (units, nav_at_deposit)
        """
        nav = await self.get_current_nav(db)
        units = amount_usd / nav
        return units, nav
    
    async def calculate_value_for_units(
        self, 
        db: AsyncSession, 
        units: float
    ) -> Tuple[float, float]:
        """
        Calculate USD value for given units
        Returns: (value_usd, current_nav)
        """
        nav = await self.get_current_nav(db)
        value = units * nav
        return value, nav
    
    async def record_nav_snapshot(self, db: AsyncSession) -> NAVHistory:
        """Record current NAV to history"""
        total_assets = await binance_service.get_total_assets_usd()
        total_units = await self.get_total_units(db)
        nav = total_assets / total_units if total_units > 0 else settings.INITIAL_NAV
        
        nav_record = NAVHistory(
            nav_value=nav,
            total_assets_usd=total_assets,
            total_units=total_units
        )
        db.add(nav_record)
        await db.commit()
        
        return nav_record
    
    async def get_nav_history(
        self, 
        db: AsyncSession, 
        days: int = 30
    ) -> list:
        """Get NAV history for specified days"""
        since = datetime.utcnow() - timedelta(days=days)
        result = await db.execute(
            select(NAVHistory)
            .where(NAVHistory.timestamp >= since)
            .order_by(NAVHistory.timestamp.asc())
        )
        return result.scalars().all()
    
    async def get_nav_change(
        self, 
        db: AsyncSession, 
        days: int
    ) -> Optional[float]:
        """Get NAV percentage change over specified days"""
        since = datetime.utcnow() - timedelta(days=days)
        
        # Get oldest NAV in period
        result = await db.execute(
            select(NAVHistory)
            .where(NAVHistory.timestamp >= since)
            .order_by(NAVHistory.timestamp.asc())
            .limit(1)
        )
        old_nav_record = result.scalar_one_or_none()
        
        if not old_nav_record:
            return None
        
        current_nav = await self.get_current_nav(db)
        old_nav = old_nav_record.nav_value
        
        if old_nav == 0:
            return None
        
        return ((current_nav - old_nav) / old_nav) * 100
    
    async def check_and_update_high_water_mark(
        self, 
        db: AsyncSession
    ) -> Tuple[bool, float]:
        """
        Check if current NAV exceeds high water mark
        Returns: (exceeded, current_hwm)
        """
        # Get or create platform stats
        result = await db.execute(select(PlatformStats).limit(1))
        stats = result.scalar_one_or_none()
        
        if not stats:
            stats = PlatformStats(high_water_mark=settings.INITIAL_NAV)
            db.add(stats)
            await db.commit()
        
        current_nav = await self.get_current_nav(db)
        
        if current_nav > stats.high_water_mark:
            old_hwm = stats.high_water_mark
            stats.high_water_mark = current_nav
            await db.commit()
            return True, old_hwm
        
        return False, stats.high_water_mark
    
    async def calculate_performance_fee(
        self, 
        db: AsyncSession
    ) -> float:
        """
        Calculate performance fee based on profits above high water mark
        Only charges fee on new profits
        """
        exceeded, old_hwm = await self.check_and_update_high_water_mark(db)
        
        if not exceeded:
            return 0.0
        
        current_nav = await self.get_current_nav(db)
        total_units = await self.get_total_units(db)
        
        # Profit per unit above HWM
        profit_per_unit = current_nav - old_hwm
        total_profit = profit_per_unit * total_units
        
        # Performance fee
        fee = total_profit * (settings.PERFORMANCE_FEE_PERCENT / 100)
        
        return fee


# Singleton instance
nav_service = NAVService()
