#!/usr/bin/env python3
"""
Script to initialize the database with initial data
Usage: python scripts/init_db.py
"""

import asyncio
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy import select

from app.core.config import settings
from app.models.transaction import PlatformStats, NAVHistory


async def init_platform_stats(session: AsyncSession):
    """Initialize platform stats"""
    result = await session.execute(select(PlatformStats).limit(1))
    stats = result.scalar_one_or_none()
    
    if not stats:
        stats = PlatformStats(
            high_water_mark=settings.INITIAL_NAV,
            total_fees_collected=0.0,
            emergency_mode="off"
        )
        session.add(stats)
        print("✅ Platform stats initialized")
    else:
        print("ℹ️  Platform stats already exist")


async def init_nav_history(session: AsyncSession):
    """Initialize NAV history with initial value"""
    result = await session.execute(select(NAVHistory).limit(1))
    nav = result.scalar_one_or_none()
    
    if not nav:
        nav = NAVHistory(
            nav_value=settings.INITIAL_NAV,
            total_assets_usd=0.0,
            total_units=0.0
        )
        session.add(nav)
        print("✅ Initial NAV record created")
    else:
        print("ℹ️  NAV history already exists")


async def main():
    """Initialize database with required data"""
    print("\n🗄️  Initializing Asinax Database...\n")
    
    # Create engine
    engine = create_async_engine(settings.DATABASE_URL, echo=False)
    async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    
    async with async_session() as session:
        try:
            await init_platform_stats(session)
            await init_nav_history(session)
            await session.commit()
            print("\n✅ Database initialization complete!\n")
        except Exception as e:
            await session.rollback()
            print(f"\n❌ Error initializing database: {e}\n")
            raise
    
    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
