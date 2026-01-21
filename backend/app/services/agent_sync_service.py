"""
═══════════════════════════════════════════════════════════════════════════════
                    🔗 AGENT SYNC SERVICE
                    خدمة مزامنة بيانات الوكيل
═══════════════════════════════════════════════════════════════════════════════
هذه الخدمة تجلب بيانات الوكيل من سيرفر الوكيل وتحدث NAV في المنصة
═══════════════════════════════════════════════════════════════════════════════
"""
import os
import httpx
import asyncio
from datetime import datetime
from typing import Optional, Dict, Any
from decimal import Decimal
import logging

logger = logging.getLogger(__name__)

# إعدادات الاتصال بالوكيل
AGENT_URL = os.getenv("AGENT_API_URL", "http://77.37.49.59:9999")
AGENT_API_KEY = os.getenv("AGENT_API_KEY", "asinax_platform_secret_key_2024")

# القيمة الأولية للصندوق (يجب تحديثها عند أول إيداع حقيقي)
INITIAL_FUND_VALUE = float(os.getenv("INITIAL_FUND_VALUE", "241.39"))


class AgentSyncService:
    """خدمة مزامنة بيانات الوكيل"""
    
    def __init__(self, db_session_factory=None):
        self.db_session_factory = db_session_factory
        self.agent_url = AGENT_URL
        self.api_key = AGENT_API_KEY
        self.initial_fund_value = INITIAL_FUND_VALUE
        self._last_nav = Decimal("1.0")
        self._last_sync = None
        
    async def fetch_agent_data(self) -> Optional[Dict[str, Any]]:
        """جلب بيانات الوكيل من API"""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    f"{self.agent_url}/platform-api/nav-data",
                    headers={"X-API-Key": self.api_key}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get("success"):
                        return data.get("data")
                        
                logger.error(f"Failed to fetch agent data: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching agent data: {e}")
            return None
    
    async def fetch_agent_trades(self, limit: int = 50) -> Optional[list]:
        """جلب صفقات الوكيل"""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    f"{self.agent_url}/platform-api/trades",
                    params={"limit": limit},
                    headers={"X-API-Key": self.api_key}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get("success"):
                        return data.get("data", {}).get("trades", [])
                        
                return None
                
        except Exception as e:
            logger.error(f"Error fetching agent trades: {e}")
            return None
    
    async def fetch_agent_performance(self) -> Optional[Dict[str, Any]]:
        """جلب بيانات أداء الوكيل"""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    f"{self.agent_url}/platform-api/performance",
                    headers={"X-API-Key": self.api_key}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get("success"):
                        return data.get("data")
                        
                return None
                
        except Exception as e:
            logger.error(f"Error fetching agent performance: {e}")
            return None
    
    def calculate_nav(self, total_portfolio_value: float, total_units: float) -> Decimal:
        """
        حساب NAV الحالي
        
        NAV = إجمالي قيمة المحفظة / إجمالي الوحدات
        
        إذا لم يكن هناك وحدات (لا مستثمرين)، نستخدم القيمة الأولية
        """
        if total_units <= 0:
            # لا يوجد مستثمرين - NAV = 1.0
            return Decimal("1.0")
        
        nav = Decimal(str(total_portfolio_value)) / Decimal(str(total_units))
        return nav.quantize(Decimal("0.0001"))
    
    async def sync_nav(self, db_session) -> Optional[Decimal]:
        """
        مزامنة NAV من الوكيل وتحديثه في قاعدة البيانات
        """
        try:
            # جلب بيانات الوكيل
            agent_data = await self.fetch_agent_data()
            if not agent_data:
                logger.warning("Could not fetch agent data, using last known NAV")
                return self._last_nav
            
            total_portfolio_value = agent_data.get("total_value", 0)
            
            # جلب إجمالي الوحدات من قاعدة البيانات
            from sqlalchemy import select, func
            from app.models.user import Balance
            
            result = await db_session.execute(
                select(func.sum(Balance.units))
            )
            total_units = result.scalar() or 0
            
            # حساب NAV
            new_nav = self.calculate_nav(total_portfolio_value, total_units)
            
            # تسجيل NAV في جدول التاريخ
            from app.models.transaction import NAVHistory
            
            nav_record = NAVHistory(
                nav_value=float(new_nav),
                total_assets_usd=total_portfolio_value,
                total_units=float(total_units)
            )
            db_session.add(nav_record)
            await db_session.commit()
            
            self._last_nav = new_nav
            self._last_sync = datetime.utcnow()
            
            logger.info(f"NAV synced: {new_nav} (Portfolio: ${total_portfolio_value}, Units: {total_units})")
            
            return new_nav
            
        except Exception as e:
            logger.error(f"Error syncing NAV: {e}")
            return self._last_nav
    
    async def get_current_nav(self, db_session) -> Decimal:
        """الحصول على NAV الحالي"""
        from sqlalchemy import select
        from app.models.transaction import NAVHistory
        
        result = await db_session.execute(
            select(NAVHistory)
            .order_by(NAVHistory.timestamp.desc())
            .limit(1)
        )
        nav_record = result.scalar_one_or_none()
        
        if nav_record:
            return Decimal(str(nav_record.nav_value))
        
        return Decimal("1.0")
    
    async def get_agent_status(self) -> Dict[str, Any]:
        """الحصول على حالة الوكيل"""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    f"{self.agent_url}/platform-api/status",
                    headers={"X-API-Key": self.api_key}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get("success"):
                        return {
                            "online": True,
                            **data.get("data", {})
                        }
                        
                return {"online": False, "status": "unreachable"}
                
        except Exception as e:
            return {"online": False, "status": "error", "error": str(e)}


# Singleton instance
_agent_sync_service: Optional[AgentSyncService] = None

def get_agent_sync_service() -> AgentSyncService:
    """الحصول على نسخة واحدة من الخدمة"""
    global _agent_sync_service
    if _agent_sync_service is None:
        _agent_sync_service = AgentSyncService()
    return _agent_sync_service
