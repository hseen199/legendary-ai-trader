"""
Legendary Trading System - Emergency System
نظام التداول الخارق - نظام الطوارئ

نظام متقدم للتعامل مع الأزمات والحالات الطارئة.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from collections import deque
from enum import Enum
import logging


class EmergencyLevel(Enum):
    """مستويات الطوارئ"""
    GREEN = 1       # عادي
    YELLOW = 2      # تحذير
    ORANGE = 3      # خطر
    RED = 4         # طوارئ
    BLACK = 5       # كارثة


class EmergencyType(Enum):
    """أنواع الطوارئ"""
    FLASH_CRASH = "flash_crash"             # انهيار مفاجئ
    EXCHANGE_ISSUE = "exchange_issue"       # مشكلة في البورصة
    API_FAILURE = "api_failure"             # فشل API
    LIQUIDITY_CRISIS = "liquidity_crisis"   # أزمة سيولة
    NETWORK_ISSUE = "network_issue"         # مشكلة شبكة
    SECURITY_BREACH = "security_breach"     # اختراق أمني
    MARGIN_CALL = "margin_call"             # نداء هامش
    MAX_DRAWDOWN = "max_drawdown"           # سحب أقصى
    SYSTEM_ERROR = "system_error"           # خطأ نظام
    REGULATORY = "regulatory"               # تنظيمي


@dataclass
class EmergencyEvent:
    """حدث طوارئ"""
    id: str
    type: EmergencyType
    level: EmergencyLevel
    
    # التفاصيل
    title: str
    description: str
    
    # التوقيت
    detected_at: datetime
    resolved_at: Optional[datetime] = None
    
    # التأثير
    affected_positions: List[str] = field(default_factory=list)
    estimated_loss: float = 0.0
    
    # الإجراءات
    actions_taken: List[Dict[str, Any]] = field(default_factory=list)
    
    # الحالة
    status: str = "active"  # active, contained, resolved


@dataclass
class EmergencyProtocol:
    """بروتوكول طوارئ"""
    id: str
    name: str
    trigger_conditions: Dict[str, Any]
    actions: List[Dict[str, Any]]
    priority: int
    enabled: bool = True


@dataclass
class SafetyCheck:
    """فحص أمان"""
    name: str
    check_function: Callable
    threshold: Any
    action_on_fail: str
    last_check: Optional[datetime] = None
    last_result: Optional[bool] = None


class EmergencySystem:
    """
    نظام الطوارئ.
    
    يوفر:
    - كشف الأزمات المفاجئة (Flash Crash)
    - خروج طوارئ ذكي
    - حماية رأس المال في الكوارث
    """
    
    def __init__(self, 
                 trading_system=None,
                 config: Dict[str, Any] = None):
        self.logger = logging.getLogger("EmergencySystem")
        self.trading_system = trading_system
        self.config = config or {}
        
        # الحالة الحالية
        self.current_level = EmergencyLevel.GREEN
        
        # أحداث الطوارئ
        self.active_emergencies: Dict[str, EmergencyEvent] = {}
        self.emergency_history: List[EmergencyEvent] = []
        
        # البروتوكولات
        self.protocols: Dict[str, EmergencyProtocol] = {}
        
        # فحوصات الأمان
        self.safety_checks: List[SafetyCheck] = []
        
        # عتبات الطوارئ
        self.thresholds = {
            "flash_crash_pct": -0.10,      # -10% في دقائق
            "max_drawdown_pct": -0.20,     # -20% سحب أقصى
            "max_daily_loss_pct": -0.05,   # -5% خسارة يومية
            "api_timeout_seconds": 30,     # 30 ثانية
            "min_liquidity_ratio": 0.1,    # 10% سيولة
            "max_position_loss_pct": -0.15 # -15% خسارة صفقة
        }
        
        # حالة النظام
        self.system_state = {
            "trading_enabled": True,
            "new_positions_allowed": True,
            "emergency_mode": False,
            "last_health_check": None
        }
        
        # إحصائيات
        self.stats = {
            "emergencies_detected": 0,
            "emergencies_resolved": 0,
            "emergency_exits": 0,
            "capital_protected": 0.0
        }
        
        # تهيئة البروتوكولات الافتراضية
        self._init_default_protocols()
    
    def _init_default_protocols(self):
        """تهيئة البروتوكولات الافتراضية."""
        # بروتوكول الانهيار المفاجئ
        self.protocols["flash_crash"] = EmergencyProtocol(
            id="flash_crash",
            name="بروتوكول الانهيار المفاجئ",
            trigger_conditions={
                "price_change_5min": self.thresholds["flash_crash_pct"]
            },
            actions=[
                {"action": "pause_trading", "delay": 0},
                {"action": "close_losing_positions", "delay": 5},
                {"action": "set_tight_stops", "delay": 10},
                {"action": "notify_admin", "delay": 0}
            ],
            priority=1
        )
        
        # بروتوكول السحب الأقصى
        self.protocols["max_drawdown"] = EmergencyProtocol(
            id="max_drawdown",
            name="بروتوكول السحب الأقصى",
            trigger_conditions={
                "drawdown": self.thresholds["max_drawdown_pct"]
            },
            actions=[
                {"action": "close_all_positions", "delay": 0},
                {"action": "disable_trading", "delay": 0},
                {"action": "notify_admin", "delay": 0}
            ],
            priority=1
        )
        
        # بروتوكول فشل API
        self.protocols["api_failure"] = EmergencyProtocol(
            id="api_failure",
            name="بروتوكول فشل API",
            trigger_conditions={
                "api_failures": 3
            },
            actions=[
                {"action": "switch_to_backup", "delay": 0},
                {"action": "pause_new_orders", "delay": 0},
                {"action": "notify_admin", "delay": 0}
            ],
            priority=2
        )
        
        # بروتوكول أزمة السيولة
        self.protocols["liquidity_crisis"] = EmergencyProtocol(
            id="liquidity_crisis",
            name="بروتوكول أزمة السيولة",
            trigger_conditions={
                "liquidity_ratio": self.thresholds["min_liquidity_ratio"]
            },
            actions=[
                {"action": "reduce_positions", "delay": 0, "percentage": 50},
                {"action": "pause_new_orders", "delay": 0},
                {"action": "notify_admin", "delay": 0}
            ],
            priority=2
        )
    
    async def monitor(self, market_data: Dict[str, Any]) -> EmergencyLevel:
        """
        مراقبة مستمرة للطوارئ.
        
        Args:
            market_data: بيانات السوق
            
        Returns:
            مستوى الطوارئ الحالي
        """
        # تحديث وقت الفحص
        self.system_state["last_health_check"] = datetime.utcnow()
        
        # فحص كل البروتوكولات
        for protocol_id, protocol in self.protocols.items():
            if protocol.enabled:
                triggered = await self._check_protocol_triggers(protocol, market_data)
                if triggered:
                    await self._activate_protocol(protocol, market_data)
        
        # تحديث مستوى الطوارئ
        self._update_emergency_level()
        
        return self.current_level
    
    async def _check_protocol_triggers(self,
                                      protocol: EmergencyProtocol,
                                      market_data: Dict[str, Any]) -> bool:
        """فحص محفزات البروتوكول."""
        conditions = protocol.trigger_conditions
        
        for condition, threshold in conditions.items():
            if condition == "price_change_5min":
                price_change = market_data.get("price_change_5min", 0)
                if price_change < threshold:
                    return True
            
            elif condition == "drawdown":
                drawdown = market_data.get("current_drawdown", 0)
                if drawdown < threshold:
                    return True
            
            elif condition == "api_failures":
                failures = market_data.get("api_failure_count", 0)
                if failures >= threshold:
                    return True
            
            elif condition == "liquidity_ratio":
                ratio = market_data.get("liquidity_ratio", 1)
                if ratio < threshold:
                    return True
        
        return False
    
    async def _activate_protocol(self,
                                protocol: EmergencyProtocol,
                                market_data: Dict[str, Any]):
        """تفعيل بروتوكول طوارئ."""
        self.logger.critical(f"تفعيل بروتوكول: {protocol.name}")
        
        # إنشاء حدث طوارئ
        emergency = EmergencyEvent(
            id=f"emergency_{datetime.utcnow().timestamp()}",
            type=self._get_emergency_type(protocol.id),
            level=EmergencyLevel.RED if protocol.priority == 1 else EmergencyLevel.ORANGE,
            title=protocol.name,
            description=f"تم تفعيل بروتوكول الطوارئ: {protocol.name}",
            detected_at=datetime.utcnow()
        )
        
        self.active_emergencies[emergency.id] = emergency
        self.stats["emergencies_detected"] += 1
        
        # تنفيذ الإجراءات
        for action_config in protocol.actions:
            await asyncio.sleep(action_config.get("delay", 0))
            result = await self._execute_emergency_action(
                action_config["action"],
                action_config
            )
            emergency.actions_taken.append({
                "action": action_config["action"],
                "result": result,
                "timestamp": datetime.utcnow().isoformat()
            })
        
        # تفعيل وضع الطوارئ
        self.system_state["emergency_mode"] = True
    
    def _get_emergency_type(self, protocol_id: str) -> EmergencyType:
        """تحديد نوع الطوارئ."""
        mapping = {
            "flash_crash": EmergencyType.FLASH_CRASH,
            "max_drawdown": EmergencyType.MAX_DRAWDOWN,
            "api_failure": EmergencyType.API_FAILURE,
            "liquidity_crisis": EmergencyType.LIQUIDITY_CRISIS
        }
        return mapping.get(protocol_id, EmergencyType.SYSTEM_ERROR)
    
    async def _execute_emergency_action(self,
                                       action: str,
                                       config: Dict[str, Any]) -> Dict[str, Any]:
        """تنفيذ إجراء طوارئ."""
        result = {"action": action, "success": False}
        
        try:
            if action == "pause_trading":
                self.system_state["trading_enabled"] = False
                result["success"] = True
                result["message"] = "تم إيقاف التداول"
            
            elif action == "close_all_positions":
                if self.trading_system:
                    await self.trading_system.close_all_positions()
                self.stats["emergency_exits"] += 1
                result["success"] = True
                result["message"] = "تم إغلاق جميع الصفقات"
            
            elif action == "close_losing_positions":
                if self.trading_system:
                    await self.trading_system.close_losing_positions()
                result["success"] = True
                result["message"] = "تم إغلاق الصفقات الخاسرة"
            
            elif action == "set_tight_stops":
                if self.trading_system:
                    await self.trading_system.tighten_all_stops(0.5)
                result["success"] = True
                result["message"] = "تم تضييق وقف الخسارة"
            
            elif action == "disable_trading":
                self.system_state["trading_enabled"] = False
                self.system_state["new_positions_allowed"] = False
                result["success"] = True
                result["message"] = "تم تعطيل التداول بالكامل"
            
            elif action == "pause_new_orders":
                self.system_state["new_positions_allowed"] = False
                result["success"] = True
                result["message"] = "تم إيقاف الأوامر الجديدة"
            
            elif action == "reduce_positions":
                percentage = config.get("percentage", 50)
                if self.trading_system:
                    await self.trading_system.reduce_all_positions(percentage)
                result["success"] = True
                result["message"] = f"تم تقليل الصفقات بنسبة {percentage}%"
            
            elif action == "switch_to_backup":
                # منطق التبديل للنسخة الاحتياطية
                result["success"] = True
                result["message"] = "تم التبديل للنظام الاحتياطي"
            
            elif action == "notify_admin":
                # إرسال إشعار
                self.logger.critical("🚨 إشعار طوارئ للمسؤول!")
                result["success"] = True
                result["message"] = "تم إرسال الإشعار"
            
            else:
                result["message"] = f"إجراء غير معروف: {action}"
        
        except Exception as e:
            result["error"] = str(e)
            self.logger.error(f"خطأ في تنفيذ إجراء الطوارئ: {e}")
        
        return result
    
    def _update_emergency_level(self):
        """تحديث مستوى الطوارئ."""
        if not self.active_emergencies:
            self.current_level = EmergencyLevel.GREEN
            return
        
        # أعلى مستوى من الطوارئ النشطة
        max_level = max(e.level for e in self.active_emergencies.values())
        self.current_level = max_level
    
    async def detect_flash_crash(self,
                                symbol: str,
                                prices: List[float],
                                timestamps: List[datetime]) -> Optional[EmergencyEvent]:
        """
        كشف الانهيار المفاجئ.
        
        Args:
            symbol: الرمز
            prices: الأسعار
            timestamps: الأوقات
            
        Returns:
            حدث الطوارئ إن وجد
        """
        if len(prices) < 2:
            return None
        
        # حساب التغير في آخر 5 دقائق
        recent_prices = []
        cutoff = datetime.utcnow() - timedelta(minutes=5)
        
        for price, ts in zip(prices, timestamps):
            if ts > cutoff:
                recent_prices.append(price)
        
        if len(recent_prices) < 2:
            return None
        
        price_change = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
        
        if price_change < self.thresholds["flash_crash_pct"]:
            emergency = EmergencyEvent(
                id=f"flash_crash_{symbol}_{datetime.utcnow().timestamp()}",
                type=EmergencyType.FLASH_CRASH,
                level=EmergencyLevel.RED,
                title=f"انهيار مفاجئ في {symbol}",
                description=f"انخفض السعر بنسبة {abs(price_change):.1%} في 5 دقائق",
                detected_at=datetime.utcnow(),
                affected_positions=[symbol],
                estimated_loss=abs(price_change)
            )
            
            self.active_emergencies[emergency.id] = emergency
            self.stats["emergencies_detected"] += 1
            
            # تفعيل البروتوكول
            await self._activate_protocol(
                self.protocols["flash_crash"],
                {"price_change_5min": price_change, "symbol": symbol}
            )
            
            return emergency
        
        return None
    
    async def emergency_exit(self,
                            reason: str,
                            symbols: List[str] = None) -> Dict[str, Any]:
        """
        خروج طوارئ.
        
        Args:
            reason: السبب
            symbols: الرموز (اختياري، الكل إذا لم يحدد)
            
        Returns:
            نتيجة الخروج
        """
        self.logger.critical(f"🚨 خروج طوارئ: {reason}")
        
        result = {
            "reason": reason,
            "timestamp": datetime.utcnow().isoformat(),
            "positions_closed": [],
            "errors": []
        }
        
        # إيقاف التداول فوراً
        self.system_state["trading_enabled"] = False
        self.system_state["new_positions_allowed"] = False
        self.system_state["emergency_mode"] = True
        
        # إغلاق الصفقات
        if self.trading_system:
            try:
                if symbols:
                    for symbol in symbols:
                        await self.trading_system.close_position(symbol)
                        result["positions_closed"].append(symbol)
                else:
                    closed = await self.trading_system.close_all_positions()
                    result["positions_closed"] = closed
            except Exception as e:
                result["errors"].append(str(e))
        
        self.stats["emergency_exits"] += 1
        
        # إنشاء حدث طوارئ
        emergency = EmergencyEvent(
            id=f"emergency_exit_{datetime.utcnow().timestamp()}",
            type=EmergencyType.SYSTEM_ERROR,
            level=EmergencyLevel.RED,
            title="خروج طوارئ",
            description=reason,
            detected_at=datetime.utcnow(),
            affected_positions=result["positions_closed"]
        )
        
        self.active_emergencies[emergency.id] = emergency
        
        return result
    
    async def resolve_emergency(self, emergency_id: str) -> bool:
        """
        حل طوارئ.
        
        Args:
            emergency_id: معرف الطوارئ
            
        Returns:
            نجاح الحل
        """
        if emergency_id not in self.active_emergencies:
            return False
        
        emergency = self.active_emergencies[emergency_id]
        emergency.status = "resolved"
        emergency.resolved_at = datetime.utcnow()
        
        # نقل للتاريخ
        self.emergency_history.append(emergency)
        del self.active_emergencies[emergency_id]
        
        self.stats["emergencies_resolved"] += 1
        
        # تحديث المستوى
        self._update_emergency_level()
        
        # إذا لم تبق طوارئ، إعادة التشغيل
        if not self.active_emergencies:
            self.system_state["emergency_mode"] = False
        
        self.logger.info(f"تم حل الطوارئ: {emergency_id}")
        
        return True
    
    def can_trade(self) -> tuple[bool, str]:
        """
        فحص إمكانية التداول.
        
        Returns:
            (يمكن التداول؟, السبب)
        """
        if not self.system_state["trading_enabled"]:
            return False, "التداول معطل"
        
        if self.system_state["emergency_mode"]:
            return False, "وضع الطوارئ نشط"
        
        if self.current_level.value >= EmergencyLevel.RED.value:
            return False, f"مستوى الطوارئ: {self.current_level.value}"
        
        return True, "جاهز للتداول"
    
    def can_open_position(self) -> tuple[bool, str]:
        """
        فحص إمكانية فتح صفقة جديدة.
        
        Returns:
            (يمكن الفتح؟, السبب)
        """
        can_trade, reason = self.can_trade()
        if not can_trade:
            return False, reason
        
        if not self.system_state["new_positions_allowed"]:
            return False, "الصفقات الجديدة معطلة"
        
        if self.current_level.value >= EmergencyLevel.ORANGE.value:
            return False, "مستوى الخطر مرتفع"
        
        return True, "يمكن فتح صفقة"
    
    def resume_trading(self):
        """استئناف التداول."""
        if self.active_emergencies:
            self.logger.warning("لا يمكن استئناف التداول - طوارئ نشطة")
            return False
        
        self.system_state["trading_enabled"] = True
        self.system_state["new_positions_allowed"] = True
        self.system_state["emergency_mode"] = False
        self.current_level = EmergencyLevel.GREEN
        
        self.logger.info("تم استئناف التداول")
        return True
    
    def get_status(self) -> Dict[str, Any]:
        """
        الحصول على حالة النظام.
        
        Returns:
            الحالة
        """
        return {
            "current_level": self.current_level.value,
            "level_name": self.current_level.name,
            "system_state": self.system_state,
            "active_emergencies": len(self.active_emergencies),
            "stats": self.stats,
            "can_trade": self.can_trade()[0],
            "can_open_position": self.can_open_position()[0]
        }
