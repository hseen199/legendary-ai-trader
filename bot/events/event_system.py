"""
Legendary Trading System - Event System
نظام التداول الخارق - نظام الأحداث

نظام متقدم لرصد ومعالجة الأحداث المهمة.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Set
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum
import logging
import json


class EventType(Enum):
    """أنواع الأحداث"""
    # أحداث السوق
    PRICE_SPIKE = "price_spike"
    PRICE_CRASH = "price_crash"
    VOLUME_SURGE = "volume_surge"
    VOLATILITY_SPIKE = "volatility_spike"
    
    # أحداث العملات
    LISTING = "listing"                 # إدراج عملة جديدة
    DELISTING = "delisting"             # إزالة عملة
    UPGRADE = "upgrade"                 # تحديث شبكة
    FORK = "fork"                       # انقسام
    AIRDROP = "airdrop"                 # توزيع مجاني
    
    # أحداث اقتصادية
    FED_MEETING = "fed_meeting"         # اجتماع الفيدرالي
    CPI_RELEASE = "cpi_release"         # بيانات التضخم
    JOBS_REPORT = "jobs_report"         # تقرير الوظائف
    GDP_RELEASE = "gdp_release"         # بيانات الناتج المحلي
    
    # أحداث تنظيمية
    REGULATION_NEWS = "regulation_news"
    EXCHANGE_HACK = "exchange_hack"
    WHALE_MOVEMENT = "whale_movement"
    
    # أحداث النظام
    SYSTEM_ALERT = "system_alert"
    TRADE_EXECUTED = "trade_executed"
    STOP_TRIGGERED = "stop_triggered"


class EventPriority(Enum):
    """أولويات الأحداث"""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4
    INFO = 5


class EventImpact(Enum):
    """تأثير الأحداث"""
    VERY_POSITIVE = "very_positive"
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    VERY_NEGATIVE = "very_negative"


@dataclass
class Event:
    """حدث"""
    id: str
    type: EventType
    priority: EventPriority
    
    # التفاصيل
    title: str
    description: str
    source: str
    
    # التوقيت
    timestamp: datetime
    scheduled_time: Optional[datetime] = None  # للأحداث المجدولة
    
    # التأثير
    impact: EventImpact = EventImpact.NEUTRAL
    affected_symbols: List[str] = field(default_factory=list)
    
    # البيانات الإضافية
    data: Dict[str, Any] = field(default_factory=dict)
    
    # الحالة
    processed: bool = False
    actions_taken: List[str] = field(default_factory=list)


@dataclass
class ScheduledEvent:
    """حدث مجدول"""
    id: str
    type: EventType
    title: str
    scheduled_time: datetime
    
    # التوقعات
    expected_impact: EventImpact
    expected_volatility: float
    
    # التحضير
    preparation_actions: List[str] = field(default_factory=list)
    
    # المصدر
    source: str = ""


@dataclass
class EventReaction:
    """رد فعل على حدث"""
    event_id: str
    action: str
    parameters: Dict[str, Any]
    executed_at: datetime
    success: bool
    result: Optional[Dict[str, Any]] = None


class EventSystem:
    """
    نظام الأحداث.
    
    يوفر:
    - رصد الأحداث المهمة (إدراج عملات، تحديثات، أخبار)
    - ردود فعل تلقائية للأحداث
    - تقويم اقتصادي مدمج
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.logger = logging.getLogger("EventSystem")
        self.config = config or {}
        
        # الأحداث النشطة
        self.active_events: Dict[str, Event] = {}
        
        # الأحداث المجدولة
        self.scheduled_events: Dict[str, ScheduledEvent] = {}
        
        # تاريخ الأحداث
        self.event_history: List[Event] = []
        
        # ردود الفعل
        self.reactions: List[EventReaction] = []
        
        # معالجات الأحداث
        self.event_handlers: Dict[EventType, List[Callable]] = defaultdict(list)
        
        # المشتركين
        self.subscribers: Dict[str, Set[EventType]] = defaultdict(set)
        
        # قواعد رد الفعل التلقائي
        self.auto_reactions: Dict[EventType, List[Dict[str, Any]]] = {}
        
        # إحصائيات
        self.stats = {
            "events_detected": 0,
            "events_processed": 0,
            "reactions_executed": 0,
            "scheduled_events": 0
        }
        
        # تهيئة قواعد رد الفعل
        self._init_auto_reactions()
    
    def _init_auto_reactions(self):
        """تهيئة قواعد رد الفعل التلقائي."""
        self.auto_reactions = {
            EventType.PRICE_CRASH: [
                {
                    "condition": {"impact": EventImpact.VERY_NEGATIVE},
                    "action": "emergency_exit",
                    "parameters": {"exit_percentage": 100}
                },
                {
                    "condition": {"impact": EventImpact.NEGATIVE},
                    "action": "reduce_exposure",
                    "parameters": {"reduce_percentage": 50}
                }
            ],
            EventType.PRICE_SPIKE: [
                {
                    "condition": {"impact": EventImpact.VERY_POSITIVE},
                    "action": "take_profit",
                    "parameters": {"profit_percentage": 50}
                }
            ],
            EventType.VOLATILITY_SPIKE: [
                {
                    "condition": {},
                    "action": "tighten_stops",
                    "parameters": {"stop_multiplier": 0.5}
                },
                {
                    "condition": {},
                    "action": "reduce_position_size",
                    "parameters": {"size_multiplier": 0.5}
                }
            ],
            EventType.LISTING: [
                {
                    "condition": {},
                    "action": "add_to_watchlist",
                    "parameters": {}
                }
            ],
            EventType.DELISTING: [
                {
                    "condition": {},
                    "action": "emergency_exit_symbol",
                    "parameters": {}
                }
            ],
            EventType.WHALE_MOVEMENT: [
                {
                    "condition": {"data.direction": "sell"},
                    "action": "alert_and_monitor",
                    "parameters": {"alert_level": "high"}
                }
            ],
            EventType.EXCHANGE_HACK: [
                {
                    "condition": {},
                    "action": "emergency_withdrawal",
                    "parameters": {}
                }
            ]
        }
    
    async def emit_event(self, event: Event) -> str:
        """
        إطلاق حدث.
        
        Args:
            event: الحدث
            
        Returns:
            معرف الحدث
        """
        # حفظ الحدث
        self.active_events[event.id] = event
        self.event_history.append(event)
        self.stats["events_detected"] += 1
        
        self.logger.info(
            f"حدث جديد: {event.type.value} - {event.title} "
            f"(أولوية: {event.priority.value})"
        )
        
        # معالجة الحدث
        await self._process_event(event)
        
        # إشعار المشتركين
        await self._notify_subscribers(event)
        
        # تنفيذ ردود الفعل التلقائية
        await self._execute_auto_reactions(event)
        
        return event.id
    
    async def _process_event(self, event: Event):
        """معالجة حدث."""
        # استدعاء المعالجات المسجلة
        handlers = self.event_handlers.get(event.type, [])
        
        for handler in handlers:
            try:
                await handler(event)
            except Exception as e:
                self.logger.error(f"خطأ في معالجة الحدث: {e}")
        
        event.processed = True
        self.stats["events_processed"] += 1
    
    async def _notify_subscribers(self, event: Event):
        """إشعار المشتركين."""
        for subscriber_id, subscribed_types in self.subscribers.items():
            if event.type in subscribed_types:
                # إرسال إشعار (يمكن تخصيصه)
                self.logger.debug(f"إشعار {subscriber_id} بالحدث {event.id}")
    
    async def _execute_auto_reactions(self, event: Event):
        """تنفيذ ردود الفعل التلقائية."""
        reactions = self.auto_reactions.get(event.type, [])
        
        for reaction_rule in reactions:
            if self._check_condition(event, reaction_rule.get("condition", {})):
                reaction = EventReaction(
                    event_id=event.id,
                    action=reaction_rule["action"],
                    parameters=reaction_rule.get("parameters", {}),
                    executed_at=datetime.utcnow(),
                    success=True
                )
                
                # تنفيذ الإجراء
                result = await self._execute_action(
                    reaction_rule["action"],
                    reaction_rule.get("parameters", {}),
                    event
                )
                
                reaction.result = result
                self.reactions.append(reaction)
                event.actions_taken.append(reaction_rule["action"])
                
                self.stats["reactions_executed"] += 1
                
                self.logger.info(
                    f"رد فعل تلقائي: {reaction_rule['action']} "
                    f"للحدث {event.type.value}"
                )
    
    def _check_condition(self, 
                        event: Event,
                        condition: Dict[str, Any]) -> bool:
        """فحص شرط."""
        if not condition:
            return True
        
        for key, expected_value in condition.items():
            if key == "impact":
                if event.impact != expected_value:
                    return False
            elif key.startswith("data."):
                data_key = key[5:]
                if event.data.get(data_key) != expected_value:
                    return False
        
        return True
    
    async def _execute_action(self,
                             action: str,
                             parameters: Dict[str, Any],
                             event: Event) -> Dict[str, Any]:
        """تنفيذ إجراء."""
        # هنا يتم تنفيذ الإجراءات الفعلية
        # في الواقع، يتم ربط هذا بنظام التداول
        
        result = {
            "action": action,
            "status": "executed",
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if action == "emergency_exit":
            result["details"] = "تم إغلاق جميع الصفقات"
        elif action == "reduce_exposure":
            result["details"] = f"تم تقليل التعرض بنسبة {parameters.get('reduce_percentage', 50)}%"
        elif action == "tighten_stops":
            result["details"] = "تم تضييق وقف الخسارة"
        elif action == "add_to_watchlist":
            result["details"] = f"تمت إضافة {event.affected_symbols} للمراقبة"
        
        return result
    
    def register_handler(self,
                        event_type: EventType,
                        handler: Callable):
        """تسجيل معالج حدث."""
        self.event_handlers[event_type].append(handler)
    
    def subscribe(self,
                 subscriber_id: str,
                 event_types: List[EventType]):
        """الاشتراك في أنواع أحداث."""
        self.subscribers[subscriber_id].update(event_types)
    
    def unsubscribe(self,
                   subscriber_id: str,
                   event_types: List[EventType] = None):
        """إلغاء الاشتراك."""
        if event_types:
            for et in event_types:
                self.subscribers[subscriber_id].discard(et)
        else:
            del self.subscribers[subscriber_id]
    
    async def schedule_event(self, scheduled: ScheduledEvent):
        """
        جدولة حدث.
        
        Args:
            scheduled: الحدث المجدول
        """
        self.scheduled_events[scheduled.id] = scheduled
        self.stats["scheduled_events"] += 1
        
        self.logger.info(
            f"حدث مجدول: {scheduled.title} "
            f"في {scheduled.scheduled_time}"
        )
    
    async def check_scheduled_events(self) -> List[ScheduledEvent]:
        """
        فحص الأحداث المجدولة القريبة.
        
        Returns:
            الأحداث القريبة
        """
        now = datetime.utcnow()
        upcoming = []
        
        for event in self.scheduled_events.values():
            time_until = (event.scheduled_time - now).total_seconds()
            
            # أحداث خلال الساعة القادمة
            if 0 < time_until <= 3600:
                upcoming.append(event)
        
        return upcoming
    
    async def detect_market_event(self,
                                 symbol: str,
                                 current_price: float,
                                 previous_price: float,
                                 current_volume: float,
                                 avg_volume: float) -> Optional[Event]:
        """
        اكتشاف حدث سوقي.
        
        Args:
            symbol: الرمز
            current_price: السعر الحالي
            previous_price: السعر السابق
            current_volume: الحجم الحالي
            avg_volume: متوسط الحجم
            
        Returns:
            الحدث المكتشف إن وجد
        """
        price_change = (current_price - previous_price) / previous_price
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
        
        event = None
        
        # ارتفاع حاد
        if price_change > 0.05:
            event = Event(
                id=f"spike_{symbol}_{datetime.utcnow().timestamp()}",
                type=EventType.PRICE_SPIKE,
                priority=EventPriority.HIGH if price_change > 0.1 else EventPriority.MEDIUM,
                title=f"ارتفاع حاد في {symbol}",
                description=f"ارتفع السعر بنسبة {price_change:.1%}",
                source="market_detector",
                timestamp=datetime.utcnow(),
                impact=EventImpact.VERY_POSITIVE if price_change > 0.1 else EventImpact.POSITIVE,
                affected_symbols=[symbol],
                data={"price_change": price_change, "volume_ratio": volume_ratio}
            )
        
        # انهيار
        elif price_change < -0.05:
            event = Event(
                id=f"crash_{symbol}_{datetime.utcnow().timestamp()}",
                type=EventType.PRICE_CRASH,
                priority=EventPriority.CRITICAL if price_change < -0.1 else EventPriority.HIGH,
                title=f"انهيار في {symbol}",
                description=f"انخفض السعر بنسبة {abs(price_change):.1%}",
                source="market_detector",
                timestamp=datetime.utcnow(),
                impact=EventImpact.VERY_NEGATIVE if price_change < -0.1 else EventImpact.NEGATIVE,
                affected_symbols=[symbol],
                data={"price_change": price_change, "volume_ratio": volume_ratio}
            )
        
        # ارتفاع حجم
        elif volume_ratio > 3:
            event = Event(
                id=f"volume_{symbol}_{datetime.utcnow().timestamp()}",
                type=EventType.VOLUME_SURGE,
                priority=EventPriority.MEDIUM,
                title=f"ارتفاع حجم في {symbol}",
                description=f"الحجم أعلى بـ {volume_ratio:.1f}x من المتوسط",
                source="market_detector",
                timestamp=datetime.utcnow(),
                impact=EventImpact.NEUTRAL,
                affected_symbols=[symbol],
                data={"price_change": price_change, "volume_ratio": volume_ratio}
            )
        
        if event:
            await self.emit_event(event)
        
        return event
    
    def add_auto_reaction(self,
                         event_type: EventType,
                         condition: Dict[str, Any],
                         action: str,
                         parameters: Dict[str, Any] = None):
        """إضافة قاعدة رد فعل تلقائي."""
        if event_type not in self.auto_reactions:
            self.auto_reactions[event_type] = []
        
        self.auto_reactions[event_type].append({
            "condition": condition,
            "action": action,
            "parameters": parameters or {}
        })
    
    def get_recent_events(self,
                         event_type: EventType = None,
                         hours: int = 24) -> List[Event]:
        """
        الحصول على الأحداث الأخيرة.
        
        Args:
            event_type: نوع الحدث (اختياري)
            hours: عدد الساعات
            
        Returns:
            الأحداث
        """
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        
        events = [
            e for e in self.event_history
            if e.timestamp > cutoff
        ]
        
        if event_type:
            events = [e for e in events if e.type == event_type]
        
        return sorted(events, key=lambda e: e.timestamp, reverse=True)
    
    def get_event_summary(self) -> Dict[str, Any]:
        """
        الحصول على ملخص الأحداث.
        
        Returns:
            الملخص
        """
        recent = self.get_recent_events(hours=24)
        
        by_type = defaultdict(int)
        by_impact = defaultdict(int)
        
        for event in recent:
            by_type[event.type.value] += 1
            by_impact[event.impact.value] += 1
        
        return {
            "total_24h": len(recent),
            "by_type": dict(by_type),
            "by_impact": dict(by_impact),
            "active_events": len(self.active_events),
            "scheduled_events": len(self.scheduled_events),
            "stats": self.stats
        }
    
    def get_economic_calendar(self,
                             days: int = 7) -> List[ScheduledEvent]:
        """
        الحصول على التقويم الاقتصادي.
        
        Args:
            days: عدد الأيام
            
        Returns:
            الأحداث المجدولة
        """
        now = datetime.utcnow()
        end = now + timedelta(days=days)
        
        events = [
            e for e in self.scheduled_events.values()
            if now <= e.scheduled_time <= end
        ]
        
        return sorted(events, key=lambda e: e.scheduled_time)
