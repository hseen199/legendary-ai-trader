"""
Legendary Trading System - Base Agent
نظام التداول الخارق - الوكيل الأساسي

الفئة الأساسية التي يرث منها جميع الوكلاء في النظام.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional
import logging
import asyncio
from dataclasses import dataclass

from .types import AnalysisResult, SignalType


@dataclass
class AgentMessage:
    """رسالة بين الوكلاء"""
    sender: str
    receiver: str
    message_type: str
    content: Any
    timestamp: datetime
    priority: int = 0  # 0 = عادي، 1 = مهم، 2 = عاجل
    
    def to_dict(self) -> Dict:
        return {
            "sender": self.sender,
            "receiver": self.receiver,
            "message_type": self.message_type,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "priority": self.priority
        }


class BaseAgent(ABC):
    """
    الفئة الأساسية لجميع الوكلاء في النظام.
    
    كل وكيل يجب أن يرث من هذه الفئة ويُنفذ الدوال المجردة.
    """
    
    def __init__(self, name: str, config: Dict[str, Any]):
        """
        تهيئة الوكيل الأساسي.
        
        Args:
            name: اسم الوكيل الفريد
            config: إعدادات الوكيل
        """
        self.name = name
        self.config = config
        self.is_active = False
        self.logger = logging.getLogger(f"Agent.{name}")
        self._message_queue: asyncio.Queue = asyncio.Queue()
        self._subscribers: List[str] = []
        self._last_activity: Optional[datetime] = None
        self._error_count: int = 0
        self._max_errors: int = config.get("max_errors", 5)
        
    @abstractmethod
    async def initialize(self) -> bool:
        """
        تهيئة الوكيل وتحميل الموارد اللازمة.
        
        Returns:
            True إذا نجحت التهيئة، False خلاف ذلك
        """
        pass
    
    @abstractmethod
    async def process(self, data: Any) -> Any:
        """
        معالجة البيانات الواردة.
        
        Args:
            data: البيانات المراد معالجتها
            
        Returns:
            نتيجة المعالجة
        """
        pass
    
    @abstractmethod
    async def shutdown(self) -> None:
        """إيقاف الوكيل وتحرير الموارد."""
        pass
    
    async def start(self) -> None:
        """بدء تشغيل الوكيل."""
        if self.is_active:
            self.logger.warning(f"الوكيل {self.name} يعمل بالفعل")
            return
            
        try:
            success = await self.initialize()
            if success:
                self.is_active = True
                self._last_activity = datetime.utcnow()
                self.logger.info(f"تم تشغيل الوكيل {self.name} بنجاح")
            else:
                self.logger.error(f"فشل في تهيئة الوكيل {self.name}")
        except Exception as e:
            self.logger.error(f"خطأ في تشغيل الوكيل {self.name}: {e}")
            self._error_count += 1
    
    async def stop(self) -> None:
        """إيقاف الوكيل."""
        if not self.is_active:
            return
            
        try:
            await self.shutdown()
            self.is_active = False
            self.logger.info(f"تم إيقاف الوكيل {self.name}")
        except Exception as e:
            self.logger.error(f"خطأ في إيقاف الوكيل {self.name}: {e}")
    
    async def send_message(self, receiver: str, message_type: str, 
                          content: Any, priority: int = 0) -> None:
        """
        إرسال رسالة إلى وكيل آخر.
        
        Args:
            receiver: اسم الوكيل المستلم
            message_type: نوع الرسالة
            content: محتوى الرسالة
            priority: أولوية الرسالة
        """
        message = AgentMessage(
            sender=self.name,
            receiver=receiver,
            message_type=message_type,
            content=content,
            timestamp=datetime.utcnow(),
            priority=priority
        )
        # سيتم إرسال الرسالة عبر منسق الوكلاء
        self.logger.debug(f"إرسال رسالة من {self.name} إلى {receiver}: {message_type}")
        return message
    
    async def receive_message(self, message: AgentMessage) -> None:
        """
        استلام رسالة من وكيل آخر.
        
        Args:
            message: الرسالة المستلمة
        """
        await self._message_queue.put(message)
        self.logger.debug(f"استلام رسالة من {message.sender}: {message.message_type}")
    
    async def get_pending_messages(self) -> List[AgentMessage]:
        """الحصول على الرسائل المعلقة."""
        messages = []
        while not self._message_queue.empty():
            messages.append(await self._message_queue.get())
        return sorted(messages, key=lambda m: m.priority, reverse=True)
    
    def subscribe(self, agent_name: str) -> None:
        """الاشتراك في تحديثات وكيل آخر."""
        if agent_name not in self._subscribers:
            self._subscribers.append(agent_name)
    
    def unsubscribe(self, agent_name: str) -> None:
        """إلغاء الاشتراك."""
        if agent_name in self._subscribers:
            self._subscribers.remove(agent_name)
    
    def get_status(self) -> Dict[str, Any]:
        """الحصول على حالة الوكيل."""
        return {
            "name": self.name,
            "is_active": self.is_active,
            "last_activity": self._last_activity.isoformat() if self._last_activity else None,
            "error_count": self._error_count,
            "pending_messages": self._message_queue.qsize(),
            "subscribers": self._subscribers
        }
    
    def reset_errors(self) -> None:
        """إعادة تعيين عداد الأخطاء."""
        self._error_count = 0
    
    def _update_activity(self) -> None:
        """تحديث وقت آخر نشاط."""
        self._last_activity = datetime.utcnow()
    
    def _handle_error(self, error: Exception) -> None:
        """معالجة الأخطاء."""
        self._error_count += 1
        self.logger.error(f"خطأ في الوكيل {self.name}: {error}")
        
        if self._error_count >= self._max_errors:
            self.logger.critical(
                f"الوكيل {self.name} تجاوز الحد الأقصى للأخطاء ({self._max_errors})"
            )


class AnalystAgent(BaseAgent):
    """
    الفئة الأساسية للوكلاء المحللين.
    
    يرث منها جميع المحللين (أساسي، تقني، مشاعر، أخبار، On-Chain).
    """
    
    def __init__(self, name: str, config: Dict[str, Any], analyst_type: str):
        super().__init__(name, config)
        self.analyst_type = analyst_type
        self.weight = config.get("weight", 0.2)
    
    @abstractmethod
    async def analyze(self, symbol: str, data: Any) -> AnalysisResult:
        """
        تحليل رمز معين.
        
        Args:
            symbol: رمز العملة
            data: البيانات المراد تحليلها
            
        Returns:
            نتيجة التحليل
        """
        pass
    
    def _calculate_confidence(self, signals: List[float]) -> float:
        """
        حساب مستوى الثقة من مجموعة إشارات.
        
        Args:
            signals: قائمة الإشارات (قيم بين -1 و 1)
            
        Returns:
            مستوى الثقة (0.0 - 1.0)
        """
        if not signals:
            return 0.0
        
        # حساب المتوسط المطلق
        avg_signal = sum(signals) / len(signals)
        
        # حساب الاتساق (كلما كانت الإشارات متقاربة، زادت الثقة)
        variance = sum((s - avg_signal) ** 2 for s in signals) / len(signals)
        consistency = max(0, 1 - variance)
        
        # الثقة = قوة الإشارة × الاتساق
        confidence = abs(avg_signal) * consistency
        
        return min(1.0, max(0.0, confidence))
    
    def _signals_to_signal_type(self, avg_signal: float) -> SignalType:
        """
        تحويل متوسط الإشارات إلى نوع إشارة.
        
        Args:
            avg_signal: متوسط الإشارات (-1 إلى 1)
            
        Returns:
            نوع الإشارة
        """
        if avg_signal >= 0.7:
            return SignalType.STRONG_BUY
        elif avg_signal >= 0.4:
            return SignalType.BUY
        elif avg_signal >= 0.15:
            return SignalType.WEAK_BUY
        elif avg_signal >= -0.15:
            return SignalType.NEUTRAL
        elif avg_signal >= -0.4:
            return SignalType.WEAK_SELL
        elif avg_signal >= -0.7:
            return SignalType.SELL
        else:
            return SignalType.STRONG_SELL


class ResearcherAgent(BaseAgent):
    """
    الفئة الأساسية للوكلاء الباحثين.
    
    يرث منها الباحث المتفائل والباحث المتشائم.
    """
    
    def __init__(self, name: str, config: Dict[str, Any], stance: str):
        super().__init__(name, config)
        self.stance = stance  # "bullish" أو "bearish"
    
    @abstractmethod
    async def research(self, symbol: str, 
                      analysis_results: List[AnalysisResult]) -> Dict[str, Any]:
        """
        إجراء البحث بناءً على نتائج التحليل.
        
        Args:
            symbol: رمز العملة
            analysis_results: نتائج التحليل من المحللين
            
        Returns:
            تقرير البحث
        """
        pass
    
    @abstractmethod
    async def debate(self, opponent_argument: str) -> str:
        """
        الرد على حجة الخصم في المناظرة.
        
        Args:
            opponent_argument: حجة الخصم
            
        Returns:
            الرد على الحجة
        """
        pass


class TradingAgent(BaseAgent):
    """
    الفئة الأساسية لوكيل التداول.
    """
    
    @abstractmethod
    async def make_decision(self, symbol: str, 
                           analysis_results: List[AnalysisResult],
                           debate_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        اتخاذ قرار التداول.
        
        Args:
            symbol: رمز العملة
            analysis_results: نتائج التحليل
            debate_result: نتيجة المناظرة
            
        Returns:
            قرار التداول
        """
        pass
    
    @abstractmethod
    async def execute_trade(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """
        تنفيذ الصفقة.
        
        Args:
            decision: قرار التداول
            
        Returns:
            نتيجة التنفيذ
        """
        pass


class RiskManagerAgent(BaseAgent):
    """
    الفئة الأساسية لوكيل إدارة المخاطر.
    """
    
    @abstractmethod
    async def assess_risk(self, symbol: str, 
                         decision: Dict[str, Any]) -> Dict[str, Any]:
        """
        تقييم مخاطر الصفقة.
        
        Args:
            symbol: رمز العملة
            decision: قرار التداول المقترح
            
        Returns:
            تقييم المخاطر
        """
        pass
    
    @abstractmethod
    async def approve_trade(self, decision: Dict[str, Any],
                           risk_assessment: Dict[str, Any]) -> bool:
        """
        الموافقة على الصفقة أو رفضها.
        
        Args:
            decision: قرار التداول
            risk_assessment: تقييم المخاطر
            
        Returns:
            True للموافقة، False للرفض
        """
        pass
    
    @abstractmethod
    async def calculate_position_size(self, symbol: str,
                                     risk_assessment: Dict[str, Any]) -> float:
        """
        حساب حجم المركز الأمثل.
        
        Args:
            symbol: رمز العملة
            risk_assessment: تقييم المخاطر
            
        Returns:
            حجم المركز
        """
        pass
