"""
Legendary Trading System - Agent Communication Protocol
نظام التداول الخارق - بروتوكول التواصل بين الوكلاء

نظام متقدم للتواصل والتنسيق بين الوكلاء.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Set
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum
import logging
import uuid
import json


class MessageType(Enum):
    """أنواع الرسائل"""
    SIGNAL = "signal"               # إشارة تداول
    ANALYSIS = "analysis"           # تحليل
    ALERT = "alert"                 # تنبيه
    REQUEST = "request"             # طلب
    RESPONSE = "response"           # رد
    VOTE = "vote"                   # تصويت
    CONSENSUS = "consensus"         # إجماع
    BROADCAST = "broadcast"         # بث عام
    CONFLICT = "conflict"           # نزاع
    RESOLUTION = "resolution"       # حل نزاع


class Priority(Enum):
    """أولويات الرسائل"""
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4


class AgentRole(Enum):
    """أدوار الوكلاء"""
    ANALYST = "analyst"
    RESEARCHER = "researcher"
    TRADER = "trader"
    RISK_MANAGER = "risk_manager"
    COORDINATOR = "coordinator"


@dataclass
class Message:
    """رسالة بين الوكلاء"""
    id: str
    type: MessageType
    sender: str
    recipients: List[str]
    content: Dict[str, Any]
    priority: Priority
    timestamp: datetime
    
    # للردود
    reply_to: Optional[str] = None
    
    # للتصويت
    vote_options: List[str] = field(default_factory=list)
    
    # حالة الرسالة
    delivered: bool = False
    acknowledged: bool = False
    
    # TTL
    expires_at: Optional[datetime] = None


@dataclass
class Vote:
    """تصويت"""
    voter: str
    option: str
    confidence: float
    reasoning: str
    timestamp: datetime


@dataclass
class Conflict:
    """نزاع بين الوكلاء"""
    id: str
    parties: List[str]
    topic: str
    positions: Dict[str, Dict[str, Any]]
    status: str  # open, resolved, escalated
    resolution: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)


class AgentCommunicationProtocol:
    """
    بروتوكول التواصل بين الوكلاء.
    
    يوفر:
    - لغة موحدة للتواصل
    - مشاركة المعرفة بين الوكلاء
    - تصويت ذكي على القرارات
    - حل النزاعات بين الوكلاء
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.logger = logging.getLogger("AgentCommunicationProtocol")
        self.config = config or {}
        
        # الوكلاء المسجلين
        self.registered_agents: Dict[str, Dict[str, Any]] = {}
        
        # صناديق الرسائل
        self.message_queues: Dict[str, asyncio.Queue] = {}
        
        # تاريخ الرسائل
        self.message_history: List[Message] = []
        
        # التصويتات النشطة
        self.active_votes: Dict[str, Dict[str, Vote]] = {}
        
        # النزاعات
        self.conflicts: Dict[str, Conflict] = {}
        
        # المشتركين في المواضيع
        self.topic_subscribers: Dict[str, Set[str]] = defaultdict(set)
        
        # معالجات الرسائل
        self.message_handlers: Dict[str, Callable] = {}
        
        # إحصائيات
        self.stats = {
            "messages_sent": 0,
            "messages_delivered": 0,
            "votes_completed": 0,
            "conflicts_resolved": 0
        }
    
    async def register_agent(self,
                            agent_id: str,
                            role: AgentRole,
                            capabilities: List[str] = None,
                            handler: Callable = None) -> bool:
        """
        تسجيل وكيل جديد.
        
        Args:
            agent_id: معرف الوكيل
            role: دور الوكيل
            capabilities: قدرات الوكيل
            handler: معالج الرسائل
            
        Returns:
            نجاح التسجيل
        """
        if agent_id in self.registered_agents:
            self.logger.warning(f"الوكيل {agent_id} مسجل مسبقاً")
            return False
        
        self.registered_agents[agent_id] = {
            "role": role,
            "capabilities": capabilities or [],
            "registered_at": datetime.utcnow(),
            "status": "active"
        }
        
        # إنشاء صندوق رسائل
        self.message_queues[agent_id] = asyncio.Queue()
        
        # تسجيل المعالج
        if handler:
            self.message_handlers[agent_id] = handler
        
        self.logger.info(f"تم تسجيل الوكيل: {agent_id} ({role.value})")
        
        return True
    
    async def send_message(self,
                          sender: str,
                          recipients: List[str],
                          content: Dict[str, Any],
                          msg_type: MessageType = MessageType.SIGNAL,
                          priority: Priority = Priority.NORMAL,
                          reply_to: str = None,
                          ttl_seconds: int = None) -> Message:
        """
        إرسال رسالة.
        
        Args:
            sender: المرسل
            recipients: المستلمين
            content: المحتوى
            msg_type: نوع الرسالة
            priority: الأولوية
            reply_to: رد على رسالة
            ttl_seconds: مدة الصلاحية
            
        Returns:
            الرسالة المرسلة
        """
        message = Message(
            id=str(uuid.uuid4()),
            type=msg_type,
            sender=sender,
            recipients=recipients,
            content=content,
            priority=priority,
            timestamp=datetime.utcnow(),
            reply_to=reply_to,
            expires_at=datetime.utcnow() + timedelta(seconds=ttl_seconds) if ttl_seconds else None
        )
        
        # توصيل الرسالة
        for recipient in recipients:
            if recipient in self.message_queues:
                await self.message_queues[recipient].put(message)
                message.delivered = True
                self.stats["messages_delivered"] += 1
        
        # حفظ في التاريخ
        self.message_history.append(message)
        self.stats["messages_sent"] += 1
        
        self.logger.debug(f"رسالة من {sender} إلى {recipients}: {msg_type.value}")
        
        return message
    
    async def broadcast(self,
                       sender: str,
                       content: Dict[str, Any],
                       topic: str = None,
                       priority: Priority = Priority.NORMAL) -> Message:
        """
        بث رسالة لجميع الوكلاء أو المشتركين في موضوع.
        
        Args:
            sender: المرسل
            content: المحتوى
            topic: الموضوع (اختياري)
            priority: الأولوية
            
        Returns:
            الرسالة المبثوثة
        """
        if topic:
            recipients = list(self.topic_subscribers.get(topic, set()))
        else:
            recipients = list(self.registered_agents.keys())
        
        # استثناء المرسل
        recipients = [r for r in recipients if r != sender]
        
        return await self.send_message(
            sender=sender,
            recipients=recipients,
            content=content,
            msg_type=MessageType.BROADCAST,
            priority=priority
        )
    
    async def receive_message(self,
                             agent_id: str,
                             timeout: float = None) -> Optional[Message]:
        """
        استلام رسالة.
        
        Args:
            agent_id: معرف الوكيل
            timeout: مهلة الانتظار
            
        Returns:
            الرسالة المستلمة
        """
        if agent_id not in self.message_queues:
            return None
        
        try:
            if timeout:
                message = await asyncio.wait_for(
                    self.message_queues[agent_id].get(),
                    timeout=timeout
                )
            else:
                message = await self.message_queues[agent_id].get()
            
            # فحص الصلاحية
            if message.expires_at and datetime.utcnow() > message.expires_at:
                return None
            
            message.acknowledged = True
            return message
            
        except asyncio.TimeoutError:
            return None
    
    def subscribe_to_topic(self, agent_id: str, topic: str):
        """الاشتراك في موضوع."""
        self.topic_subscribers[topic].add(agent_id)
    
    def unsubscribe_from_topic(self, agent_id: str, topic: str):
        """إلغاء الاشتراك من موضوع."""
        self.topic_subscribers[topic].discard(agent_id)
    
    async def start_vote(self,
                        initiator: str,
                        topic: str,
                        options: List[str],
                        voters: List[str],
                        deadline_seconds: int = 60) -> str:
        """
        بدء تصويت.
        
        Args:
            initiator: المبادر
            topic: الموضوع
            options: الخيارات
            voters: المصوتين
            deadline_seconds: مهلة التصويت
            
        Returns:
            معرف التصويت
        """
        vote_id = str(uuid.uuid4())
        
        self.active_votes[vote_id] = {}
        
        # إرسال طلب التصويت
        await self.send_message(
            sender=initiator,
            recipients=voters,
            content={
                "vote_id": vote_id,
                "topic": topic,
                "options": options,
                "deadline": (datetime.utcnow() + timedelta(seconds=deadline_seconds)).isoformat()
            },
            msg_type=MessageType.VOTE,
            priority=Priority.HIGH,
            ttl_seconds=deadline_seconds
        )
        
        self.logger.info(f"بدء تصويت: {topic}")
        
        return vote_id
    
    async def cast_vote(self,
                       vote_id: str,
                       voter: str,
                       option: str,
                       confidence: float,
                       reasoning: str = "") -> bool:
        """
        الإدلاء بصوت.
        
        Args:
            vote_id: معرف التصويت
            voter: المصوت
            option: الخيار
            confidence: الثقة
            reasoning: التبرير
            
        Returns:
            نجاح التصويت
        """
        if vote_id not in self.active_votes:
            return False
        
        self.active_votes[vote_id][voter] = Vote(
            voter=voter,
            option=option,
            confidence=confidence,
            reasoning=reasoning,
            timestamp=datetime.utcnow()
        )
        
        return True
    
    async def get_vote_result(self,
                             vote_id: str,
                             min_participation: float = 0.5) -> Dict[str, Any]:
        """
        الحصول على نتيجة التصويت.
        
        Args:
            vote_id: معرف التصويت
            min_participation: الحد الأدنى للمشاركة
            
        Returns:
            نتيجة التصويت
        """
        if vote_id not in self.active_votes:
            return {"error": "تصويت غير موجود"}
        
        votes = self.active_votes[vote_id]
        
        if not votes:
            return {"error": "لا توجد أصوات"}
        
        # حساب الأصوات المرجحة
        weighted_votes = defaultdict(float)
        total_confidence = 0
        
        for vote in votes.values():
            weighted_votes[vote.option] += vote.confidence
            total_confidence += vote.confidence
        
        # تحديد الفائز
        winner = max(weighted_votes.items(), key=lambda x: x[1])
        
        result = {
            "vote_id": vote_id,
            "total_votes": len(votes),
            "winner": winner[0],
            "winner_score": winner[1],
            "total_confidence": total_confidence,
            "consensus_strength": winner[1] / total_confidence if total_confidence > 0 else 0,
            "all_options": dict(weighted_votes),
            "voters": list(votes.keys())
        }
        
        self.stats["votes_completed"] += 1
        
        # إرسال نتيجة الإجماع
        await self.broadcast(
            sender="system",
            content=result,
            topic="vote_results",
            priority=Priority.HIGH
        )
        
        return result
    
    async def report_conflict(self,
                             reporter: str,
                             opponent: str,
                             topic: str,
                             reporter_position: Dict[str, Any],
                             opponent_position: Dict[str, Any]) -> str:
        """
        الإبلاغ عن نزاع.
        
        Args:
            reporter: المبلغ
            opponent: الخصم
            topic: الموضوع
            reporter_position: موقف المبلغ
            opponent_position: موقف الخصم
            
        Returns:
            معرف النزاع
        """
        conflict_id = str(uuid.uuid4())
        
        conflict = Conflict(
            id=conflict_id,
            parties=[reporter, opponent],
            topic=topic,
            positions={
                reporter: reporter_position,
                opponent: opponent_position
            },
            status="open"
        )
        
        self.conflicts[conflict_id] = conflict
        
        # إشعار المنسق
        await self.send_message(
            sender="system",
            recipients=self._get_coordinators(),
            content={
                "conflict_id": conflict_id,
                "parties": conflict.parties,
                "topic": topic,
                "positions": conflict.positions
            },
            msg_type=MessageType.CONFLICT,
            priority=Priority.HIGH
        )
        
        self.logger.warning(f"نزاع جديد: {topic} بين {reporter} و {opponent}")
        
        return conflict_id
    
    def _get_coordinators(self) -> List[str]:
        """الحصول على المنسقين."""
        return [
            agent_id for agent_id, info in self.registered_agents.items()
            if info["role"] == AgentRole.COORDINATOR
        ]
    
    async def resolve_conflict(self,
                              conflict_id: str,
                              resolver: str,
                              resolution: str,
                              winning_position: str = None) -> bool:
        """
        حل نزاع.
        
        Args:
            conflict_id: معرف النزاع
            resolver: الحال
            resolution: الحل
            winning_position: الموقف الفائز
            
        Returns:
            نجاح الحل
        """
        if conflict_id not in self.conflicts:
            return False
        
        conflict = self.conflicts[conflict_id]
        conflict.status = "resolved"
        conflict.resolution = resolution
        
        # إشعار الأطراف
        await self.send_message(
            sender=resolver,
            recipients=conflict.parties,
            content={
                "conflict_id": conflict_id,
                "resolution": resolution,
                "winning_position": winning_position
            },
            msg_type=MessageType.RESOLUTION,
            priority=Priority.HIGH
        )
        
        self.stats["conflicts_resolved"] += 1
        
        self.logger.info(f"تم حل النزاع: {conflict_id}")
        
        return True
    
    async def share_knowledge(self,
                             sender: str,
                             knowledge_type: str,
                             knowledge: Dict[str, Any],
                             relevance: float = 0.5) -> Message:
        """
        مشاركة المعرفة.
        
        Args:
            sender: المرسل
            knowledge_type: نوع المعرفة
            knowledge: المعرفة
            relevance: الأهمية
            
        Returns:
            رسالة المشاركة
        """
        content = {
            "knowledge_type": knowledge_type,
            "knowledge": knowledge,
            "relevance": relevance,
            "shared_at": datetime.utcnow().isoformat()
        }
        
        # تحديد المستلمين بناءً على نوع المعرفة
        recipients = self._get_relevant_agents(knowledge_type)
        
        return await self.send_message(
            sender=sender,
            recipients=recipients,
            content=content,
            msg_type=MessageType.ANALYSIS,
            priority=Priority.NORMAL if relevance < 0.7 else Priority.HIGH
        )
    
    def _get_relevant_agents(self, knowledge_type: str) -> List[str]:
        """الحصول على الوكلاء المعنيين."""
        relevance_map = {
            "technical_analysis": [AgentRole.ANALYST, AgentRole.TRADER],
            "fundamental_analysis": [AgentRole.ANALYST, AgentRole.RESEARCHER],
            "risk_assessment": [AgentRole.RISK_MANAGER, AgentRole.TRADER],
            "market_sentiment": [AgentRole.ANALYST, AgentRole.RESEARCHER],
            "trade_signal": [AgentRole.TRADER, AgentRole.RISK_MANAGER]
        }
        
        relevant_roles = relevance_map.get(knowledge_type, list(AgentRole))
        
        return [
            agent_id for agent_id, info in self.registered_agents.items()
            if info["role"] in relevant_roles
        ]
    
    async def request_analysis(self,
                              requester: str,
                              target: str,
                              analysis_type: str,
                              parameters: Dict[str, Any] = None) -> str:
        """
        طلب تحليل.
        
        Args:
            requester: الطالب
            target: المستهدف
            analysis_type: نوع التحليل
            parameters: المعاملات
            
        Returns:
            معرف الطلب
        """
        request_id = str(uuid.uuid4())
        
        await self.send_message(
            sender=requester,
            recipients=[target],
            content={
                "request_id": request_id,
                "analysis_type": analysis_type,
                "parameters": parameters or {}
            },
            msg_type=MessageType.REQUEST,
            priority=Priority.NORMAL
        )
        
        return request_id
    
    async def respond_to_request(self,
                                responder: str,
                                request_id: str,
                                requester: str,
                                response: Dict[str, Any]) -> Message:
        """
        الرد على طلب.
        
        Args:
            responder: المجيب
            request_id: معرف الطلب
            requester: الطالب
            response: الرد
            
        Returns:
            رسالة الرد
        """
        return await self.send_message(
            sender=responder,
            recipients=[requester],
            content={
                "request_id": request_id,
                "response": response
            },
            msg_type=MessageType.RESPONSE,
            reply_to=request_id
        )
    
    def get_agent_status(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """الحصول على حالة وكيل."""
        if agent_id not in self.registered_agents:
            return None
        
        info = self.registered_agents[agent_id]
        queue = self.message_queues.get(agent_id)
        
        return {
            **info,
            "pending_messages": queue.qsize() if queue else 0
        }
    
    def get_all_agents(self) -> Dict[str, Dict[str, Any]]:
        """الحصول على جميع الوكلاء."""
        return {
            agent_id: self.get_agent_status(agent_id)
            for agent_id in self.registered_agents
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """الحصول على الإحصائيات."""
        return {
            **self.stats,
            "registered_agents": len(self.registered_agents),
            "active_votes": len(self.active_votes),
            "open_conflicts": len([c for c in self.conflicts.values() if c.status == "open"]),
            "message_history_size": len(self.message_history)
        }
