"""
Legendary Trading System - Enhanced Inner Dialogue
نظام التداول الخارق - الحوار الداخلي المحسن

نظام حوار داخلي متقدم مع سياق مستمر وتعلم من التجارب.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
from enum import Enum
import logging
import hashlib


class DialogueParticipant(Enum):
    """المشاركون في الحوار"""
    ANALYST = "analyst"           # المحلل
    RISK_MANAGER = "risk_manager" # مدير المخاطر
    STRATEGIST = "strategist"     # الاستراتيجي
    INTUITION = "intuition"       # الحدس
    CRITIC = "critic"             # الناقد
    OPTIMIST = "optimist"         # المتفائل
    PESSIMIST = "pessimist"       # المتشائم
    EXECUTOR = "executor"         # المنفذ
    MEMORY = "memory"             # الذاكرة
    CONSCIENCE = "conscience"     # الضمير


class DialogueType(Enum):
    """أنواع الحوار"""
    ANALYSIS = "analysis"         # تحليل
    DECISION = "decision"         # قرار
    REFLECTION = "reflection"     # تأمل
    DEBATE = "debate"             # مناظرة
    LEARNING = "learning"         # تعلم
    EMERGENCY = "emergency"       # طوارئ


@dataclass
class DialogueMessage:
    """رسالة في الحوار"""
    id: str
    participant: DialogueParticipant
    content: str
    timestamp: datetime
    context: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.5
    references: List[str] = field(default_factory=list)  # مراجع لرسائل سابقة
    emotions: Dict[str, float] = field(default_factory=dict)  # المشاعر المرتبطة


@dataclass
class DialogueSession:
    """جلسة حوار"""
    id: str
    type: DialogueType
    topic: str
    participants: List[DialogueParticipant]
    messages: List[DialogueMessage] = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    conclusion: Optional[str] = None
    decision: Optional[Dict[str, Any]] = None
    outcome: Optional[Dict[str, Any]] = None  # نتيجة القرار بعد التنفيذ
    learned_lessons: List[str] = field(default_factory=list)


@dataclass
class DialogueContext:
    """سياق الحوار المستمر"""
    current_market_state: Dict[str, Any] = field(default_factory=dict)
    recent_decisions: deque = field(default_factory=lambda: deque(maxlen=50))
    active_positions: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    emotional_state: Dict[str, float] = field(default_factory=dict)
    pending_concerns: List[str] = field(default_factory=list)
    learned_patterns: Dict[str, Any] = field(default_factory=dict)


class EnhancedInnerDialogue:
    """
    نظام الحوار الداخلي المحسن.
    
    يوفر حواراً داخلياً متقدماً مع:
    - سياق مستمر بين الجلسات
    - ذاكرة للحوارات السابقة
    - تعلم من القرارات الناجحة والفاشلة
    - تكامل مع LLM للتفكير العميق
    """
    
    def __init__(self, llm_client=None, memory_system=None):
        self.logger = logging.getLogger("EnhancedInnerDialogue")
        self.llm_client = llm_client
        self.memory_system = memory_system
        
        # السياق المستمر
        self.context = DialogueContext()
        
        # تاريخ الجلسات
        self.session_history: deque = deque(maxlen=1000)
        self.current_session: Optional[DialogueSession] = None
        
        # أنماط الحوار المتعلمة
        self.dialogue_patterns: Dict[str, List[DialogueSession]] = {}
        
        # شخصيات المشاركين
        self.participant_personalities = self._init_personalities()
        
        # إحصائيات
        self.stats = {
            "total_sessions": 0,
            "successful_decisions": 0,
            "failed_decisions": 0,
            "lessons_learned": 0
        }
    
    def _init_personalities(self) -> Dict[DialogueParticipant, Dict[str, Any]]:
        """تهيئة شخصيات المشاركين."""
        return {
            DialogueParticipant.ANALYST: {
                "name": "المحلل",
                "style": "موضوعي ومبني على البيانات",
                "focus": "الأرقام والمؤشرات الفنية",
                "bias": "محايد",
                "prompt_template": "كمحلل فني، أرى أن {topic}. البيانات تشير إلى: {data}"
            },
            DialogueParticipant.RISK_MANAGER: {
                "name": "مدير المخاطر",
                "style": "حذر ومحافظ",
                "focus": "حماية رأس المال",
                "bias": "متحفظ",
                "prompt_template": "من منظور إدارة المخاطر، يجب أن نأخذ بعين الاعتبار: {risks}"
            },
            DialogueParticipant.STRATEGIST: {
                "name": "الاستراتيجي",
                "style": "طويل المدى ومنهجي",
                "focus": "الصورة الكبيرة",
                "bias": "محايد",
                "prompt_template": "استراتيجياً، الوضع الحالي يتطلب: {strategy}"
            },
            DialogueParticipant.INTUITION: {
                "name": "الحدس",
                "style": "غريزي وسريع",
                "focus": "الأنماط غير الواضحة",
                "bias": "متغير",
                "prompt_template": "شعوري يقول أن {feeling}. هذا مبني على {patterns}"
            },
            DialogueParticipant.CRITIC: {
                "name": "الناقد",
                "style": "متشكك ومدقق",
                "focus": "نقاط الضعف",
                "bias": "سلبي قليلاً",
                "prompt_template": "لكن ماذا عن {concerns}؟ هل فكرنا في {risks}؟"
            },
            DialogueParticipant.OPTIMIST: {
                "name": "المتفائل",
                "style": "إيجابي ومتحمس",
                "focus": "الفرص",
                "bias": "إيجابي",
                "prompt_template": "هذه فرصة ممتازة لأن {reasons}. الإيجابيات تشمل: {positives}"
            },
            DialogueParticipant.PESSIMIST: {
                "name": "المتشائم",
                "style": "حذر ومتوقع للأسوأ",
                "focus": "المخاطر المحتملة",
                "bias": "سلبي",
                "prompt_template": "ولكن ماذا لو {worst_case}؟ المخاطر المحتملة: {risks}"
            },
            DialogueParticipant.EXECUTOR: {
                "name": "المنفذ",
                "style": "عملي ومباشر",
                "focus": "التنفيذ",
                "bias": "محايد",
                "prompt_template": "للتنفيذ، نحتاج: {steps}. التوقيت المثالي: {timing}"
            },
            DialogueParticipant.MEMORY: {
                "name": "الذاكرة",
                "style": "تاريخي ومرجعي",
                "focus": "التجارب السابقة",
                "bias": "محايد",
                "prompt_template": "في حالات مشابهة سابقاً: {history}. الدروس المستفادة: {lessons}"
            },
            DialogueParticipant.CONSCIENCE: {
                "name": "الضمير",
                "style": "أخلاقي ومسؤول",
                "focus": "القرارات الصحيحة",
                "bias": "محايد",
                "prompt_template": "هل هذا القرار صحيح؟ {ethical_considerations}"
            }
        }
    
    async def start_session(self, 
                           dialogue_type: DialogueType,
                           topic: str,
                           initial_context: Dict[str, Any] = None) -> DialogueSession:
        """
        بدء جلسة حوار جديدة.
        
        Args:
            dialogue_type: نوع الحوار
            topic: موضوع الحوار
            initial_context: السياق الأولي
            
        Returns:
            جلسة الحوار الجديدة
        """
        # إنهاء الجلسة السابقة إن وجدت
        if self.current_session:
            await self.end_session()
        
        # تحديد المشاركين حسب نوع الحوار
        participants = self._select_participants(dialogue_type)
        
        # إنشاء الجلسة
        session_id = hashlib.md5(
            f"{datetime.utcnow().isoformat()}_{topic}".encode()
        ).hexdigest()[:12]
        
        self.current_session = DialogueSession(
            id=session_id,
            type=dialogue_type,
            topic=topic,
            participants=participants
        )
        
        # تحديث السياق
        if initial_context:
            self.context.current_market_state.update(initial_context)
        
        self.logger.info(f"بدء جلسة حوار: {dialogue_type.value} - {topic}")
        
        # رسالة افتتاحية
        await self._add_opening_message(topic, initial_context)
        
        return self.current_session
    
    def _select_participants(self, dialogue_type: DialogueType) -> List[DialogueParticipant]:
        """اختيار المشاركين حسب نوع الحوار."""
        if dialogue_type == DialogueType.ANALYSIS:
            return [
                DialogueParticipant.ANALYST,
                DialogueParticipant.MEMORY,
                DialogueParticipant.INTUITION
            ]
        elif dialogue_type == DialogueType.DECISION:
            return [
                DialogueParticipant.ANALYST,
                DialogueParticipant.RISK_MANAGER,
                DialogueParticipant.STRATEGIST,
                DialogueParticipant.OPTIMIST,
                DialogueParticipant.PESSIMIST,
                DialogueParticipant.EXECUTOR
            ]
        elif dialogue_type == DialogueType.DEBATE:
            return [
                DialogueParticipant.OPTIMIST,
                DialogueParticipant.PESSIMIST,
                DialogueParticipant.CRITIC,
                DialogueParticipant.STRATEGIST
            ]
        elif dialogue_type == DialogueType.REFLECTION:
            return [
                DialogueParticipant.MEMORY,
                DialogueParticipant.CONSCIENCE,
                DialogueParticipant.CRITIC
            ]
        elif dialogue_type == DialogueType.LEARNING:
            return [
                DialogueParticipant.MEMORY,
                DialogueParticipant.ANALYST,
                DialogueParticipant.STRATEGIST
            ]
        elif dialogue_type == DialogueType.EMERGENCY:
            return [
                DialogueParticipant.RISK_MANAGER,
                DialogueParticipant.EXECUTOR,
                DialogueParticipant.INTUITION
            ]
        
        return list(DialogueParticipant)
    
    async def _add_opening_message(self, topic: str, context: Dict[str, Any] = None):
        """إضافة رسالة افتتاحية."""
        # استرجاع تجارب سابقة مشابهة
        similar_sessions = await self._find_similar_sessions(topic)
        
        opening = f"موضوع النقاش: {topic}\n"
        
        if similar_sessions:
            opening += f"\nتجارب سابقة مشابهة: {len(similar_sessions)} جلسة\n"
            for session in similar_sessions[:3]:
                if session.outcome:
                    result = "ناجح" if session.outcome.get("success") else "فاشل"
                    opening += f"- {session.topic}: {result}\n"
        
        if context:
            opening += f"\nالسياق الحالي: {json.dumps(context, ensure_ascii=False, indent=2)}"
        
        await self.add_message(
            DialogueParticipant.MEMORY,
            opening,
            confidence=1.0
        )
    
    async def add_message(self,
                         participant: DialogueParticipant,
                         content: str,
                         confidence: float = 0.5,
                         context: Dict[str, Any] = None,
                         references: List[str] = None) -> DialogueMessage:
        """
        إضافة رسالة للحوار.
        
        Args:
            participant: المشارك
            content: محتوى الرسالة
            confidence: مستوى الثقة
            context: سياق إضافي
            references: مراجع لرسائل سابقة
            
        Returns:
            الرسالة المضافة
        """
        if not self.current_session:
            raise RuntimeError("لا توجد جلسة حوار نشطة")
        
        message_id = f"{self.current_session.id}_{len(self.current_session.messages)}"
        
        message = DialogueMessage(
            id=message_id,
            participant=participant,
            content=content,
            timestamp=datetime.utcnow(),
            confidence=confidence,
            context=context or {},
            references=references or []
        )
        
        self.current_session.messages.append(message)
        
        self.logger.debug(
            f"[{participant.value}] ({confidence:.0%}): {content[:100]}..."
        )
        
        return message
    
    async def conduct_dialogue(self, 
                              rounds: int = 3,
                              market_data: Dict[str, Any] = None) -> List[DialogueMessage]:
        """
        إجراء حوار كامل بين المشاركين.
        
        Args:
            rounds: عدد جولات الحوار
            market_data: بيانات السوق
            
        Returns:
            قائمة الرسائل
        """
        if not self.current_session:
            raise RuntimeError("لا توجد جلسة حوار نشطة")
        
        messages = []
        
        for round_num in range(rounds):
            self.logger.debug(f"جولة الحوار {round_num + 1}/{rounds}")
            
            for participant in self.current_session.participants:
                # توليد رد المشارك
                response = await self._generate_participant_response(
                    participant,
                    market_data,
                    round_num
                )
                
                if response:
                    message = await self.add_message(
                        participant,
                        response["content"],
                        response.get("confidence", 0.5),
                        response.get("context")
                    )
                    messages.append(message)
        
        return messages
    
    async def _generate_participant_response(self,
                                            participant: DialogueParticipant,
                                            market_data: Dict[str, Any],
                                            round_num: int) -> Optional[Dict[str, Any]]:
        """توليد رد المشارك."""
        personality = self.participant_personalities[participant]
        
        # بناء السياق للرد
        context = {
            "topic": self.current_session.topic,
            "round": round_num,
            "previous_messages": [
                {"participant": m.participant.value, "content": m.content[:200]}
                for m in self.current_session.messages[-5:]
            ],
            "market_data": market_data,
            "personality": personality
        }
        
        # استخدام LLM إن وجد
        if self.llm_client:
            return await self._generate_llm_response(participant, context)
        
        # توليد رد بسيط بدون LLM
        return self._generate_simple_response(participant, context)
    
    async def _generate_llm_response(self,
                                    participant: DialogueParticipant,
                                    context: Dict[str, Any]) -> Dict[str, Any]:
        """توليد رد باستخدام LLM."""
        personality = self.participant_personalities[participant]
        
        prompt = f"""أنت {personality['name']} في نظام تداول ذكي.
        
شخصيتك: {personality['style']}
تركيزك: {personality['focus']}
ميولك: {personality['bias']}

الموضوع: {context['topic']}

الرسائل السابقة:
{json.dumps(context['previous_messages'], ensure_ascii=False, indent=2)}

بيانات السوق:
{json.dumps(context.get('market_data', {}), ensure_ascii=False, indent=2)}

قدم رأيك بشكل مختصر (جملتين إلى ثلاث جمل) من منظور شخصيتك.
"""
        
        try:
            response = await self.llm_client.generate(prompt)
            return {
                "content": response,
                "confidence": 0.7,
                "context": {"generated_by": "llm"}
            }
        except Exception as e:
            self.logger.error(f"خطأ في توليد رد LLM: {e}")
            return self._generate_simple_response(participant, context)
    
    def _generate_simple_response(self,
                                 participant: DialogueParticipant,
                                 context: Dict[str, Any]) -> Dict[str, Any]:
        """توليد رد بسيط بدون LLM."""
        personality = self.participant_personalities[participant]
        topic = context['topic']
        
        responses = {
            DialogueParticipant.ANALYST: f"التحليل الفني يشير إلى {topic}. المؤشرات تحتاج مراجعة.",
            DialogueParticipant.RISK_MANAGER: f"من منظور المخاطر، يجب الحذر في {topic}.",
            DialogueParticipant.STRATEGIST: f"استراتيجياً، {topic} يتطلب نهجاً متوازناً.",
            DialogueParticipant.INTUITION: f"حدسي يقول أن {topic} قد يكون فرصة.",
            DialogueParticipant.CRITIC: f"لكن هل فكرنا في المخاطر المحتملة لـ {topic}؟",
            DialogueParticipant.OPTIMIST: f"أرى فرصة جيدة في {topic}!",
            DialogueParticipant.PESSIMIST: f"أخشى أن {topic} قد لا يكون الخيار الأمثل.",
            DialogueParticipant.EXECUTOR: f"للتنفيذ، نحتاج خطة واضحة لـ {topic}.",
            DialogueParticipant.MEMORY: f"في تجارب سابقة مشابهة لـ {topic}...",
            DialogueParticipant.CONSCIENCE: f"هل {topic} هو القرار الصحيح؟"
        }
        
        return {
            "content": responses.get(participant, f"رأيي في {topic}..."),
            "confidence": 0.5,
            "context": {"generated_by": "simple"}
        }
    
    async def reach_consensus(self) -> Dict[str, Any]:
        """
        الوصول لإجماع من الحوار.
        
        Returns:
            القرار النهائي مع مستوى الإجماع
        """
        if not self.current_session or not self.current_session.messages:
            return {"decision": None, "consensus": 0}
        
        # تحليل الرسائل
        sentiments = {"positive": 0, "negative": 0, "neutral": 0}
        total_confidence = 0
        
        for message in self.current_session.messages:
            # تصنيف بسيط للمشاعر
            content_lower = message.content.lower()
            if any(word in content_lower for word in ["فرصة", "إيجابي", "جيد", "ممتاز", "شراء"]):
                sentiments["positive"] += message.confidence
            elif any(word in content_lower for word in ["خطر", "حذر", "سلبي", "بيع", "خسارة"]):
                sentiments["negative"] += message.confidence
            else:
                sentiments["neutral"] += message.confidence
            
            total_confidence += message.confidence
        
        # حساب الإجماع
        if total_confidence == 0:
            return {"decision": "hold", "consensus": 0}
        
        positive_ratio = sentiments["positive"] / total_confidence
        negative_ratio = sentiments["negative"] / total_confidence
        
        # تحديد القرار
        if positive_ratio > 0.6:
            decision = "buy"
            consensus = positive_ratio
        elif negative_ratio > 0.6:
            decision = "sell"
            consensus = negative_ratio
        else:
            decision = "hold"
            consensus = sentiments["neutral"] / total_confidence
        
        result = {
            "decision": decision,
            "consensus": consensus,
            "sentiments": sentiments,
            "message_count": len(self.current_session.messages),
            "participants": [p.value for p in self.current_session.participants]
        }
        
        self.current_session.decision = result
        
        return result
    
    async def end_session(self, conclusion: str = None) -> DialogueSession:
        """
        إنهاء جلسة الحوار.
        
        Args:
            conclusion: الخلاصة النهائية
            
        Returns:
            الجلسة المنتهية
        """
        if not self.current_session:
            return None
        
        self.current_session.end_time = datetime.utcnow()
        self.current_session.conclusion = conclusion
        
        # حفظ في التاريخ
        self.session_history.append(self.current_session)
        
        # تحديث الإحصائيات
        self.stats["total_sessions"] += 1
        
        # حفظ في الذاكرة
        if self.memory_system:
            await self._save_to_memory(self.current_session)
        
        session = self.current_session
        self.current_session = None
        
        self.logger.info(f"انتهت جلسة الحوار: {session.id}")
        
        return session
    
    async def record_outcome(self, 
                            session_id: str,
                            outcome: Dict[str, Any]):
        """
        تسجيل نتيجة القرار بعد التنفيذ.
        
        Args:
            session_id: معرف الجلسة
            outcome: النتيجة
        """
        # البحث عن الجلسة
        session = None
        for s in self.session_history:
            if s.id == session_id:
                session = s
                break
        
        if not session:
            self.logger.warning(f"لم يتم العثور على الجلسة: {session_id}")
            return
        
        session.outcome = outcome
        
        # تحديث الإحصائيات
        if outcome.get("success"):
            self.stats["successful_decisions"] += 1
        else:
            self.stats["failed_decisions"] += 1
        
        # استخراج الدروس
        lessons = await self._extract_lessons(session)
        session.learned_lessons = lessons
        self.stats["lessons_learned"] += len(lessons)
        
        # تحديث أنماط الحوار
        await self._update_patterns(session)
        
        self.logger.info(
            f"تم تسجيل نتيجة الجلسة {session_id}: "
            f"{'ناجح' if outcome.get('success') else 'فاشل'}"
        )
    
    async def _extract_lessons(self, session: DialogueSession) -> List[str]:
        """استخراج الدروس من الجلسة."""
        lessons = []
        
        if not session.outcome:
            return lessons
        
        success = session.outcome.get("success", False)
        decision = session.decision.get("decision") if session.decision else None
        
        if success:
            lessons.append(
                f"القرار '{decision}' في سياق '{session.topic}' كان ناجحاً"
            )
            if session.decision and session.decision.get("consensus", 0) > 0.7:
                lessons.append("الإجماع العالي يرتبط بقرارات ناجحة")
        else:
            lessons.append(
                f"القرار '{decision}' في سياق '{session.topic}' لم يكن موفقاً"
            )
            # تحليل أسباب الفشل
            if session.decision and session.decision.get("consensus", 0) < 0.5:
                lessons.append("الإجماع المنخفض قد يشير لقرار محفوف بالمخاطر")
        
        return lessons
    
    async def _update_patterns(self, session: DialogueSession):
        """تحديث أنماط الحوار المتعلمة."""
        pattern_key = f"{session.type.value}_{session.decision.get('decision') if session.decision else 'unknown'}"
        
        if pattern_key not in self.dialogue_patterns:
            self.dialogue_patterns[pattern_key] = []
        
        self.dialogue_patterns[pattern_key].append(session)
    
    async def _find_similar_sessions(self, topic: str) -> List[DialogueSession]:
        """البحث عن جلسات مشابهة."""
        similar = []
        
        topic_words = set(topic.lower().split())
        
        for session in self.session_history:
            session_words = set(session.topic.lower().split())
            overlap = len(topic_words & session_words)
            
            if overlap > 0:
                similar.append(session)
        
        # ترتيب حسب التشابه والحداثة
        similar.sort(key=lambda s: s.start_time, reverse=True)
        
        return similar[:10]
    
    async def _save_to_memory(self, session: DialogueSession):
        """حفظ الجلسة في الذاكرة."""
        if not self.memory_system:
            return
        
        memory_entry = {
            "type": "dialogue_session",
            "session_id": session.id,
            "topic": session.topic,
            "dialogue_type": session.type.value,
            "decision": session.decision,
            "outcome": session.outcome,
            "lessons": session.learned_lessons,
            "timestamp": session.start_time.isoformat()
        }
        
        await self.memory_system.store(memory_entry)
    
    async def get_advice_from_history(self, 
                                     topic: str,
                                     context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        الحصول على نصيحة من التاريخ.
        
        Args:
            topic: الموضوع
            context: السياق
            
        Returns:
            النصيحة المبنية على التجارب السابقة
        """
        similar_sessions = await self._find_similar_sessions(topic)
        
        if not similar_sessions:
            return {
                "advice": "لا توجد تجارب سابقة مشابهة",
                "confidence": 0.3,
                "based_on": 0
            }
        
        # تحليل النتائج السابقة
        successful = [s for s in similar_sessions if s.outcome and s.outcome.get("success")]
        failed = [s for s in similar_sessions if s.outcome and not s.outcome.get("success")]
        
        success_rate = len(successful) / len(similar_sessions) if similar_sessions else 0
        
        # جمع الدروس
        all_lessons = []
        for session in similar_sessions:
            all_lessons.extend(session.learned_lessons)
        
        advice = {
            "advice": f"بناءً على {len(similar_sessions)} تجربة سابقة مشابهة",
            "success_rate": success_rate,
            "successful_count": len(successful),
            "failed_count": len(failed),
            "lessons": all_lessons[:5],
            "confidence": min(0.9, 0.3 + (len(similar_sessions) * 0.1)),
            "recommendation": "proceed" if success_rate > 0.6 else "caution" if success_rate > 0.4 else "avoid"
        }
        
        return advice
    
    def get_stats(self) -> Dict[str, Any]:
        """الحصول على الإحصائيات."""
        success_rate = 0
        if self.stats["successful_decisions"] + self.stats["failed_decisions"] > 0:
            success_rate = self.stats["successful_decisions"] / (
                self.stats["successful_decisions"] + self.stats["failed_decisions"]
            )
        
        return {
            **self.stats,
            "success_rate": success_rate,
            "patterns_learned": len(self.dialogue_patterns),
            "history_size": len(self.session_history)
        }
