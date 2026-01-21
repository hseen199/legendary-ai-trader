"""
═══════════════════════════════════════════════════════════════
LEGENDARY AGENT - Inner Dialogue
نظام الحوار الداخلي (التفكير بصوت عالٍ)
═══════════════════════════════════════════════════════════════
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from loguru import logger


class ThoughtType(Enum):
    """أنواع الأفكار"""
    OBSERVATION = "ملاحظة"
    ANALYSIS = "تحليل"
    HYPOTHESIS = "فرضية"
    DOUBT = "شك"
    CONFIDENCE = "ثقة"
    WARNING = "تحذير"
    DECISION = "قرار"
    REFLECTION = "تأمل"


class Persona(Enum):
    """الشخصيات الداخلية"""
    ANALYST = "المحلل"          # يركز على البيانات والأرقام
    RISK_MANAGER = "مدير المخاطر"  # يحذر من المخاطر
    OPTIMIST = "المتفائل"        # يرى الفرص
    SKEPTIC = "المتشكك"          # يشكك في كل شيء
    STRATEGIST = "الاستراتيجي"   # يفكر طويل المدى
    INTUITIVE = "الحدسي"         # يعتمد على الحدس


@dataclass
class Thought:
    """فكرة"""
    content: str
    type: ThoughtType
    persona: Persona
    confidence: float
    supporting_data: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class DialogueExchange:
    """تبادل حواري"""
    speaker: Persona
    message: str
    response_to: Optional[str] = None
    agreement_level: float = 0.5  # 0 = disagreement, 1 = full agreement


@dataclass
class InternalDebate:
    """نقاش داخلي"""
    topic: str
    exchanges: List[DialogueExchange]
    conclusion: Optional[str] = None
    consensus_level: float = 0.0
    final_decision: Optional[str] = None


class InnerDialogue:
    """
    نظام الحوار الداخلي
    
    يحاكي التفكير البشري من خلال:
    - شخصيات داخلية متعددة
    - نقاشات وحوارات
    - التوصل لقرارات متوازنة
    """
    
    def __init__(self):
        """تهيئة نظام الحوار الداخلي"""
        self.thoughts: List[Thought] = []
        self.debates: List[InternalDebate] = []
        self.current_mood: Dict[Persona, float] = {
            persona: 0.5 for persona in Persona
        }
        
        # قوالب الردود لكل شخصية
        self._init_response_templates()
        
        logger.info("💭 InnerDialogue initialized")
    
    def _init_response_templates(self):
        """تهيئة قوالب الردود"""
        self.templates = {
            Persona.ANALYST: {
                'bullish': [
                    "البيانات تشير إلى {indicator} عند {value}، وهذا يدعم الصعود",
                    "من الناحية الفنية، {pattern} يعطي إشارة إيجابية",
                    "الأرقام واضحة: {metric} يؤكد الاتجاه الصاعد"
                ],
                'bearish': [
                    "التحليل يظهر ضعف في {indicator}",
                    "الأرقام لا تدعم الشراء: {metric} سلبي",
                    "البيانات تحذر من {warning}"
                ],
                'neutral': [
                    "المؤشرات متضاربة، {indicator} محايد",
                    "لا توجد إشارة واضحة في البيانات",
                    "ننتظر تأكيد من {indicator}"
                ]
            },
            Persona.RISK_MANAGER: {
                'bullish': [
                    "المخاطر مقبولة، لكن يجب وضع وقف خسارة عند {stop_loss}",
                    "يمكن الدخول بحجم {size}% فقط",
                    "الفرصة جيدة لكن لا تنسَ إدارة المخاطر"
                ],
                'bearish': [
                    "⚠️ تحذير: المخاطر عالية جداً!",
                    "السحب المحتمل {drawdown}% - هل أنت مستعد؟",
                    "لا أنصح بالدخول، المخاطر تفوق العائد المتوقع"
                ],
                'neutral': [
                    "المخاطر متوسطة، قلل الحجم إذا دخلت",
                    "انتظر تأكيد قبل المخاطرة",
                    "الحذر مطلوب في هذه الظروف"
                ]
            },
            Persona.OPTIMIST: {
                'bullish': [
                    "فرصة ذهبية! 🚀 {reason}",
                    "هذا هو الوقت المثالي للشراء",
                    "أرى إمكانية ربح {potential}%"
                ],
                'bearish': [
                    "حتى في الهبوط هناك فرص",
                    "ربما هذا قاع جيد للشراء",
                    "السوق سيتعافى، الصبر مطلوب"
                ],
                'neutral': [
                    "الفرصة قادمة، فقط انتظر",
                    "أشعر بتحسن قريب في السوق",
                    "الإيجابية ستعود"
                ]
            },
            Persona.SKEPTIC: {
                'bullish': [
                    "هل أنت متأكد؟ {doubt}",
                    "لا تنخدع بالإشارات الإيجابية",
                    "ماذا لو كانت فخ؟"
                ],
                'bearish': [
                    "كما توقعت، الوضع سيء",
                    "هذا ما حذرت منه",
                    "لا تثق بأي ارتداد"
                ],
                'neutral': [
                    "لا أثق بهذا السوق",
                    "شيء ما ليس صحيحاً",
                    "أفضل البقاء خارجاً"
                ]
            },
            Persona.STRATEGIST: {
                'bullish': [
                    "على المدى الطويل، هذا يتوافق مع خطتنا",
                    "الدخول الآن يخدم الاستراتيجية العامة",
                    "هذه الصفقة جزء من خطة أكبر"
                ],
                'bearish': [
                    "استراتيجياً، الانتظار أفضل",
                    "هذا لا يتوافق مع أهدافنا طويلة المدى",
                    "نحتاج إعادة تقييم الخطة"
                ],
                'neutral': [
                    "الوضع يتطلب مراجعة الاستراتيجية",
                    "ربما نحتاج تعديل الخطة",
                    "المرونة مطلوبة الآن"
                ]
            },
            Persona.INTUITIVE: {
                'bullish': [
                    "شعوري يقول هذه فرصة جيدة",
                    "هناك شيء إيجابي في الأجواء",
                    "حدسي يدفعني للشراء"
                ],
                'bearish': [
                    "لا أشعر بالراحة تجاه هذا",
                    "شيء ما يقلقني",
                    "حدسي يحذرني"
                ],
                'neutral': [
                    "لا أستطيع تحديد شعوري",
                    "الحدس صامت الآن",
                    "أحتاج وقت لأشعر بالسوق"
                ]
            }
        }
    
    # ═══════════════════════════════════════════════════════════════
    # THOUGHT GENERATION
    # ═══════════════════════════════════════════════════════════════
    
    def think(
        self,
        observation: Dict[str, Any],
        context: Dict[str, Any]
    ) -> List[Thought]:
        """
        توليد أفكار من ملاحظة
        
        Args:
            observation: الملاحظة
            context: السياق
            
        Returns:
            قائمة الأفكار
        """
        thoughts = []
        
        # تحديد حالة السوق
        market_state = self._assess_market_state(observation)
        
        # كل شخصية تعطي رأيها
        for persona in Persona:
            thought = self._generate_persona_thought(
                persona, observation, context, market_state
            )
            thoughts.append(thought)
            self.thoughts.append(thought)
        
        return thoughts
    
    def _assess_market_state(self, observation: Dict) -> str:
        """تقييم حالة السوق"""
        features = observation.get('features', {})
        
        bullish_signals = 0
        bearish_signals = 0
        
        # RSI
        rsi = features.get('rsi_14', 50)
        if rsi < 30:
            bullish_signals += 1
        elif rsi > 70:
            bearish_signals += 1
        
        # MACD
        macd = features.get('macd', 0)
        macd_signal = features.get('macd_signal', 0)
        if macd > macd_signal:
            bullish_signals += 1
        else:
            bearish_signals += 1
        
        # Price vs MA
        close = features.get('close', 0)
        sma_50 = features.get('sma_50', close)
        if close > sma_50:
            bullish_signals += 1
        else:
            bearish_signals += 1
        
        if bullish_signals > bearish_signals:
            return 'bullish'
        elif bearish_signals > bullish_signals:
            return 'bearish'
        return 'neutral'
    
    def _generate_persona_thought(
        self,
        persona: Persona,
        observation: Dict,
        context: Dict,
        market_state: str
    ) -> Thought:
        """توليد فكرة من شخصية"""
        templates = self.templates.get(persona, {}).get(market_state, [""])
        template = np.random.choice(templates)
        
        # ملء القالب بالبيانات
        features = observation.get('features', {})
        content = template.format(
            indicator='RSI',
            value=features.get('rsi_14', 'N/A'),
            pattern='تقاطع المتوسطات',
            metric='MACD',
            warning='تقلب عالي',
            stop_loss='-2%',
            size=10,
            drawdown=5,
            reason='زخم قوي',
            potential=3,
            doubt='الحجم ضعيف'
        )
        
        # تحديد نوع الفكرة
        thought_type = self._get_thought_type(persona, market_state)
        
        # حساب الثقة بناءً على مزاج الشخصية
        base_confidence = self.current_mood[persona]
        if market_state == 'bullish' and persona in [Persona.OPTIMIST, Persona.ANALYST]:
            confidence = min(1.0, base_confidence + 0.2)
        elif market_state == 'bearish' and persona in [Persona.SKEPTIC, Persona.RISK_MANAGER]:
            confidence = min(1.0, base_confidence + 0.2)
        else:
            confidence = base_confidence
        
        return Thought(
            content=content,
            type=thought_type,
            persona=persona,
            confidence=confidence,
            supporting_data=features
        )
    
    def _get_thought_type(self, persona: Persona, market_state: str) -> ThoughtType:
        """تحديد نوع الفكرة"""
        mapping = {
            Persona.ANALYST: ThoughtType.ANALYSIS,
            Persona.RISK_MANAGER: ThoughtType.WARNING,
            Persona.OPTIMIST: ThoughtType.CONFIDENCE,
            Persona.SKEPTIC: ThoughtType.DOUBT,
            Persona.STRATEGIST: ThoughtType.REFLECTION,
            Persona.INTUITIVE: ThoughtType.OBSERVATION
        }
        return mapping.get(persona, ThoughtType.OBSERVATION)
    
    # ═══════════════════════════════════════════════════════════════
    # INTERNAL DEBATE
    # ═══════════════════════════════════════════════════════════════
    
    def debate(
        self,
        topic: str,
        observation: Dict[str, Any],
        max_rounds: int = 3
    ) -> InternalDebate:
        """
        إجراء نقاش داخلي
        
        Args:
            topic: موضوع النقاش
            observation: الملاحظة
            max_rounds: عدد الجولات
            
        Returns:
            النقاش
        """
        debate = InternalDebate(topic=topic, exchanges=[])
        
        # تحديد المشاركين بناءً على الموضوع
        if 'شراء' in topic or 'buy' in topic.lower():
            participants = [
                Persona.ANALYST, Persona.RISK_MANAGER,
                Persona.OPTIMIST, Persona.SKEPTIC
            ]
        elif 'بيع' in topic or 'sell' in topic.lower():
            participants = [
                Persona.ANALYST, Persona.RISK_MANAGER,
                Persona.STRATEGIST, Persona.SKEPTIC
            ]
        else:
            participants = list(Persona)[:4]
        
        market_state = self._assess_market_state(observation)
        
        # جولات النقاش
        for round_num in range(max_rounds):
            for persona in participants:
                # توليد رد
                response = self._generate_debate_response(
                    persona, topic, observation, market_state,
                    debate.exchanges
                )
                debate.exchanges.append(response)
        
        # التوصل لاستنتاج
        debate.conclusion, debate.consensus_level = self._reach_conclusion(
            debate.exchanges
        )
        
        # القرار النهائي
        debate.final_decision = self._make_decision(
            debate.conclusion, debate.consensus_level
        )
        
        self.debates.append(debate)
        return debate
    
    def _generate_debate_response(
        self,
        persona: Persona,
        topic: str,
        observation: Dict,
        market_state: str,
        previous_exchanges: List[DialogueExchange]
    ) -> DialogueExchange:
        """توليد رد في النقاش"""
        templates = self.templates.get(persona, {}).get(market_state, [""])
        template = np.random.choice(templates)
        
        features = observation.get('features', {})
        message = template.format(
            indicator='RSI',
            value=features.get('rsi_14', 'N/A'),
            pattern='تقاطع المتوسطات',
            metric='MACD',
            warning='تقلب عالي',
            stop_loss='-2%',
            size=10,
            drawdown=5,
            reason='زخم قوي',
            potential=3,
            doubt='الحجم ضعيف'
        )
        
        # تحديد مستوى الاتفاق مع الردود السابقة
        agreement = 0.5
        if previous_exchanges:
            last_exchange = previous_exchanges[-1]
            if self._opinions_align(persona, last_exchange.speaker, market_state):
                agreement = 0.7
            else:
                agreement = 0.3
        
        return DialogueExchange(
            speaker=persona,
            message=message,
            response_to=previous_exchanges[-1].message if previous_exchanges else None,
            agreement_level=agreement
        )
    
    def _opinions_align(
        self,
        persona1: Persona,
        persona2: Persona,
        market_state: str
    ) -> bool:
        """التحقق من توافق الآراء"""
        bullish_personas = {Persona.OPTIMIST, Persona.ANALYST}
        bearish_personas = {Persona.SKEPTIC, Persona.RISK_MANAGER}
        
        if market_state == 'bullish':
            return persona1 in bullish_personas and persona2 in bullish_personas
        elif market_state == 'bearish':
            return persona1 in bearish_personas and persona2 in bearish_personas
        return False
    
    def _reach_conclusion(
        self,
        exchanges: List[DialogueExchange]
    ) -> Tuple[str, float]:
        """التوصل لاستنتاج"""
        if not exchanges:
            return "لا يوجد استنتاج", 0.0
        
        # حساب متوسط الاتفاق
        avg_agreement = np.mean([e.agreement_level for e in exchanges])
        
        # تحديد الرأي السائد
        bullish_count = sum(
            1 for e in exchanges
            if e.speaker in [Persona.OPTIMIST, Persona.ANALYST]
            and e.agreement_level > 0.5
        )
        bearish_count = sum(
            1 for e in exchanges
            if e.speaker in [Persona.SKEPTIC, Persona.RISK_MANAGER]
            and e.agreement_level > 0.5
        )
        
        if bullish_count > bearish_count:
            conclusion = "الأغلبية تميل للشراء"
        elif bearish_count > bullish_count:
            conclusion = "الأغلبية تميل للحذر"
        else:
            conclusion = "الآراء متساوية"
        
        return conclusion, avg_agreement
    
    def _make_decision(
        self,
        conclusion: str,
        consensus_level: float
    ) -> str:
        """اتخاذ القرار النهائي"""
        if consensus_level > 0.7:
            if 'شراء' in conclusion:
                return "BUY"
            elif 'حذر' in conclusion:
                return "HOLD"
        elif consensus_level > 0.5:
            return "HOLD"
        else:
            return "WAIT"
        
        return "HOLD"
    
    # ═══════════════════════════════════════════════════════════════
    # SELF REFLECTION
    # ═══════════════════════════════════════════════════════════════
    
    def reflect(
        self,
        decision: str,
        outcome: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        التأمل في قرار سابق
        
        Args:
            decision: القرار
            outcome: النتيجة
            
        Returns:
            التأمل
        """
        was_correct = outcome.get('profitable', False)
        
        reflection = {
            'decision': decision,
            'outcome': outcome,
            'was_correct': was_correct,
            'lessons': [],
            'mood_adjustments': {}
        }
        
        # تحديث مزاج الشخصيات
        if was_correct:
            # تعزيز الشخصيات التي كانت محقة
            if decision == 'BUY':
                self.current_mood[Persona.OPTIMIST] = min(1.0, self.current_mood[Persona.OPTIMIST] + 0.1)
                self.current_mood[Persona.ANALYST] = min(1.0, self.current_mood[Persona.ANALYST] + 0.05)
            else:
                self.current_mood[Persona.SKEPTIC] = min(1.0, self.current_mood[Persona.SKEPTIC] + 0.1)
                self.current_mood[Persona.RISK_MANAGER] = min(1.0, self.current_mood[Persona.RISK_MANAGER] + 0.05)
            
            reflection['lessons'].append("القرار كان صحيحاً - استمر بنفس النهج")
        else:
            # تقليل ثقة الشخصيات التي أخطأت
            if decision == 'BUY':
                self.current_mood[Persona.OPTIMIST] = max(0.2, self.current_mood[Persona.OPTIMIST] - 0.1)
                reflection['lessons'].append("كان يجب الاستماع أكثر للمتشكك")
            else:
                self.current_mood[Persona.SKEPTIC] = max(0.2, self.current_mood[Persona.SKEPTIC] - 0.1)
                reflection['lessons'].append("كان يجب الاستماع أكثر للمتفائل")
        
        reflection['mood_adjustments'] = {
            p.value: self.current_mood[p] for p in Persona
        }
        
        return reflection
    
    # ═══════════════════════════════════════════════════════════════
    # DIALOGUE SUMMARY
    # ═══════════════════════════════════════════════════════════════
    
    def get_inner_voice(
        self,
        observation: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        الحصول على الصوت الداخلي (ملخص التفكير)
        
        Args:
            observation: الملاحظة
            context: السياق
            
        Returns:
            الصوت الداخلي
        """
        # توليد الأفكار
        thoughts = self.think(observation, context)
        
        # إجراء نقاش
        debate = self.debate("هل يجب التداول الآن؟", observation)
        
        # تجميع الصوت الداخلي
        inner_voice = {
            'thoughts': [
                {
                    'persona': t.persona.value,
                    'content': t.content,
                    'type': t.type.value,
                    'confidence': t.confidence
                }
                for t in thoughts
            ],
            'debate_conclusion': debate.conclusion,
            'consensus_level': debate.consensus_level,
            'decision': debate.final_decision,
            'dominant_persona': self._get_dominant_persona(),
            'overall_sentiment': self._calculate_sentiment(thoughts),
            'confidence': self._calculate_overall_confidence(thoughts, debate)
        }
        
        return inner_voice
    
    def _get_dominant_persona(self) -> str:
        """الحصول على الشخصية المسيطرة"""
        return max(self.current_mood, key=self.current_mood.get).value
    
    def _calculate_sentiment(self, thoughts: List[Thought]) -> str:
        """حساب المشاعر العامة"""
        positive = sum(
            1 for t in thoughts
            if t.type in [ThoughtType.CONFIDENCE, ThoughtType.DECISION]
        )
        negative = sum(
            1 for t in thoughts
            if t.type in [ThoughtType.DOUBT, ThoughtType.WARNING]
        )
        
        if positive > negative:
            return 'positive'
        elif negative > positive:
            return 'negative'
        return 'neutral'
    
    def _calculate_overall_confidence(
        self,
        thoughts: List[Thought],
        debate: InternalDebate
    ) -> float:
        """حساب الثقة الإجمالية"""
        thought_confidence = np.mean([t.confidence for t in thoughts])
        debate_confidence = debate.consensus_level
        
        return thought_confidence * 0.4 + debate_confidence * 0.6


# ═══════════════════════════════════════════════════════════════
# STANDALONE EXECUTION
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # اختبار نظام الحوار الداخلي
    dialogue = InnerDialogue()
    
    observation = {
        'features': {
            'rsi_14': 28,
            'macd': 0.5,
            'macd_signal': 0.3,
            'close': 50000,
            'sma_50': 48000
        }
    }
    
    context = {'market': 'crypto', 'symbol': 'BTCUSDT'}
    
    # الحصول على الصوت الداخلي
    inner_voice = dialogue.get_inner_voice(observation, context)
    
    print("🧠 Inner Voice:")
    print(f"Decision: {inner_voice['decision']}")
    print(f"Confidence: {inner_voice['confidence']:.2%}")
    print(f"Sentiment: {inner_voice['overall_sentiment']}")
    print(f"Dominant Persona: {inner_voice['dominant_persona']}")
    print(f"\nDebate Conclusion: {inner_voice['debate_conclusion']}")
    print(f"Consensus: {inner_voice['consensus_level']:.2%}")
    
    print("\n💭 Thoughts:")
    for thought in inner_voice['thoughts']:
        print(f"  [{thought['persona']}]: {thought['content']}")
