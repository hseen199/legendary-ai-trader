"""
═══════════════════════════════════════════════════════════════
LEGENDARY AGENT - Creative Mind
العقل المبدع - يجمع كل مكونات التفكير
═══════════════════════════════════════════════════════════════
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
from loguru import logger

from .reasoning_engine import ReasoningEngine, Premise, ReasoningType
from .strategy_inventor import StrategyInventor, Strategy, StrategyType
from .inner_dialogue import InnerDialogue, Persona


class MindState(Enum):
    """حالات العقل"""
    OBSERVING = "مراقبة"
    ANALYZING = "تحليل"
    HYPOTHESIZING = "افتراض"
    DEBATING = "مناقشة"
    DECIDING = "قرار"
    REFLECTING = "تأمل"
    LEARNING = "تعلم"


@dataclass
class Decision:
    """قرار"""
    action: str  # BUY, SELL, HOLD
    confidence: float
    reasoning: str
    supporting_factors: List[str]
    risk_factors: List[str]
    strategy_used: Optional[str] = None
    inner_voice_summary: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class LearningInsight:
    """رؤية تعلم"""
    insight: str
    source: str
    confidence: float
    applicable_conditions: List[str]
    timestamp: datetime = field(default_factory=datetime.now)


class CreativeMind:
    """
    العقل المبدع
    
    يجمع بين:
    - محرك التفكير المنطقي
    - مبتكر الاستراتيجيات
    - نظام الحوار الداخلي
    
    لإنتاج قرارات ذكية ومبدعة
    """
    
    def __init__(self):
        """تهيئة العقل المبدع"""
        # المكونات الفرعية
        self.reasoning_engine = ReasoningEngine()
        self.strategy_inventor = StrategyInventor()
        self.inner_dialogue = InnerDialogue()
        
        # الحالة
        self.state = MindState.OBSERVING
        self.decisions_history: List[Decision] = []
        self.insights: List[LearningInsight] = []
        self.active_strategy: Optional[Strategy] = None
        
        # الذاكرة قصيرة المدى
        self.short_term_memory: List[Dict] = []
        self.memory_capacity = 100
        
        # إحصائيات
        self.stats = {
            'total_decisions': 0,
            'correct_decisions': 0,
            'strategies_invented': 0,
            'insights_gained': 0
        }
        
        logger.info("🧠 CreativeMind initialized - Ready to think!")
    
    # ═══════════════════════════════════════════════════════════════
    # MAIN THINKING PROCESS
    # ═══════════════════════════════════════════════════════════════
    
    def think(
        self,
        observation: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Decision:
        """
        عملية التفكير الرئيسية
        
        Args:
            observation: الملاحظة الحالية
            context: السياق
            
        Returns:
            القرار
        """
        logger.info("🧠 Starting thinking process...")
        
        # 1. المراقبة
        self.state = MindState.OBSERVING
        self._observe(observation)
        
        # 2. التحليل
        self.state = MindState.ANALYZING
        analysis = self._analyze(observation, context)
        
        # 3. توليد الفرضيات
        self.state = MindState.HYPOTHESIZING
        hypotheses = self._hypothesize(observation, analysis)
        
        # 4. النقاش الداخلي
        self.state = MindState.DEBATING
        inner_voice = self._debate(observation, context, analysis)
        
        # 5. اتخاذ القرار
        self.state = MindState.DECIDING
        decision = self._decide(analysis, hypotheses, inner_voice, context)
        
        # 6. حفظ القرار
        self.decisions_history.append(decision)
        self.stats['total_decisions'] += 1
        
        logger.info(f"🎯 Decision: {decision.action} (Confidence: {decision.confidence:.2%})")
        
        return decision
    
    def _observe(self, observation: Dict[str, Any]) -> None:
        """المراقبة وتخزين الملاحظة"""
        # إضافة للذاكرة قصيرة المدى
        self.short_term_memory.append({
            'type': 'observation',
            'data': observation,
            'timestamp': datetime.now().isoformat()
        })
        
        # الحفاظ على سعة الذاكرة
        if len(self.short_term_memory) > self.memory_capacity:
            self.short_term_memory = self.short_term_memory[-self.memory_capacity:]
    
    def _analyze(
        self,
        observation: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """التحليل"""
        features = observation.get('features', {})
        
        analysis = {
            'market_condition': self._assess_market_condition(features),
            'trend': self._identify_trend(features),
            'momentum': self._assess_momentum(features),
            'volatility': self._assess_volatility(features),
            'volume_analysis': self._analyze_volume(features),
            'key_levels': self._identify_key_levels(features),
            'patterns': self._detect_patterns(features)
        }
        
        # استخدام محرك التفكير
        premises = self._create_premises(analysis)
        conclusion = self.reasoning_engine.deduce(premises, context)
        
        if conclusion:
            analysis['reasoning_conclusion'] = {
                'statement': conclusion.statement,
                'confidence': conclusion.confidence,
                'type': conclusion.reasoning_type.value
            }
        
        return analysis
    
    def _assess_market_condition(self, features: Dict) -> str:
        """تقييم حالة السوق"""
        rsi = features.get('rsi_14', 50)
        adx = features.get('adx', 25)
        
        if adx > 25:
            if rsi > 50:
                return 'TRENDING_UP'
            else:
                return 'TRENDING_DOWN'
        else:
            return 'RANGING'
    
    def _identify_trend(self, features: Dict) -> Dict[str, Any]:
        """تحديد الاتجاه"""
        close = features.get('close', 0)
        sma_20 = features.get('sma_20', close)
        sma_50 = features.get('sma_50', close)
        sma_200 = features.get('sma_200', close)
        
        short_term = 'UP' if close > sma_20 else 'DOWN'
        medium_term = 'UP' if close > sma_50 else 'DOWN'
        long_term = 'UP' if close > sma_200 else 'DOWN'
        
        return {
            'short_term': short_term,
            'medium_term': medium_term,
            'long_term': long_term,
            'aligned': short_term == medium_term == long_term
        }
    
    def _assess_momentum(self, features: Dict) -> Dict[str, Any]:
        """تقييم الزخم"""
        rsi = features.get('rsi_14', 50)
        macd = features.get('macd', 0)
        macd_signal = features.get('macd_signal', 0)
        
        return {
            'rsi': rsi,
            'rsi_condition': 'OVERSOLD' if rsi < 30 else 'OVERBOUGHT' if rsi > 70 else 'NEUTRAL',
            'macd_bullish': macd > macd_signal,
            'strength': abs(macd - macd_signal)
        }
    
    def _assess_volatility(self, features: Dict) -> Dict[str, Any]:
        """تقييم التقلب"""
        atr = features.get('atr_14', 0)
        bb_width = features.get('bb_width', 0)
        
        return {
            'atr': atr,
            'bb_width': bb_width,
            'level': 'HIGH' if bb_width > 5 else 'LOW' if bb_width < 2 else 'MEDIUM'
        }
    
    def _analyze_volume(self, features: Dict) -> Dict[str, Any]:
        """تحليل الحجم"""
        volume = features.get('volume', 0)
        volume_sma = features.get('volume_sma_20', volume)
        
        ratio = volume / volume_sma if volume_sma > 0 else 1
        
        return {
            'current': volume,
            'average': volume_sma,
            'ratio': ratio,
            'spike': ratio > 2,
            'weak': ratio < 0.5
        }
    
    def _identify_key_levels(self, features: Dict) -> Dict[str, Any]:
        """تحديد المستويات الرئيسية"""
        close = features.get('close', 0)
        high_20 = features.get('highest_20', close)
        low_20 = features.get('lowest_20', close)
        bb_upper = features.get('bb_upper', close)
        bb_lower = features.get('bb_lower', close)
        
        return {
            'resistance': [high_20, bb_upper],
            'support': [low_20, bb_lower],
            'near_resistance': close > high_20 * 0.98,
            'near_support': close < low_20 * 1.02
        }
    
    def _detect_patterns(self, features: Dict) -> List[str]:
        """كشف الأنماط"""
        patterns = []
        
        rsi = features.get('rsi_14', 50)
        macd = features.get('macd', 0)
        macd_signal = features.get('macd_signal', 0)
        bb_percent = features.get('bb_percent', 0.5)
        
        # أنماط RSI
        if rsi < 30:
            patterns.append('RSI_OVERSOLD')
        elif rsi > 70:
            patterns.append('RSI_OVERBOUGHT')
        
        # أنماط MACD
        if macd > macd_signal and macd > 0:
            patterns.append('MACD_BULLISH_STRONG')
        elif macd > macd_signal:
            patterns.append('MACD_BULLISH')
        elif macd < macd_signal and macd < 0:
            patterns.append('MACD_BEARISH_STRONG')
        elif macd < macd_signal:
            patterns.append('MACD_BEARISH')
        
        # أنماط Bollinger
        if bb_percent > 1:
            patterns.append('BB_BREAKOUT_UP')
        elif bb_percent < 0:
            patterns.append('BB_BREAKOUT_DOWN')
        
        return patterns
    
    def _create_premises(self, analysis: Dict) -> List[Premise]:
        """إنشاء المقدمات المنطقية"""
        premises = []
        
        # مقدمة الاتجاه
        trend = analysis.get('trend', {})
        if trend.get('aligned'):
            premises.append(Premise(
                statement=f"الاتجاه متوافق: {trend.get('short_term')}",
                confidence=0.8,
                source="trend_analysis"
            ))
        
        # مقدمة الزخم
        momentum = analysis.get('momentum', {})
        if momentum.get('rsi_condition') == 'OVERSOLD':
            premises.append(Premise(
                statement="RSI في منطقة التشبع البيعي",
                confidence=0.85,
                source="momentum_analysis"
            ))
        elif momentum.get('rsi_condition') == 'OVERBOUGHT':
            premises.append(Premise(
                statement="RSI في منطقة التشبع الشرائي",
                confidence=0.85,
                source="momentum_analysis"
            ))
        
        # مقدمة الحجم
        volume = analysis.get('volume_analysis', {})
        if volume.get('spike'):
            premises.append(Premise(
                statement="ارتفاع كبير في الحجم",
                confidence=0.75,
                source="volume_analysis"
            ))
        
        return premises
    
    def _hypothesize(
        self,
        observation: Dict[str, Any],
        analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """توليد الفرضيات"""
        hypotheses = []
        
        # فرضية بناءً على الاتجاه
        trend = analysis.get('trend', {})
        if trend.get('aligned') and trend.get('short_term') == 'UP':
            hypotheses.append({
                'statement': 'استمرار الصعود محتمل',
                'probability': 0.7,
                'conditions': ['الاتجاه متوافق صعوداً']
            })
        elif trend.get('aligned') and trend.get('short_term') == 'DOWN':
            hypotheses.append({
                'statement': 'استمرار الهبوط محتمل',
                'probability': 0.7,
                'conditions': ['الاتجاه متوافق هبوطاً']
            })
        
        # فرضية بناءً على الزخم
        momentum = analysis.get('momentum', {})
        if momentum.get('rsi_condition') == 'OVERSOLD' and momentum.get('macd_bullish'):
            hypotheses.append({
                'statement': 'انعكاس صعودي محتمل',
                'probability': 0.65,
                'conditions': ['تشبع بيعي', 'MACD إيجابي']
            })
        
        # فرضية بناءً على الأنماط
        patterns = analysis.get('patterns', [])
        if 'BB_BREAKOUT_UP' in patterns:
            hypotheses.append({
                'statement': 'اختراق صعودي',
                'probability': 0.6,
                'conditions': ['اختراق البولنجر العلوي']
            })
        
        return hypotheses
    
    def _debate(
        self,
        observation: Dict[str, Any],
        context: Dict[str, Any],
        analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """النقاش الداخلي"""
        return self.inner_dialogue.get_inner_voice(observation, context)
    
    def _decide(
        self,
        analysis: Dict[str, Any],
        hypotheses: List[Dict],
        inner_voice: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Decision:
        """اتخاذ القرار"""
        # جمع العوامل
        supporting_factors = []
        risk_factors = []
        
        # تحليل الاتجاه
        trend = analysis.get('trend', {})
        if trend.get('aligned'):
            if trend.get('short_term') == 'UP':
                supporting_factors.append("الاتجاه متوافق صعوداً")
            else:
                risk_factors.append("الاتجاه متوافق هبوطاً")
        else:
            risk_factors.append("الاتجاه غير متوافق")
        
        # تحليل الزخم
        momentum = analysis.get('momentum', {})
        if momentum.get('rsi_condition') == 'OVERSOLD':
            supporting_factors.append("تشبع بيعي - فرصة شراء")
        elif momentum.get('rsi_condition') == 'OVERBOUGHT':
            risk_factors.append("تشبع شرائي - خطر الهبوط")
        
        if momentum.get('macd_bullish'):
            supporting_factors.append("MACD إيجابي")
        else:
            risk_factors.append("MACD سلبي")
        
        # تحليل التقلب
        volatility = analysis.get('volatility', {})
        if volatility.get('level') == 'HIGH':
            risk_factors.append("تقلب عالي")
        
        # تحليل الحجم
        volume = analysis.get('volume_analysis', {})
        if volume.get('spike'):
            supporting_factors.append("حجم قوي")
        elif volume.get('weak'):
            risk_factors.append("حجم ضعيف")
        
        # الفرضيات
        bullish_hypotheses = [h for h in hypotheses if 'صعود' in h['statement'] or 'صعودي' in h['statement']]
        bearish_hypotheses = [h for h in hypotheses if 'هبوط' in h['statement'] or 'هبوطي' in h['statement']]
        
        # الصوت الداخلي
        inner_decision = inner_voice.get('decision', 'HOLD')
        inner_confidence = inner_voice.get('confidence', 0.5)
        
        # حساب الدرجات
        buy_score = len(supporting_factors) * 0.3 + len(bullish_hypotheses) * 0.2
        sell_score = len(risk_factors) * 0.3 + len(bearish_hypotheses) * 0.2
        
        # تعديل بناءً على الصوت الداخلي
        if inner_decision == 'BUY':
            buy_score += inner_confidence * 0.3
        elif inner_decision == 'SELL':
            sell_score += inner_confidence * 0.3
        
        # القرار النهائي
        if buy_score > sell_score + 0.2:
            action = 'BUY'
            confidence = min(0.95, buy_score / (buy_score + sell_score + 0.1))
        elif sell_score > buy_score + 0.2:
            action = 'SELL'
            confidence = min(0.95, sell_score / (buy_score + sell_score + 0.1))
        else:
            action = 'HOLD'
            confidence = 0.5
        
        # بناء التبرير
        reasoning = self._build_reasoning(
            action, supporting_factors, risk_factors, hypotheses, inner_voice
        )
        
        return Decision(
            action=action,
            confidence=confidence,
            reasoning=reasoning,
            supporting_factors=supporting_factors,
            risk_factors=risk_factors,
            strategy_used=self.active_strategy.name if self.active_strategy else None,
            inner_voice_summary=inner_voice.get('debate_conclusion')
        )
    
    def _build_reasoning(
        self,
        action: str,
        supporting: List[str],
        risks: List[str],
        hypotheses: List[Dict],
        inner_voice: Dict
    ) -> str:
        """بناء التبرير"""
        parts = []
        
        if action == 'BUY':
            parts.append("قررت الشراء للأسباب التالية:")
            for factor in supporting[:3]:
                parts.append(f"  • {factor}")
        elif action == 'SELL':
            parts.append("قررت البيع/الانتظار للأسباب التالية:")
            for factor in risks[:3]:
                parts.append(f"  • {factor}")
        else:
            parts.append("قررت الانتظار لأن الإشارات متضاربة")
        
        if hypotheses:
            parts.append(f"\nالفرضية الرئيسية: {hypotheses[0]['statement']}")
        
        parts.append(f"\nالصوت الداخلي: {inner_voice.get('debate_conclusion', 'لا يوجد')}")
        
        return "\n".join(parts)
    
    # ═══════════════════════════════════════════════════════════════
    # LEARNING & REFLECTION
    # ═══════════════════════════════════════════════════════════════
    
    def learn_from_outcome(
        self,
        decision: Decision,
        outcome: Dict[str, Any]
    ) -> LearningInsight:
        """
        التعلم من النتيجة
        
        Args:
            decision: القرار
            outcome: النتيجة
            
        Returns:
            رؤية التعلم
        """
        self.state = MindState.LEARNING
        
        was_profitable = outcome.get('pnl', 0) > 0
        
        # تحديث الإحصائيات
        if (decision.action == 'BUY' and was_profitable) or \
           (decision.action in ['SELL', 'HOLD'] and not was_profitable):
            self.stats['correct_decisions'] += 1
        
        # التأمل في الحوار الداخلي
        reflection = self.inner_dialogue.reflect(decision.action, outcome)
        
        # استخراج الرؤية
        if was_profitable:
            insight_text = f"القرار بـ {decision.action} كان صحيحاً عندما: {', '.join(decision.supporting_factors[:2])}"
        else:
            insight_text = f"القرار بـ {decision.action} كان خاطئاً. كان يجب الانتباه لـ: {', '.join(decision.risk_factors[:2])}"
        
        insight = LearningInsight(
            insight=insight_text,
            source='outcome_analysis',
            confidence=0.7 if was_profitable else 0.6,
            applicable_conditions=decision.supporting_factors if was_profitable else decision.risk_factors
        )
        
        self.insights.append(insight)
        self.stats['insights_gained'] += 1
        
        # تحديث الأنماط المتعلمة
        self._update_learned_patterns(decision, outcome)
        
        logger.info(f"📚 Learned: {insight_text[:50]}...")
        
        return insight
    
    def _update_learned_patterns(
        self,
        decision: Decision,
        outcome: Dict[str, Any]
    ):
        """تحديث الأنماط المتعلمة"""
        pattern_key = f"{decision.action}_{hash(str(decision.supporting_factors))}"
        
        observation = {
            'features': decision.supporting_factors,
            'outcome': 'profitable' if outcome.get('pnl', 0) > 0 else 'loss'
        }
        
        # استخدام محرك التفكير للتعلم الاستقرائي
        if len(self.decisions_history) >= 10:
            recent_observations = [
                {
                    'features': d.supporting_factors,
                    'outcome': 'profitable'  # نفترض الربحية للتبسيط
                }
                for d in self.decisions_history[-10:]
            ]
            self.reasoning_engine.induce(recent_observations)
    
    # ═══════════════════════════════════════════════════════════════
    # STRATEGY MANAGEMENT
    # ═══════════════════════════════════════════════════════════════
    
    def invent_new_strategy(
        self,
        market_condition: str = None,
        risk_level: str = 'medium'
    ) -> Strategy:
        """ابتكار استراتيجية جديدة"""
        strategy = self.strategy_inventor.invent_strategy(
            market_condition=market_condition,
            risk_level=risk_level
        )
        
        self.stats['strategies_invented'] += 1
        logger.info(f"💡 Invented new strategy: {strategy.name}")
        
        return strategy
    
    def set_active_strategy(self, strategy: Strategy) -> None:
        """تعيين الاستراتيجية النشطة"""
        self.active_strategy = strategy
        logger.info(f"📋 Active strategy set to: {strategy.name}")
    
    # ═══════════════════════════════════════════════════════════════
    # STATUS & EXPORT
    # ═══════════════════════════════════════════════════════════════
    
    def get_status(self) -> Dict[str, Any]:
        """الحصول على حالة العقل"""
        accuracy = (
            self.stats['correct_decisions'] / self.stats['total_decisions']
            if self.stats['total_decisions'] > 0 else 0
        )
        
        return {
            'state': self.state.value,
            'stats': self.stats,
            'accuracy': accuracy,
            'active_strategy': self.active_strategy.name if self.active_strategy else None,
            'memory_usage': len(self.short_term_memory),
            'insights_count': len(self.insights),
            'dominant_persona': self.inner_dialogue._get_dominant_persona()
        }
    
    def export_mind_state(self) -> Dict[str, Any]:
        """تصدير حالة العقل"""
        return {
            'state': self.state.value,
            'stats': self.stats,
            'recent_decisions': [
                {
                    'action': d.action,
                    'confidence': d.confidence,
                    'reasoning': d.reasoning[:100],
                    'timestamp': d.timestamp.isoformat()
                }
                for d in self.decisions_history[-10:]
            ],
            'insights': [
                {
                    'insight': i.insight,
                    'confidence': i.confidence
                }
                for i in self.insights[-10:]
            ],
            'learned_patterns': len(self.reasoning_engine.learned_patterns),
            'strategies_available': len(self.strategy_inventor.strategies)
        }


# ═══════════════════════════════════════════════════════════════
# STANDALONE EXECUTION
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # اختبار العقل المبدع
    mind = CreativeMind()
    
    observation = {
        'features': {
            'close': 50000,
            'rsi_14': 28,
            'macd': 0.5,
            'macd_signal': 0.3,
            'adx': 30,
            'sma_20': 49000,
            'sma_50': 48000,
            'sma_200': 45000,
            'bb_upper': 52000,
            'bb_lower': 47000,
            'bb_percent': 0.6,
            'bb_width': 3.5,
            'atr_14': 1500,
            'volume': 1000000,
            'volume_sma_20': 800000,
            'highest_20': 51000,
            'lowest_20': 46000
        }
    }
    
    context = {
        'symbol': 'BTCUSDT',
        'market': 'crypto',
        'timeframe': '1h'
    }
    
    # التفكير واتخاذ القرار
    decision = mind.think(observation, context)
    
    print("🧠 Creative Mind Decision:")
    print(f"Action: {decision.action}")
    print(f"Confidence: {decision.confidence:.2%}")
    print(f"\nReasoning:\n{decision.reasoning}")
    print(f"\nSupporting Factors: {decision.supporting_factors}")
    print(f"Risk Factors: {decision.risk_factors}")
    
    # حالة العقل
    print(f"\n📊 Mind Status: {mind.get_status()}")
