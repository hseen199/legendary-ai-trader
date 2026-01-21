"""
Legendary Trading System - Enhanced Reasoning Engine
نظام التداول الخارق - محرك التفكير المحسن

محرك تفكير متقدم مع سلاسل تفكير وتفسير القرارات.
"""

import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
import json


class ReasoningType(Enum):
    """أنواع التفكير"""
    DEDUCTIVE = "deductive"           # استنتاجي
    INDUCTIVE = "inductive"           # استقرائي
    ABDUCTIVE = "abductive"           # تفسيري
    ANALOGICAL = "analogical"         # تشبيهي
    CAUSAL = "causal"                 # سببي
    PROBABILISTIC = "probabilistic"   # احتمالي
    COUNTERFACTUAL = "counterfactual" # افتراضي معاكس


class ConfidenceLevel(Enum):
    """مستويات الثقة"""
    VERY_LOW = 0.2
    LOW = 0.4
    MEDIUM = 0.6
    HIGH = 0.8
    VERY_HIGH = 0.95


@dataclass
class ThoughtStep:
    """خطوة تفكير واحدة"""
    id: int
    reasoning_type: ReasoningType
    premise: str                      # المقدمة
    logic: str                        # المنطق المستخدم
    conclusion: str                   # الاستنتاج
    confidence: float
    evidence: List[str] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)
    uncertainties: List[str] = field(default_factory=list)


@dataclass
class ThoughtChain:
    """سلسلة تفكير كاملة"""
    id: str
    question: str
    steps: List[ThoughtStep] = field(default_factory=list)
    final_conclusion: Optional[str] = None
    overall_confidence: float = 0.0
    reasoning_path: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class CausalRelation:
    """علاقة سببية"""
    cause: str
    effect: str
    strength: float  # 0-1
    evidence: List[str] = field(default_factory=list)
    confounders: List[str] = field(default_factory=list)


class EnhancedReasoningEngine:
    """
    محرك التفكير المحسن.
    
    يوفر:
    - سلاسل تفكير متعددة المستويات (Chain of Thought)
    - تفكير سببي (Causal Reasoning)
    - استدلال احتمالي (Probabilistic Reasoning)
    - تفسير القرارات (Explainable AI)
    """
    
    def __init__(self, llm_client=None, memory_system=None):
        self.logger = logging.getLogger("EnhancedReasoningEngine")
        self.llm_client = llm_client
        self.memory_system = memory_system
        
        # قاعدة المعرفة السببية
        self.causal_knowledge: Dict[str, List[CausalRelation]] = {}
        
        # تاريخ سلاسل التفكير
        self.thought_history: List[ThoughtChain] = []
        
        # قواعد الاستدلال
        self.inference_rules = self._init_inference_rules()
        
        # أنماط التفكير المتعلمة
        self.learned_patterns: Dict[str, Any] = {}
    
    def _init_inference_rules(self) -> Dict[str, Any]:
        """تهيئة قواعد الاستدلال."""
        return {
            # قواعد السوق
            "trend_continuation": {
                "if": ["strong_trend", "high_volume", "no_divergence"],
                "then": "trend_likely_continues",
                "confidence": 0.7
            },
            "trend_reversal": {
                "if": ["divergence", "exhaustion_pattern", "volume_decline"],
                "then": "reversal_possible",
                "confidence": 0.6
            },
            "breakout": {
                "if": ["consolidation", "volume_spike", "price_break"],
                "then": "breakout_confirmed",
                "confidence": 0.65
            },
            "false_breakout": {
                "if": ["breakout", "quick_reversal", "low_volume"],
                "then": "false_breakout",
                "confidence": 0.55
            },
            
            # قواعد المخاطر
            "high_risk": {
                "if": ["high_volatility", "low_liquidity", "uncertain_trend"],
                "then": "reduce_position",
                "confidence": 0.8
            },
            "low_risk_opportunity": {
                "if": ["clear_trend", "high_liquidity", "low_volatility"],
                "then": "increase_position",
                "confidence": 0.7
            },
            
            # قواعد المشاعر
            "extreme_fear": {
                "if": ["fear_index_high", "mass_selling", "negative_news"],
                "then": "potential_bottom",
                "confidence": 0.5
            },
            "extreme_greed": {
                "if": ["greed_index_high", "fomo_buying", "euphoric_news"],
                "then": "potential_top",
                "confidence": 0.5
            }
        }
    
    async def think(self, 
                   question: str,
                   context: Dict[str, Any],
                   depth: int = 3) -> ThoughtChain:
        """
        التفكير في سؤال أو مشكلة.
        
        Args:
            question: السؤال أو المشكلة
            context: السياق
            depth: عمق التفكير
            
        Returns:
            سلسلة التفكير
        """
        chain_id = f"thought_{datetime.utcnow().timestamp()}"
        chain = ThoughtChain(id=chain_id, question=question)
        
        self.logger.info(f"بدء التفكير في: {question}")
        
        # الخطوة 1: تحليل السؤال
        step1 = await self._analyze_question(question, context)
        chain.steps.append(step1)
        
        # الخطوة 2: جمع الأدلة
        step2 = await self._gather_evidence(question, context)
        chain.steps.append(step2)
        
        # الخطوة 3: تطبيق التفكير السببي
        step3 = await self._apply_causal_reasoning(context)
        chain.steps.append(step3)
        
        # الخطوة 4: التفكير الاحتمالي
        step4 = await self._apply_probabilistic_reasoning(context)
        chain.steps.append(step4)
        
        # خطوات إضافية حسب العمق
        for i in range(depth - 4):
            step = await self._deep_reasoning_step(chain, context, i)
            chain.steps.append(step)
        
        # الاستنتاج النهائي
        chain.final_conclusion = await self._synthesize_conclusion(chain)
        chain.overall_confidence = self._calculate_chain_confidence(chain)
        chain.reasoning_path = self._build_reasoning_path(chain)
        
        # حفظ في التاريخ
        self.thought_history.append(chain)
        
        return chain
    
    async def _analyze_question(self, 
                               question: str,
                               context: Dict[str, Any]) -> ThoughtStep:
        """تحليل السؤال."""
        # تحديد نوع السؤال
        question_type = self._classify_question(question)
        
        # استخراج المتغيرات الرئيسية
        key_variables = self._extract_key_variables(question, context)
        
        return ThoughtStep(
            id=1,
            reasoning_type=ReasoningType.DEDUCTIVE,
            premise=f"السؤال: {question}",
            logic=f"نوع السؤال: {question_type}. المتغيرات الرئيسية: {key_variables}",
            conclusion=f"يتطلب تحليل {question_type} مع التركيز على {key_variables}",
            confidence=0.9,
            evidence=[],
            assumptions=["السؤال واضح ومحدد"],
            uncertainties=[]
        )
    
    def _classify_question(self, question: str) -> str:
        """تصنيف نوع السؤال."""
        question_lower = question.lower()
        
        if any(word in question_lower for word in ["هل", "أ", "سوف"]):
            return "yes_no"
        elif any(word in question_lower for word in ["لماذا", "سبب"]):
            return "causal"
        elif any(word in question_lower for word in ["كيف", "طريقة"]):
            return "procedural"
        elif any(word in question_lower for word in ["ما", "ماذا"]):
            return "descriptive"
        elif any(word in question_lower for word in ["متى", "وقت"]):
            return "temporal"
        elif any(word in question_lower for word in ["أين", "مكان"]):
            return "spatial"
        else:
            return "analytical"
    
    def _extract_key_variables(self, 
                              question: str,
                              context: Dict[str, Any]) -> List[str]:
        """استخراج المتغيرات الرئيسية."""
        variables = []
        
        # من السياق
        if "symbol" in context:
            variables.append(f"العملة: {context['symbol']}")
        if "price" in context:
            variables.append(f"السعر: {context['price']}")
        if "trend" in context:
            variables.append(f"الاتجاه: {context['trend']}")
        
        return variables
    
    async def _gather_evidence(self,
                              question: str,
                              context: Dict[str, Any]) -> ThoughtStep:
        """جمع الأدلة."""
        evidence = []
        
        # أدلة من البيانات
        if "indicators" in context:
            for indicator, value in context["indicators"].items():
                evidence.append(f"{indicator}: {value}")
        
        # أدلة من التاريخ
        if self.memory_system:
            historical = await self.memory_system.recall(question, limit=5)
            for item in historical:
                evidence.append(f"تجربة سابقة: {item.get('summary', '')}")
        
        return ThoughtStep(
            id=2,
            reasoning_type=ReasoningType.INDUCTIVE,
            premise="جمع الأدلة المتاحة",
            logic="تحليل البيانات والمؤشرات والتجارب السابقة",
            conclusion=f"تم جمع {len(evidence)} دليل",
            confidence=0.8 if evidence else 0.4,
            evidence=evidence,
            assumptions=["البيانات دقيقة ومحدثة"],
            uncertainties=["قد تكون هناك أدلة غير متاحة"]
        )
    
    async def _apply_causal_reasoning(self,
                                     context: Dict[str, Any]) -> ThoughtStep:
        """تطبيق التفكير السببي."""
        causal_chains = []
        
        # البحث عن علاقات سببية
        if "price_change" in context:
            change = context["price_change"]
            
            if change > 0.05:
                causal_chains.append(
                    CausalRelation(
                        cause="زيادة الطلب أو أخبار إيجابية",
                        effect="ارتفاع السعر",
                        strength=0.7,
                        evidence=["تغير السعر إيجابي"]
                    )
                )
            elif change < -0.05:
                causal_chains.append(
                    CausalRelation(
                        cause="زيادة العرض أو أخبار سلبية",
                        effect="انخفاض السعر",
                        strength=0.7,
                        evidence=["تغير السعر سلبي"]
                    )
                )
        
        if "volume_spike" in context and context["volume_spike"]:
            causal_chains.append(
                CausalRelation(
                    cause="اهتمام متزايد أو حدث مهم",
                    effect="ارتفاع حجم التداول",
                    strength=0.6,
                    evidence=["حجم تداول غير عادي"]
                )
            )
        
        conclusion = "لا توجد علاقات سببية واضحة"
        if causal_chains:
            conclusion = f"تم تحديد {len(causal_chains)} علاقة سببية محتملة"
        
        return ThoughtStep(
            id=3,
            reasoning_type=ReasoningType.CAUSAL,
            premise="تحليل العلاقات السببية",
            logic="ربط الأسباب بالنتائج بناءً على الأنماط المعروفة",
            conclusion=conclusion,
            confidence=0.6,
            evidence=[f"{c.cause} -> {c.effect}" for c in causal_chains],
            assumptions=["العلاقات السببية ثابتة نسبياً"],
            uncertainties=["قد تكون هناك عوامل مخفية"]
        )
    
    async def _apply_probabilistic_reasoning(self,
                                            context: Dict[str, Any]) -> ThoughtStep:
        """تطبيق التفكير الاحتمالي."""
        probabilities = {}
        
        # حساب احتمالات السيناريوهات
        base_prob = 0.5
        
        # تعديل بناءً على الاتجاه
        if context.get("trend") == "bullish":
            probabilities["continuation_up"] = base_prob + 0.15
            probabilities["reversal_down"] = base_prob - 0.15
        elif context.get("trend") == "bearish":
            probabilities["continuation_down"] = base_prob + 0.15
            probabilities["reversal_up"] = base_prob - 0.15
        else:
            probabilities["sideways"] = base_prob + 0.1
        
        # تعديل بناءً على المؤشرات
        if context.get("rsi", 50) > 70:
            probabilities["overbought_correction"] = 0.6
        elif context.get("rsi", 50) < 30:
            probabilities["oversold_bounce"] = 0.6
        
        # تعديل بناءً على الحجم
        if context.get("volume_trend") == "increasing":
            for key in probabilities:
                probabilities[key] *= 1.1
        
        # تطبيع الاحتمالات
        total = sum(probabilities.values())
        if total > 0:
            probabilities = {k: v/total for k, v in probabilities.items()}
        
        most_likely = max(probabilities.items(), key=lambda x: x[1]) if probabilities else ("unknown", 0)
        
        return ThoughtStep(
            id=4,
            reasoning_type=ReasoningType.PROBABILISTIC,
            premise="تحليل احتمالي للسيناريوهات",
            logic="حساب احتمالات كل سيناريو بناءً على المعطيات",
            conclusion=f"السيناريو الأكثر احتمالاً: {most_likely[0]} ({most_likely[1]:.0%})",
            confidence=most_likely[1],
            evidence=[f"{k}: {v:.0%}" for k, v in probabilities.items()],
            assumptions=["الاحتمالات مبنية على أنماط تاريخية"],
            uncertainties=["الأسواق غير قابلة للتنبؤ بشكل كامل"]
        )
    
    async def _deep_reasoning_step(self,
                                  chain: ThoughtChain,
                                  context: Dict[str, Any],
                                  step_num: int) -> ThoughtStep:
        """خطوة تفكير عميقة إضافية."""
        # تحليل الخطوات السابقة
        previous_conclusions = [s.conclusion for s in chain.steps]
        
        # البحث عن تناقضات
        contradictions = self._find_contradictions(previous_conclusions)
        
        # البحث عن أنماط
        patterns = self._find_patterns(previous_conclusions)
        
        reasoning_type = ReasoningType.ANALOGICAL if patterns else ReasoningType.ABDUCTIVE
        
        return ThoughtStep(
            id=5 + step_num,
            reasoning_type=reasoning_type,
            premise=f"تحليل عميق للاستنتاجات السابقة ({len(previous_conclusions)} استنتاج)",
            logic="البحث عن تناقضات وأنماط في التفكير",
            conclusion=f"تناقضات: {len(contradictions)}, أنماط: {len(patterns)}",
            confidence=0.7,
            evidence=patterns[:3],
            assumptions=["التحليل السابق صحيح"],
            uncertainties=contradictions[:3]
        )
    
    def _find_contradictions(self, conclusions: List[str]) -> List[str]:
        """البحث عن تناقضات."""
        contradictions = []
        
        # تحليل بسيط للتناقضات
        positive_words = ["إيجابي", "صعود", "شراء", "فرصة"]
        negative_words = ["سلبي", "هبوط", "بيع", "خطر"]
        
        has_positive = any(
            any(word in c for word in positive_words) 
            for c in conclusions
        )
        has_negative = any(
            any(word in c for word in negative_words) 
            for c in conclusions
        )
        
        if has_positive and has_negative:
            contradictions.append("تضارب بين إشارات إيجابية وسلبية")
        
        return contradictions
    
    def _find_patterns(self, conclusions: List[str]) -> List[str]:
        """البحث عن أنماط."""
        patterns = []
        
        # تحليل بسيط للأنماط
        all_text = " ".join(conclusions).lower()
        
        if all_text.count("صعود") > 1 or all_text.count("إيجابي") > 1:
            patterns.append("نمط إيجابي متكرر")
        
        if all_text.count("هبوط") > 1 or all_text.count("سلبي") > 1:
            patterns.append("نمط سلبي متكرر")
        
        if "احتمال" in all_text:
            patterns.append("تفكير احتمالي")
        
        return patterns
    
    async def _synthesize_conclusion(self, chain: ThoughtChain) -> str:
        """تجميع الاستنتاج النهائي."""
        if not chain.steps:
            return "لا توجد معلومات كافية للاستنتاج"
        
        # جمع الاستنتاجات
        conclusions = [s.conclusion for s in chain.steps]
        confidences = [s.confidence for s in chain.steps]
        
        # حساب المتوسط المرجح
        weighted_sum = sum(c * conf for c, conf in zip(range(len(conclusions)), confidences))
        
        # تحديد الاتجاه العام
        positive_score = sum(
            conf for c, conf in zip(conclusions, confidences)
            if any(word in c.lower() for word in ["إيجابي", "صعود", "فرصة"])
        )
        negative_score = sum(
            conf for c, conf in zip(conclusions, confidences)
            if any(word in c.lower() for word in ["سلبي", "هبوط", "خطر"])
        )
        
        if positive_score > negative_score * 1.2:
            direction = "إيجابي"
            action = "الشراء قد يكون مناسباً"
        elif negative_score > positive_score * 1.2:
            direction = "سلبي"
            action = "البيع أو الانتظار قد يكون أفضل"
        else:
            direction = "محايد"
            action = "الانتظار ومراقبة السوق"
        
        return f"الاستنتاج: الوضع {direction}. التوصية: {action}"
    
    def _calculate_chain_confidence(self, chain: ThoughtChain) -> float:
        """حساب الثقة الإجمالية لسلسلة التفكير."""
        if not chain.steps:
            return 0.0
        
        # المتوسط الهندسي للثقة
        product = 1.0
        for step in chain.steps:
            product *= step.confidence
        
        return product ** (1.0 / len(chain.steps))
    
    def _build_reasoning_path(self, chain: ThoughtChain) -> str:
        """بناء مسار التفكير."""
        path_parts = []
        
        for step in chain.steps:
            path_parts.append(
                f"[{step.reasoning_type.value}] {step.conclusion}"
            )
        
        return " → ".join(path_parts)
    
    async def explain_decision(self, 
                              decision: str,
                              context: Dict[str, Any]) -> Dict[str, Any]:
        """
        تفسير قرار (Explainable AI).
        
        Args:
            decision: القرار
            context: السياق
            
        Returns:
            التفسير الكامل
        """
        explanation = {
            "decision": decision,
            "timestamp": datetime.utcnow().isoformat(),
            "factors": [],
            "reasoning_chain": [],
            "confidence_breakdown": {},
            "alternative_decisions": [],
            "risks": [],
            "assumptions": []
        }
        
        # العوامل المؤثرة
        if "indicators" in context:
            for indicator, value in context["indicators"].items():
                impact = self._assess_indicator_impact(indicator, value, decision)
                explanation["factors"].append({
                    "name": indicator,
                    "value": value,
                    "impact": impact["direction"],
                    "weight": impact["weight"]
                })
        
        # سلسلة التفكير
        chain = await self.think(f"لماذا {decision}؟", context)
        explanation["reasoning_chain"] = [
            {
                "step": s.id,
                "type": s.reasoning_type.value,
                "conclusion": s.conclusion,
                "confidence": s.confidence
            }
            for s in chain.steps
        ]
        
        # القرارات البديلة
        alternatives = ["buy", "sell", "hold"]
        for alt in alternatives:
            if alt != decision:
                explanation["alternative_decisions"].append({
                    "decision": alt,
                    "why_not": self._explain_why_not(alt, context)
                })
        
        # المخاطر
        explanation["risks"] = self._identify_risks(decision, context)
        
        # الافتراضات
        explanation["assumptions"] = [
            "البيانات المستخدمة دقيقة ومحدثة",
            "ظروف السوق ستبقى مستقرة نسبياً",
            "لا توجد أحداث غير متوقعة"
        ]
        
        return explanation
    
    def _assess_indicator_impact(self,
                                indicator: str,
                                value: Any,
                                decision: str) -> Dict[str, Any]:
        """تقييم تأثير مؤشر على القرار."""
        impact = {"direction": "neutral", "weight": 0.5}
        
        if indicator == "rsi":
            if value > 70:
                impact = {"direction": "bearish", "weight": 0.7}
            elif value < 30:
                impact = {"direction": "bullish", "weight": 0.7}
        elif indicator == "macd":
            if value > 0:
                impact = {"direction": "bullish", "weight": 0.6}
            else:
                impact = {"direction": "bearish", "weight": 0.6}
        elif indicator == "volume":
            if value > 1.5:  # نسبة للمتوسط
                impact = {"direction": "confirming", "weight": 0.5}
        
        return impact
    
    def _explain_why_not(self, 
                        alternative: str,
                        context: Dict[str, Any]) -> str:
        """تفسير لماذا لم يتم اختيار بديل."""
        if alternative == "buy":
            if context.get("rsi", 50) > 70:
                return "RSI في منطقة التشبع الشرائي"
            if context.get("trend") == "bearish":
                return "الاتجاه العام هابط"
        elif alternative == "sell":
            if context.get("rsi", 50) < 30:
                return "RSI في منطقة التشبع البيعي"
            if context.get("trend") == "bullish":
                return "الاتجاه العام صاعد"
        elif alternative == "hold":
            if context.get("signal_strength", 0) > 0.7:
                return "إشارة قوية تستدعي التحرك"
        
        return "لا يتوافق مع المعطيات الحالية"
    
    def _identify_risks(self,
                       decision: str,
                       context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """تحديد المخاطر المرتبطة بالقرار."""
        risks = []
        
        if decision == "buy":
            risks.append({
                "risk": "انخفاض السعر بعد الشراء",
                "probability": 0.3,
                "mitigation": "استخدام وقف خسارة"
            })
            if context.get("volatility", 0) > 0.05:
                risks.append({
                    "risk": "تقلب عالي قد يؤدي لخسائر سريعة",
                    "probability": 0.4,
                    "mitigation": "تقليل حجم الصفقة"
                })
        elif decision == "sell":
            risks.append({
                "risk": "ارتفاع السعر بعد البيع",
                "probability": 0.3,
                "mitigation": "البيع الجزئي"
            })
        
        return risks
    
    async def counterfactual_analysis(self,
                                     decision: str,
                                     outcome: Dict[str, Any],
                                     context: Dict[str, Any]) -> Dict[str, Any]:
        """
        تحليل افتراضي معاكس (ماذا لو؟).
        
        Args:
            decision: القرار الذي تم اتخاذه
            outcome: النتيجة الفعلية
            context: السياق
            
        Returns:
            تحليل السيناريوهات البديلة
        """
        analysis = {
            "actual_decision": decision,
            "actual_outcome": outcome,
            "counterfactuals": []
        }
        
        alternatives = ["buy", "sell", "hold"]
        
        for alt in alternatives:
            if alt != decision:
                hypothetical = await self._simulate_alternative(alt, context, outcome)
                analysis["counterfactuals"].append({
                    "alternative": alt,
                    "hypothetical_outcome": hypothetical,
                    "comparison": self._compare_outcomes(outcome, hypothetical)
                })
        
        # الدرس المستفاد
        analysis["lesson"] = self._extract_counterfactual_lesson(analysis)
        
        return analysis
    
    async def _simulate_alternative(self,
                                   alternative: str,
                                   context: Dict[str, Any],
                                   actual_outcome: Dict[str, Any]) -> Dict[str, Any]:
        """محاكاة نتيجة بديل."""
        # محاكاة بسيطة بناءً على تغير السعر الفعلي
        price_change = actual_outcome.get("price_change", 0)
        
        if alternative == "buy":
            return {
                "pnl": price_change * context.get("position_size", 100),
                "success": price_change > 0
            }
        elif alternative == "sell":
            return {
                "pnl": -price_change * context.get("position_size", 100),
                "success": price_change < 0
            }
        else:  # hold
            return {
                "pnl": 0,
                "success": abs(price_change) < 0.02
            }
    
    def _compare_outcomes(self,
                         actual: Dict[str, Any],
                         hypothetical: Dict[str, Any]) -> str:
        """مقارنة النتائج."""
        actual_pnl = actual.get("pnl", 0)
        hypo_pnl = hypothetical.get("pnl", 0)
        
        if hypo_pnl > actual_pnl:
            return f"البديل كان أفضل بـ {hypo_pnl - actual_pnl:.2f}"
        elif hypo_pnl < actual_pnl:
            return f"القرار الفعلي كان أفضل بـ {actual_pnl - hypo_pnl:.2f}"
        else:
            return "النتيجة متساوية تقريباً"
    
    def _extract_counterfactual_lesson(self, analysis: Dict[str, Any]) -> str:
        """استخراج الدرس من التحليل الافتراضي."""
        actual_success = analysis["actual_outcome"].get("success", False)
        
        better_alternatives = [
            cf for cf in analysis["counterfactuals"]
            if cf["hypothetical_outcome"].get("pnl", 0) > 
               analysis["actual_outcome"].get("pnl", 0)
        ]
        
        if not better_alternatives:
            return "القرار كان الأفضل من بين البدائل"
        else:
            best_alt = max(
                better_alternatives,
                key=lambda x: x["hypothetical_outcome"].get("pnl", 0)
            )
            return f"كان يمكن أن يكون '{best_alt['alternative']}' خياراً أفضل"
    
    def get_stats(self) -> Dict[str, Any]:
        """الحصول على الإحصائيات."""
        return {
            "total_thought_chains": len(self.thought_history),
            "causal_relations": sum(len(v) for v in self.causal_knowledge.values()),
            "learned_patterns": len(self.learned_patterns),
            "inference_rules": len(self.inference_rules)
        }
