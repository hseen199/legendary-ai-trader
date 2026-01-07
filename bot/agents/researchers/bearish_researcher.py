"""
Legendary Trading System - Bearish Researcher Agent
نظام التداول الخارق - وكيل الباحث المتشائم

يبني الحجج الهبوطية ويدافع عنها في المناظرات.
"""

import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging
import json

from ...core.base_agent import ResearcherAgent
from ...core.types import (
    AnalysisResult, SignalType, ResearchReport, ResearcherStance
)


class BearishResearcherAgent(ResearcherAgent):
    """
    وكيل الباحث المتشائم.
    
    يبحث عن:
    - المخاطر والتحذيرات
    - إشارات التوزيع
    - نقاط المقاومة
    - الأنماط الهبوطية
    """
    
    def __init__(self, config: Dict[str, Any], llm_client=None):
        super().__init__(
            name="BearishResearcher",
            config=config,
            stance="bearish"
        )
        self.llm_client = llm_client
        self._current_thesis = ""
        self._arguments = []
    
    async def initialize(self) -> bool:
        """تهيئة الباحث المتشائم."""
        self.logger.info("تهيئة الباحث المتشائم...")
        return True
    
    async def process(self, data: Any) -> Any:
        """معالجة البيانات."""
        return await self.research(data.get("symbol"), data.get("analyses", []))
    
    async def shutdown(self) -> None:
        """إيقاف الباحث."""
        self.logger.info("إيقاف الباحث المتشائم")
    
    async def research(self, symbol: str, 
                      analysis_results: List[AnalysisResult]) -> Dict[str, Any]:
        """
        إجراء البحث الهبوطي.
        
        Args:
            symbol: رمز العملة
            analysis_results: نتائج التحليل من المحللين
            
        Returns:
            تقرير البحث
        """
        self._update_activity()
        
        try:
            # جمع الأدلة الهبوطية
            bearish_evidence = self._gather_bearish_evidence(analysis_results)
            
            # بناء الأطروحة
            if self.llm_client:
                thesis = await self._build_thesis_with_llm(symbol, bearish_evidence)
            else:
                thesis = self._build_thesis_basic(symbol, bearish_evidence)
            
            self._current_thesis = thesis["thesis"]
            self._arguments = thesis["arguments"]
            
            # تقييم المخاطر من منظور هبوطي
            risk_assessment = self._assess_risks(analysis_results)
            
            # حساب الثقة
            confidence = self._calculate_confidence(bearish_evidence, analysis_results)
            
            return {
                "stance": ResearcherStance.BEARISH.value,
                "symbol": symbol,
                "timestamp": datetime.utcnow().isoformat(),
                "thesis": self._current_thesis,
                "arguments": self._arguments,
                "counter_arguments": thesis.get("counter_arguments", []),
                "confidence": confidence,
                "risk_assessment": risk_assessment,
                "evidence": bearish_evidence
            }
            
        except Exception as e:
            self.logger.error(f"خطأ في البحث الهبوطي: {e}")
            self._handle_error(e)
            return self._create_default_report(symbol)
    
    async def debate(self, opponent_argument: str) -> str:
        """
        الرد على حجة الخصم.
        
        Args:
            opponent_argument: حجة الباحث المتفائل
            
        Returns:
            الرد على الحجة
        """
        if self.llm_client:
            return await self._debate_with_llm(opponent_argument)
        else:
            return self._debate_basic(opponent_argument)
    
    def _gather_bearish_evidence(self, analyses: List[AnalysisResult]) -> Dict[str, Any]:
        """جمع الأدلة الهبوطية من التحليلات."""
        evidence = {
            "strong_signals": [],
            "moderate_signals": [],
            "warning_signs": [],
            "supporting_data": {},
            "bearish_count": 0,
            "total_confidence": 0
        }
        
        for analysis in analyses:
            signal_value = self._signal_to_value(analysis.signal)
            
            if signal_value <= -0.5:
                evidence["strong_signals"].append({
                    "source": analysis.analyst_type.value,
                    "signal": analysis.signal.value,
                    "confidence": analysis.confidence,
                    "reasoning": analysis.reasoning
                })
                evidence["bearish_count"] += 1
            elif signal_value <= -0.2:
                evidence["moderate_signals"].append({
                    "source": analysis.analyst_type.value,
                    "signal": analysis.signal.value,
                    "confidence": analysis.confidence,
                    "reasoning": analysis.reasoning
                })
                evidence["bearish_count"] += 1
            
            # جمع علامات التحذير حتى من الإشارات المحايدة
            if analysis.data:
                warnings = self._extract_warnings(analysis.data)
                evidence["warning_signs"].extend(warnings)
                evidence["supporting_data"][analysis.analyst_type.value] = analysis.data
            
            evidence["total_confidence"] += analysis.confidence * abs(min(0, signal_value))
        
        return evidence
    
    def _extract_warnings(self, data: Dict) -> List[str]:
        """استخراج علامات التحذير من البيانات."""
        warnings = []
        
        # تحذيرات RSI
        rsi = data.get("indicators", {}).get("rsi")
        if rsi and rsi > 70:
            warnings.append(f"RSI في منطقة ذروة الشراء ({rsi:.1f})")
        
        # تحذيرات الحجم
        volume_sentiment = data.get("volume_sentiment")
        if volume_sentiment and volume_sentiment < -0.3:
            warnings.append("ضغط بيع من الحجم")
        
        # تحذيرات On-Chain
        net_flow = data.get("net_flow")
        if net_flow and net_flow > 0:
            warnings.append("تدفق داخلي للمنصات (ضغط بيع محتمل)")
        
        whale_score = data.get("whale_score")
        if whale_score and whale_score < -0.3:
            warnings.append("الحيتان تبيع")
        
        return warnings
    
    def _signal_to_value(self, signal: SignalType) -> float:
        """تحويل الإشارة إلى قيمة رقمية."""
        mapping = {
            SignalType.STRONG_BUY: 1.0,
            SignalType.BUY: 0.6,
            SignalType.WEAK_BUY: 0.3,
            SignalType.NEUTRAL: 0.0,
            SignalType.WEAK_SELL: -0.3,
            SignalType.SELL: -0.6,
            SignalType.STRONG_SELL: -1.0
        }
        return mapping.get(signal, 0.0)
    
    async def _build_thesis_with_llm(self, symbol: str, 
                                     evidence: Dict) -> Dict[str, Any]:
        """بناء الأطروحة باستخدام LLM."""
        try:
            prompt = f"""
            أنت باحث متشائم/حذر في سوق العملات المشفرة. بناءً على الأدلة التالية،
            قم ببناء حجة هبوطية أو تحذيرية للعملة {symbol}:
            
            إشارات هبوطية قوية: {json.dumps(evidence['strong_signals'], ensure_ascii=False)}
            إشارات هبوطية معتدلة: {json.dumps(evidence['moderate_signals'], ensure_ascii=False)}
            علامات تحذير: {json.dumps(evidence['warning_signs'], ensure_ascii=False)}
            
            أجب بصيغة JSON:
            {{
                "thesis": "الأطروحة الرئيسية في جملة واحدة",
                "arguments": ["حجة 1", "حجة 2", "حجة 3"],
                "counter_arguments": ["نقطة قوة محتملة للخصم 1", "نقطة قوة 2"],
                "risks": ["خطر 1", "خطر 2"],
                "support_levels": ["مستوى دعم 1", "مستوى دعم 2"]
            }}
            """
            
            response = await self.llm_client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=1000
            )
            
            result = json.loads(response.choices[0].message.content)
            return result
            
        except Exception as e:
            self.logger.error(f"خطأ في LLM: {e}")
            return self._build_thesis_basic(symbol, evidence)
    
    def _build_thesis_basic(self, symbol: str, evidence: Dict) -> Dict[str, Any]:
        """بناء أطروحة أساسية بدون LLM."""
        arguments = []
        
        # بناء الحجج من الإشارات الهبوطية
        for signal in evidence["strong_signals"]:
            arguments.append(f"{signal['source']}: {signal['reasoning']}")
        
        # إضافة حجج من الإشارات المعتدلة
        for signal in evidence["moderate_signals"][:2]:
            arguments.append(f"{signal['source']}: {signal['reasoning']}")
        
        # إضافة علامات التحذير
        for warning in evidence["warning_signs"][:2]:
            arguments.append(f"تحذير: {warning}")
        
        # بناء الأطروحة
        if evidence["bearish_count"] >= 3:
            thesis = f"توجد مخاطر هبوطية قوية على {symbol} مدعومة بـ {evidence['bearish_count']} إشارات سلبية"
        elif evidence["bearish_count"] >= 1:
            thesis = f"توجد مخاطر هبوطية محتملة على {symbol} يجب الحذر منها"
        elif evidence["warning_signs"]:
            thesis = f"توجد علامات تحذير على {symbol} رغم غياب إشارات هبوطية قوية"
        else:
            thesis = f"لا توجد أدلة هبوطية قوية على {symbol} حالياً"
        
        return {
            "thesis": thesis,
            "arguments": arguments[:5],
            "counter_arguments": ["قد تكون هناك محفزات إيجابية غير متوقعة", "الدعم قد يصمد"]
        }
    
    async def _debate_with_llm(self, opponent_argument: str) -> str:
        """الرد على الخصم باستخدام LLM."""
        try:
            prompt = f"""
            أنت باحث متشائم/حذر تدافع عن موقفك.
            
            أطروحتك: {self._current_thesis}
            حججك: {json.dumps(self._arguments, ensure_ascii=False)}
            
            حجة الخصم المتفائل: {opponent_argument}
            
            قم بالرد على حجة الخصم بشكل مقنع ومنطقي، مع التركيز على المخاطر.
            أجب بفقرة واحدة فقط.
            """
            
            response = await self.llm_client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=300
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            self.logger.error(f"خطأ في LLM: {e}")
            return self._debate_basic(opponent_argument)
    
    def _debate_basic(self, opponent_argument: str) -> str:
        """رد أساسي بدون LLM."""
        responses = [
            "التفاؤل المفرط خطير في الأسواق. المؤشرات الفنية تظهر ضعفاً واضحاً يجب عدم تجاهله.",
            "التاريخ يعلمنا أن الأسواق تميل للمبالغة في التفاؤل قبل التصحيحات الكبيرة.",
            "البيانات على السلسلة تظهر توزيعاً من الحيتان، وهذا عادة ما يسبق الهبوط.",
            "المخاطر الحالية تفوق الفرص المحتملة. الحكمة تقتضي الحذر.",
            "نسبة المخاطرة للعائد غير مواتية في المستويات الحالية."
        ]
        
        import random
        return random.choice(responses)
    
    def _assess_risks(self, analyses: List[AnalysisResult]) -> str:
        """تقييم المخاطر من منظور هبوطي."""
        high_risks = []
        moderate_risks = []
        
        for analysis in analyses:
            signal_value = self._signal_to_value(analysis.signal)
            
            if signal_value <= -0.5:
                high_risks.append(f"{analysis.analyst_type.value}: {analysis.reasoning}")
            elif signal_value <= -0.2:
                moderate_risks.append(f"{analysis.analyst_type.value}: {analysis.reasoning}")
        
        if high_risks:
            return f"مخاطر عالية: {' | '.join(high_risks[:2])}"
        elif moderate_risks:
            return f"مخاطر معتدلة: {' | '.join(moderate_risks[:2])}"
        else:
            return "المخاطر منخفضة نسبياً - لكن يجب البقاء حذراً"
    
    def _calculate_confidence(self, evidence: Dict, 
                             analyses: List[AnalysisResult]) -> float:
        """حساب مستوى الثقة."""
        if not analyses:
            return 0.0
        
        # نسبة الإشارات الهبوطية
        bearish_ratio = evidence["bearish_count"] / len(analyses)
        
        # إضافة وزن لعلامات التحذير
        warning_weight = min(0.3, len(evidence["warning_signs"]) * 0.1)
        
        # متوسط الثقة المرجح
        if evidence["bearish_count"] > 0:
            avg_confidence = evidence["total_confidence"] / evidence["bearish_count"]
        else:
            avg_confidence = 0.3 + warning_weight
        
        # الثقة النهائية
        confidence = bearish_ratio * 0.4 + avg_confidence * 0.4 + warning_weight * 0.2
        
        return min(1.0, max(0.0, confidence))
    
    def _create_default_report(self, symbol: str) -> Dict[str, Any]:
        """إنشاء تقرير افتراضي."""
        return {
            "stance": ResearcherStance.BEARISH.value,
            "symbol": symbol,
            "timestamp": datetime.utcnow().isoformat(),
            "thesis": "لا تتوفر بيانات كافية لبناء أطروحة هبوطية",
            "arguments": [],
            "counter_arguments": [],
            "confidence": 0.0,
            "risk_assessment": "غير محدد",
            "evidence": {}
        }
