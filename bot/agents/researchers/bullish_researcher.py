"""
Legendary Trading System - Bullish Researcher Agent
نظام التداول الخارق - وكيل الباحث المتفائل

يبني الحجج الصعودية ويدافع عنها في المناظرات.
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


class BullishResearcherAgent(ResearcherAgent):
    """
    وكيل الباحث المتفائل.
    
    يبحث عن:
    - الفرص الصعودية
    - المحفزات الإيجابية
    - نقاط الدعم القوية
    - إشارات التجميع
    """
    
    def __init__(self, config: Dict[str, Any], llm_client=None):
        super().__init__(
            name="BullishResearcher",
            config=config,
            stance="bullish"
        )
        self.llm_client = llm_client
        self._current_thesis = ""
        self._arguments = []
    
    async def initialize(self) -> bool:
        """تهيئة الباحث المتفائل."""
        self.logger.info("تهيئة الباحث المتفائل...")
        return True
    
    async def process(self, data: Any) -> Any:
        """معالجة البيانات."""
        return await self.research(data.get("symbol"), data.get("analyses", []))
    
    async def shutdown(self) -> None:
        """إيقاف الباحث."""
        self.logger.info("إيقاف الباحث المتفائل")
    
    async def research(self, symbol: str, 
                      analysis_results: List[AnalysisResult]) -> Dict[str, Any]:
        """
        إجراء البحث الصعودي.
        
        Args:
            symbol: رمز العملة
            analysis_results: نتائج التحليل من المحللين
            
        Returns:
            تقرير البحث
        """
        self._update_activity()
        
        try:
            # جمع الأدلة الصعودية
            bullish_evidence = self._gather_bullish_evidence(analysis_results)
            
            # بناء الأطروحة
            if self.llm_client:
                thesis = await self._build_thesis_with_llm(symbol, bullish_evidence)
            else:
                thesis = self._build_thesis_basic(symbol, bullish_evidence)
            
            self._current_thesis = thesis["thesis"]
            self._arguments = thesis["arguments"]
            
            # تقييم المخاطر من منظور صعودي
            risk_assessment = self._assess_risks(analysis_results)
            
            # حساب الثقة
            confidence = self._calculate_confidence(bullish_evidence, analysis_results)
            
            return {
                "stance": ResearcherStance.BULLISH.value,
                "symbol": symbol,
                "timestamp": datetime.utcnow().isoformat(),
                "thesis": self._current_thesis,
                "arguments": self._arguments,
                "counter_arguments": thesis.get("counter_arguments", []),
                "confidence": confidence,
                "risk_assessment": risk_assessment,
                "evidence": bullish_evidence
            }
            
        except Exception as e:
            self.logger.error(f"خطأ في البحث الصعودي: {e}")
            self._handle_error(e)
            return self._create_default_report(symbol)
    
    async def debate(self, opponent_argument: str) -> str:
        """
        الرد على حجة الخصم.
        
        Args:
            opponent_argument: حجة الباحث المتشائم
            
        Returns:
            الرد على الحجة
        """
        if self.llm_client:
            return await self._debate_with_llm(opponent_argument)
        else:
            return self._debate_basic(opponent_argument)
    
    def _gather_bullish_evidence(self, analyses: List[AnalysisResult]) -> Dict[str, Any]:
        """جمع الأدلة الصعودية من التحليلات."""
        evidence = {
            "strong_signals": [],
            "moderate_signals": [],
            "supporting_data": {},
            "bullish_count": 0,
            "total_confidence": 0
        }
        
        for analysis in analyses:
            signal_value = self._signal_to_value(analysis.signal)
            
            if signal_value >= 0.5:
                evidence["strong_signals"].append({
                    "source": analysis.analyst_type.value,
                    "signal": analysis.signal.value,
                    "confidence": analysis.confidence,
                    "reasoning": analysis.reasoning
                })
                evidence["bullish_count"] += 1
            elif signal_value >= 0.2:
                evidence["moderate_signals"].append({
                    "source": analysis.analyst_type.value,
                    "signal": analysis.signal.value,
                    "confidence": analysis.confidence,
                    "reasoning": analysis.reasoning
                })
                evidence["bullish_count"] += 1
            
            evidence["total_confidence"] += analysis.confidence * max(0, signal_value)
            
            # جمع البيانات الداعمة
            if analysis.data:
                evidence["supporting_data"][analysis.analyst_type.value] = analysis.data
        
        return evidence
    
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
            أنت باحث متفائل في سوق العملات المشفرة. بناءً على الأدلة التالية،
            قم ببناء حجة صعودية قوية للعملة {symbol}:
            
            إشارات قوية: {json.dumps(evidence['strong_signals'], ensure_ascii=False)}
            إشارات معتدلة: {json.dumps(evidence['moderate_signals'], ensure_ascii=False)}
            
            أجب بصيغة JSON:
            {{
                "thesis": "الأطروحة الرئيسية في جملة واحدة",
                "arguments": ["حجة 1", "حجة 2", "حجة 3"],
                "counter_arguments": ["نقطة ضعف محتملة 1", "نقطة ضعف 2"],
                "catalysts": ["محفز 1", "محفز 2"],
                "price_targets": {{"short_term": "هدف قصير المدى", "medium_term": "هدف متوسط"}}
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
        
        # بناء الحجج من الإشارات القوية
        for signal in evidence["strong_signals"]:
            arguments.append(f"{signal['source']}: {signal['reasoning']}")
        
        # إضافة حجج من الإشارات المعتدلة
        for signal in evidence["moderate_signals"][:2]:
            arguments.append(f"{signal['source']}: {signal['reasoning']}")
        
        # بناء الأطروحة
        if evidence["bullish_count"] >= 3:
            thesis = f"توجد فرصة صعودية قوية على {symbol} مدعومة بـ {evidence['bullish_count']} إشارات إيجابية"
        elif evidence["bullish_count"] >= 1:
            thesis = f"توجد فرصة صعودية محتملة على {symbol}"
        else:
            thesis = f"لا توجد أدلة صعودية قوية على {symbol} حالياً"
        
        return {
            "thesis": thesis,
            "arguments": arguments[:5],
            "counter_arguments": ["يجب مراقبة مستويات الدعم", "التقلبات قد تكون عالية"]
        }
    
    async def _debate_with_llm(self, opponent_argument: str) -> str:
        """الرد على الخصم باستخدام LLM."""
        try:
            prompt = f"""
            أنت باحث متفائل تدافع عن موقفك الصعودي.
            
            أطروحتك: {self._current_thesis}
            حججك: {json.dumps(self._arguments, ensure_ascii=False)}
            
            حجة الخصم المتشائم: {opponent_argument}
            
            قم بالرد على حجة الخصم بشكل مقنع ومنطقي. أجب بفقرة واحدة فقط.
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
            "على الرغم من المخاوف المذكورة، تظل الأساسيات قوية والمؤشرات الفنية تدعم الاتجاه الصعودي.",
            "التصحيحات طبيعية في الأسواق الصاعدة وتوفر فرص شراء أفضل.",
            "البيانات على السلسلة تظهر تجميعاً من المستثمرين الكبار مما يدعم النظرة الإيجابية.",
            "المحفزات القادمة قد تتغلب على المخاوف الحالية."
        ]
        
        import random
        return random.choice(responses)
    
    def _assess_risks(self, analyses: List[AnalysisResult]) -> str:
        """تقييم المخاطر من منظور صعودي."""
        risks = []
        
        for analysis in analyses:
            signal_value = self._signal_to_value(analysis.signal)
            
            if signal_value < -0.3:
                risks.append(f"تحذير من {analysis.analyst_type.value}: {analysis.reasoning}")
        
        if not risks:
            return "المخاطر منخفضة - معظم المؤشرات إيجابية"
        elif len(risks) <= 2:
            return f"مخاطر معتدلة: {' | '.join(risks[:2])}"
        else:
            return f"مخاطر مرتفعة يجب مراقبتها: {' | '.join(risks[:3])}"
    
    def _calculate_confidence(self, evidence: Dict, 
                             analyses: List[AnalysisResult]) -> float:
        """حساب مستوى الثقة."""
        if not analyses:
            return 0.0
        
        # نسبة الإشارات الصعودية
        bullish_ratio = evidence["bullish_count"] / len(analyses)
        
        # متوسط الثقة المرجح
        if evidence["bullish_count"] > 0:
            avg_confidence = evidence["total_confidence"] / evidence["bullish_count"]
        else:
            avg_confidence = 0.3
        
        # الثقة النهائية
        confidence = bullish_ratio * 0.5 + avg_confidence * 0.5
        
        return min(1.0, max(0.0, confidence))
    
    def _create_default_report(self, symbol: str) -> Dict[str, Any]:
        """إنشاء تقرير افتراضي."""
        return {
            "stance": ResearcherStance.BULLISH.value,
            "symbol": symbol,
            "timestamp": datetime.utcnow().isoformat(),
            "thesis": "لا تتوفر بيانات كافية لبناء أطروحة صعودية",
            "arguments": [],
            "counter_arguments": [],
            "confidence": 0.0,
            "risk_assessment": "غير محدد",
            "evidence": {}
        }
