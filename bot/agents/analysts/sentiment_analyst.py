"""
Legendary Trading System - Sentiment Analyst Agent
نظام التداول الخارق - وكيل محلل المشاعر

يحلل مشاعر السوق من مصادر متعددة باستخدام LLM.
"""

import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging
import json

from ...core.base_agent import AnalystAgent
from ...core.types import (
    AnalysisResult, SignalType, AnalystType, SentimentData
)


class SentimentAnalystAgent(AnalystAgent):
    """
    وكيل محلل المشاعر.
    
    يحلل مشاعر السوق من:
    - مؤشر الخوف والطمع
    - وسائل التواصل الاجتماعي
    - حجم التداول والنشاط
    - تحليل LLM للنصوص
    """
    
    def __init__(self, config: Dict[str, Any], llm_client=None):
        super().__init__(
            name="SentimentAnalyst",
            config=config,
            analyst_type="sentiment"
        )
        self.weight = config.get("analyst_weights", {}).get("sentiment", 0.15)
        self.llm_client = llm_client
        
        # عتبات المشاعر
        self.extreme_fear_threshold = 20
        self.fear_threshold = 40
        self.greed_threshold = 60
        self.extreme_greed_threshold = 80
    
    async def initialize(self) -> bool:
        """تهيئة محلل المشاعر."""
        self.logger.info("تهيئة محلل المشاعر...")
        return True
    
    async def process(self, data: Any) -> Any:
        """معالجة البيانات."""
        return await self.analyze(data.get("symbol"), data)
    
    async def shutdown(self) -> None:
        """إيقاف المحلل."""
        self.logger.info("إيقاف محلل المشاعر")
    
    async def analyze(self, symbol: str, data: Dict[str, Any]) -> AnalysisResult:
        """
        تحليل مشاعر السوق.
        
        Args:
            symbol: رمز العملة
            data: بيانات المشاعر
            
        Returns:
            نتيجة التحليل
        """
        self._update_activity()
        
        try:
            # جمع بيانات المشاعر
            sentiment_data = await self._collect_sentiment_data(symbol, data)
            
            # تحليل المشاعر
            analysis = self._analyze_sentiment(sentiment_data)
            
            # استخدام LLM للتحليل العميق إذا متاح
            if self.llm_client and data.get("news_texts"):
                llm_analysis = await self._llm_sentiment_analysis(
                    symbol, data.get("news_texts", [])
                )
                analysis = self._merge_analyses(analysis, llm_analysis)
            
            return AnalysisResult(
                analyst_type=AnalystType.SENTIMENT,
                symbol=symbol,
                timestamp=datetime.utcnow(),
                signal=analysis["signal"],
                confidence=analysis["confidence"],
                reasoning=analysis["reasoning"],
                data={
                    "sentiment_score": analysis["sentiment_score"],
                    "fear_greed_index": sentiment_data.get("fear_greed_index"),
                    "social_sentiment": sentiment_data.get("social_sentiment"),
                    "volume_sentiment": sentiment_data.get("volume_sentiment")
                }
            )
            
        except Exception as e:
            self.logger.error(f"خطأ في تحليل المشاعر: {e}")
            self._handle_error(e)
            return self._create_neutral_result(symbol, f"خطأ: {str(e)}")
    
    async def _collect_sentiment_data(self, symbol: str, 
                                      data: Dict[str, Any]) -> Dict[str, Any]:
        """جمع بيانات المشاعر من مصادر مختلفة."""
        sentiment_data = {
            "symbol": symbol,
            "timestamp": datetime.utcnow()
        }
        
        # مؤشر الخوف والطمع
        fear_greed = data.get("fear_greed_index")
        if fear_greed is not None:
            sentiment_data["fear_greed_index"] = fear_greed
        else:
            # قيمة افتراضية محايدة
            sentiment_data["fear_greed_index"] = 50
        
        # مشاعر وسائل التواصل
        social_data = data.get("social_data", {})
        sentiment_data["social_sentiment"] = self._calculate_social_sentiment(social_data)
        
        # مشاعر الحجم
        volume_data = data.get("volume_data", {})
        sentiment_data["volume_sentiment"] = self._calculate_volume_sentiment(volume_data)
        
        return sentiment_data
    
    def _calculate_social_sentiment(self, social_data: Dict) -> float:
        """حساب مشاعر وسائل التواصل الاجتماعي."""
        if not social_data:
            return 0.0
        
        # حساب المتوسط المرجح
        twitter_sentiment = social_data.get("twitter", 0)
        reddit_sentiment = social_data.get("reddit", 0)
        telegram_sentiment = social_data.get("telegram", 0)
        
        weights = {"twitter": 0.4, "reddit": 0.35, "telegram": 0.25}
        
        total = (
            twitter_sentiment * weights["twitter"] +
            reddit_sentiment * weights["reddit"] +
            telegram_sentiment * weights["telegram"]
        )
        
        return total
    
    def _calculate_volume_sentiment(self, volume_data: Dict) -> float:
        """حساب مشاعر الحجم."""
        if not volume_data:
            return 0.0
        
        current_volume = volume_data.get("current", 0)
        avg_volume = volume_data.get("average", 1)
        buy_volume = volume_data.get("buy_volume", 0)
        sell_volume = volume_data.get("sell_volume", 0)
        
        # نسبة الحجم
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
        
        # نسبة الشراء للبيع
        total_volume = buy_volume + sell_volume
        if total_volume > 0:
            buy_ratio = buy_volume / total_volume
        else:
            buy_ratio = 0.5
        
        # تحويل إلى مشاعر (-1 إلى 1)
        volume_sentiment = (buy_ratio - 0.5) * 2
        
        # تعديل بناءً على حجم التداول
        if volume_ratio > 1.5:
            volume_sentiment *= 1.2  # تضخيم عند حجم عالي
        elif volume_ratio < 0.5:
            volume_sentiment *= 0.8  # تخفيف عند حجم منخفض
        
        return max(-1, min(1, volume_sentiment))
    
    def _analyze_sentiment(self, sentiment_data: Dict) -> Dict[str, Any]:
        """تحليل بيانات المشاعر وتوليد الإشارة."""
        signals = []
        reasoning_parts = []
        
        # تحليل مؤشر الخوف والطمع
        fear_greed = sentiment_data.get("fear_greed_index", 50)
        fg_signal, fg_reason = self._analyze_fear_greed(fear_greed)
        signals.append(("fear_greed", fg_signal, 0.4))
        reasoning_parts.append(fg_reason)
        
        # تحليل المشاعر الاجتماعية
        social = sentiment_data.get("social_sentiment", 0)
        social_signal = social * 0.6  # تطبيع
        signals.append(("social", social_signal, 0.3))
        if abs(social) > 0.3:
            reasoning_parts.append(
                f"المشاعر الاجتماعية {'إيجابية' if social > 0 else 'سلبية'}"
            )
        
        # تحليل مشاعر الحجم
        volume = sentiment_data.get("volume_sentiment", 0)
        signals.append(("volume", volume, 0.3))
        if abs(volume) > 0.3:
            reasoning_parts.append(
                f"ضغط {'شراء' if volume > 0 else 'بيع'} من الحجم"
            )
        
        # حساب الإشارة النهائية
        weighted_sum = sum(s[1] * s[2] for s in signals)
        total_weight = sum(s[2] for s in signals)
        final_score = weighted_sum / total_weight if total_weight > 0 else 0
        
        # تحديد نوع الإشارة
        signal_type = self._score_to_signal(final_score)
        
        # حساب الثقة
        confidence = self._calculate_sentiment_confidence(signals)
        
        return {
            "signal": signal_type,
            "confidence": confidence,
            "sentiment_score": final_score,
            "reasoning": "تحليل المشاعر: " + " | ".join(reasoning_parts)
        }
    
    def _analyze_fear_greed(self, index: float) -> tuple:
        """تحليل مؤشر الخوف والطمع."""
        if index <= self.extreme_fear_threshold:
            return 0.8, f"خوف شديد ({index}) - فرصة شراء محتملة"
        elif index <= self.fear_threshold:
            return 0.4, f"خوف ({index}) - إشارة شراء"
        elif index >= self.extreme_greed_threshold:
            return -0.8, f"طمع شديد ({index}) - فرصة بيع محتملة"
        elif index >= self.greed_threshold:
            return -0.4, f"طمع ({index}) - إشارة بيع"
        else:
            return 0, f"محايد ({index})"
    
    def _score_to_signal(self, score: float) -> SignalType:
        """تحويل النتيجة إلى نوع إشارة."""
        if score >= 0.6:
            return SignalType.STRONG_BUY
        elif score >= 0.3:
            return SignalType.BUY
        elif score >= 0.1:
            return SignalType.WEAK_BUY
        elif score <= -0.6:
            return SignalType.STRONG_SELL
        elif score <= -0.3:
            return SignalType.SELL
        elif score <= -0.1:
            return SignalType.WEAK_SELL
        else:
            return SignalType.NEUTRAL
    
    def _calculate_sentiment_confidence(self, signals: List[tuple]) -> float:
        """حساب مستوى الثقة من الإشارات."""
        if not signals:
            return 0.0
        
        values = [s[1] for s in signals]
        
        # الاتساق بين الإشارات
        avg = sum(values) / len(values)
        variance = sum((v - avg) ** 2 for v in values) / len(values)
        consistency = max(0, 1 - variance)
        
        # قوة الإشارة
        strength = abs(avg)
        
        return min(1.0, consistency * strength * 1.5)
    
    async def _llm_sentiment_analysis(self, symbol: str, 
                                      texts: List[str]) -> Dict[str, Any]:
        """تحليل المشاعر باستخدام LLM."""
        if not self.llm_client or not texts:
            return {"signal": SignalType.NEUTRAL, "confidence": 0, "reasoning": ""}
        
        try:
            # إعداد النص للتحليل
            combined_text = "\n".join(texts[:10])  # أول 10 نصوص
            
            prompt = f"""
            حلل المشاعر التالية المتعلقة بالعملة {symbol}:
            
            {combined_text}
            
            أجب بصيغة JSON فقط:
            {{
                "sentiment": "bullish" أو "bearish" أو "neutral",
                "score": رقم من -1 إلى 1,
                "confidence": رقم من 0 إلى 1,
                "key_points": ["نقطة 1", "نقطة 2"]
            }}
            """
            
            response = await self.llm_client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=500
            )
            
            result = json.loads(response.choices[0].message.content)
            
            sentiment_map = {
                "bullish": SignalType.BUY,
                "bearish": SignalType.SELL,
                "neutral": SignalType.NEUTRAL
            }
            
            return {
                "signal": sentiment_map.get(result.get("sentiment"), SignalType.NEUTRAL),
                "confidence": result.get("confidence", 0.5),
                "reasoning": " | ".join(result.get("key_points", []))
            }
            
        except Exception as e:
            self.logger.error(f"خطأ في تحليل LLM: {e}")
            return {"signal": SignalType.NEUTRAL, "confidence": 0, "reasoning": ""}
    
    def _merge_analyses(self, base: Dict, llm: Dict) -> Dict:
        """دمج التحليلات."""
        if llm.get("confidence", 0) < 0.3:
            return base
        
        # دمج الثقة
        merged_confidence = (base["confidence"] * 0.6 + llm["confidence"] * 0.4)
        
        # دمج التفسير
        merged_reasoning = base["reasoning"]
        if llm.get("reasoning"):
            merged_reasoning += f" | LLM: {llm['reasoning']}"
        
        return {
            "signal": base["signal"],  # الاحتفاظ بالإشارة الأساسية
            "confidence": merged_confidence,
            "sentiment_score": base["sentiment_score"],
            "reasoning": merged_reasoning
        }
    
    def _create_neutral_result(self, symbol: str, reason: str) -> AnalysisResult:
        """إنشاء نتيجة محايدة."""
        return AnalysisResult(
            analyst_type=AnalystType.SENTIMENT,
            symbol=symbol,
            timestamp=datetime.utcnow(),
            signal=SignalType.NEUTRAL,
            confidence=0.0,
            reasoning=reason,
            data={}
        )
