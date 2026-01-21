"""
Legendary Trading System - News Analyst Agent
نظام التداول الخارق - وكيل محلل الأخبار

يحلل الأخبار والأحداث المؤثرة على السوق باستخدام LLM.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging
import json
import re

from ...core.base_agent import AnalystAgent
from ...core.types import AnalysisResult, SignalType, AnalystType


class NewsAnalystAgent(AnalystAgent):
    """
    وكيل محلل الأخبار.
    
    يحلل:
    - أخبار العملات المشفرة
    - إعلانات المشاريع
    - الأحداث التنظيمية
    - تصريحات الشخصيات المؤثرة
    """
    
    def __init__(self, config: Dict[str, Any], llm_client=None):
        super().__init__(
            name="NewsAnalyst",
            config=config,
            analyst_type="news"
        )
        self.weight = config.get("analyst_weights", {}).get("news", 0.15)
        self.llm_client = llm_client
        
        # كلمات مفتاحية للتصنيف
        self.bullish_keywords = [
            "partnership", "adoption", "launch", "upgrade", "bullish",
            "institutional", "etf approved", "integration", "milestone",
            "شراكة", "تبني", "إطلاق", "ترقية", "صعودي", "مؤسسي"
        ]
        
        self.bearish_keywords = [
            "hack", "exploit", "ban", "regulation", "lawsuit", "bearish",
            "crash", "dump", "scam", "fraud", "investigation",
            "اختراق", "حظر", "تنظيم", "دعوى", "هبوطي", "احتيال"
        ]
        
        self.high_impact_keywords = [
            "sec", "fed", "regulation", "etf", "institutional",
            "blackrock", "grayscale", "coinbase", "binance"
        ]
    
    async def initialize(self) -> bool:
        """تهيئة محلل الأخبار."""
        self.logger.info("تهيئة محلل الأخبار...")
        return True
    
    async def process(self, data: Any) -> Any:
        """معالجة البيانات."""
        return await self.analyze(data.get("symbol"), data)
    
    async def shutdown(self) -> None:
        """إيقاف المحلل."""
        self.logger.info("إيقاف محلل الأخبار")
    
    async def analyze(self, symbol: str, data: Dict[str, Any]) -> AnalysisResult:
        """
        تحليل الأخبار المتعلقة بالرمز.
        
        Args:
            symbol: رمز العملة
            data: بيانات الأخبار
            
        Returns:
            نتيجة التحليل
        """
        self._update_activity()
        
        try:
            # استخراج الأخبار
            news_items = data.get("news", [])
            
            if not news_items:
                return self._create_neutral_result(symbol, "لا توجد أخبار حديثة")
            
            # تحليل أولي بالكلمات المفتاحية
            keyword_analysis = self._keyword_analysis(news_items)
            
            # تحليل LLM إذا متاح
            if self.llm_client:
                llm_analysis = await self._llm_news_analysis(symbol, news_items)
                final_analysis = self._merge_analyses(keyword_analysis, llm_analysis)
            else:
                final_analysis = keyword_analysis
            
            return AnalysisResult(
                analyst_type=AnalystType.NEWS,
                symbol=symbol,
                timestamp=datetime.utcnow(),
                signal=final_analysis["signal"],
                confidence=final_analysis["confidence"],
                reasoning=final_analysis["reasoning"],
                data={
                    "news_count": len(news_items),
                    "sentiment_score": final_analysis["score"],
                    "high_impact_news": final_analysis.get("high_impact", []),
                    "recent_headlines": [n.get("title", "")[:100] for n in news_items[:5]]
                }
            )
            
        except Exception as e:
            self.logger.error(f"خطأ في تحليل الأخبار: {e}")
            self._handle_error(e)
            return self._create_neutral_result(symbol, f"خطأ: {str(e)}")
    
    def _keyword_analysis(self, news_items: List[Dict]) -> Dict[str, Any]:
        """تحليل الأخبار بالكلمات المفتاحية."""
        bullish_count = 0
        bearish_count = 0
        high_impact_news = []
        total_weight = 0
        weighted_sentiment = 0
        
        for news in news_items:
            title = news.get("title", "").lower()
            content = news.get("content", "").lower()
            text = f"{title} {content}"
            
            # حساب عمر الخبر
            published = news.get("published_at")
            if published:
                try:
                    if isinstance(published, str):
                        published = datetime.fromisoformat(published.replace("Z", "+00:00"))
                    age_hours = (datetime.utcnow() - published.replace(tzinfo=None)).total_seconds() / 3600
                except:
                    age_hours = 24
            else:
                age_hours = 24
            
            # وزن الخبر بناءً على العمر
            recency_weight = max(0.1, 1 - (age_hours / 72))  # يتناقص خلال 72 ساعة
            
            # تحليل المشاعر
            bullish_matches = sum(1 for kw in self.bullish_keywords if kw in text)
            bearish_matches = sum(1 for kw in self.bearish_keywords if kw in text)
            
            if bullish_matches > bearish_matches:
                bullish_count += 1
                sentiment = 0.5 + (bullish_matches * 0.1)
            elif bearish_matches > bullish_matches:
                bearish_count += 1
                sentiment = -0.5 - (bearish_matches * 0.1)
            else:
                sentiment = 0
            
            # التحقق من الأخبار عالية التأثير
            is_high_impact = any(kw in text for kw in self.high_impact_keywords)
            if is_high_impact:
                high_impact_news.append({
                    "title": news.get("title", ""),
                    "sentiment": sentiment
                })
                recency_weight *= 1.5  # زيادة وزن الأخبار المهمة
            
            weighted_sentiment += sentiment * recency_weight
            total_weight += recency_weight
        
        # حساب النتيجة النهائية
        if total_weight > 0:
            final_score = weighted_sentiment / total_weight
        else:
            final_score = 0
        
        # تحديد الإشارة
        signal = self._score_to_signal(final_score)
        
        # حساب الثقة
        total_news = bullish_count + bearish_count
        if total_news > 0:
            dominance = abs(bullish_count - bearish_count) / total_news
            confidence = min(1.0, dominance * abs(final_score) * 2)
        else:
            confidence = 0.3
        
        # بناء التفسير
        reasoning_parts = []
        if bullish_count > bearish_count:
            reasoning_parts.append(f"{bullish_count} خبر إيجابي مقابل {bearish_count} سلبي")
        elif bearish_count > bullish_count:
            reasoning_parts.append(f"{bearish_count} خبر سلبي مقابل {bullish_count} إيجابي")
        
        if high_impact_news:
            reasoning_parts.append(f"{len(high_impact_news)} خبر عالي التأثير")
        
        return {
            "signal": signal,
            "confidence": confidence,
            "score": final_score,
            "reasoning": "تحليل الأخبار: " + " | ".join(reasoning_parts) if reasoning_parts else "أخبار محايدة",
            "high_impact": high_impact_news,
            "bullish_count": bullish_count,
            "bearish_count": bearish_count
        }
    
    async def _llm_news_analysis(self, symbol: str, 
                                 news_items: List[Dict]) -> Dict[str, Any]:
        """تحليل الأخبار باستخدام LLM."""
        if not self.llm_client:
            return None
        
        try:
            # إعداد الأخبار للتحليل
            news_text = ""
            for i, news in enumerate(news_items[:10], 1):
                title = news.get("title", "")
                content = news.get("content", "")[:500]
                news_text += f"\n{i}. {title}\n{content}\n"
            
            prompt = f"""
            حلل الأخبار التالية المتعلقة بالعملة المشفرة {symbol} وحدد تأثيرها على السعر:
            
            {news_text}
            
            أجب بصيغة JSON فقط:
            {{
                "overall_sentiment": "bullish" أو "bearish" أو "neutral",
                "sentiment_score": رقم من -1 (سلبي جداً) إلى 1 (إيجابي جداً),
                "confidence": رقم من 0 إلى 1,
                "key_events": ["حدث 1", "حدث 2"],
                "potential_impact": "high" أو "medium" أو "low",
                "time_horizon": "short" أو "medium" أو "long",
                "summary": "ملخص قصير"
            }}
            """
            
            response = await self.llm_client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=800
            )
            
            # استخراج JSON من الرد
            response_text = response.choices[0].message.content
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
            else:
                return None
            
            # تحويل النتيجة
            sentiment_map = {
                "bullish": SignalType.BUY,
                "bearish": SignalType.SELL,
                "neutral": SignalType.NEUTRAL
            }
            
            return {
                "signal": sentiment_map.get(result.get("overall_sentiment"), SignalType.NEUTRAL),
                "confidence": result.get("confidence", 0.5),
                "score": result.get("sentiment_score", 0),
                "reasoning": result.get("summary", ""),
                "key_events": result.get("key_events", []),
                "impact": result.get("potential_impact", "medium")
            }
            
        except Exception as e:
            self.logger.error(f"خطأ في تحليل LLM للأخبار: {e}")
            return None
    
    def _merge_analyses(self, keyword: Dict, llm: Optional[Dict]) -> Dict[str, Any]:
        """دمج تحليل الكلمات المفتاحية مع تحليل LLM."""
        if not llm:
            return keyword
        
        # أوزان الدمج
        keyword_weight = 0.4
        llm_weight = 0.6
        
        # دمج النتائج
        merged_score = (
            keyword["score"] * keyword_weight +
            llm["score"] * llm_weight
        )
        
        merged_confidence = (
            keyword["confidence"] * keyword_weight +
            llm["confidence"] * llm_weight
        )
        
        # تحديد الإشارة
        signal = self._score_to_signal(merged_score)
        
        # دمج التفسير
        reasoning = keyword["reasoning"]
        if llm.get("reasoning"):
            reasoning += f" | LLM: {llm['reasoning']}"
        
        return {
            "signal": signal,
            "confidence": merged_confidence,
            "score": merged_score,
            "reasoning": reasoning,
            "high_impact": keyword.get("high_impact", []),
            "key_events": llm.get("key_events", [])
        }
    
    def _score_to_signal(self, score: float) -> SignalType:
        """تحويل النتيجة إلى إشارة."""
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
    
    def _create_neutral_result(self, symbol: str, reason: str) -> AnalysisResult:
        """إنشاء نتيجة محايدة."""
        return AnalysisResult(
            analyst_type=AnalystType.NEWS,
            symbol=symbol,
            timestamp=datetime.utcnow(),
            signal=SignalType.NEUTRAL,
            confidence=0.0,
            reasoning=reason,
            data={}
        )
