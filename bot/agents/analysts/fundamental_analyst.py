"""
Legendary Trading System - Fundamental Analyst Agent
نظام التداول الخارق - وكيل المحلل الأساسي

يحلل الأساسيات والمقاييس الجوهرية للعملات المشفرة.
"""

from datetime import datetime
from typing import Dict, List, Any, Optional
import logging
import math

from ...core.base_agent import AnalystAgent
from ...core.types import AnalysisResult, SignalType, AnalystType


class FundamentalAnalystAgent(AnalystAgent):
    """
    وكيل المحلل الأساسي.
    
    يحلل:
    - القيمة السوقية والتقييم
    - حجم التداول والسيولة
    - العرض والتوزيع
    - مقاييس المشروع (TVL, المستخدمين, النمو)
    - مقارنة مع المنافسين
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(
            name="FundamentalAnalyst",
            config=config,
            analyst_type="fundamental"
        )
        self.weight = config.get("analyst_weights", {}).get("fundamental", 0.20)
        
        # معايير التقييم
        self.market_cap_tiers = {
            "mega": 50_000_000_000,    # > 50B
            "large": 10_000_000_000,   # > 10B
            "mid": 1_000_000_000,      # > 1B
            "small": 100_000_000,      # > 100M
            "micro": 0                  # < 100M
        }
    
    async def initialize(self) -> bool:
        """تهيئة المحلل الأساسي."""
        self.logger.info("تهيئة المحلل الأساسي...")
        return True
    
    async def process(self, data: Any) -> Any:
        """معالجة البيانات."""
        return await self.analyze(data.get("symbol"), data)
    
    async def shutdown(self) -> None:
        """إيقاف المحلل."""
        self.logger.info("إيقاف المحلل الأساسي")
    
    async def analyze(self, symbol: str, data: Dict[str, Any]) -> AnalysisResult:
        """
        تحليل أساسي شامل للعملة.
        
        Args:
            symbol: رمز العملة
            data: بيانات العملة
            
        Returns:
            نتيجة التحليل
        """
        self._update_activity()
        
        try:
            # استخراج البيانات الأساسية
            fundamentals = data.get("fundamentals", {})
            market_data = data.get("market", {})
            
            # تحليل القيمة السوقية
            mcap_analysis = self._analyze_market_cap(market_data)
            
            # تحليل حجم التداول والسيولة
            volume_analysis = self._analyze_volume(market_data)
            
            # تحليل العرض
            supply_analysis = self._analyze_supply(fundamentals, market_data)
            
            # تحليل مقاييس المشروع
            project_analysis = self._analyze_project_metrics(fundamentals)
            
            # تحليل التقييم النسبي
            valuation_analysis = self._analyze_valuation(market_data, fundamentals)
            
            # دمج التحليلات
            final_analysis = self._combine_analyses(
                mcap_analysis, volume_analysis, supply_analysis,
                project_analysis, valuation_analysis
            )
            
            return AnalysisResult(
                analyst_type=AnalystType.FUNDAMENTAL,
                symbol=symbol,
                timestamp=datetime.utcnow(),
                signal=final_analysis["signal"],
                confidence=final_analysis["confidence"],
                reasoning=final_analysis["reasoning"],
                data={
                    "market_cap_tier": mcap_analysis.get("tier"),
                    "volume_score": volume_analysis.get("score"),
                    "supply_score": supply_analysis.get("score"),
                    "project_score": project_analysis.get("score"),
                    "valuation_score": valuation_analysis.get("score"),
                    "overall_score": final_analysis["score"]
                }
            )
            
        except Exception as e:
            self.logger.error(f"خطأ في التحليل الأساسي: {e}")
            self._handle_error(e)
            return self._create_neutral_result(symbol, f"خطأ: {str(e)}")
    
    def _analyze_market_cap(self, market_data: Dict) -> Dict[str, Any]:
        """تحليل القيمة السوقية."""
        market_cap = market_data.get("market_cap", 0)
        mcap_change_24h = market_data.get("market_cap_change_24h", 0)
        mcap_rank = market_data.get("market_cap_rank", 999)
        
        # تحديد الفئة
        tier = "micro"
        for tier_name, threshold in self.market_cap_tiers.items():
            if market_cap >= threshold:
                tier = tier_name
                break
        
        # حساب النتيجة
        score = 0
        descriptions = []
        
        # نقاط للفئة (العملات الكبيرة أكثر استقراراً)
        tier_scores = {
            "mega": 0.2,
            "large": 0.15,
            "mid": 0.1,
            "small": 0.0,
            "micro": -0.1
        }
        score += tier_scores.get(tier, 0)
        descriptions.append(f"فئة {tier} (المرتبة #{mcap_rank})")
        
        # نقاط للتغير
        if mcap_change_24h > 5:
            score += 0.2
            descriptions.append(f"نمو قوي في القيمة السوقية (+{mcap_change_24h:.1f}%)")
        elif mcap_change_24h > 2:
            score += 0.1
            descriptions.append(f"نمو معتدل في القيمة السوقية (+{mcap_change_24h:.1f}%)")
        elif mcap_change_24h < -5:
            score -= 0.2
            descriptions.append(f"انخفاض حاد في القيمة السوقية ({mcap_change_24h:.1f}%)")
        elif mcap_change_24h < -2:
            score -= 0.1
            descriptions.append(f"انخفاض معتدل في القيمة السوقية ({mcap_change_24h:.1f}%)")
        
        return {
            "score": score,
            "tier": tier,
            "market_cap": market_cap,
            "rank": mcap_rank,
            "description": " | ".join(descriptions)
        }
    
    def _analyze_volume(self, market_data: Dict) -> Dict[str, Any]:
        """تحليل حجم التداول والسيولة."""
        volume_24h = market_data.get("volume_24h", 0)
        market_cap = market_data.get("market_cap", 1)
        volume_change = market_data.get("volume_change_24h", 0)
        
        # نسبة الحجم للقيمة السوقية
        volume_mcap_ratio = volume_24h / market_cap if market_cap > 0 else 0
        
        score = 0
        descriptions = []
        
        # تحليل نسبة الحجم
        if volume_mcap_ratio > 0.3:
            score += 0.15
            descriptions.append("سيولة عالية جداً")
        elif volume_mcap_ratio > 0.1:
            score += 0.1
            descriptions.append("سيولة جيدة")
        elif volume_mcap_ratio < 0.02:
            score -= 0.15
            descriptions.append("سيولة منخفضة")
        elif volume_mcap_ratio < 0.05:
            score -= 0.05
            descriptions.append("سيولة متوسطة")
        
        # تحليل تغير الحجم
        if volume_change > 50:
            score += 0.2
            descriptions.append(f"ارتفاع كبير في الحجم (+{volume_change:.0f}%)")
        elif volume_change > 20:
            score += 0.1
            descriptions.append(f"ارتفاع في الحجم (+{volume_change:.0f}%)")
        elif volume_change < -30:
            score -= 0.1
            descriptions.append(f"انخفاض في الحجم ({volume_change:.0f}%)")
        
        return {
            "score": score,
            "volume_24h": volume_24h,
            "volume_mcap_ratio": volume_mcap_ratio,
            "description": " | ".join(descriptions) if descriptions else "حجم تداول عادي"
        }
    
    def _analyze_supply(self, fundamentals: Dict, market_data: Dict) -> Dict[str, Any]:
        """تحليل العرض والتوزيع."""
        circulating = fundamentals.get("circulating_supply", 0)
        total = fundamentals.get("total_supply", 0)
        max_supply = fundamentals.get("max_supply")
        
        score = 0
        descriptions = []
        
        # نسبة العرض المتداول
        if total > 0:
            circulation_ratio = circulating / total
            
            if circulation_ratio > 0.9:
                score += 0.1
                descriptions.append(f"معظم العرض متداول ({circulation_ratio*100:.0f}%)")
            elif circulation_ratio < 0.3:
                score -= 0.15
                descriptions.append(f"عرض متداول منخفض ({circulation_ratio*100:.0f}%) - خطر تخفيف")
            elif circulation_ratio < 0.5:
                score -= 0.05
                descriptions.append(f"عرض متداول متوسط ({circulation_ratio*100:.0f}%)")
        
        # تحليل الحد الأقصى للعرض
        if max_supply:
            if circulating / max_supply > 0.8:
                score += 0.1
                descriptions.append("قرب من الحد الأقصى للعرض (ندرة)")
        else:
            score -= 0.05
            descriptions.append("لا يوجد حد أقصى للعرض")
        
        # معدل التضخم
        inflation_rate = fundamentals.get("inflation_rate")
        if inflation_rate is not None:
            if inflation_rate < 2:
                score += 0.1
                descriptions.append(f"تضخم منخفض ({inflation_rate:.1f}%)")
            elif inflation_rate > 10:
                score -= 0.15
                descriptions.append(f"تضخم مرتفع ({inflation_rate:.1f}%)")
        
        return {
            "score": score,
            "circulating_supply": circulating,
            "total_supply": total,
            "max_supply": max_supply,
            "description": " | ".join(descriptions) if descriptions else "عرض عادي"
        }
    
    def _analyze_project_metrics(self, fundamentals: Dict) -> Dict[str, Any]:
        """تحليل مقاييس المشروع."""
        tvl = fundamentals.get("tvl")
        tvl_change = fundamentals.get("tvl_change_24h", 0)
        active_users = fundamentals.get("active_users")
        user_growth = fundamentals.get("user_growth_30d", 0)
        developer_activity = fundamentals.get("developer_activity")
        
        score = 0
        descriptions = []
        
        # تحليل TVL (للمشاريع DeFi)
        if tvl is not None:
            if tvl > 1_000_000_000:
                score += 0.2
                descriptions.append(f"TVL قوي (${tvl/1e9:.1f}B)")
            elif tvl > 100_000_000:
                score += 0.1
                descriptions.append(f"TVL جيد (${tvl/1e6:.0f}M)")
            
            if tvl_change > 10:
                score += 0.1
                descriptions.append(f"نمو TVL (+{tvl_change:.0f}%)")
            elif tvl_change < -10:
                score -= 0.1
                descriptions.append(f"انخفاض TVL ({tvl_change:.0f}%)")
        
        # تحليل المستخدمين النشطين
        if active_users is not None:
            if active_users > 100000:
                score += 0.15
                descriptions.append(f"قاعدة مستخدمين كبيرة ({active_users:,})")
            elif active_users > 10000:
                score += 0.05
                descriptions.append(f"قاعدة مستخدمين جيدة ({active_users:,})")
            
            if user_growth > 20:
                score += 0.1
                descriptions.append(f"نمو مستخدمين قوي (+{user_growth:.0f}%)")
        
        # نشاط المطورين
        if developer_activity is not None:
            if developer_activity > 80:
                score += 0.15
                descriptions.append("نشاط تطوير مرتفع")
            elif developer_activity > 50:
                score += 0.05
                descriptions.append("نشاط تطوير جيد")
            elif developer_activity < 20:
                score -= 0.1
                descriptions.append("نشاط تطوير منخفض")
        
        return {
            "score": score,
            "tvl": tvl,
            "active_users": active_users,
            "developer_activity": developer_activity,
            "description": " | ".join(descriptions) if descriptions else "مقاييس مشروع غير متاحة"
        }
    
    def _analyze_valuation(self, market_data: Dict, fundamentals: Dict) -> Dict[str, Any]:
        """تحليل التقييم النسبي."""
        market_cap = market_data.get("market_cap", 0)
        tvl = fundamentals.get("tvl")
        revenue = fundamentals.get("revenue_30d")
        
        score = 0
        descriptions = []
        
        # نسبة MC/TVL (للمشاريع DeFi)
        if tvl and tvl > 0:
            mc_tvl_ratio = market_cap / tvl
            
            if mc_tvl_ratio < 1:
                score += 0.2
                descriptions.append(f"تقييم منخفض (MC/TVL={mc_tvl_ratio:.2f})")
            elif mc_tvl_ratio < 2:
                score += 0.1
                descriptions.append(f"تقييم معقول (MC/TVL={mc_tvl_ratio:.2f})")
            elif mc_tvl_ratio > 10:
                score -= 0.15
                descriptions.append(f"تقييم مرتفع (MC/TVL={mc_tvl_ratio:.2f})")
            elif mc_tvl_ratio > 5:
                score -= 0.05
                descriptions.append(f"تقييم مرتفع نسبياً (MC/TVL={mc_tvl_ratio:.2f})")
        
        # نسبة P/S (السعر للإيرادات)
        if revenue and revenue > 0:
            ps_ratio = market_cap / (revenue * 12)  # سنوي
            
            if ps_ratio < 10:
                score += 0.15
                descriptions.append(f"P/S جذاب ({ps_ratio:.1f})")
            elif ps_ratio < 30:
                score += 0.05
                descriptions.append(f"P/S معقول ({ps_ratio:.1f})")
            elif ps_ratio > 100:
                score -= 0.1
                descriptions.append(f"P/S مرتفع ({ps_ratio:.1f})")
        
        # مقارنة مع ATH
        ath = market_data.get("ath")
        current_price = market_data.get("price", 0)
        if ath and current_price:
            ath_ratio = current_price / ath
            
            if ath_ratio < 0.2:
                score += 0.15
                descriptions.append(f"بعيد عن ATH ({ath_ratio*100:.0f}%)")
            elif ath_ratio < 0.5:
                score += 0.05
                descriptions.append(f"دون ATH ({ath_ratio*100:.0f}%)")
            elif ath_ratio > 0.9:
                score -= 0.1
                descriptions.append(f"قرب ATH ({ath_ratio*100:.0f}%)")
        
        return {
            "score": score,
            "description": " | ".join(descriptions) if descriptions else "تقييم غير محدد"
        }
    
    def _combine_analyses(self, mcap: Dict, volume: Dict, supply: Dict,
                         project: Dict, valuation: Dict) -> Dict[str, Any]:
        """دمج جميع التحليلات."""
        # أوزان التحليلات
        weights = {
            "mcap": 0.15,
            "volume": 0.20,
            "supply": 0.15,
            "project": 0.25,
            "valuation": 0.25
        }
        
        # حساب النتيجة المرجحة
        weighted_score = (
            mcap["score"] * weights["mcap"] +
            volume["score"] * weights["volume"] +
            supply["score"] * weights["supply"] +
            project["score"] * weights["project"] +
            valuation["score"] * weights["valuation"]
        )
        
        # تحديد الإشارة
        if weighted_score >= 0.3:
            signal = SignalType.STRONG_BUY
        elif weighted_score >= 0.15:
            signal = SignalType.BUY
        elif weighted_score >= 0.05:
            signal = SignalType.WEAK_BUY
        elif weighted_score <= -0.3:
            signal = SignalType.STRONG_SELL
        elif weighted_score <= -0.15:
            signal = SignalType.SELL
        elif weighted_score <= -0.05:
            signal = SignalType.WEAK_SELL
        else:
            signal = SignalType.NEUTRAL
        
        # حساب الثقة
        scores = [mcap["score"], volume["score"], supply["score"],
                  project["score"], valuation["score"]]
        avg = sum(scores) / len(scores)
        variance = sum((s - avg) ** 2 for s in scores) / len(scores)
        consistency = max(0, 1 - variance * 5)
        confidence = min(1.0, abs(weighted_score) * consistency * 2)
        
        # بناء التفسير
        reasoning_parts = []
        for name, analysis in [("القيمة السوقية", mcap), ("الحجم", volume),
                               ("العرض", supply), ("المشروع", project),
                               ("التقييم", valuation)]:
            if abs(analysis["score"]) > 0.1:
                reasoning_parts.append(analysis.get("description", ""))
        
        return {
            "signal": signal,
            "confidence": confidence,
            "score": weighted_score,
            "reasoning": "التحليل الأساسي: " + " | ".join(reasoning_parts[:3]) if reasoning_parts else "تحليل أساسي محايد"
        }
    
    def _create_neutral_result(self, symbol: str, reason: str) -> AnalysisResult:
        """إنشاء نتيجة محايدة."""
        return AnalysisResult(
            analyst_type=AnalystType.FUNDAMENTAL,
            symbol=symbol,
            timestamp=datetime.utcnow(),
            signal=SignalType.NEUTRAL,
            confidence=0.0,
            reasoning=reason,
            data={}
        )
