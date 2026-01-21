"""
═══════════════════════════════════════════════════════════════
LEGENDARY AGENT - Understanding Layer
طبقة الفهم - فهم حالة السوق ونوايا اللاعبين
═══════════════════════════════════════════════════════════════
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from loguru import logger


class MarketRegime(Enum):
    """أنظمة السوق"""
    STRONG_BULL = "صعود قوي"
    BULL = "صعود"
    WEAK_BULL = "صعود ضعيف"
    RANGING = "تذبذب"
    WEAK_BEAR = "هبوط ضعيف"
    BEAR = "هبوط"
    STRONG_BEAR = "هبوط قوي"
    CRASH = "انهيار"
    EUPHORIA = "نشوة"


class MarketPhase(Enum):
    """مراحل السوق"""
    ACCUMULATION = "تجميع"
    MARKUP = "صعود"
    DISTRIBUTION = "توزيع"
    MARKDOWN = "هبوط"


class PlayerIntent(Enum):
    """نوايا اللاعبين"""
    ACCUMULATING = "يجمّع"
    DISTRIBUTING = "يوزّع"
    WAITING = "ينتظر"
    AGGRESSIVE_BUYING = "شراء عنيف"
    AGGRESSIVE_SELLING = "بيع عنيف"
    NEUTRAL = "محايد"


@dataclass
class MarketContext:
    """سياق السوق"""
    regime: MarketRegime
    phase: MarketPhase
    trend_strength: float  # 0-1
    volatility_level: str  # LOW, MEDIUM, HIGH, EXTREME
    momentum: str  # STRONG_UP, UP, NEUTRAL, DOWN, STRONG_DOWN
    volume_profile: str  # INCREASING, STABLE, DECREASING
    support_levels: List[float]
    resistance_levels: List[float]
    key_level_proximity: Optional[str] = None  # NEAR_SUPPORT, NEAR_RESISTANCE, BETWEEN


@dataclass
class PlayerAnalysis:
    """تحليل اللاعبين"""
    whale_intent: PlayerIntent
    retail_sentiment: str  # BULLISH, BEARISH, NEUTRAL
    smart_money_flow: str  # INFLOW, OUTFLOW, NEUTRAL
    liquidation_risk: str  # LOW, MEDIUM, HIGH
    funding_bias: str  # LONG, SHORT, NEUTRAL


@dataclass
class UnderstandingState:
    """حالة الفهم"""
    symbol: str
    timestamp: datetime
    market_context: MarketContext
    player_analysis: PlayerAnalysis
    narrative: str  # السردية
    confidence: float
    warnings: List[str] = field(default_factory=list)
    opportunities: List[str] = field(default_factory=list)


class UnderstandingLayer:
    """
    طبقة الفهم
    
    مسؤولة عن:
    - فهم حالة السوق الحالية
    - تحديد نظام السوق
    - تحليل نوايا اللاعبين
    - بناء السردية
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        تهيئة طبقة الفهم
        
        Args:
            config: إعدادات الطبقة
        """
        self.config = config or {}
        
        # تاريخ الفهم
        self.understanding_history: List[UnderstandingState] = []
        self.max_history = 100
        
        # عتبات
        self.thresholds = {
            'strong_trend': 0.7,
            'weak_trend': 0.3,
            'high_volatility': 5.0,
            'low_volatility': 1.5,
            'volume_spike': 2.0,
            'near_level_percent': 0.02
        }
        
        logger.info("🧠 UnderstandingLayer initialized")
    
    # ═══════════════════════════════════════════════════════════════
    # MAIN UNDERSTANDING
    # ═══════════════════════════════════════════════════════════════
    
    def understand(
        self,
        symbol: str,
        features: Dict[str, float],
        historical_features: List[Dict[str, float]] = None
    ) -> UnderstandingState:
        """
        فهم حالة السوق
        
        Args:
            symbol: رمز العملة
            features: الميزات الحالية
            historical_features: الميزات التاريخية
            
        Returns:
            حالة الفهم
        """
        # تحليل السياق
        market_context = self._analyze_market_context(features, historical_features)
        
        # تحليل اللاعبين
        player_analysis = self._analyze_players(features)
        
        # بناء السردية
        narrative = self._build_narrative(market_context, player_analysis)
        
        # تحديد التحذيرات والفرص
        warnings = self._identify_warnings(market_context, player_analysis)
        opportunities = self._identify_opportunities(market_context, player_analysis)
        
        # حساب الثقة
        confidence = self._calculate_confidence(features, market_context)
        
        state = UnderstandingState(
            symbol=symbol,
            timestamp=datetime.now(),
            market_context=market_context,
            player_analysis=player_analysis,
            narrative=narrative,
            confidence=confidence,
            warnings=warnings,
            opportunities=opportunities
        )
        
        # حفظ في التاريخ
        self.understanding_history.append(state)
        if len(self.understanding_history) > self.max_history:
            self.understanding_history = self.understanding_history[-self.max_history:]
        
        return state
    
    # ═══════════════════════════════════════════════════════════════
    # MARKET CONTEXT ANALYSIS
    # ═══════════════════════════════════════════════════════════════
    
    def _analyze_market_context(
        self,
        features: Dict[str, float],
        historical: List[Dict] = None
    ) -> MarketContext:
        """تحليل سياق السوق"""
        # تحديد النظام
        regime = self._determine_regime(features)
        
        # تحديد المرحلة
        phase = self._determine_phase(features, historical)
        
        # قوة الاتجاه
        trend_strength = self._calculate_trend_strength(features)
        
        # مستوى التقلب
        volatility_level = self._assess_volatility_level(features)
        
        # الزخم
        momentum = self._assess_momentum(features)
        
        # ملف الحجم
        volume_profile = self._analyze_volume_profile(features, historical)
        
        # المستويات الرئيسية
        support_levels, resistance_levels = self._identify_levels(features)
        
        # القرب من المستويات
        key_level_proximity = self._check_level_proximity(
            features.get('close', 0),
            support_levels,
            resistance_levels
        )
        
        return MarketContext(
            regime=regime,
            phase=phase,
            trend_strength=trend_strength,
            volatility_level=volatility_level,
            momentum=momentum,
            volume_profile=volume_profile,
            support_levels=support_levels,
            resistance_levels=resistance_levels,
            key_level_proximity=key_level_proximity
        )
    
    def _determine_regime(self, features: Dict) -> MarketRegime:
        """تحديد نظام السوق"""
        rsi = features.get('rsi_14', 50)
        adx = features.get('adx', 25)
        macd = features.get('macd', 0)
        macd_signal = features.get('macd_signal', 0)
        close = features.get('close', 0)
        sma_50 = features.get('sma_50', close)
        sma_200 = features.get('sma_200', close)
        
        # حالات خاصة
        if rsi > 85:
            return MarketRegime.EUPHORIA
        if rsi < 15:
            return MarketRegime.CRASH
        
        # تحديد الاتجاه
        trend_score = 0
        
        if close > sma_50:
            trend_score += 1
        else:
            trend_score -= 1
        
        if close > sma_200:
            trend_score += 1
        else:
            trend_score -= 1
        
        if macd > macd_signal:
            trend_score += 1
        else:
            trend_score -= 1
        
        if rsi > 50:
            trend_score += 0.5
        else:
            trend_score -= 0.5
        
        # تحديد القوة
        if adx > 40:
            strength = 'STRONG'
        elif adx > 25:
            strength = 'NORMAL'
        else:
            strength = 'WEAK'
        
        # تحديد النظام
        if trend_score >= 3:
            return MarketRegime.STRONG_BULL if strength == 'STRONG' else MarketRegime.BULL
        elif trend_score >= 1.5:
            return MarketRegime.BULL if strength != 'WEAK' else MarketRegime.WEAK_BULL
        elif trend_score <= -3:
            return MarketRegime.STRONG_BEAR if strength == 'STRONG' else MarketRegime.BEAR
        elif trend_score <= -1.5:
            return MarketRegime.BEAR if strength != 'WEAK' else MarketRegime.WEAK_BEAR
        else:
            return MarketRegime.RANGING
    
    def _determine_phase(
        self,
        features: Dict,
        historical: List[Dict] = None
    ) -> MarketPhase:
        """تحديد مرحلة السوق"""
        volume_ratio = features.get('volume', 1) / features.get('volume_sma_20', 1)
        rsi = features.get('rsi_14', 50)
        close = features.get('close', 0)
        sma_50 = features.get('sma_50', close)
        
        # تجميع: حجم منخفض، سعر مستقر، RSI منخفض
        if volume_ratio < 0.8 and rsi < 40 and close < sma_50:
            return MarketPhase.ACCUMULATION
        
        # صعود: حجم متزايد، سعر صاعد
        if volume_ratio > 1.2 and close > sma_50 and rsi > 50:
            return MarketPhase.MARKUP
        
        # توزيع: حجم عالي، RSI عالي
        if volume_ratio > 1.5 and rsi > 65:
            return MarketPhase.DISTRIBUTION
        
        # هبوط: سعر هابط
        if close < sma_50 and rsi < 50:
            return MarketPhase.MARKDOWN
        
        return MarketPhase.ACCUMULATION
    
    def _calculate_trend_strength(self, features: Dict) -> float:
        """حساب قوة الاتجاه"""
        adx = features.get('adx', 25)
        
        # تطبيع ADX إلى 0-1
        strength = min(1.0, adx / 50)
        
        return strength
    
    def _assess_volatility_level(self, features: Dict) -> str:
        """تقييم مستوى التقلب"""
        atr_percent = features.get('atr_percent', 2)
        bb_width = features.get('bb_width', 3)
        
        avg_volatility = (atr_percent + bb_width) / 2
        
        if avg_volatility > 7:
            return 'EXTREME'
        elif avg_volatility > 5:
            return 'HIGH'
        elif avg_volatility > 2:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def _assess_momentum(self, features: Dict) -> str:
        """تقييم الزخم"""
        rsi = features.get('rsi_14', 50)
        macd = features.get('macd', 0)
        macd_signal = features.get('macd_signal', 0)
        
        macd_diff = macd - macd_signal
        
        if rsi > 70 and macd_diff > 0:
            return 'STRONG_UP'
        elif rsi > 55 and macd_diff > 0:
            return 'UP'
        elif rsi < 30 and macd_diff < 0:
            return 'STRONG_DOWN'
        elif rsi < 45 and macd_diff < 0:
            return 'DOWN'
        else:
            return 'NEUTRAL'
    
    def _analyze_volume_profile(
        self,
        features: Dict,
        historical: List[Dict] = None
    ) -> str:
        """تحليل ملف الحجم"""
        volume = features.get('volume', 0)
        volume_sma = features.get('volume_sma_20', volume)
        
        ratio = volume / volume_sma if volume_sma > 0 else 1
        
        if ratio > 1.5:
            return 'INCREASING'
        elif ratio < 0.7:
            return 'DECREASING'
        else:
            return 'STABLE'
    
    def _identify_levels(
        self,
        features: Dict
    ) -> Tuple[List[float], List[float]]:
        """تحديد مستويات الدعم والمقاومة"""
        close = features.get('close', 0)
        
        # مستويات من البولنجر
        bb_upper = features.get('bb_upper', close * 1.02)
        bb_lower = features.get('bb_lower', close * 0.98)
        bb_middle = features.get('bb_middle', close)
        
        # مستويات من القمم والقيعان
        highest_20 = features.get('highest_20', close * 1.03)
        lowest_20 = features.get('lowest_20', close * 0.97)
        
        # المتوسطات كمستويات
        sma_50 = features.get('sma_50', close)
        sma_200 = features.get('sma_200', close)
        
        support_levels = sorted([
            lowest_20,
            bb_lower,
            min(sma_50, sma_200)
        ])
        
        resistance_levels = sorted([
            highest_20,
            bb_upper,
            max(sma_50, sma_200)
        ])
        
        return support_levels, resistance_levels
    
    def _check_level_proximity(
        self,
        price: float,
        support_levels: List[float],
        resistance_levels: List[float]
    ) -> Optional[str]:
        """التحقق من القرب من المستويات"""
        if not price or price == 0:
            return None
        
        threshold = self.thresholds['near_level_percent']
        
        for support in support_levels:
            if abs(price - support) / price < threshold:
                return 'NEAR_SUPPORT'
        
        for resistance in resistance_levels:
            if abs(price - resistance) / price < threshold:
                return 'NEAR_RESISTANCE'
        
        return 'BETWEEN'
    
    # ═══════════════════════════════════════════════════════════════
    # PLAYER ANALYSIS
    # ═══════════════════════════════════════════════════════════════
    
    def _analyze_players(self, features: Dict) -> PlayerAnalysis:
        """تحليل اللاعبين"""
        # نية الحيتان
        whale_intent = self._analyze_whale_intent(features)
        
        # مشاعر المتداولين الصغار
        retail_sentiment = self._analyze_retail_sentiment(features)
        
        # تدفق الأموال الذكية
        smart_money_flow = self._analyze_smart_money(features)
        
        # مخاطر التصفية
        liquidation_risk = self._assess_liquidation_risk(features)
        
        # انحياز التمويل
        funding_bias = self._analyze_funding_bias(features)
        
        return PlayerAnalysis(
            whale_intent=whale_intent,
            retail_sentiment=retail_sentiment,
            smart_money_flow=smart_money_flow,
            liquidation_risk=liquidation_risk,
            funding_bias=funding_bias
        )
    
    def _analyze_whale_intent(self, features: Dict) -> PlayerIntent:
        """تحليل نية الحيتان"""
        orderbook_imbalance = features.get('orderbook_imbalance', 0)
        whale_accumulating = features.get('whale_accumulating', 0)
        whale_distributing = features.get('whale_distributing', 0)
        volume_ratio = features.get('volume', 1) / features.get('volume_sma_20', 1)
        
        if whale_accumulating:
            return PlayerIntent.ACCUMULATING
        if whale_distributing:
            return PlayerIntent.DISTRIBUTING
        
        if orderbook_imbalance > 0.3 and volume_ratio > 1.5:
            return PlayerIntent.AGGRESSIVE_BUYING
        if orderbook_imbalance < -0.3 and volume_ratio > 1.5:
            return PlayerIntent.AGGRESSIVE_SELLING
        
        if orderbook_imbalance > 0.1:
            return PlayerIntent.ACCUMULATING
        if orderbook_imbalance < -0.1:
            return PlayerIntent.DISTRIBUTING
        
        return PlayerIntent.NEUTRAL
    
    def _analyze_retail_sentiment(self, features: Dict) -> str:
        """تحليل مشاعر المتداولين الصغار"""
        fear_greed = features.get('fear_greed', 50)
        social_sentiment = features.get('social_sentiment', 0)
        
        combined = (fear_greed / 100 + (social_sentiment + 1) / 2) / 2
        
        if combined > 0.65:
            return 'BULLISH'
        elif combined < 0.35:
            return 'BEARISH'
        else:
            return 'NEUTRAL'
    
    def _analyze_smart_money(self, features: Dict) -> str:
        """تحليل تدفق الأموال الذكية"""
        # استخدام عدم توازن دفتر الأوامر كمؤشر
        imbalance = features.get('orderbook_imbalance', 0)
        volume_ratio = features.get('volume', 1) / features.get('volume_sma_20', 1)
        
        if imbalance > 0.2 and volume_ratio > 1.2:
            return 'INFLOW'
        elif imbalance < -0.2 and volume_ratio > 1.2:
            return 'OUTFLOW'
        else:
            return 'NEUTRAL'
    
    def _assess_liquidation_risk(self, features: Dict) -> str:
        """تقييم مخاطر التصفية"""
        rsi = features.get('rsi_14', 50)
        volatility = features.get('atr_percent', 2)
        
        if (rsi > 80 or rsi < 20) and volatility > 5:
            return 'HIGH'
        elif (rsi > 70 or rsi < 30) and volatility > 3:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def _analyze_funding_bias(self, features: Dict) -> str:
        """تحليل انحياز التمويل"""
        funding_rate = features.get('funding_rate', 0)
        
        if funding_rate > 0.01:
            return 'LONG'
        elif funding_rate < -0.01:
            return 'SHORT'
        else:
            return 'NEUTRAL'
    
    # ═══════════════════════════════════════════════════════════════
    # NARRATIVE BUILDING
    # ═══════════════════════════════════════════════════════════════
    
    def _build_narrative(
        self,
        context: MarketContext,
        players: PlayerAnalysis
    ) -> str:
        """بناء السردية"""
        parts = []
        
        # وصف النظام
        regime_desc = {
            MarketRegime.STRONG_BULL: "السوق في صعود قوي",
            MarketRegime.BULL: "السوق صاعد",
            MarketRegime.WEAK_BULL: "السوق يميل للصعود لكن بضعف",
            MarketRegime.RANGING: "السوق في حالة تذبذب",
            MarketRegime.WEAK_BEAR: "السوق يميل للهبوط",
            MarketRegime.BEAR: "السوق هابط",
            MarketRegime.STRONG_BEAR: "السوق في هبوط حاد",
            MarketRegime.CRASH: "السوق في حالة انهيار!",
            MarketRegime.EUPHORIA: "السوق في حالة نشوة!"
        }
        parts.append(regime_desc.get(context.regime, "حالة السوق غير واضحة"))
        
        # وصف المرحلة
        phase_desc = {
            MarketPhase.ACCUMULATION: "مرحلة تجميع",
            MarketPhase.MARKUP: "مرحلة صعود",
            MarketPhase.DISTRIBUTION: "مرحلة توزيع",
            MarketPhase.MARKDOWN: "مرحلة هبوط"
        }
        parts.append(f"({phase_desc.get(context.phase, '')})")
        
        # وصف الحيتان
        whale_desc = {
            PlayerIntent.ACCUMULATING: "الحيتان تجمّع",
            PlayerIntent.DISTRIBUTING: "الحيتان توزّع",
            PlayerIntent.AGGRESSIVE_BUYING: "شراء عنيف من الحيتان",
            PlayerIntent.AGGRESSIVE_SELLING: "بيع عنيف من الحيتان",
            PlayerIntent.NEUTRAL: "الحيتان محايدة"
        }
        parts.append(whale_desc.get(players.whale_intent, ""))
        
        # القرب من المستويات
        if context.key_level_proximity == 'NEAR_SUPPORT':
            parts.append("السعر قريب من الدعم")
        elif context.key_level_proximity == 'NEAR_RESISTANCE':
            parts.append("السعر قريب من المقاومة")
        
        return ". ".join(filter(None, parts)) + "."
    
    # ═══════════════════════════════════════════════════════════════
    # WARNINGS & OPPORTUNITIES
    # ═══════════════════════════════════════════════════════════════
    
    def _identify_warnings(
        self,
        context: MarketContext,
        players: PlayerAnalysis
    ) -> List[str]:
        """تحديد التحذيرات"""
        warnings = []
        
        if context.regime in [MarketRegime.CRASH, MarketRegime.EUPHORIA]:
            warnings.append("⚠️ السوق في حالة متطرفة!")
        
        if context.volatility_level == 'EXTREME':
            warnings.append("⚠️ تقلب شديد!")
        
        if players.liquidation_risk == 'HIGH':
            warnings.append("⚠️ مخاطر تصفية عالية!")
        
        if players.whale_intent == PlayerIntent.DISTRIBUTING:
            warnings.append("⚠️ الحيتان توزّع - احذر!")
        
        if context.phase == MarketPhase.DISTRIBUTION:
            warnings.append("⚠️ مرحلة توزيع - قد يبدأ الهبوط")
        
        return warnings
    
    def _identify_opportunities(
        self,
        context: MarketContext,
        players: PlayerAnalysis
    ) -> List[str]:
        """تحديد الفرص"""
        opportunities = []
        
        if context.phase == MarketPhase.ACCUMULATION and players.whale_intent == PlayerIntent.ACCUMULATING:
            opportunities.append("🎯 فرصة شراء: تجميع + حيتان تجمّع")
        
        if context.key_level_proximity == 'NEAR_SUPPORT' and context.momentum != 'STRONG_DOWN':
            opportunities.append("🎯 فرصة شراء: قرب الدعم")
        
        if context.regime in [MarketRegime.BULL, MarketRegime.STRONG_BULL] and players.smart_money_flow == 'INFLOW':
            opportunities.append("🎯 فرصة شراء: اتجاه صاعد + تدفق أموال")
        
        return opportunities
    
    def _calculate_confidence(
        self,
        features: Dict,
        context: MarketContext
    ) -> float:
        """حساب الثقة"""
        confidence = 0.5
        
        # زيادة الثقة إذا كان الاتجاه قوي
        confidence += context.trend_strength * 0.2
        
        # تقليل الثقة في التقلب العالي
        if context.volatility_level in ['HIGH', 'EXTREME']:
            confidence -= 0.1
        
        # زيادة الثقة في الأنظمة الواضحة
        if context.regime in [MarketRegime.STRONG_BULL, MarketRegime.STRONG_BEAR]:
            confidence += 0.1
        
        return max(0.1, min(0.95, confidence))
    
    # ═══════════════════════════════════════════════════════════════
    # STATUS
    # ═══════════════════════════════════════════════════════════════
    
    def get_status(self) -> Dict[str, Any]:
        """الحصول على حالة الطبقة"""
        return {
            'history_size': len(self.understanding_history),
            'last_understanding': (
                self.understanding_history[-1].narrative
                if self.understanding_history else None
            )
        }


# ═══════════════════════════════════════════════════════════════
# STANDALONE EXECUTION
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # اختبار طبقة الفهم
    understanding = UnderstandingLayer()
    
    features = {
        'close': 50000,
        'rsi_14': 65,
        'adx': 35,
        'macd': 0.5,
        'macd_signal': 0.3,
        'sma_50': 48000,
        'sma_200': 45000,
        'bb_upper': 52000,
        'bb_lower': 47000,
        'bb_middle': 49500,
        'bb_width': 3.5,
        'atr_percent': 2.5,
        'volume': 1000000,
        'volume_sma_20': 800000,
        'highest_20': 51000,
        'lowest_20': 46000,
        'orderbook_imbalance': 0.15,
        'fear_greed': 65,
        'social_sentiment': 0.3
    }
    
    state = understanding.understand('BTCUSDT', features)
    
    print("🧠 Understanding State:")
    print(f"Regime: {state.market_context.regime.value}")
    print(f"Phase: {state.market_context.phase.value}")
    print(f"Trend Strength: {state.market_context.trend_strength:.2f}")
    print(f"Volatility: {state.market_context.volatility_level}")
    print(f"\nWhale Intent: {state.player_analysis.whale_intent.value}")
    print(f"Retail Sentiment: {state.player_analysis.retail_sentiment}")
    print(f"\nNarrative: {state.narrative}")
    print(f"Confidence: {state.confidence:.2%}")
    print(f"\nWarnings: {state.warnings}")
    print(f"Opportunities: {state.opportunities}")
