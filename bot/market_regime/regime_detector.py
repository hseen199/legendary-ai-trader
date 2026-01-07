"""
Legendary Trading System - Market Regime Detection
نظام التداول الخارق - كشف الأنظمة السوقية

نظام متقدم للتنبؤ بحالة السوق وتغيراتها.
"""

import asyncio
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
from enum import Enum
import logging


class MarketRegime(Enum):
    """أنظمة السوق"""
    STRONG_BULL = "strong_bull"           # صعود قوي
    BULL = "bull"                          # صعود
    WEAK_BULL = "weak_bull"               # صعود ضعيف
    SIDEWAYS = "sideways"                  # جانبي
    WEAK_BEAR = "weak_bear"               # هبوط ضعيف
    BEAR = "bear"                          # هبوط
    STRONG_BEAR = "strong_bear"           # هبوط قوي
    HIGH_VOLATILITY = "high_volatility"   # تقلب عالي
    CRASH = "crash"                        # انهيار
    RECOVERY = "recovery"                  # تعافي


class VolatilityRegime(Enum):
    """أنظمة التقلب"""
    VERY_LOW = "very_low"
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    EXTREME = "extreme"


@dataclass
class RegimeState:
    """حالة النظام السوقي"""
    timestamp: datetime
    regime: MarketRegime
    volatility_regime: VolatilityRegime
    confidence: float
    trend_strength: float
    momentum: float
    volume_profile: str
    support_levels: List[float]
    resistance_levels: List[float]
    
    # احتمالات التحول
    transition_probabilities: Dict[str, float] = field(default_factory=dict)
    
    # مدة النظام الحالي
    regime_duration: int = 0  # عدد الشموع


@dataclass
class RegimeTransition:
    """تحول بين الأنظمة"""
    from_regime: MarketRegime
    to_regime: MarketRegime
    timestamp: datetime
    trigger: str
    confidence: float


class MarketRegimeDetector:
    """
    نظام كشف الأنظمة السوقية.
    
    يوفر:
    - تحديد حالة السوق (صاعد، هابط، جانبي، متقلب)
    - تغيير الاستراتيجية حسب حالة السوق
    - التنبؤ بتغير حالة السوق قبل حدوثه
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.logger = logging.getLogger("MarketRegimeDetector")
        self.config = config or {}
        
        # تاريخ الأنظمة
        self.regime_history: deque = deque(maxlen=1000)
        
        # تحولات الأنظمة
        self.transitions: List[RegimeTransition] = []
        
        # الحالة الحالية
        self.current_state: Optional[RegimeState] = None
        
        # مصفوفة الانتقال (Markov)
        self.transition_matrix: Dict[str, Dict[str, float]] = self._init_transition_matrix()
        
        # عتبات الكشف
        self.thresholds = {
            "strong_trend": 0.03,      # 3% تغير يومي
            "weak_trend": 0.01,        # 1% تغير يومي
            "high_volatility": 0.04,   # 4% تقلب
            "extreme_volatility": 0.08, # 8% تقلب
            "crash_threshold": -0.10,  # -10% انهيار
            "recovery_threshold": 0.05  # 5% تعافي
        }
        
        # إحصائيات
        self.stats = {
            "regimes_detected": 0,
            "transitions_count": 0,
            "prediction_accuracy": 0.0,
            "correct_predictions": 0,
            "total_predictions": 0
        }
    
    def _init_transition_matrix(self) -> Dict[str, Dict[str, float]]:
        """تهيئة مصفوفة الانتقال."""
        regimes = [r.value for r in MarketRegime]
        matrix = {}
        
        for regime in regimes:
            matrix[regime] = {}
            for target in regimes:
                if regime == target:
                    matrix[regime][target] = 0.6  # احتمال البقاء
                else:
                    matrix[regime][target] = 0.4 / (len(regimes) - 1)
        
        # تعديلات منطقية
        # من الصعود القوي
        matrix["strong_bull"]["bull"] = 0.25
        matrix["strong_bull"]["weak_bull"] = 0.1
        matrix["strong_bull"]["strong_bull"] = 0.5
        
        # من الهبوط القوي
        matrix["strong_bear"]["bear"] = 0.25
        matrix["strong_bear"]["weak_bear"] = 0.1
        matrix["strong_bear"]["strong_bear"] = 0.5
        
        # من الجانبي
        matrix["sideways"]["bull"] = 0.15
        matrix["sideways"]["bear"] = 0.15
        matrix["sideways"]["sideways"] = 0.5
        
        return matrix
    
    async def detect_regime(self, 
                           market_data: Dict[str, Any]) -> RegimeState:
        """
        كشف النظام السوقي الحالي.
        
        Args:
            market_data: بيانات السوق
            
        Returns:
            حالة النظام
        """
        # استخراج البيانات
        prices = market_data.get("prices", [])
        volumes = market_data.get("volumes", [])
        
        if len(prices) < 20:
            return self._default_state()
        
        # حساب المؤشرات
        returns = self._calculate_returns(prices)
        volatility = self._calculate_volatility(returns)
        trend = self._calculate_trend(prices)
        momentum = self._calculate_momentum(prices)
        volume_profile = self._analyze_volume(volumes)
        
        # تحديد النظام
        regime = self._classify_regime(returns, volatility, trend, momentum)
        volatility_regime = self._classify_volatility(volatility)
        
        # حساب مستويات الدعم والمقاومة
        support, resistance = self._find_sr_levels(prices)
        
        # حساب احتمالات التحول
        transition_probs = self._calculate_transition_probabilities(regime)
        
        # حساب مدة النظام
        duration = self._calculate_regime_duration(regime)
        
        # إنشاء الحالة
        state = RegimeState(
            timestamp=datetime.utcnow(),
            regime=regime,
            volatility_regime=volatility_regime,
            confidence=self._calculate_confidence(returns, volatility, trend),
            trend_strength=abs(trend),
            momentum=momentum,
            volume_profile=volume_profile,
            support_levels=support,
            resistance_levels=resistance,
            transition_probabilities=transition_probs,
            regime_duration=duration
        )
        
        # تحديث التاريخ
        self._update_history(state)
        
        self.current_state = state
        self.stats["regimes_detected"] += 1
        
        self.logger.debug(f"النظام: {regime.value} (ثقة: {state.confidence:.1%})")
        
        return state
    
    def _calculate_returns(self, prices: List[float]) -> List[float]:
        """حساب العوائد."""
        returns = []
        for i in range(1, len(prices)):
            ret = (prices[i] - prices[i-1]) / prices[i-1]
            returns.append(ret)
        return returns
    
    def _calculate_volatility(self, returns: List[float]) -> float:
        """حساب التقلب."""
        if not returns:
            return 0
        return float(np.std(returns))
    
    def _calculate_trend(self, prices: List[float]) -> float:
        """حساب الاتجاه."""
        if len(prices) < 2:
            return 0
        
        # اتجاه قصير المدى
        short_trend = (prices[-1] - prices[-5]) / prices[-5] if len(prices) >= 5 else 0
        
        # اتجاه متوسط المدى
        mid_trend = (prices[-1] - prices[-20]) / prices[-20] if len(prices) >= 20 else short_trend
        
        # المتوسط المرجح
        return short_trend * 0.6 + mid_trend * 0.4
    
    def _calculate_momentum(self, prices: List[float]) -> float:
        """حساب الزخم."""
        if len(prices) < 14:
            return 0
        
        # RSI-like momentum
        gains = []
        losses = []
        
        for i in range(1, min(15, len(prices))):
            change = prices[-i] - prices[-i-1]
            if change > 0:
                gains.append(change)
            else:
                losses.append(abs(change))
        
        avg_gain = sum(gains) / len(gains) if gains else 0
        avg_loss = sum(losses) / len(losses) if losses else 0.0001
        
        rs = avg_gain / avg_loss
        momentum = (rs - 1) / (rs + 1)  # تطبيع بين -1 و 1
        
        return momentum
    
    def _analyze_volume(self, volumes: List[float]) -> str:
        """تحليل الحجم."""
        if len(volumes) < 20:
            return "normal"
        
        recent_avg = sum(volumes[-5:]) / 5
        historical_avg = sum(volumes[-20:]) / 20
        
        ratio = recent_avg / historical_avg if historical_avg > 0 else 1
        
        if ratio > 2:
            return "very_high"
        elif ratio > 1.5:
            return "high"
        elif ratio < 0.5:
            return "low"
        else:
            return "normal"
    
    def _classify_regime(self,
                        returns: List[float],
                        volatility: float,
                        trend: float,
                        momentum: float) -> MarketRegime:
        """تصنيف النظام السوقي."""
        # فحص الانهيار
        if returns and returns[-1] < self.thresholds["crash_threshold"]:
            return MarketRegime.CRASH
        
        # فحص التعافي
        if returns and returns[-1] > self.thresholds["recovery_threshold"]:
            return MarketRegime.RECOVERY
        
        # فحص التقلب العالي
        if volatility > self.thresholds["extreme_volatility"]:
            return MarketRegime.HIGH_VOLATILITY
        
        # تصنيف الاتجاه
        if trend > self.thresholds["strong_trend"]:
            return MarketRegime.STRONG_BULL
        elif trend > self.thresholds["weak_trend"]:
            if momentum > 0.3:
                return MarketRegime.BULL
            else:
                return MarketRegime.WEAK_BULL
        elif trend < -self.thresholds["strong_trend"]:
            return MarketRegime.STRONG_BEAR
        elif trend < -self.thresholds["weak_trend"]:
            if momentum < -0.3:
                return MarketRegime.BEAR
            else:
                return MarketRegime.WEAK_BEAR
        else:
            return MarketRegime.SIDEWAYS
    
    def _classify_volatility(self, volatility: float) -> VolatilityRegime:
        """تصنيف التقلب."""
        if volatility < 0.005:
            return VolatilityRegime.VERY_LOW
        elif volatility < 0.015:
            return VolatilityRegime.LOW
        elif volatility < 0.03:
            return VolatilityRegime.NORMAL
        elif volatility < 0.05:
            return VolatilityRegime.HIGH
        else:
            return VolatilityRegime.EXTREME
    
    def _find_sr_levels(self, prices: List[float]) -> Tuple[List[float], List[float]]:
        """إيجاد مستويات الدعم والمقاومة."""
        if len(prices) < 20:
            return [], []
        
        support = []
        resistance = []
        
        # البحث عن القمم والقيعان
        for i in range(2, len(prices) - 2):
            # قاع محلي (دعم)
            if prices[i] < prices[i-1] and prices[i] < prices[i-2] and \
               prices[i] < prices[i+1] and prices[i] < prices[i+2]:
                support.append(prices[i])
            
            # قمة محلية (مقاومة)
            if prices[i] > prices[i-1] and prices[i] > prices[i-2] and \
               prices[i] > prices[i+1] and prices[i] > prices[i+2]:
                resistance.append(prices[i])
        
        # ترتيب وتصفية
        support = sorted(set(support))[-3:]
        resistance = sorted(set(resistance))[:3]
        
        return support, resistance
    
    def _calculate_transition_probabilities(self,
                                           current_regime: MarketRegime) -> Dict[str, float]:
        """حساب احتمالات التحول."""
        return self.transition_matrix.get(current_regime.value, {})
    
    def _calculate_regime_duration(self, regime: MarketRegime) -> int:
        """حساب مدة النظام الحالي."""
        if not self.regime_history:
            return 1
        
        duration = 1
        for state in reversed(list(self.regime_history)):
            if state.regime == regime:
                duration += 1
            else:
                break
        
        return duration
    
    def _calculate_confidence(self,
                             returns: List[float],
                             volatility: float,
                             trend: float) -> float:
        """حساب الثقة في التصنيف."""
        confidence = 0.5
        
        # قوة الاتجاه تزيد الثقة
        confidence += min(0.3, abs(trend) * 5)
        
        # التقلب المنخفض يزيد الثقة
        if volatility < 0.02:
            confidence += 0.1
        elif volatility > 0.05:
            confidence -= 0.1
        
        # اتساق العوائد
        if returns:
            same_sign = sum(1 for r in returns[-10:] if (r > 0) == (trend > 0))
            confidence += (same_sign / 10 - 0.5) * 0.2
        
        return max(0.1, min(0.95, confidence))
    
    def _update_history(self, state: RegimeState):
        """تحديث التاريخ."""
        # فحص التحول
        if self.current_state and self.current_state.regime != state.regime:
            transition = RegimeTransition(
                from_regime=self.current_state.regime,
                to_regime=state.regime,
                timestamp=datetime.utcnow(),
                trigger=self._identify_trigger(self.current_state, state),
                confidence=state.confidence
            )
            self.transitions.append(transition)
            self.stats["transitions_count"] += 1
            
            # تحديث مصفوفة الانتقال
            self._update_transition_matrix(transition)
            
            self.logger.info(
                f"تحول النظام: {transition.from_regime.value} -> {transition.to_regime.value}"
            )
        
        self.regime_history.append(state)
    
    def _identify_trigger(self,
                         old_state: RegimeState,
                         new_state: RegimeState) -> str:
        """تحديد سبب التحول."""
        triggers = []
        
        if new_state.trend_strength > old_state.trend_strength * 1.5:
            triggers.append("زيادة قوة الاتجاه")
        elif new_state.trend_strength < old_state.trend_strength * 0.5:
            triggers.append("ضعف الاتجاه")
        
        if new_state.volatility_regime != old_state.volatility_regime:
            triggers.append(f"تغير التقلب إلى {new_state.volatility_regime.value}")
        
        if abs(new_state.momentum - old_state.momentum) > 0.3:
            triggers.append("تغير كبير في الزخم")
        
        return ", ".join(triggers) if triggers else "تحول طبيعي"
    
    def _update_transition_matrix(self, transition: RegimeTransition):
        """تحديث مصفوفة الانتقال."""
        from_regime = transition.from_regime.value
        to_regime = transition.to_regime.value
        
        # زيادة احتمال هذا الانتقال
        if from_regime in self.transition_matrix:
            current = self.transition_matrix[from_regime].get(to_regime, 0)
            self.transition_matrix[from_regime][to_regime] = min(0.9, current + 0.05)
            
            # تطبيع
            total = sum(self.transition_matrix[from_regime].values())
            for key in self.transition_matrix[from_regime]:
                self.transition_matrix[from_regime][key] /= total
    
    async def predict_regime_change(self,
                                   horizon: int = 10) -> Dict[str, Any]:
        """
        التنبؤ بتغير النظام.
        
        Args:
            horizon: أفق التنبؤ (عدد الشموع)
            
        Returns:
            التنبؤ
        """
        if not self.current_state:
            return {"prediction": None, "confidence": 0}
        
        current = self.current_state.regime.value
        probs = self.transition_matrix.get(current, {})
        
        # حساب الاحتمالات المتراكمة
        cumulative_probs = {}
        for regime, prob in probs.items():
            # احتمال التحول خلال الأفق
            cumulative_probs[regime] = 1 - (1 - prob) ** horizon
        
        # أكثر التحولات احتمالاً
        most_likely = max(cumulative_probs.items(), key=lambda x: x[1])
        
        # تعديل بناءً على مدة النظام الحالي
        duration_factor = min(2, self.current_state.regime_duration / 50)
        change_probability = 1 - (1 / (1 + duration_factor))
        
        prediction = {
            "current_regime": current,
            "most_likely_next": most_likely[0],
            "probability": most_likely[1] * change_probability,
            "all_probabilities": cumulative_probs,
            "regime_duration": self.current_state.regime_duration,
            "confidence": self.current_state.confidence,
            "horizon": horizon
        }
        
        self.stats["total_predictions"] += 1
        
        return prediction
    
    def get_strategy_recommendation(self) -> Dict[str, Any]:
        """
        الحصول على توصية الاستراتيجية.
        
        Returns:
            توصية الاستراتيجية
        """
        if not self.current_state:
            return {"strategy": "wait", "reason": "لا توجد بيانات كافية"}
        
        regime = self.current_state.regime
        volatility = self.current_state.volatility_regime
        
        recommendations = {
            MarketRegime.STRONG_BULL: {
                "strategy": "trend_following",
                "position": "long",
                "size": "large",
                "stop_loss": "wide",
                "reason": "اتجاه صعودي قوي"
            },
            MarketRegime.BULL: {
                "strategy": "trend_following",
                "position": "long",
                "size": "medium",
                "stop_loss": "normal",
                "reason": "اتجاه صعودي"
            },
            MarketRegime.WEAK_BULL: {
                "strategy": "swing",
                "position": "long",
                "size": "small",
                "stop_loss": "tight",
                "reason": "صعود ضعيف - حذر"
            },
            MarketRegime.SIDEWAYS: {
                "strategy": "mean_reversion",
                "position": "both",
                "size": "small",
                "stop_loss": "tight",
                "reason": "سوق جانبي"
            },
            MarketRegime.WEAK_BEAR: {
                "strategy": "swing",
                "position": "short",
                "size": "small",
                "stop_loss": "tight",
                "reason": "هبوط ضعيف - حذر"
            },
            MarketRegime.BEAR: {
                "strategy": "trend_following",
                "position": "short",
                "size": "medium",
                "stop_loss": "normal",
                "reason": "اتجاه هبوطي"
            },
            MarketRegime.STRONG_BEAR: {
                "strategy": "trend_following",
                "position": "short",
                "size": "large",
                "stop_loss": "wide",
                "reason": "اتجاه هبوطي قوي"
            },
            MarketRegime.HIGH_VOLATILITY: {
                "strategy": "wait",
                "position": "none",
                "size": "none",
                "stop_loss": "none",
                "reason": "تقلب عالي - انتظار"
            },
            MarketRegime.CRASH: {
                "strategy": "emergency_exit",
                "position": "close_all",
                "size": "none",
                "stop_loss": "none",
                "reason": "انهيار - خروج طوارئ"
            },
            MarketRegime.RECOVERY: {
                "strategy": "scalping",
                "position": "long",
                "size": "small",
                "stop_loss": "tight",
                "reason": "تعافي - فرص سريعة"
            }
        }
        
        rec = recommendations.get(regime, {"strategy": "wait", "reason": "غير محدد"})
        
        # تعديل حسب التقلب
        if volatility == VolatilityRegime.EXTREME:
            rec["size"] = "minimal"
            rec["note"] = "تقلب شديد - تقليل الحجم"
        elif volatility == VolatilityRegime.VERY_LOW:
            rec["note"] = "تقلب منخفض جداً - قد تكون الفرص محدودة"
        
        rec["regime"] = regime.value
        rec["volatility"] = volatility.value
        rec["confidence"] = self.current_state.confidence
        
        return rec
    
    def _default_state(self) -> RegimeState:
        """الحالة الافتراضية."""
        return RegimeState(
            timestamp=datetime.utcnow(),
            regime=MarketRegime.SIDEWAYS,
            volatility_regime=VolatilityRegime.NORMAL,
            confidence=0.3,
            trend_strength=0,
            momentum=0,
            volume_profile="normal",
            support_levels=[],
            resistance_levels=[],
            transition_probabilities={},
            regime_duration=0
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """الحصول على الإحصائيات."""
        return {
            **self.stats,
            "current_regime": self.current_state.regime.value if self.current_state else None,
            "history_size": len(self.regime_history),
            "transitions_count": len(self.transitions)
        }
