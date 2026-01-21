"""
Legendary Trading System - AI Intuition System
نظام التداول الخارق - نظام الحدس الاصطناعي

نظام متقدم للتعلم من الأنماط غير الواضحة واتخاذ قرارات سريعة.
"""

import asyncio
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
from enum import Enum
import logging
import hashlib
import json


class IntuitionSignal(Enum):
    """إشارات الحدس"""
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    WEAK_BUY = "weak_buy"
    NEUTRAL = "neutral"
    WEAK_SELL = "weak_sell"
    SELL = "sell"
    STRONG_SELL = "strong_sell"
    DANGER = "danger"
    OPPORTUNITY = "opportunity"


class PatternType(Enum):
    """أنواع الأنماط"""
    PRICE_ACTION = "price_action"
    VOLUME_PATTERN = "volume_pattern"
    TIME_PATTERN = "time_pattern"
    CORRELATION = "correlation"
    SENTIMENT = "sentiment"
    ANOMALY = "anomaly"


@dataclass
class LearnedPattern:
    """نمط متعلم"""
    id: str
    type: PatternType
    description: str
    features: Dict[str, Any]
    
    # إحصائيات
    occurrences: int = 0
    successful_predictions: int = 0
    failed_predictions: int = 0
    
    # الأداء
    avg_return: float = 0.0
    win_rate: float = 0.0
    
    # التعلم
    confidence: float = 0.5
    last_seen: datetime = field(default_factory=datetime.utcnow)
    first_seen: datetime = field(default_factory=datetime.utcnow)


@dataclass
class IntuitionReading:
    """قراءة الحدس"""
    timestamp: datetime
    signal: IntuitionSignal
    strength: float  # 0-1
    confidence: float  # 0-1
    
    # المكونات
    components: Dict[str, float] = field(default_factory=dict)
    
    # الأنماط المكتشفة
    patterns_detected: List[str] = field(default_factory=list)
    
    # التفسير
    reasoning: str = ""
    
    # للتحقق لاحقاً
    outcome: Optional[Dict[str, Any]] = None


class AIIntuitionSystem:
    """
    نظام الحدس الاصطناعي.
    
    يوفر:
    - تعلم الأنماط غير الواضحة
    - "شعور" بالسوق مبني على آلاف الصفقات
    - قرارات سريعة في الحالات الطارئة
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.logger = logging.getLogger("AIIntuitionSystem")
        self.config = config or {}
        
        # الأنماط المتعلمة
        self.learned_patterns: Dict[str, LearnedPattern] = {}
        
        # تاريخ القراءات
        self.reading_history: deque = deque(maxlen=5000)
        
        # ذاكرة قصيرة المدى
        self.short_term_memory: deque = deque(maxlen=100)
        
        # الحالة العاطفية للسوق
        self.market_mood = {
            "fear": 0.5,
            "greed": 0.5,
            "uncertainty": 0.5,
            "momentum": 0.0
        }
        
        # عتبات الحدس
        self.thresholds = {
            "strong_signal": 0.8,
            "weak_signal": 0.3,
            "danger_threshold": 0.85,
            "opportunity_threshold": 0.75,
            "min_confidence": 0.4
        }
        
        # إحصائيات
        self.stats = {
            "total_readings": 0,
            "patterns_learned": 0,
            "accurate_intuitions": 0,
            "total_verified": 0
        }
        
        # تهيئة الأنماط الأساسية
        self._init_base_patterns()
    
    def _init_base_patterns(self):
        """تهيئة الأنماط الأساسية."""
        base_patterns = [
            {
                "id": "volume_spike_before_move",
                "type": PatternType.VOLUME_PATTERN,
                "description": "ارتفاع الحجم قبل حركة كبيرة",
                "features": {"volume_ratio": 2.0, "price_change": 0.02}
            },
            {
                "id": "divergence_reversal",
                "type": PatternType.PRICE_ACTION,
                "description": "تباعد يسبق انعكاس",
                "features": {"divergence": True, "trend_weakening": True}
            },
            {
                "id": "time_of_day_pattern",
                "type": PatternType.TIME_PATTERN,
                "description": "نمط وقت معين من اليوم",
                "features": {"hour_range": [14, 16], "volatility": "high"}
            },
            {
                "id": "whale_accumulation",
                "type": PatternType.VOLUME_PATTERN,
                "description": "تجميع من الحيتان",
                "features": {"large_orders": True, "price_stable": True}
            },
            {
                "id": "fear_capitulation",
                "type": PatternType.SENTIMENT,
                "description": "استسلام من الخوف",
                "features": {"fear_index": 0.9, "selling_pressure": "extreme"}
            },
            {
                "id": "greed_exhaustion",
                "type": PatternType.SENTIMENT,
                "description": "إرهاق من الطمع",
                "features": {"greed_index": 0.9, "buying_pressure": "extreme"}
            }
        ]
        
        for pattern_data in base_patterns:
            pattern = LearnedPattern(
                id=pattern_data["id"],
                type=pattern_data["type"],
                description=pattern_data["description"],
                features=pattern_data["features"],
                confidence=0.5
            )
            self.learned_patterns[pattern.id] = pattern
        
        self.stats["patterns_learned"] = len(self.learned_patterns)
    
    async def get_intuition(self, 
                           market_data: Dict[str, Any],
                           context: Dict[str, Any] = None) -> IntuitionReading:
        """
        الحصول على قراءة الحدس.
        
        Args:
            market_data: بيانات السوق
            context: سياق إضافي
            
        Returns:
            قراءة الحدس
        """
        # تحديث الذاكرة قصيرة المدى
        self.short_term_memory.append({
            "timestamp": datetime.utcnow(),
            "data": market_data
        })
        
        # تحليل المكونات
        components = await self._analyze_components(market_data, context)
        
        # اكتشاف الأنماط
        detected_patterns = await self._detect_patterns(market_data)
        
        # حساب الإشارة
        signal, strength = self._calculate_signal(components, detected_patterns)
        
        # حساب الثقة
        confidence = self._calculate_confidence(components, detected_patterns)
        
        # توليد التفسير
        reasoning = self._generate_reasoning(signal, components, detected_patterns)
        
        # إنشاء القراءة
        reading = IntuitionReading(
            timestamp=datetime.utcnow(),
            signal=signal,
            strength=strength,
            confidence=confidence,
            components=components,
            patterns_detected=[p.id for p in detected_patterns],
            reasoning=reasoning
        )
        
        # حفظ في التاريخ
        self.reading_history.append(reading)
        self.stats["total_readings"] += 1
        
        self.logger.debug(f"حدس: {signal.value} (قوة: {strength:.1%}, ثقة: {confidence:.1%})")
        
        return reading
    
    async def _analyze_components(self,
                                 market_data: Dict[str, Any],
                                 context: Dict[str, Any] = None) -> Dict[str, float]:
        """تحليل مكونات الحدس."""
        components = {}
        
        # مكون الاتجاه
        prices = market_data.get("prices", [])
        if len(prices) >= 20:
            short_ma = sum(prices[-5:]) / 5
            long_ma = sum(prices[-20:]) / 20
            trend_component = (short_ma - long_ma) / long_ma
            components["trend"] = max(-1, min(1, trend_component * 10))
        else:
            components["trend"] = 0
        
        # مكون الزخم
        if len(prices) >= 14:
            momentum = (prices[-1] - prices[-14]) / prices[-14]
            components["momentum"] = max(-1, min(1, momentum * 5))
        else:
            components["momentum"] = 0
        
        # مكون الحجم
        volumes = market_data.get("volumes", [])
        if len(volumes) >= 20:
            recent_vol = sum(volumes[-5:]) / 5
            avg_vol = sum(volumes[-20:]) / 20
            vol_ratio = recent_vol / avg_vol if avg_vol > 0 else 1
            components["volume"] = max(0, min(1, (vol_ratio - 1) / 2))
        else:
            components["volume"] = 0.5
        
        # مكون التقلب
        if len(prices) >= 20:
            returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
            volatility = np.std(returns[-20:]) if len(returns) >= 20 else 0.02
            components["volatility"] = max(0, min(1, volatility / 0.05))
        else:
            components["volatility"] = 0.5
        
        # مكون المشاعر
        if context:
            fear_greed = context.get("fear_greed_index", 50) / 100
            components["sentiment"] = fear_greed
        else:
            components["sentiment"] = 0.5
        
        # مكون الوقت
        hour = datetime.utcnow().hour
        # ساعات التداول النشطة
        if 8 <= hour <= 16 or 20 <= hour <= 23:
            components["time_quality"] = 0.8
        else:
            components["time_quality"] = 0.4
        
        # تحديث مزاج السوق
        self._update_market_mood(components)
        
        return components
    
    def _update_market_mood(self, components: Dict[str, float]):
        """تحديث مزاج السوق."""
        # تحديث تدريجي
        alpha = 0.1
        
        # الخوف يزيد مع التقلب العالي والزخم السلبي
        fear_signal = components.get("volatility", 0.5) * 0.5 + \
                     max(0, -components.get("momentum", 0)) * 0.5
        self.market_mood["fear"] = (1 - alpha) * self.market_mood["fear"] + alpha * fear_signal
        
        # الطمع يزيد مع الزخم الإيجابي والحجم العالي
        greed_signal = max(0, components.get("momentum", 0)) * 0.5 + \
                      components.get("volume", 0.5) * 0.5
        self.market_mood["greed"] = (1 - alpha) * self.market_mood["greed"] + alpha * greed_signal
        
        # عدم اليقين
        uncertainty = abs(components.get("trend", 0)) < 0.2
        self.market_mood["uncertainty"] = (1 - alpha) * self.market_mood["uncertainty"] + \
                                         alpha * (0.7 if uncertainty else 0.3)
        
        # الزخم العام
        self.market_mood["momentum"] = components.get("momentum", 0)
    
    async def _detect_patterns(self, 
                              market_data: Dict[str, Any]) -> List[LearnedPattern]:
        """اكتشاف الأنماط."""
        detected = []
        
        for pattern_id, pattern in self.learned_patterns.items():
            if await self._pattern_matches(pattern, market_data):
                detected.append(pattern)
                pattern.occurrences += 1
                pattern.last_seen = datetime.utcnow()
        
        return detected
    
    async def _pattern_matches(self,
                              pattern: LearnedPattern,
                              market_data: Dict[str, Any]) -> bool:
        """فحص تطابق نمط."""
        features = pattern.features
        
        if pattern.type == PatternType.VOLUME_PATTERN:
            volumes = market_data.get("volumes", [])
            if len(volumes) >= 20:
                recent = sum(volumes[-5:]) / 5
                avg = sum(volumes[-20:]) / 20
                ratio = recent / avg if avg > 0 else 1
                
                if "volume_ratio" in features:
                    if ratio >= features["volume_ratio"]:
                        return True
        
        elif pattern.type == PatternType.PRICE_ACTION:
            prices = market_data.get("prices", [])
            if len(prices) >= 20:
                # فحص التباعد
                if features.get("divergence"):
                    # منطق بسيط للتباعد
                    price_trend = prices[-1] > prices[-10]
                    momentum = market_data.get("rsi", 50)
                    momentum_trend = momentum > 50
                    
                    if price_trend != momentum_trend:
                        return True
        
        elif pattern.type == PatternType.TIME_PATTERN:
            hour = datetime.utcnow().hour
            hour_range = features.get("hour_range", [0, 24])
            if hour_range[0] <= hour <= hour_range[1]:
                return True
        
        elif pattern.type == PatternType.SENTIMENT:
            fear_index = market_data.get("fear_index", 0.5)
            greed_index = market_data.get("greed_index", 0.5)
            
            if features.get("fear_index") and fear_index >= features["fear_index"]:
                return True
            if features.get("greed_index") and greed_index >= features["greed_index"]:
                return True
        
        return False
    
    def _calculate_signal(self,
                         components: Dict[str, float],
                         patterns: List[LearnedPattern]) -> Tuple[IntuitionSignal, float]:
        """حساب الإشارة."""
        # حساب النتيجة الأساسية
        score = 0
        
        # من المكونات
        score += components.get("trend", 0) * 0.3
        score += components.get("momentum", 0) * 0.25
        score += (components.get("sentiment", 0.5) - 0.5) * 0.2
        score += (components.get("volume", 0.5) - 0.5) * 0.15
        
        # من الأنماط
        for pattern in patterns:
            if pattern.win_rate > 0.6:
                score += 0.1 * pattern.confidence
            elif pattern.win_rate < 0.4:
                score -= 0.1 * pattern.confidence
        
        # فحص الخطر
        if self.market_mood["fear"] > self.thresholds["danger_threshold"]:
            return IntuitionSignal.DANGER, self.market_mood["fear"]
        
        # فحص الفرصة
        if self.market_mood["greed"] < 0.3 and score > 0.5:
            return IntuitionSignal.OPPORTUNITY, score
        
        # تحديد الإشارة
        strength = abs(score)
        
        if score > self.thresholds["strong_signal"]:
            return IntuitionSignal.STRONG_BUY, strength
        elif score > self.thresholds["weak_signal"]:
            return IntuitionSignal.BUY, strength
        elif score > 0:
            return IntuitionSignal.WEAK_BUY, strength
        elif score < -self.thresholds["strong_signal"]:
            return IntuitionSignal.STRONG_SELL, strength
        elif score < -self.thresholds["weak_signal"]:
            return IntuitionSignal.SELL, strength
        elif score < 0:
            return IntuitionSignal.WEAK_SELL, strength
        else:
            return IntuitionSignal.NEUTRAL, 0
    
    def _calculate_confidence(self,
                             components: Dict[str, float],
                             patterns: List[LearnedPattern]) -> float:
        """حساب الثقة."""
        confidence = 0.5
        
        # من اتساق المكونات
        values = list(components.values())
        if values:
            variance = np.var(values)
            confidence += (1 - variance) * 0.2
        
        # من الأنماط المكتشفة
        if patterns:
            avg_pattern_confidence = sum(p.confidence for p in patterns) / len(patterns)
            confidence += avg_pattern_confidence * 0.2
        
        # من جودة الوقت
        confidence *= components.get("time_quality", 0.5) + 0.5
        
        # من عدم اليقين
        confidence *= (1 - self.market_mood["uncertainty"] * 0.3)
        
        return max(0.1, min(0.95, confidence))
    
    def _generate_reasoning(self,
                           signal: IntuitionSignal,
                           components: Dict[str, float],
                           patterns: List[LearnedPattern]) -> str:
        """توليد التفسير."""
        reasons = []
        
        # من المكونات
        if components.get("trend", 0) > 0.3:
            reasons.append("اتجاه صعودي واضح")
        elif components.get("trend", 0) < -0.3:
            reasons.append("اتجاه هبوطي واضح")
        
        if components.get("momentum", 0) > 0.3:
            reasons.append("زخم إيجابي قوي")
        elif components.get("momentum", 0) < -0.3:
            reasons.append("زخم سلبي قوي")
        
        if components.get("volume", 0.5) > 0.7:
            reasons.append("حجم تداول مرتفع")
        
        # من الأنماط
        for pattern in patterns[:3]:
            reasons.append(f"نمط: {pattern.description}")
        
        # من مزاج السوق
        if self.market_mood["fear"] > 0.7:
            reasons.append("خوف مرتفع في السوق")
        elif self.market_mood["greed"] > 0.7:
            reasons.append("طمع مرتفع في السوق")
        
        return ". ".join(reasons) if reasons else "لا توجد أسباب واضحة"
    
    async def quick_decision(self,
                            market_data: Dict[str, Any],
                            urgency: str = "normal") -> Dict[str, Any]:
        """
        قرار سريع في حالة طوارئ.
        
        Args:
            market_data: بيانات السوق
            urgency: مستوى الاستعجال
            
        Returns:
            القرار السريع
        """
        # قراءة سريعة
        reading = await self.get_intuition(market_data)
        
        decision = {
            "action": "hold",
            "confidence": reading.confidence,
            "reasoning": reading.reasoning,
            "urgency": urgency
        }
        
        # في حالة الخطر
        if reading.signal == IntuitionSignal.DANGER:
            decision["action"] = "exit_all"
            decision["priority"] = "immediate"
            decision["reasoning"] = "خطر مكتشف - خروج فوري"
        
        # في حالة الفرصة
        elif reading.signal == IntuitionSignal.OPPORTUNITY:
            decision["action"] = "enter"
            decision["priority"] = "high"
        
        # إشارات قوية
        elif reading.signal == IntuitionSignal.STRONG_BUY and reading.confidence > 0.7:
            decision["action"] = "buy"
            decision["size"] = "large"
        
        elif reading.signal == IntuitionSignal.STRONG_SELL and reading.confidence > 0.7:
            decision["action"] = "sell"
            decision["size"] = "large"
        
        # إشارات عادية
        elif reading.signal in [IntuitionSignal.BUY, IntuitionSignal.WEAK_BUY]:
            decision["action"] = "buy"
            decision["size"] = "small" if reading.signal == IntuitionSignal.WEAK_BUY else "medium"
        
        elif reading.signal in [IntuitionSignal.SELL, IntuitionSignal.WEAK_SELL]:
            decision["action"] = "sell"
            decision["size"] = "small" if reading.signal == IntuitionSignal.WEAK_SELL else "medium"
        
        return decision
    
    async def learn_from_outcome(self,
                                reading_timestamp: datetime,
                                outcome: Dict[str, Any]):
        """
        التعلم من النتيجة.
        
        Args:
            reading_timestamp: وقت القراءة
            outcome: النتيجة الفعلية
        """
        # البحث عن القراءة
        reading = None
        for r in self.reading_history:
            if abs((r.timestamp - reading_timestamp).total_seconds()) < 60:
                reading = r
                break
        
        if not reading:
            return
        
        reading.outcome = outcome
        success = outcome.get("success", False)
        
        # تحديث الأنماط
        for pattern_id in reading.patterns_detected:
            if pattern_id in self.learned_patterns:
                pattern = self.learned_patterns[pattern_id]
                
                if success:
                    pattern.successful_predictions += 1
                else:
                    pattern.failed_predictions += 1
                
                # تحديث معدل النجاح
                total = pattern.successful_predictions + pattern.failed_predictions
                if total > 0:
                    pattern.win_rate = pattern.successful_predictions / total
                
                # تحديث الثقة
                pattern.confidence = 0.3 + pattern.win_rate * 0.6
                
                # تحديث متوسط العائد
                if "return" in outcome:
                    alpha = 1 / (pattern.occurrences + 1)
                    pattern.avg_return = (1 - alpha) * pattern.avg_return + alpha * outcome["return"]
        
        # تحديث الإحصائيات
        self.stats["total_verified"] += 1
        if success:
            self.stats["accurate_intuitions"] += 1
    
    async def discover_new_pattern(self,
                                  market_data: Dict[str, Any],
                                  outcome: Dict[str, Any]) -> Optional[LearnedPattern]:
        """
        اكتشاف نمط جديد.
        
        Args:
            market_data: بيانات السوق
            outcome: النتيجة
            
        Returns:
            النمط الجديد إن وجد
        """
        # استخراج الميزات
        features = self._extract_features(market_data)
        
        # فحص إذا كان نمطاً جديداً
        feature_hash = hashlib.md5(json.dumps(features, sort_keys=True).encode()).hexdigest()[:8]
        pattern_id = f"discovered_{feature_hash}"
        
        if pattern_id in self.learned_patterns:
            return None
        
        # إنشاء نمط جديد
        pattern = LearnedPattern(
            id=pattern_id,
            type=PatternType.ANOMALY,
            description=f"نمط مكتشف تلقائياً",
            features=features,
            occurrences=1,
            successful_predictions=1 if outcome.get("success") else 0,
            failed_predictions=0 if outcome.get("success") else 1,
            confidence=0.4
        )
        
        self.learned_patterns[pattern_id] = pattern
        self.stats["patterns_learned"] += 1
        
        self.logger.info(f"تم اكتشاف نمط جديد: {pattern_id}")
        
        return pattern
    
    def _extract_features(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """استخراج الميزات."""
        features = {}
        
        prices = market_data.get("prices", [])
        volumes = market_data.get("volumes", [])
        
        if len(prices) >= 20:
            # ميزات السعر
            features["price_change_5"] = (prices[-1] - prices[-5]) / prices[-5]
            features["price_change_20"] = (prices[-1] - prices[-20]) / prices[-20]
            
            # ميزات التقلب
            returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
            features["volatility"] = float(np.std(returns[-20:]))
        
        if len(volumes) >= 20:
            # ميزات الحجم
            features["volume_ratio"] = sum(volumes[-5:]) / sum(volumes[-20:]) * 4
        
        # ميزات الوقت
        features["hour"] = datetime.utcnow().hour
        features["day_of_week"] = datetime.utcnow().weekday()
        
        return features
    
    def get_market_mood(self) -> Dict[str, float]:
        """الحصول على مزاج السوق."""
        return self.market_mood.copy()
    
    def get_stats(self) -> Dict[str, Any]:
        """الحصول على الإحصائيات."""
        accuracy = 0
        if self.stats["total_verified"] > 0:
            accuracy = self.stats["accurate_intuitions"] / self.stats["total_verified"]
        
        return {
            **self.stats,
            "accuracy": accuracy,
            "market_mood": self.market_mood,
            "active_patterns": len(self.learned_patterns)
        }
