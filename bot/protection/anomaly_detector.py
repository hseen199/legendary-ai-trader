"""
═══════════════════════════════════════════════════════════════
LEGENDARY AGENT - Anomaly Detector
كاشف الشذوذ - اكتشاف الأنماط غير الطبيعية
═══════════════════════════════════════════════════════════════
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import deque
from loguru import logger


class AnomalyType(Enum):
    """أنواع الشذوذ"""
    PRICE_SPIKE = "قفزة سعرية"
    VOLUME_SPIKE = "قفزة حجم"
    SPREAD_ANOMALY = "شذوذ فارق"
    ORDERBOOK_IMBALANCE = "عدم توازن دفتر الأوامر"
    CORRELATION_BREAK = "كسر الارتباط"
    PATTERN_DEVIATION = "انحراف عن النمط"
    LIQUIDITY_DRAIN = "استنزاف السيولة"
    WHALE_ACTIVITY = "نشاط حوت"


class AnomalySeverity(Enum):
    """شدة الشذوذ"""
    LOW = "منخفض"
    MEDIUM = "متوسط"
    HIGH = "عالي"
    CRITICAL = "حرج"


@dataclass
class Anomaly:
    """شذوذ مكتشف"""
    type: AnomalyType
    severity: AnomalySeverity
    symbol: str
    timestamp: datetime
    score: float  # 0-1
    details: Dict[str, Any]
    recommendation: str
    expires_at: datetime = field(default_factory=lambda: datetime.now() + timedelta(minutes=15))


@dataclass
class DetectionResult:
    """نتيجة الكشف"""
    timestamp: datetime
    symbol: str
    anomalies: List[Anomaly]
    risk_score: float
    is_safe: bool
    recommendations: List[str]


class AnomalyDetector:
    """
    كاشف الشذوذ
    
    يكتشف الأنماط غير الطبيعية في السوق
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        تهيئة كاشف الشذوذ
        
        Args:
            config: إعدادات الكاشف
        """
        self.config = config or {}
        
        # عتبات الكشف
        self.thresholds = {
            'price_spike_percent': self.config.get('price_spike', 3.0),
            'volume_spike_ratio': self.config.get('volume_spike', 5.0),
            'spread_anomaly_percent': self.config.get('spread_anomaly', 1.0),
            'orderbook_imbalance': self.config.get('orderbook_imbalance', 0.7),
            'correlation_break': self.config.get('correlation_break', 0.5),
            'liquidity_drain_percent': self.config.get('liquidity_drain', 50.0),
            'whale_threshold_usd': self.config.get('whale_threshold', 1000000)
        }
        
        # تاريخ البيانات
        self.price_history: Dict[str, deque] = {}
        self.volume_history: Dict[str, deque] = {}
        self.spread_history: Dict[str, deque] = {}
        self.history_size = 100
        
        # الشذوذات النشطة
        self.active_anomalies: List[Anomaly] = []
        
        # الإحصائيات
        self.stats = {
            'total_detections': 0,
            'by_type': {t.value: 0 for t in AnomalyType},
            'by_severity': {s.value: 0 for s in AnomalySeverity}
        }
        
        logger.info("🔍 AnomalyDetector initialized")
    
    # ═══════════════════════════════════════════════════════════════
    # MAIN DETECTION
    # ═══════════════════════════════════════════════════════════════
    
    def detect(
        self,
        symbol: str,
        current_price: float,
        features: Dict[str, float],
        orderbook: Dict[str, Any] = None
    ) -> DetectionResult:
        """
        كشف الشذوذ
        
        Args:
            symbol: رمز العملة
            current_price: السعر الحالي
            features: الميزات
            orderbook: دفتر الأوامر
            
        Returns:
            نتيجة الكشف
        """
        self.stats['total_detections'] += 1
        anomalies = []
        
        # تحديث التاريخ
        self._update_history(symbol, current_price, features)
        
        # 1. كشف قفزة السعر
        price_anomaly = self._detect_price_spike(symbol, current_price)
        if price_anomaly:
            anomalies.append(price_anomaly)
        
        # 2. كشف قفزة الحجم
        volume_anomaly = self._detect_volume_spike(symbol, features)
        if volume_anomaly:
            anomalies.append(volume_anomaly)
        
        # 3. كشف شذوذ الفارق
        spread_anomaly = self._detect_spread_anomaly(symbol, features)
        if spread_anomaly:
            anomalies.append(spread_anomaly)
        
        # 4. كشف عدم توازن دفتر الأوامر
        if orderbook:
            orderbook_anomaly = self._detect_orderbook_imbalance(symbol, orderbook)
            if orderbook_anomaly:
                anomalies.append(orderbook_anomaly)
        
        # 5. كشف استنزاف السيولة
        liquidity_anomaly = self._detect_liquidity_drain(symbol, features, orderbook)
        if liquidity_anomaly:
            anomalies.append(liquidity_anomaly)
        
        # 6. كشف نشاط الحيتان
        whale_anomaly = self._detect_whale_activity(symbol, features)
        if whale_anomaly:
            anomalies.append(whale_anomaly)
        
        # تحديث الإحصائيات
        for anomaly in anomalies:
            self.stats['by_type'][anomaly.type.value] += 1
            self.stats['by_severity'][anomaly.severity.value] += 1
        
        # تحديث الشذوذات النشطة
        self._update_active_anomalies(anomalies)
        
        # حساب درجة المخاطر
        risk_score = self._calculate_risk_score(anomalies)
        
        # التوصيات
        recommendations = self._generate_recommendations(anomalies)
        
        return DetectionResult(
            timestamp=datetime.now(),
            symbol=symbol,
            anomalies=anomalies,
            risk_score=risk_score,
            is_safe=len(anomalies) == 0,
            recommendations=recommendations
        )
    
    # ═══════════════════════════════════════════════════════════════
    # DETECTION METHODS
    # ═══════════════════════════════════════════════════════════════
    
    def _detect_price_spike(
        self,
        symbol: str,
        current_price: float
    ) -> Optional[Anomaly]:
        """كشف قفزة السعر"""
        if symbol not in self.price_history:
            return None
        
        history = list(self.price_history[symbol])
        if len(history) < 5:
            return None
        
        # حساب التغير
        recent_avg = np.mean(history[-10:])
        if recent_avg == 0:
            return None
        
        change_percent = abs(current_price - recent_avg) / recent_avg * 100
        
        if change_percent >= self.thresholds['price_spike_percent']:
            severity = self._determine_severity(change_percent, [3, 5, 10])
            
            return Anomaly(
                type=AnomalyType.PRICE_SPIKE,
                severity=severity,
                symbol=symbol,
                timestamp=datetime.now(),
                score=min(1.0, change_percent / 10),
                details={
                    'change_percent': change_percent,
                    'current_price': current_price,
                    'recent_avg': recent_avg
                },
                recommendation="تجنب الدخول - انتظر استقرار السعر"
            )
        
        return None
    
    def _detect_volume_spike(
        self,
        symbol: str,
        features: Dict[str, float]
    ) -> Optional[Anomaly]:
        """كشف قفزة الحجم"""
        volume = features.get('volume', 0)
        volume_sma = features.get('volume_sma_20', volume)
        
        if volume_sma == 0:
            return None
        
        ratio = volume / volume_sma
        
        if ratio >= self.thresholds['volume_spike_ratio']:
            severity = self._determine_severity(ratio, [5, 10, 20])
            
            return Anomaly(
                type=AnomalyType.VOLUME_SPIKE,
                severity=severity,
                symbol=symbol,
                timestamp=datetime.now(),
                score=min(1.0, ratio / 20),
                details={
                    'volume': volume,
                    'volume_sma': volume_sma,
                    'ratio': ratio
                },
                recommendation="حجم غير طبيعي - قد يكون تلاعب أو خبر"
            )
        
        return None
    
    def _detect_spread_anomaly(
        self,
        symbol: str,
        features: Dict[str, float]
    ) -> Optional[Anomaly]:
        """كشف شذوذ الفارق"""
        spread = features.get('orderbook_spread', 0)
        
        if spread >= self.thresholds['spread_anomaly_percent']:
            severity = self._determine_severity(spread, [1, 2, 5])
            
            return Anomaly(
                type=AnomalyType.SPREAD_ANOMALY,
                severity=severity,
                symbol=symbol,
                timestamp=datetime.now(),
                score=min(1.0, spread / 5),
                details={'spread_percent': spread},
                recommendation="فارق عالي - تكلفة تداول مرتفعة"
            )
        
        return None
    
    def _detect_orderbook_imbalance(
        self,
        symbol: str,
        orderbook: Dict[str, Any]
    ) -> Optional[Anomaly]:
        """كشف عدم توازن دفتر الأوامر"""
        bid_volume = orderbook.get('bid_volume', 0)
        ask_volume = orderbook.get('ask_volume', 0)
        
        total = bid_volume + ask_volume
        if total == 0:
            return None
        
        imbalance = abs(bid_volume - ask_volume) / total
        
        if imbalance >= self.thresholds['orderbook_imbalance']:
            direction = "شراء" if bid_volume > ask_volume else "بيع"
            severity = self._determine_severity(imbalance, [0.7, 0.8, 0.9])
            
            return Anomaly(
                type=AnomalyType.ORDERBOOK_IMBALANCE,
                severity=severity,
                symbol=symbol,
                timestamp=datetime.now(),
                score=imbalance,
                details={
                    'bid_volume': bid_volume,
                    'ask_volume': ask_volume,
                    'imbalance': imbalance,
                    'direction': direction
                },
                recommendation=f"ضغط {direction} قوي - قد يتحرك السعر"
            )
        
        return None
    
    def _detect_liquidity_drain(
        self,
        symbol: str,
        features: Dict[str, float],
        orderbook: Dict[str, Any] = None
    ) -> Optional[Anomaly]:
        """كشف استنزاف السيولة"""
        if not orderbook:
            return None
        
        bid_depth = orderbook.get('bid_depth', 0)
        ask_depth = orderbook.get('ask_depth', 0)
        
        # مقارنة مع التاريخ
        if symbol in self.volume_history:
            history = list(self.volume_history[symbol])
            if len(history) >= 10:
                avg_volume = np.mean(history[-10:])
                current_depth = bid_depth + ask_depth
                
                if avg_volume > 0:
                    drain_percent = (1 - current_depth / avg_volume) * 100
                    
                    if drain_percent >= self.thresholds['liquidity_drain_percent']:
                        severity = self._determine_severity(drain_percent, [50, 70, 90])
                        
                        return Anomaly(
                            type=AnomalyType.LIQUIDITY_DRAIN,
                            severity=severity,
                            symbol=symbol,
                            timestamp=datetime.now(),
                            score=min(1.0, drain_percent / 100),
                            details={
                                'drain_percent': drain_percent,
                                'current_depth': current_depth,
                                'avg_volume': avg_volume
                            },
                            recommendation="سيولة منخفضة - خطر انزلاق عالي"
                        )
        
        return None
    
    def _detect_whale_activity(
        self,
        symbol: str,
        features: Dict[str, float]
    ) -> Optional[Anomaly]:
        """كشف نشاط الحيتان"""
        # يمكن استخدام بيانات خارجية للكشف
        large_trades = features.get('large_trades_volume', 0)
        
        if large_trades >= self.thresholds['whale_threshold_usd']:
            severity = self._determine_severity(
                large_trades,
                [1000000, 5000000, 10000000]
            )
            
            return Anomaly(
                type=AnomalyType.WHALE_ACTIVITY,
                severity=severity,
                symbol=symbol,
                timestamp=datetime.now(),
                score=min(1.0, large_trades / 10000000),
                details={
                    'large_trades_volume': large_trades
                },
                recommendation="نشاط حوت - راقب الاتجاه"
            )
        
        return None
    
    # ═══════════════════════════════════════════════════════════════
    # HELPERS
    # ═══════════════════════════════════════════════════════════════
    
    def _update_history(
        self,
        symbol: str,
        price: float,
        features: Dict[str, float]
    ) -> None:
        """تحديث التاريخ"""
        # تاريخ السعر
        if symbol not in self.price_history:
            self.price_history[symbol] = deque(maxlen=self.history_size)
        self.price_history[symbol].append(price)
        
        # تاريخ الحجم
        if symbol not in self.volume_history:
            self.volume_history[symbol] = deque(maxlen=self.history_size)
        self.volume_history[symbol].append(features.get('volume', 0))
        
        # تاريخ الفارق
        if symbol not in self.spread_history:
            self.spread_history[symbol] = deque(maxlen=self.history_size)
        self.spread_history[symbol].append(features.get('orderbook_spread', 0))
    
    def _determine_severity(
        self,
        value: float,
        thresholds: List[float]
    ) -> AnomalySeverity:
        """تحديد الشدة"""
        if value >= thresholds[2]:
            return AnomalySeverity.CRITICAL
        elif value >= thresholds[1]:
            return AnomalySeverity.HIGH
        elif value >= thresholds[0]:
            return AnomalySeverity.MEDIUM
        else:
            return AnomalySeverity.LOW
    
    def _update_active_anomalies(self, new_anomalies: List[Anomaly]) -> None:
        """تحديث الشذوذات النشطة"""
        # إزالة المنتهية
        now = datetime.now()
        self.active_anomalies = [
            a for a in self.active_anomalies
            if a.expires_at > now
        ]
        
        # إضافة الجديدة
        self.active_anomalies.extend(new_anomalies)
    
    def _calculate_risk_score(self, anomalies: List[Anomaly]) -> float:
        """حساب درجة المخاطر"""
        if not anomalies:
            return 0.0
        
        severity_weights = {
            AnomalySeverity.LOW: 0.2,
            AnomalySeverity.MEDIUM: 0.4,
            AnomalySeverity.HIGH: 0.7,
            AnomalySeverity.CRITICAL: 1.0
        }
        
        max_score = max(
            severity_weights.get(a.severity, 0) * a.score
            for a in anomalies
        )
        
        avg_score = np.mean([
            severity_weights.get(a.severity, 0) * a.score
            for a in anomalies
        ])
        
        return max_score * 0.7 + avg_score * 0.3
    
    def _generate_recommendations(self, anomalies: List[Anomaly]) -> List[str]:
        """توليد التوصيات"""
        recommendations = []
        
        for anomaly in anomalies:
            if anomaly.recommendation:
                recommendations.append(
                    f"[{anomaly.severity.value}] {anomaly.recommendation}"
                )
        
        if any(a.severity == AnomalySeverity.CRITICAL for a in anomalies):
            recommendations.insert(0, "🚨 شذوذ حرج - تجنب التداول!")
        
        return list(set(recommendations))
    
    # ═══════════════════════════════════════════════════════════════
    # STATUS
    # ═══════════════════════════════════════════════════════════════
    
    def get_active_anomalies(self, symbol: str = None) -> List[Anomaly]:
        """الحصول على الشذوذات النشطة"""
        anomalies = self.active_anomalies
        
        if symbol:
            anomalies = [a for a in anomalies if a.symbol == symbol]
        
        return anomalies
    
    def get_stats(self) -> Dict[str, Any]:
        """الحصول على الإحصائيات"""
        return {
            **self.stats,
            'active_anomalies': len(self.active_anomalies)
        }


# ═══════════════════════════════════════════════════════════════
# STANDALONE EXECUTION
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # اختبار كاشف الشذوذ
    detector = AnomalyDetector()
    
    # محاكاة تاريخ
    for i in range(20):
        detector._update_history(
            'BTCUSDT',
            50000 + i * 10,
            {'volume': 1000000, 'orderbook_spread': 0.05}
        )
    
    # كشف مع بيانات طبيعية
    result1 = detector.detect(
        'BTCUSDT',
        50200,
        {
            'volume': 1200000,
            'volume_sma_20': 1000000,
            'orderbook_spread': 0.05
        }
    )
    print(f"Normal: {len(result1.anomalies)} anomalies, Risk: {result1.risk_score:.2f}")
    
    # كشف مع قفزة سعر
    result2 = detector.detect(
        'BTCUSDT',
        55000,  # قفزة كبيرة
        {
            'volume': 5000000,  # حجم عالي
            'volume_sma_20': 1000000,
            'orderbook_spread': 2.0  # فارق عالي
        }
    )
    print(f"\nAnomalous: {len(result2.anomalies)} anomalies, Risk: {result2.risk_score:.2f}")
    
    for anomaly in result2.anomalies:
        print(f"  - {anomaly.type.value} [{anomaly.severity.value}]: {anomaly.recommendation}")
    
    print(f"\nRecommendations: {result2.recommendations}")
    print(f"\nStats: {detector.get_stats()}")
