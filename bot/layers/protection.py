"""
═══════════════════════════════════════════════════════════════
LEGENDARY AGENT - Protection Layer
طبقة الحماية - حماية المحفظة من المخاطر
═══════════════════════════════════════════════════════════════
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import deque
from loguru import logger


class AlertLevel(Enum):
    """مستويات التنبيه"""
    INFO = "معلومات"
    WARNING = "تحذير"
    DANGER = "خطر"
    CRITICAL = "حرج"


class ProtectionType(Enum):
    """أنواع الحماية"""
    FLASH_CRASH = "انهيار سريع"
    MANIPULATION = "تلاعب"
    VOLATILITY = "تقلب"
    CORRELATION = "ارتباط"
    LIQUIDITY = "سيولة"
    NEWS = "أخبار"
    DAILY_LOSS = "خسارة يومية"
    WEEKLY_LOSS = "خسارة أسبوعية"
    POSITION_LIMIT = "حد المراكز"


@dataclass
class Alert:
    """تنبيه"""
    type: ProtectionType
    level: AlertLevel
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    data: Dict[str, Any] = field(default_factory=dict)
    action_required: str = ""
    expires_at: Optional[datetime] = None


@dataclass
class ProtectionStatus:
    """حالة الحماية"""
    timestamp: datetime
    is_safe: bool
    alerts: List[Alert]
    blocked_actions: List[str]
    risk_level: float  # 0-1
    circuit_breaker_active: bool = False
    emergency_mode: bool = False
    recommendations: List[str] = field(default_factory=list)


class ProtectionLayer:
    """
    طبقة الحماية
    
    مسؤولة عن:
    - كشف الانهيارات السريعة
    - كشف التلاعب
    - مراقبة التقلب
    - حماية من الخسائر
    - قاطع الدائرة
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        تهيئة طبقة الحماية
        
        Args:
            config: إعدادات الطبقة
        """
        self.config = config or {}
        
        # حدود الحماية
        self.limits = {
            'max_daily_loss': self.config.get('max_daily_loss', 5.0),
            'max_weekly_loss': self.config.get('max_weekly_loss', 10.0),
            'flash_crash_threshold': self.config.get('flash_crash', 5.0),
            'flash_crash_window': self.config.get('flash_crash_window', 5),  # دقائق
            'max_volatility': self.config.get('max_volatility', 10.0),
            'manipulation_threshold': self.config.get('manipulation', 3.0),
            'max_portfolio_heat': self.config.get('max_heat', 80.0),
            'correlation_threshold': self.config.get('correlation', 0.8),
            'min_liquidity_ratio': self.config.get('min_liquidity', 0.1)
        }
        
        # حالة الحماية
        self.state = {
            'daily_pnl': 0.0,
            'weekly_pnl': 0.0,
            'portfolio_heat': 0.0,
            'circuit_breaker_until': None,
            'blocked_symbols': [],
            'consecutive_losses': {}
        }
        
        # تاريخ الأسعار للكشف عن الانهيار
        self.price_history: Dict[str, deque] = {}
        self.history_size = 100
        
        # التنبيهات النشطة
        self.active_alerts: List[Alert] = []
        
        # إحصائيات
        self.stats = {
            'total_checks': 0,
            'alerts_triggered': 0,
            'circuit_breaker_activations': 0,
            'blocked_trades': 0
        }
        
        logger.info("🛡️ ProtectionLayer initialized")
    
    # ═══════════════════════════════════════════════════════════════
    # MAIN PROTECTION CHECK
    # ═══════════════════════════════════════════════════════════════
    
    def check(
        self,
        symbol: str,
        current_price: float,
        features: Dict[str, float],
        portfolio_state: Dict[str, Any]
    ) -> ProtectionStatus:
        """
        فحص الحماية
        
        Args:
            symbol: رمز العملة
            current_price: السعر الحالي
            features: الميزات
            portfolio_state: حالة المحفظة
            
        Returns:
            حالة الحماية
        """
        self.stats['total_checks'] += 1
        alerts = []
        blocked_actions = []
        
        # تحديث تاريخ الأسعار
        self._update_price_history(symbol, current_price)
        
        # تحديث حالة المحفظة
        self._update_portfolio_state(portfolio_state)
        
        # 1. فحص الانهيار السريع
        flash_crash_alert = self._check_flash_crash(symbol, current_price)
        if flash_crash_alert:
            alerts.append(flash_crash_alert)
            blocked_actions.extend(['BUY', 'SELL'])
        
        # 2. فحص التلاعب
        manipulation_alert = self._check_manipulation(symbol, features)
        if manipulation_alert:
            alerts.append(manipulation_alert)
            blocked_actions.append('BUY')
        
        # 3. فحص التقلب
        volatility_alert = self._check_volatility(features)
        if volatility_alert:
            alerts.append(volatility_alert)
        
        # 4. فحص حد الخسارة اليومي
        daily_loss_alert = self._check_daily_loss()
        if daily_loss_alert:
            alerts.append(daily_loss_alert)
            blocked_actions.extend(['BUY'])
        
        # 5. فحص حد الخسارة الأسبوعي
        weekly_loss_alert = self._check_weekly_loss()
        if weekly_loss_alert:
            alerts.append(weekly_loss_alert)
            blocked_actions.extend(['BUY'])
        
        # 6. فحص حرارة المحفظة
        heat_alert = self._check_portfolio_heat()
        if heat_alert:
            alerts.append(heat_alert)
            blocked_actions.append('BUY')
        
        # 7. فحص السيولة
        liquidity_alert = self._check_liquidity(features)
        if liquidity_alert:
            alerts.append(liquidity_alert)
        
        # 8. فحص الخسائر المتتالية
        consecutive_alert = self._check_consecutive_losses(symbol)
        if consecutive_alert:
            alerts.append(consecutive_alert)
            blocked_actions.append('BUY')
        
        # تحديث التنبيهات النشطة
        self.active_alerts = alerts
        
        # حساب مستوى المخاطر
        risk_level = self._calculate_risk_level(alerts)
        
        # فحص قاطع الدائرة
        circuit_breaker = self._check_circuit_breaker(alerts)
        
        # وضع الطوارئ
        emergency = any(a.level == AlertLevel.CRITICAL for a in alerts)
        
        # التوصيات
        recommendations = self._generate_recommendations(alerts, risk_level)
        
        # تحديث الإحصائيات
        self.stats['alerts_triggered'] += len(alerts)
        if blocked_actions:
            self.stats['blocked_trades'] += 1
        
        return ProtectionStatus(
            timestamp=datetime.now(),
            is_safe=len(alerts) == 0,
            alerts=alerts,
            blocked_actions=list(set(blocked_actions)),
            risk_level=risk_level,
            circuit_breaker_active=circuit_breaker,
            emergency_mode=emergency,
            recommendations=recommendations
        )
    
    # ═══════════════════════════════════════════════════════════════
    # FLASH CRASH DETECTION
    # ═══════════════════════════════════════════════════════════════
    
    def _check_flash_crash(
        self,
        symbol: str,
        current_price: float
    ) -> Optional[Alert]:
        """فحص الانهيار السريع"""
        if symbol not in self.price_history:
            return None
        
        history = list(self.price_history[symbol])
        if len(history) < 5:
            return None
        
        # حساب التغير في آخر 5 دقائق
        window = min(self.limits['flash_crash_window'], len(history))
        recent_prices = history[-window:]
        
        if not recent_prices:
            return None
        
        max_price = max(recent_prices)
        min_price = min(recent_prices)
        
        if max_price == 0:
            return None
        
        change_percent = (max_price - min_price) / max_price * 100
        
        if change_percent >= self.limits['flash_crash_threshold']:
            return Alert(
                type=ProtectionType.FLASH_CRASH,
                level=AlertLevel.CRITICAL,
                message=f"⚠️ انهيار سريع! تغير {change_percent:.1f}% في {window} دقائق",
                data={
                    'change_percent': change_percent,
                    'max_price': max_price,
                    'min_price': min_price,
                    'current_price': current_price
                },
                action_required="إيقاف جميع الصفقات",
                expires_at=datetime.now() + timedelta(minutes=15)
            )
        
        return None
    
    # ═══════════════════════════════════════════════════════════════
    # MANIPULATION DETECTION
    # ═══════════════════════════════════════════════════════════════
    
    def _check_manipulation(
        self,
        symbol: str,
        features: Dict[str, float]
    ) -> Optional[Alert]:
        """فحص التلاعب"""
        # مؤشرات التلاعب:
        # 1. حجم غير طبيعي
        # 2. فارق كبير في دفتر الأوامر
        # 3. تحركات سعرية غير منطقية
        
        volume = features.get('volume', 0)
        volume_sma = features.get('volume_sma_20', volume)
        orderbook_imbalance = features.get('orderbook_imbalance', 0)
        spread = features.get('orderbook_spread', 0)
        
        manipulation_score = 0
        
        # حجم غير طبيعي
        if volume_sma > 0:
            volume_ratio = volume / volume_sma
            if volume_ratio > 5:
                manipulation_score += 0.4
            elif volume_ratio > 3:
                manipulation_score += 0.2
        
        # عدم توازن شديد في دفتر الأوامر
        if abs(orderbook_imbalance) > 0.5:
            manipulation_score += 0.3
        
        # فارق كبير
        if spread > 0.01:  # 1%
            manipulation_score += 0.3
        
        if manipulation_score >= self.limits['manipulation_threshold'] / 10:
            return Alert(
                type=ProtectionType.MANIPULATION,
                level=AlertLevel.DANGER,
                message=f"🚨 احتمال تلاعب في {symbol}",
                data={
                    'manipulation_score': manipulation_score,
                    'volume_ratio': volume / volume_sma if volume_sma > 0 else 0,
                    'orderbook_imbalance': orderbook_imbalance
                },
                action_required="تجنب الشراء",
                expires_at=datetime.now() + timedelta(minutes=30)
            )
        
        return None
    
    # ═══════════════════════════════════════════════════════════════
    # VOLATILITY CHECK
    # ═══════════════════════════════════════════════════════════════
    
    def _check_volatility(self, features: Dict[str, float]) -> Optional[Alert]:
        """فحص التقلب"""
        atr_percent = features.get('atr_percent', 0)
        bb_width = features.get('bb_width', 0)
        
        avg_volatility = (atr_percent + bb_width) / 2
        
        if avg_volatility >= self.limits['max_volatility']:
            return Alert(
                type=ProtectionType.VOLATILITY,
                level=AlertLevel.WARNING,
                message=f"⚡ تقلب عالي جداً: {avg_volatility:.1f}%",
                data={
                    'atr_percent': atr_percent,
                    'bb_width': bb_width,
                    'avg_volatility': avg_volatility
                },
                action_required="تقليل حجم المراكز"
            )
        
        return None
    
    # ═══════════════════════════════════════════════════════════════
    # LOSS LIMITS
    # ═══════════════════════════════════════════════════════════════
    
    def _check_daily_loss(self) -> Optional[Alert]:
        """فحص حد الخسارة اليومي"""
        daily_pnl = self.state['daily_pnl']
        limit = self.limits['max_daily_loss']
        
        if daily_pnl <= -limit:
            return Alert(
                type=ProtectionType.DAILY_LOSS,
                level=AlertLevel.CRITICAL,
                message=f"🛑 تم الوصول لحد الخسارة اليومي: {daily_pnl:.1f}%",
                data={'daily_pnl': daily_pnl, 'limit': limit},
                action_required="إيقاف التداول لليوم",
                expires_at=datetime.now().replace(hour=0, minute=0, second=0) + timedelta(days=1)
            )
        elif daily_pnl <= -limit * 0.8:
            return Alert(
                type=ProtectionType.DAILY_LOSS,
                level=AlertLevel.DANGER,
                message=f"⚠️ اقتراب من حد الخسارة اليومي: {daily_pnl:.1f}%",
                data={'daily_pnl': daily_pnl, 'limit': limit},
                action_required="تقليل المخاطر"
            )
        
        return None
    
    def _check_weekly_loss(self) -> Optional[Alert]:
        """فحص حد الخسارة الأسبوعي"""
        weekly_pnl = self.state['weekly_pnl']
        limit = self.limits['max_weekly_loss']
        
        if weekly_pnl <= -limit:
            return Alert(
                type=ProtectionType.WEEKLY_LOSS,
                level=AlertLevel.CRITICAL,
                message=f"🛑 تم الوصول لحد الخسارة الأسبوعي: {weekly_pnl:.1f}%",
                data={'weekly_pnl': weekly_pnl, 'limit': limit},
                action_required="إيقاف التداول للأسبوع"
            )
        
        return None
    
    # ═══════════════════════════════════════════════════════════════
    # PORTFOLIO HEAT
    # ═══════════════════════════════════════════════════════════════
    
    def _check_portfolio_heat(self) -> Optional[Alert]:
        """فحص حرارة المحفظة"""
        heat = self.state['portfolio_heat']
        limit = self.limits['max_portfolio_heat']
        
        if heat >= limit:
            return Alert(
                type=ProtectionType.POSITION_LIMIT,
                level=AlertLevel.WARNING,
                message=f"🔥 حرارة المحفظة عالية: {heat:.1f}%",
                data={'heat': heat, 'limit': limit},
                action_required="لا تفتح مراكز جديدة"
            )
        
        return None
    
    # ═══════════════════════════════════════════════════════════════
    # LIQUIDITY CHECK
    # ═══════════════════════════════════════════════════════════════
    
    def _check_liquidity(self, features: Dict[str, float]) -> Optional[Alert]:
        """فحص السيولة"""
        bid_depth = features.get('bid_depth', 0)
        ask_depth = features.get('ask_depth', 0)
        volume = features.get('volume', 0)
        
        if volume == 0:
            return None
        
        # نسبة العمق للحجم
        total_depth = bid_depth + ask_depth
        liquidity_ratio = total_depth / volume if volume > 0 else 0
        
        if liquidity_ratio < self.limits['min_liquidity_ratio']:
            return Alert(
                type=ProtectionType.LIQUIDITY,
                level=AlertLevel.WARNING,
                message="💧 سيولة منخفضة",
                data={
                    'liquidity_ratio': liquidity_ratio,
                    'bid_depth': bid_depth,
                    'ask_depth': ask_depth
                },
                action_required="تقليل حجم الصفقة"
            )
        
        return None
    
    # ═══════════════════════════════════════════════════════════════
    # CONSECUTIVE LOSSES
    # ═══════════════════════════════════════════════════════════════
    
    def _check_consecutive_losses(self, symbol: str) -> Optional[Alert]:
        """فحص الخسائر المتتالية"""
        losses = self.state['consecutive_losses'].get(symbol, 0)
        
        if losses >= 3:
            return Alert(
                type=ProtectionType.DAILY_LOSS,
                level=AlertLevel.DANGER,
                message=f"📉 {losses} خسائر متتالية في {symbol}",
                data={'consecutive_losses': losses, 'symbol': symbol},
                action_required=f"تجنب {symbol} مؤقتاً",
                expires_at=datetime.now() + timedelta(hours=4)
            )
        
        return None
    
    def record_trade_result(self, symbol: str, profitable: bool) -> None:
        """تسجيل نتيجة صفقة"""
        if profitable:
            self.state['consecutive_losses'][symbol] = 0
        else:
            current = self.state['consecutive_losses'].get(symbol, 0)
            self.state['consecutive_losses'][symbol] = current + 1
    
    # ═══════════════════════════════════════════════════════════════
    # CIRCUIT BREAKER
    # ═══════════════════════════════════════════════════════════════
    
    def _check_circuit_breaker(self, alerts: List[Alert]) -> bool:
        """فحص قاطع الدائرة"""
        # تفعيل قاطع الدائرة إذا كان هناك تنبيه حرج
        critical_alerts = [a for a in alerts if a.level == AlertLevel.CRITICAL]
        
        if critical_alerts:
            self.state['circuit_breaker_until'] = datetime.now() + timedelta(minutes=30)
            self.stats['circuit_breaker_activations'] += 1
            logger.warning("🔌 Circuit breaker activated!")
            return True
        
        # التحقق من انتهاء قاطع الدائرة
        if self.state['circuit_breaker_until']:
            if datetime.now() < self.state['circuit_breaker_until']:
                return True
            else:
                self.state['circuit_breaker_until'] = None
        
        return False
    
    # ═══════════════════════════════════════════════════════════════
    # HELPERS
    # ═══════════════════════════════════════════════════════════════
    
    def _update_price_history(self, symbol: str, price: float) -> None:
        """تحديث تاريخ الأسعار"""
        if symbol not in self.price_history:
            self.price_history[symbol] = deque(maxlen=self.history_size)
        
        self.price_history[symbol].append(price)
    
    def _update_portfolio_state(self, portfolio_state: Dict) -> None:
        """تحديث حالة المحفظة"""
        self.state['daily_pnl'] = portfolio_state.get('daily_pnl', 0)
        self.state['weekly_pnl'] = portfolio_state.get('weekly_pnl', 0)
        self.state['portfolio_heat'] = portfolio_state.get('portfolio_heat', 0)
    
    def _calculate_risk_level(self, alerts: List[Alert]) -> float:
        """حساب مستوى المخاطر"""
        if not alerts:
            return 0.0
        
        level_scores = {
            AlertLevel.INFO: 0.1,
            AlertLevel.WARNING: 0.3,
            AlertLevel.DANGER: 0.6,
            AlertLevel.CRITICAL: 1.0
        }
        
        max_score = max(level_scores.get(a.level, 0) for a in alerts)
        avg_score = np.mean([level_scores.get(a.level, 0) for a in alerts])
        
        return max_score * 0.7 + avg_score * 0.3
    
    def _generate_recommendations(
        self,
        alerts: List[Alert],
        risk_level: float
    ) -> List[str]:
        """توليد التوصيات"""
        recommendations = []
        
        if risk_level > 0.8:
            recommendations.append("🛑 أوقف جميع الصفقات فوراً")
        elif risk_level > 0.5:
            recommendations.append("⚠️ قلل التعرض للسوق")
        
        for alert in alerts:
            if alert.action_required:
                recommendations.append(alert.action_required)
        
        return list(set(recommendations))
    
    # ═══════════════════════════════════════════════════════════════
    # STATUS
    # ═══════════════════════════════════════════════════════════════
    
    def get_status(self) -> Dict[str, Any]:
        """الحصول على حالة الطبقة"""
        return {
            'state': self.state,
            'limits': self.limits,
            'stats': self.stats,
            'active_alerts': len(self.active_alerts),
            'circuit_breaker_active': self.state['circuit_breaker_until'] is not None
        }
    
    def reset_daily(self) -> None:
        """إعادة تعيين يومي"""
        self.state['daily_pnl'] = 0
        self.state['consecutive_losses'] = {}
        logger.info("🔄 Daily protection state reset")
    
    def reset_weekly(self) -> None:
        """إعادة تعيين أسبوعي"""
        self.state['weekly_pnl'] = 0
        logger.info("🔄 Weekly protection state reset")


# ═══════════════════════════════════════════════════════════════
# STANDALONE EXECUTION
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # اختبار طبقة الحماية
    protection = ProtectionLayer()
    
    features = {
        'volume': 1000000,
        'volume_sma_20': 500000,
        'orderbook_imbalance': 0.1,
        'orderbook_spread': 0.005,
        'atr_percent': 3.0,
        'bb_width': 4.0,
        'bid_depth': 100000,
        'ask_depth': 120000
    }
    
    portfolio_state = {
        'daily_pnl': -3.5,
        'weekly_pnl': -6.0,
        'portfolio_heat': 60
    }
    
    # محاكاة تاريخ أسعار
    for i in range(10):
        protection._update_price_history('BTCUSDT', 50000 - i * 100)
    
    status = protection.check('BTCUSDT', 49000, features, portfolio_state)
    
    print("🛡️ Protection Status:")
    print(f"Is Safe: {status.is_safe}")
    print(f"Risk Level: {status.risk_level:.2%}")
    print(f"Circuit Breaker: {status.circuit_breaker_active}")
    print(f"Emergency Mode: {status.emergency_mode}")
    print(f"Blocked Actions: {status.blocked_actions}")
    
    print("\n📢 Alerts:")
    for alert in status.alerts:
        print(f"  [{alert.level.value}] {alert.type.value}: {alert.message}")
    
    print(f"\n💡 Recommendations: {status.recommendations}")
