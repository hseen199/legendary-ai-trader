"""
Legendary Trading System - Protection System
نظام التداول الخارق - نظام الحماية المتقدم

يوفر حماية شاملة للنظام ورأس المال.
"""

import asyncio
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from collections import deque
from enum import Enum
import logging


class AlertLevel(Enum):
    """مستويات التنبيه"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class ProtectionAction(Enum):
    """إجراءات الحماية"""
    NONE = "none"
    REDUCE_POSITION = "reduce_position"
    CLOSE_POSITION = "close_position"
    PAUSE_TRADING = "pause_trading"
    EMERGENCY_EXIT = "emergency_exit"


@dataclass
class Alert:
    """تنبيه"""
    id: str
    level: AlertLevel
    source: str
    message: str
    timestamp: datetime
    data: Dict = field(default_factory=dict)
    acknowledged: bool = False


@dataclass
class ProtectionConfig:
    """إعدادات الحماية"""
    # حدود الخسارة
    max_daily_loss_percent: float = 5.0
    max_weekly_loss_percent: float = 10.0
    max_monthly_loss_percent: float = 15.0
    max_drawdown_percent: float = 20.0
    
    # حدود الصفقات
    max_position_size_percent: float = 10.0
    max_open_positions: int = 10
    max_trades_per_hour: int = 20
    max_trades_per_day: int = 100
    
    # حدود التقلب
    volatility_threshold: float = 0.05
    correlation_threshold: float = 0.8
    
    # إعدادات قاطع الدائرة
    circuit_breaker_threshold: int = 3
    circuit_breaker_cooldown: int = 3600  # ثانية
    
    # إعدادات الكشف عن الشذوذ
    anomaly_sensitivity: float = 2.0
    anomaly_window: int = 100


class CircuitBreaker:
    """
    قاطع الدائرة - يوقف التداول عند حدوث أخطاء متكررة.
    """
    
    def __init__(self, config: ProtectionConfig):
        self.config = config
        self.logger = logging.getLogger("CircuitBreaker")
        
        self._failure_count = 0
        self._last_failure = None
        self._is_open = False
        self._open_time = None
        self._failure_history: deque = deque(maxlen=100)
    
    def record_failure(self, error: str) -> bool:
        """
        تسجيل فشل.
        
        Returns:
            True إذا تم فتح القاطع
        """
        now = datetime.utcnow()
        self._failure_count += 1
        self._last_failure = now
        self._failure_history.append({
            "time": now,
            "error": error
        })
        
        # التحقق من الحد
        if self._failure_count >= self.config.circuit_breaker_threshold:
            self._open_circuit()
            return True
        
        return False
    
    def record_success(self):
        """تسجيل نجاح."""
        self._failure_count = max(0, self._failure_count - 1)
    
    def is_open(self) -> bool:
        """التحقق من حالة القاطع."""
        if not self._is_open:
            return False
        
        # التحقق من انتهاء فترة التبريد
        if self._open_time:
            elapsed = (datetime.utcnow() - self._open_time).total_seconds()
            if elapsed >= self.config.circuit_breaker_cooldown:
                self._close_circuit()
                return False
        
        return True
    
    def _open_circuit(self):
        """فتح القاطع."""
        self._is_open = True
        self._open_time = datetime.utcnow()
        self.logger.warning("تم فتح قاطع الدائرة - إيقاف التداول مؤقتاً")
    
    def _close_circuit(self):
        """إغلاق القاطع."""
        self._is_open = False
        self._open_time = None
        self._failure_count = 0
        self.logger.info("تم إغلاق قاطع الدائرة - استئناف التداول")
    
    def get_status(self) -> Dict[str, Any]:
        """الحصول على الحالة."""
        return {
            "is_open": self._is_open,
            "failure_count": self._failure_count,
            "last_failure": self._last_failure.isoformat() if self._last_failure else None,
            "cooldown_remaining": self._get_cooldown_remaining()
        }
    
    def _get_cooldown_remaining(self) -> int:
        """الحصول على الوقت المتبقي للتبريد."""
        if not self._is_open or not self._open_time:
            return 0
        
        elapsed = (datetime.utcnow() - self._open_time).total_seconds()
        remaining = self.config.circuit_breaker_cooldown - elapsed
        return max(0, int(remaining))


class AnomalyDetector:
    """
    كاشف الشذوذ - يكتشف السلوكيات غير الطبيعية.
    """
    
    def __init__(self, config: ProtectionConfig):
        self.config = config
        self.logger = logging.getLogger("AnomalyDetector")
        
        self._price_history: Dict[str, deque] = {}
        self._volume_history: Dict[str, deque] = {}
        self._return_history: Dict[str, deque] = {}
    
    def update(self, symbol: str, price: float, volume: float):
        """تحديث البيانات."""
        if symbol not in self._price_history:
            self._price_history[symbol] = deque(maxlen=self.config.anomaly_window)
            self._volume_history[symbol] = deque(maxlen=self.config.anomaly_window)
            self._return_history[symbol] = deque(maxlen=self.config.anomaly_window)
        
        # حساب العائد
        if len(self._price_history[symbol]) > 0:
            prev_price = self._price_history[symbol][-1]
            ret = (price - prev_price) / prev_price if prev_price > 0 else 0
            self._return_history[symbol].append(ret)
        
        self._price_history[symbol].append(price)
        self._volume_history[symbol].append(volume)
    
    def detect_price_anomaly(self, symbol: str, price: float) -> Optional[Alert]:
        """كشف شذوذ السعر."""
        if symbol not in self._price_history or len(self._price_history[symbol]) < 10:
            return None
        
        prices = np.array(self._price_history[symbol])
        mean = np.mean(prices)
        std = np.std(prices)
        
        if std == 0:
            return None
        
        z_score = abs(price - mean) / std
        
        if z_score > self.config.anomaly_sensitivity * 2:
            return Alert(
                id=f"price_anomaly_{symbol}_{datetime.utcnow().timestamp()}",
                level=AlertLevel.CRITICAL,
                source="AnomalyDetector",
                message=f"شذوذ سعري كبير في {symbol}: z-score = {z_score:.2f}",
                timestamp=datetime.utcnow(),
                data={"symbol": symbol, "price": price, "z_score": z_score}
            )
        elif z_score > self.config.anomaly_sensitivity:
            return Alert(
                id=f"price_anomaly_{symbol}_{datetime.utcnow().timestamp()}",
                level=AlertLevel.WARNING,
                source="AnomalyDetector",
                message=f"شذوذ سعري في {symbol}: z-score = {z_score:.2f}",
                timestamp=datetime.utcnow(),
                data={"symbol": symbol, "price": price, "z_score": z_score}
            )
        
        return None
    
    def detect_volume_anomaly(self, symbol: str, volume: float) -> Optional[Alert]:
        """كشف شذوذ الحجم."""
        if symbol not in self._volume_history or len(self._volume_history[symbol]) < 10:
            return None
        
        volumes = np.array(self._volume_history[symbol])
        mean = np.mean(volumes)
        std = np.std(volumes)
        
        if std == 0 or mean == 0:
            return None
        
        z_score = abs(volume - mean) / std
        volume_ratio = volume / mean
        
        if z_score > self.config.anomaly_sensitivity * 2 and volume_ratio > 3:
            return Alert(
                id=f"volume_anomaly_{symbol}_{datetime.utcnow().timestamp()}",
                level=AlertLevel.WARNING,
                source="AnomalyDetector",
                message=f"حجم تداول غير عادي في {symbol}: {volume_ratio:.1f}x المتوسط",
                timestamp=datetime.utcnow(),
                data={"symbol": symbol, "volume": volume, "ratio": volume_ratio}
            )
        
        return None
    
    def detect_volatility_spike(self, symbol: str) -> Optional[Alert]:
        """كشف ارتفاع التقلب."""
        if symbol not in self._return_history or len(self._return_history[symbol]) < 20:
            return None
        
        returns = np.array(self._return_history[symbol])
        volatility = np.std(returns[-20:])
        historical_vol = np.std(returns)
        
        if historical_vol == 0:
            return None
        
        vol_ratio = volatility / historical_vol
        
        if vol_ratio > 2:
            return Alert(
                id=f"volatility_spike_{symbol}_{datetime.utcnow().timestamp()}",
                level=AlertLevel.WARNING,
                source="AnomalyDetector",
                message=f"ارتفاع التقلب في {symbol}: {vol_ratio:.1f}x المعتاد",
                timestamp=datetime.utcnow(),
                data={"symbol": symbol, "volatility": volatility, "ratio": vol_ratio}
            )
        
        return None


class DrawdownMonitor:
    """
    مراقب السحب - يتتبع الخسائر ويحمي رأس المال.
    """
    
    def __init__(self, config: ProtectionConfig):
        self.config = config
        self.logger = logging.getLogger("DrawdownMonitor")
        
        self._peak_equity = 0.0
        self._current_equity = 0.0
        self._daily_start_equity = 0.0
        self._weekly_start_equity = 0.0
        self._monthly_start_equity = 0.0
        
        self._last_daily_reset = datetime.utcnow().date()
        self._last_weekly_reset = datetime.utcnow().date()
        self._last_monthly_reset = datetime.utcnow().date()
    
    def update_equity(self, equity: float) -> List[Alert]:
        """تحديث رأس المال."""
        alerts = []
        self._current_equity = equity
        
        # تحديث القمة
        if equity > self._peak_equity:
            self._peak_equity = equity
        
        # إعادة تعيين الفترات
        self._check_period_resets()
        
        # التحقق من الحدود
        alerts.extend(self._check_drawdown())
        alerts.extend(self._check_daily_loss())
        alerts.extend(self._check_weekly_loss())
        alerts.extend(self._check_monthly_loss())
        
        return alerts
    
    def _check_period_resets(self):
        """التحقق من إعادة تعيين الفترات."""
        today = datetime.utcnow().date()
        
        # يومي
        if today != self._last_daily_reset:
            self._daily_start_equity = self._current_equity
            self._last_daily_reset = today
        
        # أسبوعي
        if today.isocalendar()[1] != self._last_weekly_reset.isocalendar()[1]:
            self._weekly_start_equity = self._current_equity
            self._last_weekly_reset = today
        
        # شهري
        if today.month != self._last_monthly_reset.month:
            self._monthly_start_equity = self._current_equity
            self._last_monthly_reset = today
    
    def _check_drawdown(self) -> List[Alert]:
        """التحقق من السحب."""
        if self._peak_equity == 0:
            return []
        
        drawdown = (self._peak_equity - self._current_equity) / self._peak_equity * 100
        
        if drawdown >= self.config.max_drawdown_percent:
            return [Alert(
                id=f"max_drawdown_{datetime.utcnow().timestamp()}",
                level=AlertLevel.EMERGENCY,
                source="DrawdownMonitor",
                message=f"تجاوز الحد الأقصى للسحب: {drawdown:.1f}%",
                timestamp=datetime.utcnow(),
                data={"drawdown": drawdown, "peak": self._peak_equity, "current": self._current_equity}
            )]
        elif drawdown >= self.config.max_drawdown_percent * 0.8:
            return [Alert(
                id=f"drawdown_warning_{datetime.utcnow().timestamp()}",
                level=AlertLevel.CRITICAL,
                source="DrawdownMonitor",
                message=f"اقتراب من الحد الأقصى للسحب: {drawdown:.1f}%",
                timestamp=datetime.utcnow(),
                data={"drawdown": drawdown}
            )]
        
        return []
    
    def _check_daily_loss(self) -> List[Alert]:
        """التحقق من الخسارة اليومية."""
        if self._daily_start_equity == 0:
            return []
        
        loss = (self._daily_start_equity - self._current_equity) / self._daily_start_equity * 100
        
        if loss >= self.config.max_daily_loss_percent:
            return [Alert(
                id=f"daily_loss_{datetime.utcnow().timestamp()}",
                level=AlertLevel.CRITICAL,
                source="DrawdownMonitor",
                message=f"تجاوز الحد اليومي للخسارة: {loss:.1f}%",
                timestamp=datetime.utcnow(),
                data={"loss": loss, "period": "daily"}
            )]
        
        return []
    
    def _check_weekly_loss(self) -> List[Alert]:
        """التحقق من الخسارة الأسبوعية."""
        if self._weekly_start_equity == 0:
            return []
        
        loss = (self._weekly_start_equity - self._current_equity) / self._weekly_start_equity * 100
        
        if loss >= self.config.max_weekly_loss_percent:
            return [Alert(
                id=f"weekly_loss_{datetime.utcnow().timestamp()}",
                level=AlertLevel.CRITICAL,
                source="DrawdownMonitor",
                message=f"تجاوز الحد الأسبوعي للخسارة: {loss:.1f}%",
                timestamp=datetime.utcnow(),
                data={"loss": loss, "period": "weekly"}
            )]
        
        return []
    
    def _check_monthly_loss(self) -> List[Alert]:
        """التحقق من الخسارة الشهرية."""
        if self._monthly_start_equity == 0:
            return []
        
        loss = (self._monthly_start_equity - self._current_equity) / self._monthly_start_equity * 100
        
        if loss >= self.config.max_monthly_loss_percent:
            return [Alert(
                id=f"monthly_loss_{datetime.utcnow().timestamp()}",
                level=AlertLevel.EMERGENCY,
                source="DrawdownMonitor",
                message=f"تجاوز الحد الشهري للخسارة: {loss:.1f}%",
                timestamp=datetime.utcnow(),
                data={"loss": loss, "period": "monthly"}
            )]
        
        return []
    
    def get_status(self) -> Dict[str, Any]:
        """الحصول على الحالة."""
        drawdown = 0
        if self._peak_equity > 0:
            drawdown = (self._peak_equity - self._current_equity) / self._peak_equity * 100
        
        daily_pnl = 0
        if self._daily_start_equity > 0:
            daily_pnl = (self._current_equity - self._daily_start_equity) / self._daily_start_equity * 100
        
        return {
            "current_equity": self._current_equity,
            "peak_equity": self._peak_equity,
            "drawdown_percent": drawdown,
            "daily_pnl_percent": daily_pnl
        }


class RateLimiter:
    """
    محدد المعدل - يحد من عدد الصفقات.
    """
    
    def __init__(self, config: ProtectionConfig):
        self.config = config
        self.logger = logging.getLogger("RateLimiter")
        
        self._hourly_trades: deque = deque(maxlen=1000)
        self._daily_trades: deque = deque(maxlen=10000)
    
    def record_trade(self):
        """تسجيل صفقة."""
        now = datetime.utcnow()
        self._hourly_trades.append(now)
        self._daily_trades.append(now)
    
    def can_trade(self) -> Tuple[bool, Optional[str]]:
        """التحقق من إمكانية التداول."""
        now = datetime.utcnow()
        
        # تنظيف القديم
        hour_ago = now - timedelta(hours=1)
        day_ago = now - timedelta(days=1)
        
        hourly_count = sum(1 for t in self._hourly_trades if t > hour_ago)
        daily_count = sum(1 for t in self._daily_trades if t > day_ago)
        
        if hourly_count >= self.config.max_trades_per_hour:
            return False, f"تجاوز الحد الساعي ({hourly_count}/{self.config.max_trades_per_hour})"
        
        if daily_count >= self.config.max_trades_per_day:
            return False, f"تجاوز الحد اليومي ({daily_count}/{self.config.max_trades_per_day})"
        
        return True, None
    
    def get_status(self) -> Dict[str, Any]:
        """الحصول على الحالة."""
        now = datetime.utcnow()
        hour_ago = now - timedelta(hours=1)
        day_ago = now - timedelta(days=1)
        
        return {
            "hourly_trades": sum(1 for t in self._hourly_trades if t > hour_ago),
            "daily_trades": sum(1 for t in self._daily_trades if t > day_ago),
            "hourly_limit": self.config.max_trades_per_hour,
            "daily_limit": self.config.max_trades_per_day
        }


class ProtectionSystem:
    """
    نظام الحماية المتكامل.
    
    يجمع جميع آليات الحماية ويوفر واجهة موحدة.
    """
    
    def __init__(self, config: ProtectionConfig = None):
        self.config = config or ProtectionConfig()
        self.logger = logging.getLogger("ProtectionSystem")
        
        # إنشاء المكونات
        self.circuit_breaker = CircuitBreaker(self.config)
        self.anomaly_detector = AnomalyDetector(self.config)
        self.drawdown_monitor = DrawdownMonitor(self.config)
        self.rate_limiter = RateLimiter(self.config)
        
        # التنبيهات
        self._alerts: deque = deque(maxlen=1000)
        self._alert_handlers: List[Callable] = []
        
        # حالة التداول
        self._trading_enabled = True
        self._pause_reason: Optional[str] = None
    
    async def initialize(self, initial_equity: float):
        """تهيئة النظام."""
        self.logger.info("تهيئة نظام الحماية...")
        
        self.drawdown_monitor._peak_equity = initial_equity
        self.drawdown_monitor._current_equity = initial_equity
        self.drawdown_monitor._daily_start_equity = initial_equity
        self.drawdown_monitor._weekly_start_equity = initial_equity
        self.drawdown_monitor._monthly_start_equity = initial_equity
        
        self.logger.info(f"تم تهيئة نظام الحماية برأس مال: {initial_equity}")
    
    def add_alert_handler(self, handler: Callable):
        """إضافة معالج تنبيهات."""
        self._alert_handlers.append(handler)
    
    async def check_trade_allowed(self, symbol: str, 
                                 action: str,
                                 size: float,
                                 current_positions: int) -> Tuple[bool, Optional[str]]:
        """
        التحقق من السماح بالتداول.
        
        Args:
            symbol: رمز العملة
            action: الإجراء
            size: حجم الصفقة
            current_positions: عدد الصفقات الحالية
            
        Returns:
            (مسموح، السبب)
        """
        # التحقق من حالة التداول
        if not self._trading_enabled:
            return False, f"التداول متوقف: {self._pause_reason}"
        
        # التحقق من قاطع الدائرة
        if self.circuit_breaker.is_open():
            return False, "قاطع الدائرة مفتوح"
        
        # التحقق من محدد المعدل
        can_trade, reason = self.rate_limiter.can_trade()
        if not can_trade:
            return False, reason
        
        # التحقق من عدد الصفقات
        if current_positions >= self.config.max_open_positions:
            return False, f"تجاوز الحد الأقصى للصفقات ({current_positions})"
        
        return True, None
    
    async def update_market_data(self, symbol: str, price: float, volume: float):
        """تحديث بيانات السوق."""
        # تحديث كاشف الشذوذ
        self.anomaly_detector.update(symbol, price, volume)
        
        # التحقق من الشذوذ
        alerts = []
        
        price_alert = self.anomaly_detector.detect_price_anomaly(symbol, price)
        if price_alert:
            alerts.append(price_alert)
        
        volume_alert = self.anomaly_detector.detect_volume_anomaly(symbol, volume)
        if volume_alert:
            alerts.append(volume_alert)
        
        vol_alert = self.anomaly_detector.detect_volatility_spike(symbol)
        if vol_alert:
            alerts.append(vol_alert)
        
        # معالجة التنبيهات
        for alert in alerts:
            await self._handle_alert(alert)
    
    async def update_equity(self, equity: float):
        """تحديث رأس المال."""
        alerts = self.drawdown_monitor.update_equity(equity)
        
        for alert in alerts:
            await self._handle_alert(alert)
            
            # إيقاف التداول عند التنبيهات الطارئة
            if alert.level == AlertLevel.EMERGENCY:
                await self.pause_trading(alert.message)
    
    async def record_trade_result(self, success: bool, error: str = None):
        """تسجيل نتيجة صفقة."""
        if success:
            self.circuit_breaker.record_success()
            self.rate_limiter.record_trade()
        else:
            opened = self.circuit_breaker.record_failure(error or "خطأ غير معروف")
            if opened:
                await self._handle_alert(Alert(
                    id=f"circuit_breaker_{datetime.utcnow().timestamp()}",
                    level=AlertLevel.CRITICAL,
                    source="CircuitBreaker",
                    message="تم فتح قاطع الدائرة بسبب أخطاء متكررة",
                    timestamp=datetime.utcnow()
                ))
    
    async def pause_trading(self, reason: str):
        """إيقاف التداول."""
        self._trading_enabled = False
        self._pause_reason = reason
        self.logger.warning(f"تم إيقاف التداول: {reason}")
        
        await self._handle_alert(Alert(
            id=f"trading_paused_{datetime.utcnow().timestamp()}",
            level=AlertLevel.CRITICAL,
            source="ProtectionSystem",
            message=f"تم إيقاف التداول: {reason}",
            timestamp=datetime.utcnow()
        ))
    
    async def resume_trading(self):
        """استئناف التداول."""
        self._trading_enabled = True
        self._pause_reason = None
        self.logger.info("تم استئناف التداول")
    
    async def _handle_alert(self, alert: Alert):
        """معالجة تنبيه."""
        self._alerts.append(alert)
        
        # تسجيل
        log_method = {
            AlertLevel.INFO: self.logger.info,
            AlertLevel.WARNING: self.logger.warning,
            AlertLevel.CRITICAL: self.logger.error,
            AlertLevel.EMERGENCY: self.logger.critical
        }.get(alert.level, self.logger.info)
        
        log_method(f"[{alert.source}] {alert.message}")
        
        # استدعاء المعالجات
        for handler in self._alert_handlers:
            try:
                await handler(alert)
            except Exception as e:
                self.logger.error(f"خطأ في معالج التنبيه: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """الحصول على حالة النظام."""
        return {
            "trading_enabled": self._trading_enabled,
            "pause_reason": self._pause_reason,
            "circuit_breaker": self.circuit_breaker.get_status(),
            "drawdown": self.drawdown_monitor.get_status(),
            "rate_limiter": self.rate_limiter.get_status(),
            "recent_alerts": [
                {
                    "level": a.level.value,
                    "message": a.message,
                    "time": a.timestamp.isoformat()
                }
                for a in list(self._alerts)[-10:]
            ]
        }
    
    def get_recommended_action(self) -> ProtectionAction:
        """الحصول على الإجراء الموصى به."""
        status = self.drawdown_monitor.get_status()
        
        # طوارئ
        if status["drawdown_percent"] >= self.config.max_drawdown_percent:
            return ProtectionAction.EMERGENCY_EXIT
        
        # إيقاف
        if self.circuit_breaker.is_open():
            return ProtectionAction.PAUSE_TRADING
        
        # تقليل
        if status["drawdown_percent"] >= self.config.max_drawdown_percent * 0.7:
            return ProtectionAction.REDUCE_POSITION
        
        return ProtectionAction.NONE
