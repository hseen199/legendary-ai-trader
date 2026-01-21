"""
═══════════════════════════════════════════════════════════════
LEGENDARY AGENT - Circuit Breaker
قاطع الدائرة - إيقاف التداول في حالات الطوارئ
═══════════════════════════════════════════════════════════════
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from loguru import logger


class BreakerState(Enum):
    """حالات القاطع"""
    CLOSED = "مغلق"      # عادي - التداول مسموح
    OPEN = "مفتوح"       # طوارئ - التداول ممنوع
    HALF_OPEN = "نصف مفتوح"  # اختبار - تداول محدود


class TripReason(Enum):
    """أسباب التفعيل"""
    DAILY_LOSS_LIMIT = "حد الخسارة اليومي"
    WEEKLY_LOSS_LIMIT = "حد الخسارة الأسبوعي"
    CONSECUTIVE_LOSSES = "خسائر متتالية"
    FLASH_CRASH = "انهيار سريع"
    HIGH_VOLATILITY = "تقلب عالي"
    SYSTEM_ERROR = "خطأ في النظام"
    MANUAL = "يدوي"


@dataclass
class TripEvent:
    """حدث تفعيل"""
    reason: TripReason
    timestamp: datetime
    details: Dict[str, Any]
    duration_minutes: int
    auto_reset: bool = True


@dataclass
class BreakerStatus:
    """حالة القاطع"""
    state: BreakerState
    is_trading_allowed: bool
    current_trip: Optional[TripEvent] = None
    reset_at: Optional[datetime] = None
    trip_count_today: int = 0
    last_trip_reason: Optional[str] = None


class CircuitBreaker:
    """
    قاطع الدائرة
    
    يحمي المحفظة من الخسائر الكبيرة عن طريق
    إيقاف التداول في حالات الطوارئ
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        تهيئة قاطع الدائرة
        
        Args:
            config: إعدادات القاطع
        """
        self.config = config or {}
        
        # الحدود
        self.limits = {
            'daily_loss': self.config.get('daily_loss', 5.0),
            'weekly_loss': self.config.get('weekly_loss', 10.0),
            'consecutive_losses': self.config.get('consecutive_losses', 5),
            'flash_crash_percent': self.config.get('flash_crash', 5.0),
            'max_volatility': self.config.get('max_volatility', 15.0),
            'max_trips_per_day': self.config.get('max_trips', 3)
        }
        
        # فترات الإيقاف (بالدقائق)
        self.cooldown_periods = {
            TripReason.DAILY_LOSS_LIMIT: 1440,  # يوم كامل
            TripReason.WEEKLY_LOSS_LIMIT: 10080,  # أسبوع
            TripReason.CONSECUTIVE_LOSSES: 120,  # ساعتين
            TripReason.FLASH_CRASH: 60,  # ساعة
            TripReason.HIGH_VOLATILITY: 30,  # نصف ساعة
            TripReason.SYSTEM_ERROR: 15,  # 15 دقيقة
            TripReason.MANUAL: 60  # ساعة
        }
        
        # الحالة
        self.state = BreakerState.CLOSED
        self.current_trip: Optional[TripEvent] = None
        self.reset_at: Optional[datetime] = None
        
        # التاريخ
        self.trip_history: List[TripEvent] = []
        self.trips_today = 0
        self.last_reset_date = datetime.now().date()
        
        # الإحصائيات
        self.stats = {
            'total_trips': 0,
            'trips_by_reason': {r.value: 0 for r in TripReason},
            'total_downtime_minutes': 0
        }
        
        logger.info("🔌 CircuitBreaker initialized")
    
    # ═══════════════════════════════════════════════════════════════
    # MAIN INTERFACE
    # ═══════════════════════════════════════════════════════════════
    
    def check(
        self,
        daily_pnl: float = 0,
        weekly_pnl: float = 0,
        consecutive_losses: int = 0,
        price_change_5m: float = 0,
        volatility: float = 0
    ) -> BreakerStatus:
        """
        فحص وتحديث حالة القاطع
        
        Args:
            daily_pnl: الربح/الخسارة اليومية
            weekly_pnl: الربح/الخسارة الأسبوعية
            consecutive_losses: الخسائر المتتالية
            price_change_5m: تغير السعر في 5 دقائق
            volatility: التقلب الحالي
            
        Returns:
            حالة القاطع
        """
        # إعادة تعيين يومي
        self._check_daily_reset()
        
        # التحقق من انتهاء فترة الإيقاف
        if self.state == BreakerState.OPEN:
            self._check_reset()
        
        # إذا كان مغلقاً، تحقق من شروط التفعيل
        if self.state == BreakerState.CLOSED:
            trip_reason = self._check_trip_conditions(
                daily_pnl, weekly_pnl, consecutive_losses,
                price_change_5m, volatility
            )
            
            if trip_reason:
                self._trip(trip_reason, {
                    'daily_pnl': daily_pnl,
                    'weekly_pnl': weekly_pnl,
                    'consecutive_losses': consecutive_losses,
                    'price_change_5m': price_change_5m,
                    'volatility': volatility
                })
        
        # إذا كان نصف مفتوح، تحقق من إمكانية الإغلاق
        elif self.state == BreakerState.HALF_OPEN:
            if self._can_close():
                self._close()
        
        return self.get_status()
    
    def _check_trip_conditions(
        self,
        daily_pnl: float,
        weekly_pnl: float,
        consecutive_losses: int,
        price_change_5m: float,
        volatility: float
    ) -> Optional[TripReason]:
        """التحقق من شروط التفعيل"""
        # حد الخسارة اليومي
        if daily_pnl <= -self.limits['daily_loss']:
            return TripReason.DAILY_LOSS_LIMIT
        
        # حد الخسارة الأسبوعي
        if weekly_pnl <= -self.limits['weekly_loss']:
            return TripReason.WEEKLY_LOSS_LIMIT
        
        # الخسائر المتتالية
        if consecutive_losses >= self.limits['consecutive_losses']:
            return TripReason.CONSECUTIVE_LOSSES
        
        # الانهيار السريع
        if abs(price_change_5m) >= self.limits['flash_crash_percent']:
            return TripReason.FLASH_CRASH
        
        # التقلب العالي
        if volatility >= self.limits['max_volatility']:
            return TripReason.HIGH_VOLATILITY
        
        return None
    
    # ═══════════════════════════════════════════════════════════════
    # STATE MANAGEMENT
    # ═══════════════════════════════════════════════════════════════
    
    def _trip(self, reason: TripReason, details: Dict) -> None:
        """تفعيل القاطع"""
        # التحقق من عدد التفعيلات اليومية
        if self.trips_today >= self.limits['max_trips_per_day']:
            # إيقاف لبقية اليوم
            duration = self._minutes_until_midnight()
        else:
            duration = self.cooldown_periods.get(reason, 60)
        
        self.current_trip = TripEvent(
            reason=reason,
            timestamp=datetime.now(),
            details=details,
            duration_minutes=duration,
            auto_reset=reason not in [TripReason.DAILY_LOSS_LIMIT, TripReason.WEEKLY_LOSS_LIMIT]
        )
        
        self.state = BreakerState.OPEN
        self.reset_at = datetime.now() + timedelta(minutes=duration)
        
        # تحديث الإحصائيات
        self.trips_today += 1
        self.stats['total_trips'] += 1
        self.stats['trips_by_reason'][reason.value] += 1
        
        # حفظ في التاريخ
        self.trip_history.append(self.current_trip)
        
        logger.warning(
            f"🔴 Circuit breaker TRIPPED! "
            f"Reason: {reason.value}, Duration: {duration} minutes"
        )
    
    def _check_reset(self) -> None:
        """التحقق من إمكانية إعادة التعيين"""
        if self.reset_at and datetime.now() >= self.reset_at:
            if self.current_trip and self.current_trip.auto_reset:
                self.state = BreakerState.HALF_OPEN
                logger.info("🟡 Circuit breaker entering HALF_OPEN state")
            else:
                # يحتاج إعادة تعيين يدوية
                logger.info("⚠️ Circuit breaker requires manual reset")
    
    def _can_close(self) -> bool:
        """التحقق من إمكانية الإغلاق"""
        # يمكن إضافة شروط إضافية هنا
        return True
    
    def _close(self) -> None:
        """إغلاق القاطع"""
        if self.current_trip:
            # حساب وقت التوقف
            downtime = (datetime.now() - self.current_trip.timestamp).total_seconds() / 60
            self.stats['total_downtime_minutes'] += downtime
        
        self.state = BreakerState.CLOSED
        self.current_trip = None
        self.reset_at = None
        
        logger.info("🟢 Circuit breaker CLOSED - Trading resumed")
    
    def _check_daily_reset(self) -> None:
        """إعادة تعيين يومي"""
        today = datetime.now().date()
        if today > self.last_reset_date:
            self.trips_today = 0
            self.last_reset_date = today
    
    def _minutes_until_midnight(self) -> int:
        """حساب الدقائق حتى منتصف الليل"""
        now = datetime.now()
        midnight = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
        return int((midnight - now).total_seconds() / 60)
    
    # ═══════════════════════════════════════════════════════════════
    # MANUAL CONTROL
    # ═══════════════════════════════════════════════════════════════
    
    def manual_trip(self, reason: str = "Manual intervention", duration_minutes: int = 60) -> None:
        """تفعيل يدوي"""
        self._trip(TripReason.MANUAL, {'reason': reason})
        self.reset_at = datetime.now() + timedelta(minutes=duration_minutes)
        logger.warning(f"🔴 Manual circuit breaker trip: {reason}")
    
    def manual_reset(self) -> bool:
        """إعادة تعيين يدوية"""
        if self.state != BreakerState.CLOSED:
            self._close()
            logger.info("🟢 Manual circuit breaker reset")
            return True
        return False
    
    def force_open(self, duration_minutes: int = 60) -> None:
        """فتح إجباري"""
        self.state = BreakerState.OPEN
        self.reset_at = datetime.now() + timedelta(minutes=duration_minutes)
        self.current_trip = TripEvent(
            reason=TripReason.MANUAL,
            timestamp=datetime.now(),
            details={'forced': True},
            duration_minutes=duration_minutes,
            auto_reset=True
        )
        logger.warning(f"🔴 Circuit breaker forced OPEN for {duration_minutes} minutes")
    
    # ═══════════════════════════════════════════════════════════════
    # STATUS
    # ═══════════════════════════════════════════════════════════════
    
    def get_status(self) -> BreakerStatus:
        """الحصول على حالة القاطع"""
        return BreakerStatus(
            state=self.state,
            is_trading_allowed=self.state == BreakerState.CLOSED,
            current_trip=self.current_trip,
            reset_at=self.reset_at,
            trip_count_today=self.trips_today,
            last_trip_reason=(
                self.current_trip.reason.value
                if self.current_trip else None
            )
        )
    
    def is_trading_allowed(self) -> bool:
        """هل التداول مسموح"""
        return self.state == BreakerState.CLOSED
    
    def get_stats(self) -> Dict[str, Any]:
        """الحصول على الإحصائيات"""
        return {
            **self.stats,
            'current_state': self.state.value,
            'trips_today': self.trips_today,
            'history_count': len(self.trip_history)
        }


# ═══════════════════════════════════════════════════════════════
# STANDALONE EXECUTION
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # اختبار قاطع الدائرة
    breaker = CircuitBreaker()
    
    # فحص عادي
    status = breaker.check(
        daily_pnl=-2.0,
        weekly_pnl=-5.0,
        consecutive_losses=2,
        price_change_5m=1.0,
        volatility=5.0
    )
    print(f"Status 1: {status.state.value}, Trading: {status.is_trading_allowed}")
    
    # فحص مع خسارة كبيرة
    status = breaker.check(
        daily_pnl=-6.0,  # تجاوز الحد
        weekly_pnl=-8.0,
        consecutive_losses=3,
        price_change_5m=1.0,
        volatility=5.0
    )
    print(f"Status 2: {status.state.value}, Trading: {status.is_trading_allowed}")
    print(f"Trip Reason: {status.last_trip_reason}")
    print(f"Reset At: {status.reset_at}")
    
    # إعادة تعيين يدوية
    breaker.manual_reset()
    status = breaker.get_status()
    print(f"Status 3: {status.state.value}, Trading: {status.is_trading_allowed}")
    
    # الإحصائيات
    print(f"\nStats: {breaker.get_stats()}")
