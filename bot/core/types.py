"""
═══════════════════════════════════════════════════════════════
LEGENDARY AGENT - Types & Definitions
أنواع البيانات والتعريفات المشتركة
═══════════════════════════════════════════════════════════════
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import numpy as np


# ═══════════════════════════════════════════════════════════════
# ENUMS - التعدادات
# ═══════════════════════════════════════════════════════════════

class Action(Enum):
    """إجراءات التداول"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class MarketRegime(Enum):
    """حالات السوق"""
    STRONG_BULLISH = "STRONG_BULLISH"
    BULLISH = "BULLISH"
    NEUTRAL = "NEUTRAL"
    BEARISH = "BEARISH"
    STRONG_BEARISH = "STRONG_BEARISH"
    VOLATILE = "VOLATILE"
    RANGING = "RANGING"


class RiskLevel(Enum):
    """مستويات المخاطر"""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    EXTREME = "EXTREME"


class TradingMode(Enum):
    """أوضاع التداول"""
    AGGRESSIVE = "aggressive"
    BALANCED = "balanced"
    CONSERVATIVE = "conservative"


class SignalStrength(Enum):
    """قوة الإشارة"""
    VERY_WEAK = 1
    WEAK = 2
    MODERATE = 3
    STRONG = 4
    VERY_STRONG = 5


class PositionStatus(Enum):
    """حالة الصفقة"""
    OPEN = "OPEN"
    PARTIAL_1 = "PARTIAL_1"
    PARTIAL_2 = "PARTIAL_2"
    CLOSED = "CLOSED"
    STOPPED = "STOPPED"


# ═══════════════════════════════════════════════════════════════
# DATA CLASSES - كلاسات البيانات
# ═══════════════════════════════════════════════════════════════

@dataclass
class OHLCV:
    """بيانات الشمعة"""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat(),
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume
        }


@dataclass
class TechnicalIndicators:
    """المؤشرات التقنية"""
    rsi: float
    macd: float
    macd_signal: float
    macd_histogram: float
    bb_upper: float
    bb_middle: float
    bb_lower: float
    ema_9: float
    ema_21: float
    ema_50: float
    sma_20: float
    sma_50: float
    sma_200: float
    atr: float
    adx: float
    obv: float
    vwap: Optional[float] = None
    
    def to_array(self) -> np.ndarray:
        """تحويل إلى مصفوفة للنموذج"""
        return np.array([
            self.rsi, self.macd, self.macd_signal, self.macd_histogram,
            self.bb_upper, self.bb_middle, self.bb_lower,
            self.ema_9, self.ema_21, self.ema_50,
            self.sma_20, self.sma_50, self.sma_200,
            self.atr, self.adx, self.obv
        ], dtype=np.float32)


@dataclass
class MarketContext:
    """سياق السوق"""
    regime: MarketRegime
    volatility: float
    trend_strength: float
    momentum: float
    volume_ratio: float
    btc_correlation: float
    fear_greed_index: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'regime': self.regime.value,
            'volatility': self.volatility,
            'trend_strength': self.trend_strength,
            'momentum': self.momentum,
            'volume_ratio': self.volume_ratio,
            'btc_correlation': self.btc_correlation,
            'fear_greed_index': self.fear_greed_index
        }


@dataclass
class TradingSignal:
    """إشارة التداول"""
    symbol: str
    action: Action
    confidence: float
    price: float
    timestamp: datetime
    source: str
    reasoning: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'action': self.action.value,
            'confidence': self.confidence,
            'price': self.price,
            'timestamp': self.timestamp.isoformat(),
            'source': self.source,
            'reasoning': self.reasoning
        }


@dataclass
class Position:
    """الصفقة"""
    symbol: str
    entry_price: float
    quantity: float
    side: str  # "LONG" or "SHORT"
    entry_time: datetime
    status: PositionStatus
    stop_loss: float
    take_profit_levels: List[float]
    current_price: Optional[float] = None
    unrealized_pnl: Optional[float] = None
    realized_pnl: float = 0.0
    partial_exits: List[Dict[str, Any]] = field(default_factory=list)
    
    @property
    def pnl_percent(self) -> float:
        """نسبة الربح/الخسارة"""
        if self.current_price is None:
            return 0.0
        return ((self.current_price - self.entry_price) / self.entry_price) * 100


@dataclass
class TakeProfitLevel:
    """مستوى جني الأرباح"""
    target_percent: float
    sell_percent: float
    triggered: bool = False
    triggered_at: Optional[datetime] = None
    triggered_price: Optional[float] = None


@dataclass
class RiskMetrics:
    """مقاييس المخاطر"""
    daily_pnl: float
    weekly_pnl: float
    monthly_pnl: float
    portfolio_heat: float
    max_drawdown: float
    consecutive_losses: int
    risk_level: RiskLevel
    
    def is_safe_to_trade(self, config: Dict) -> Tuple[bool, str]:
        """التحقق من أمان التداول"""
        if self.daily_pnl <= config['risk']['max_daily_loss']:
            return False, f"Daily loss limit reached: {self.daily_pnl}%"
        if self.weekly_pnl <= config['risk']['max_weekly_loss']:
            return False, f"Weekly loss limit reached: {self.weekly_pnl}%"
        if self.portfolio_heat >= config['risk']['portfolio_heat_limit']:
            return False, f"Portfolio heat limit reached: {self.portfolio_heat}%"
        return True, "Safe to trade"


@dataclass
class ThinkingProcess:
    """عملية التفكير"""
    observation: str
    hypothesis: str
    supporting_evidence: List[str]
    counter_arguments: List[str]
    resolution: str
    confidence_adjustment: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'observation': self.observation,
            'hypothesis': self.hypothesis,
            'supporting_evidence': self.supporting_evidence,
            'counter_arguments': self.counter_arguments,
            'resolution': self.resolution,
            'confidence_adjustment': self.confidence_adjustment
        }


@dataclass
class Strategy:
    """الاستراتيجية"""
    name: str
    origin: str  # "PREDEFINED" or "SELF_INVENTED"
    description: str
    conditions: Dict[str, Any]
    success_rate: float
    times_used: int
    total_profit: float
    last_used: Optional[datetime] = None
    created_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'origin': self.origin,
            'description': self.description,
            'success_rate': self.success_rate,
            'times_used': self.times_used,
            'total_profit': self.total_profit
        }


@dataclass
class AgentDecision:
    """قرار الوكيل النهائي"""
    symbol: str
    action: Action
    confidence: float
    position_size_percent: float
    entry_price: float
    stop_loss: float
    take_profit_levels: List[float]
    trailing_stop_activation: Optional[float]
    
    # معلومات إضافية
    market_regime: MarketRegime
    risk_score: float
    reasoning: str
    thinking_process: ThinkingProcess
    strategy_used: Strategy
    
    # تحذيرات وبدائل
    warnings: List[str] = field(default_factory=list)
    alternative_symbols: List[str] = field(default_factory=list)
    
    # توقيت
    timestamp: datetime = field(default_factory=datetime.now)
    time_horizon: str = "4-8 hours"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'action': self.action.value,
            'confidence': self.confidence,
            'position_size_percent': self.position_size_percent,
            'entry_price': self.entry_price,
            'stop_loss': self.stop_loss,
            'take_profit_levels': self.take_profit_levels,
            'trailing_stop_activation': self.trailing_stop_activation,
            'market_regime': self.market_regime.value,
            'risk_score': self.risk_score,
            'reasoning': self.reasoning,
            'thinking_process': self.thinking_process.to_dict(),
            'strategy_used': self.strategy_used.to_dict(),
            'warnings': self.warnings,
            'alternative_symbols': self.alternative_symbols,
            'timestamp': self.timestamp.isoformat(),
            'time_horizon': self.time_horizon
        }


@dataclass
class ModelPrediction:
    """تنبؤ النموذج"""
    model_name: str
    action: Action
    confidence: float
    raw_scores: Dict[str, float]
    features_importance: Optional[Dict[str, float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'model_name': self.model_name,
            'action': self.action.value,
            'confidence': self.confidence,
            'raw_scores': self.raw_scores
        }


@dataclass
class BacktestResult:
    """نتيجة الاختبار التاريخي"""
    strategy_name: str
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_profit: float
    max_drawdown: float
    sharpe_ratio: float
    profit_factor: float
    avg_trade_duration: float
    
    def is_viable(self, min_win_rate: float = 0.55, min_profit_factor: float = 1.5) -> bool:
        """التحقق من جدوى الاستراتيجية"""
        return self.win_rate >= min_win_rate and self.profit_factor >= min_profit_factor


@dataclass
class SymbolProfile:
    """ملف تعريف العملة"""
    symbol: str
    total_trades: int
    wins: int
    losses: int
    win_rate: float
    avg_profit: float
    avg_loss: float
    best_timeframe: str
    worst_timeframe: str
    correlation_with_btc: float
    sector: str
    is_banned: bool = False
    ban_until: Optional[datetime] = None
    consecutive_losses: int = 0
    
    def should_trade(self) -> Tuple[bool, str]:
        """هل يجب التداول على هذه العملة؟"""
        if self.is_banned:
            return False, f"Symbol banned until {self.ban_until}"
        if self.win_rate < 0.3:
            return False, f"Low win rate: {self.win_rate:.1%}"
        return True, "OK"


# ═══════════════════════════════════════════════════════════════
# TYPE ALIASES - أسماء مستعارة للأنواع
# ═══════════════════════════════════════════════════════════════

FeatureVector = np.ndarray
PriceHistory = List[OHLCV]
IndicatorHistory = List[TechnicalIndicators]
Predictions = Dict[str, ModelPrediction]
