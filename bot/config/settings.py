"""
Legendary Trading System - Configuration Settings
نظام التداول الخارق - إعدادات التكوين

هذا الملف يحتوي على جميع الإعدادات الأساسية للنظام.
يجب تعيين المتغيرات البيئية قبل تشغيل النظام.
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum


class TradingMode(Enum):
    """أوضاع التداول المتاحة"""
    LIVE = "live"           # تداول حقيقي
    PAPER = "paper"         # تداول ورقي (محاكاة)
    BACKTEST = "backtest"   # اختبار خلفي


class RiskLevel(Enum):
    """مستويات المخاطرة"""
    CONSERVATIVE = "conservative"   # محافظ
    MODERATE = "moderate"           # معتدل
    AGGRESSIVE = "aggressive"       # عدواني


@dataclass
class BinanceConfig:
    """إعدادات Binance"""
    api_key: str = field(default_factory=lambda: os.getenv("BINANCE_API_KEY", ""))
    api_secret: str = field(default_factory=lambda: os.getenv("BINANCE_API_SECRET", ""))
    testnet: bool = False  # التداول على الشبكة الحقيقية
    base_url: str = "https://api.binance.com"
    ws_url: str = "wss://stream.binance.com:9443/ws"
    
    def __post_init__(self):
        if not self.api_key or not self.api_secret:
            raise ValueError("يجب تعيين BINANCE_API_KEY و BINANCE_API_SECRET في متغيرات البيئة")


@dataclass
class OpenAIConfig:
    """إعدادات OpenAI للـ LLM"""
    api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    model: str = "gpt-4-turbo-preview"
    temperature: float = 0.7
    max_tokens: int = 4096
    
    def __post_init__(self):
        if not self.api_key:
            raise ValueError("يجب تعيين OPENAI_API_KEY في متغيرات البيئة")


@dataclass
class TradingConfig:
    """إعدادات التداول"""
    mode: TradingMode = TradingMode.LIVE
    quote_currency: str = "USDT"
    
    # أهم 100 عملة على Binance مقابل USDT
    target_symbols: List[str] = field(default_factory=lambda: [
        "BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT",
        "DOGEUSDT", "SOLUSDT", "DOTUSDT", "MATICUSDT", "LTCUSDT",
        "SHIBUSDT", "TRXUSDT", "AVAXUSDT", "LINKUSDT", "ATOMUSDT",
        "UNIUSDT", "ETCUSDT", "XLMUSDT", "BCHUSDT", "APTUSDT",
        "FILUSDT", "LDOUSDT", "ARBUSDT", "OPUSDT", "NEARUSDT",
        "VETUSDT", "ICPUSDT", "QNTUSDT", "AAVEUSDT", "GRTUSDT",
        "ALGOUSDT", "FTMUSDT", "SANDUSDT", "MANAUSDT", "AXSUSDT",
        "THETAUSDT", "EGLDUSDT", "XTZUSDT", "EOSUSDT", "FLOWUSDT",
        "CHZUSDT", "LRCUSDT", "MKRUSDT", "SNXUSDT", "APEUSDT",
        "CRVUSDT", "ENJUSDT", "BATUSDT", "COMPUSDT", "YFIUSDT",
        "ZECUSDT", "DASHUSDT", "NEOUSDT", "WAVESUSDT", "KSMUSDT",
        "ZILUSDT", "RUNEUSDT", "KAVAUSDT", "ANKRUSDT", "1INCHUSDT",
        "GALAUSDT", "ROSEUSDT", "CELOUSDT", "IOTAUSDT", "ONTUSDT",
        "SUSHIUSDT", "ZENUSDT", "HOTUSDT", "RVNUSDT", "STXUSDT",
        "HBARUSDT", "MINAUSDT", "OCEANUSDT", "SKLUSDT", "IMXUSDT",
        "GMXUSDT", "INJUSDT", "CFXUSDT", "AGIXUSDT", "FETUSDT",
        "RNDRUSDT", "MASKUSDT", "HIGHUSDT", "MAGICUSDT", "WOOUSDT",
        "SSVUSDT", "BLURUSDT", "SUIUSDT", "PEPEUSDT", "FLOKIUSDT",
        "WLDUSDT", "SEIUSDT", "CYBERUSDT", "ARKMUSDT", "PENDLEUSDT",
        "TIAUSDT", "JUPUSDT", "STRKUSDT", "DYMUSDT", "PIXELUSDT"
    ])
    
    # إعدادات الصفقات
    max_open_trades: int = 10
    min_trade_amount_usdt: float = 10.0
    max_trade_amount_usdt: float = 1000.0
    
    # الفترات الزمنية للتحليل
    timeframes: List[str] = field(default_factory=lambda: ["1m", "5m", "15m", "1h", "4h", "1d"])
    primary_timeframe: str = "15m"


@dataclass
class RiskConfig:
    """إعدادات إدارة المخاطر"""
    risk_level: RiskLevel = RiskLevel.MODERATE
    
    # نسب المخاطرة
    max_portfolio_risk: float = 0.02      # 2% من المحفظة كحد أقصى للمخاطرة
    max_single_trade_risk: float = 0.01   # 1% من المحفظة لكل صفقة
    max_daily_loss: float = 0.05          # 5% حد الخسارة اليومية
    max_drawdown: float = 0.15            # 15% حد السحب الأقصى
    
    # وقف الخسارة وجني الأرباح
    default_stop_loss: float = 0.02       # 2% وقف خسارة افتراضي
    default_take_profit: float = 0.04     # 4% جني أرباح افتراضي
    trailing_stop: bool = True
    trailing_stop_distance: float = 0.015  # 1.5% مسافة الوقف المتحرك
    
    # Kelly Criterion
    use_kelly_criterion: bool = True
    kelly_fraction: float = 0.25          # استخدام ربع Kelly للحذر


@dataclass
class ModelConfig:
    """إعدادات النماذج"""
    # نماذج DRL
    drl_models: List[str] = field(default_factory=lambda: ["A2C", "PPO", "SAC", "TD3"])
    active_drl_model: str = "PPO"
    
    # إعدادات التدريب
    learning_rate: float = 0.0003
    batch_size: int = 64
    buffer_size: int = 100000
    gamma: float = 0.99
    tau: float = 0.005
    
    # Ensemble
    use_ensemble: bool = True
    ensemble_voting: str = "weighted"  # weighted, majority, average


@dataclass
class AutoLearningConfig:
    """إعدادات التعلم التلقائي"""
    enabled: bool = True
    
    # فترات التدريب
    retrain_interval_hours: int = 24      # إعادة التدريب كل 24 ساعة
    evaluation_interval_hours: int = 6    # تقييم الأداء كل 6 ساعات
    
    # معايير إعادة التدريب
    min_trades_for_retrain: int = 100     # الحد الأدنى من الصفقات قبل إعادة التدريب
    performance_threshold: float = -0.05   # إعادة التدريب إذا انخفض الأداء 5%
    
    # Hyperparameter Optimization
    use_hyperopt: bool = True
    hyperopt_trials: int = 100
    hyperopt_timeout_hours: int = 4


@dataclass
class MemoryConfig:
    """إعدادات الذاكرة"""
    # أنواع الذاكرة
    short_term_capacity: int = 1000       # سعة الذاكرة قصيرة المدى
    working_memory_capacity: int = 100    # سعة الذاكرة العاملة
    long_term_enabled: bool = True
    
    # قاعدة البيانات
    database_path: str = "data/memory.db"
    
    # التنظيف
    cleanup_interval_hours: int = 24
    max_memory_age_days: int = 90


@dataclass
class AgentConfig:
    """إعدادات الوكلاء"""
    # فريق المحللين
    analysts: List[str] = field(default_factory=lambda: [
        "fundamental", "technical", "sentiment", "news", "onchain"
    ])
    
    # فريق الباحثين
    researchers: List[str] = field(default_factory=lambda: ["bullish", "bearish"])
    
    # إعدادات المناظرات
    max_debate_rounds: int = 3
    consensus_threshold: float = 0.7      # نسبة الإجماع المطلوبة
    
    # أوزان الوكلاء
    analyst_weights: Dict[str, float] = field(default_factory=lambda: {
        "fundamental": 0.20,
        "technical": 0.30,
        "sentiment": 0.15,
        "news": 0.15,
        "onchain": 0.20
    })


@dataclass
class SystemConfig:
    """الإعدادات العامة للنظام"""
    binance: BinanceConfig = field(default_factory=BinanceConfig)
    openai: OpenAIConfig = field(default_factory=OpenAIConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    auto_learning: AutoLearningConfig = field(default_factory=AutoLearningConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    
    # إعدادات التسجيل
    log_level: str = "INFO"
    log_to_file: bool = True
    log_path: str = "logs/"
    
    # إعدادات النظام
    timezone: str = "UTC"
    heartbeat_interval: int = 60          # نبضة كل 60 ثانية


def load_config() -> SystemConfig:
    """تحميل الإعدادات من متغيرات البيئة"""
    return SystemConfig()


# تصدير الإعدادات الافتراضية
DEFAULT_CONFIG = SystemConfig
