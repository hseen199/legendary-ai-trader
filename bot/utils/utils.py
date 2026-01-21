"""
═══════════════════════════════════════════════════════════════
LEGENDARY AGENT - Utilities
الأدوات المساعدة
═══════════════════════════════════════════════════════════════
"""

import os
import yaml
import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import numpy as np
import pandas as pd
from loguru import logger


# ═══════════════════════════════════════════════════════════════
# CONFIGURATION - الإعدادات
# ═══════════════════════════════════════════════════════════════

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """تحميل ملف الإعدادات"""
    # البحث عن الملف في عدة مسارات
    possible_paths = [
        config_path,
        os.path.join(os.path.dirname(__file__), config_path),
        os.path.join(os.path.dirname(__file__), "..", config_path),
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                logger.info(f"✅ Config loaded from: {path}")
                return config
    
    raise FileNotFoundError(f"Config file not found in: {possible_paths}")


def save_config(config: Dict[str, Any], config_path: str = "config.yaml") -> None:
    """حفظ ملف الإعدادات"""
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    logger.info(f"✅ Config saved to: {config_path}")


# ═══════════════════════════════════════════════════════════════
# DATA PROCESSING - معالجة البيانات
# ═══════════════════════════════════════════════════════════════

def normalize_data(data: np.ndarray, method: str = "minmax") -> np.ndarray:
    """تطبيع البيانات"""
    if method == "minmax":
        min_val = np.min(data, axis=0)
        max_val = np.max(data, axis=0)
        range_val = max_val - min_val
        range_val[range_val == 0] = 1  # تجنب القسمة على صفر
        return (data - min_val) / range_val
    elif method == "zscore":
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        std[std == 0] = 1  # تجنب القسمة على صفر
        return (data - mean) / std
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def denormalize_data(
    normalized: np.ndarray, 
    original_min: np.ndarray, 
    original_max: np.ndarray
) -> np.ndarray:
    """إلغاء تطبيع البيانات"""
    return normalized * (original_max - original_min) + original_min


def calculate_returns(prices: np.ndarray, log_returns: bool = False) -> np.ndarray:
    """حساب العوائد"""
    if log_returns:
        return np.log(prices[1:] / prices[:-1])
    else:
        return (prices[1:] - prices[:-1]) / prices[:-1]


def calculate_volatility(returns: np.ndarray, window: int = 20) -> np.ndarray:
    """حساب التقلب"""
    if len(returns) < window:
        return np.array([np.std(returns)])
    
    volatility = []
    for i in range(len(returns) - window + 1):
        vol = np.std(returns[i:i+window]) * np.sqrt(252)  # سنوي
        volatility.append(vol)
    
    return np.array(volatility)


def resample_ohlcv(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """إعادة تجميع بيانات OHLCV لإطار زمني مختلف"""
    timeframe_map = {
        '1m': '1min', '5m': '5min', '15m': '15min',
        '30m': '30min', '1h': '1H', '4h': '4H',
        '1d': '1D', '1w': '1W'
    }
    
    if timeframe not in timeframe_map:
        raise ValueError(f"Unknown timeframe: {timeframe}")
    
    rule = timeframe_map[timeframe]
    
    resampled = df.resample(rule).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    return resampled


# ═══════════════════════════════════════════════════════════════
# FEATURE ENGINEERING - هندسة الميزات
# ═══════════════════════════════════════════════════════════════

def create_sequences(
    data: np.ndarray, 
    sequence_length: int,
    target_column: int = -1,
    prediction_horizon: int = 1
) -> tuple:
    """إنشاء تسلسلات للتدريب"""
    X, y = [], []
    
    for i in range(len(data) - sequence_length - prediction_horizon + 1):
        X.append(data[i:i+sequence_length])
        y.append(data[i+sequence_length+prediction_horizon-1, target_column])
    
    return np.array(X), np.array(y)


def create_labels(
    prices: np.ndarray, 
    threshold: float = 0.01,
    horizon: int = 1
) -> np.ndarray:
    """إنشاء تصنيفات (شراء/بيع/انتظار)"""
    labels = []
    
    for i in range(len(prices) - horizon):
        future_return = (prices[i + horizon] - prices[i]) / prices[i]
        
        if future_return > threshold:
            labels.append(0)  # BUY
        elif future_return < -threshold:
            labels.append(1)  # SELL
        else:
            labels.append(2)  # HOLD
    
    return np.array(labels)


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """إضافة ميزات الوقت"""
    df = df.copy()
    
    if 'timestamp' in df.columns:
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['day_of_month'] = df['timestamp'].dt.day
        df['month'] = df['timestamp'].dt.month
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # ترميز دائري للساعة
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        # ترميز دائري ليوم الأسبوع
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    return df


# ═══════════════════════════════════════════════════════════════
# TECHNICAL ANALYSIS HELPERS - مساعدات التحليل الفني
# ═══════════════════════════════════════════════════════════════

def calculate_sma(prices: np.ndarray, period: int) -> np.ndarray:
    """حساب المتوسط المتحرك البسيط"""
    if len(prices) < period:
        return np.full(len(prices), np.nan)
    
    sma = np.convolve(prices, np.ones(period)/period, mode='valid')
    padding = np.full(period - 1, np.nan)
    return np.concatenate([padding, sma])


def calculate_ema(prices: np.ndarray, period: int) -> np.ndarray:
    """حساب المتوسط المتحرك الأسي"""
    if len(prices) < period:
        return np.full(len(prices), np.nan)
    
    multiplier = 2 / (period + 1)
    ema = np.zeros(len(prices))
    ema[:period] = np.nan
    ema[period-1] = np.mean(prices[:period])
    
    for i in range(period, len(prices)):
        ema[i] = (prices[i] - ema[i-1]) * multiplier + ema[i-1]
    
    return ema


def calculate_rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
    """حساب مؤشر القوة النسبية"""
    if len(prices) < period + 1:
        return np.full(len(prices), 50.0)
    
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    avg_gain = np.zeros(len(prices))
    avg_loss = np.zeros(len(prices))
    
    avg_gain[period] = np.mean(gains[:period])
    avg_loss[period] = np.mean(losses[:period])
    
    for i in range(period + 1, len(prices)):
        avg_gain[i] = (avg_gain[i-1] * (period - 1) + gains[i-1]) / period
        avg_loss[i] = (avg_loss[i-1] * (period - 1) + losses[i-1]) / period
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    rsi[:period] = np.nan
    
    return rsi


def calculate_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """حساب متوسط المدى الحقيقي"""
    if len(high) < 2:
        return np.array([0])
    
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[tr[0]], tr])
    
    atr = calculate_ema(tr, period)
    return atr


# ═══════════════════════════════════════════════════════════════
# RISK CALCULATIONS - حسابات المخاطر
# ═══════════════════════════════════════════════════════════════

def calculate_position_size(
    portfolio_value: float,
    risk_percent: float,
    entry_price: float,
    stop_loss_price: float,
    max_position_percent: float = 15.0
) -> float:
    """حساب حجم الصفقة بناءً على المخاطرة"""
    risk_amount = portfolio_value * (risk_percent / 100)
    price_risk = abs(entry_price - stop_loss_price)
    
    if price_risk == 0:
        return 0
    
    position_size = risk_amount / price_risk
    position_value = position_size * entry_price
    
    # تطبيق الحد الأقصى
    max_position_value = portfolio_value * (max_position_percent / 100)
    if position_value > max_position_value:
        position_size = max_position_value / entry_price
    
    return position_size


def calculate_kelly_criterion(win_rate: float, avg_win: float, avg_loss: float) -> float:
    """حساب معيار كيلي لحجم الصفقة الأمثل"""
    if avg_loss == 0:
        return 0
    
    win_loss_ratio = avg_win / abs(avg_loss)
    kelly = win_rate - ((1 - win_rate) / win_loss_ratio)
    
    # تطبيق نصف كيلي للأمان
    return max(0, kelly * 0.5)


def calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
    """حساب نسبة شارب"""
    if len(returns) == 0 or np.std(returns) == 0:
        return 0
    
    excess_returns = np.mean(returns) - (risk_free_rate / 252)
    return (excess_returns / np.std(returns)) * np.sqrt(252)


def calculate_max_drawdown(equity_curve: np.ndarray) -> float:
    """حساب أقصى سحب"""
    if len(equity_curve) == 0:
        return 0
    
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - peak) / peak
    return np.min(drawdown) * 100


# ═══════════════════════════════════════════════════════════════
# FILE & LOGGING - الملفات والسجلات
# ═══════════════════════════════════════════════════════════════

def setup_logging(log_file: str = "logs/legendary_agent.log", level: str = "INFO"):
    """إعداد نظام السجلات"""
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    logger.add(
        log_file,
        rotation="10 MB",
        retention="7 days",
        level=level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}"
    )
    
    return logger


def save_json(data: Any, filepath: str) -> None:
    """حفظ بيانات JSON"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


def load_json(filepath: str) -> Any:
    """تحميل بيانات JSON"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def generate_id(data: str) -> str:
    """توليد معرف فريد"""
    return hashlib.md5(f"{data}{datetime.now().isoformat()}".encode()).hexdigest()[:12]


# ═══════════════════════════════════════════════════════════════
# TIME UTILITIES - أدوات الوقت
# ═══════════════════════════════════════════════════════════════

def get_current_timestamp() -> datetime:
    """الحصول على الوقت الحالي"""
    return datetime.utcnow()


def timestamp_to_ms(dt: datetime) -> int:
    """تحويل التاريخ إلى ميلي ثانية"""
    return int(dt.timestamp() * 1000)


def ms_to_timestamp(ms: int) -> datetime:
    """تحويل ميلي ثانية إلى تاريخ"""
    return datetime.fromtimestamp(ms / 1000)


def get_timeframe_minutes(timeframe: str) -> int:
    """الحصول على دقائق الإطار الزمني"""
    mapping = {
        '1m': 1, '3m': 3, '5m': 5, '15m': 15, '30m': 30,
        '1h': 60, '2h': 120, '4h': 240, '6h': 360, '8h': 480,
        '12h': 720, '1d': 1440, '3d': 4320, '1w': 10080
    }
    return mapping.get(timeframe, 60)


# ═══════════════════════════════════════════════════════════════
# VALIDATION - التحقق
# ═══════════════════════════════════════════════════════════════

def validate_ohlcv(df: pd.DataFrame) -> tuple:
    """التحقق من صحة بيانات OHLCV"""
    errors = []
    warnings = []
    
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    for col in required_columns:
        if col not in df.columns:
            errors.append(f"Missing column: {col}")
    
    if errors:
        return False, errors, warnings
    
    # التحقق من القيم
    if (df['high'] < df['low']).any():
        errors.append("High < Low in some rows")
    
    if (df['high'] < df['open']).any() or (df['high'] < df['close']).any():
        warnings.append("High < Open or Close in some rows")
    
    if (df['low'] > df['open']).any() or (df['low'] > df['close']).any():
        warnings.append("Low > Open or Close in some rows")
    
    if (df['volume'] < 0).any():
        errors.append("Negative volume detected")
    
    if df.isnull().any().any():
        warnings.append("NaN values detected")
    
    return len(errors) == 0, errors, warnings


def validate_symbol(symbol: str) -> bool:
    """التحقق من صحة رمز العملة"""
    if not symbol or not isinstance(symbol, str):
        return False
    
    # يجب أن ينتهي بـ USDT أو BUSD أو BTC
    valid_quotes = ['USDT', 'BUSD', 'BTC', 'ETH']
    return any(symbol.endswith(quote) for quote in valid_quotes)
