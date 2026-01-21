"""
Legendary Trading System - Data Pipeline
نظام التداول الخارق - خط أنابيب البيانات

يجمع ويحضر البيانات للتدريب والتداول.
"""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import logging
import aiohttp
from pathlib import Path
import json


@dataclass
class DataConfig:
    """إعدادات البيانات"""
    symbols: List[str] = None
    timeframes: List[str] = None
    lookback_days: int = 365
    cache_dir: str = "data_cache"
    update_interval: int = 60  # ثواني
    
    def __post_init__(self):
        if self.symbols is None:
            self.symbols = ["BTCUSDT", "ETHUSDT"]
        if self.timeframes is None:
            self.timeframes = ["1h", "4h", "1d"]


class BinanceDataFetcher:
    """جلب البيانات من Binance."""
    
    BASE_URL = "https://api.binance.com/api/v3"
    
    def __init__(self, api_key: str = "", api_secret: str = ""):
        self.api_key = api_key
        self.api_secret = api_secret
        self.logger = logging.getLogger("BinanceDataFetcher")
        self._session: Optional[aiohttp.ClientSession] = None
    
    async def initialize(self):
        """تهيئة الجلسة."""
        self._session = aiohttp.ClientSession()
    
    async def close(self):
        """إغلاق الجلسة."""
        if self._session:
            await self._session.close()
    
    async def get_klines(self, symbol: str, interval: str,
                        limit: int = 1000,
                        start_time: Optional[int] = None,
                        end_time: Optional[int] = None) -> pd.DataFrame:
        """
        جلب بيانات الشموع.
        
        Args:
            symbol: رمز العملة
            interval: الإطار الزمني
            limit: عدد الشموع
            start_time: وقت البداية (milliseconds)
            end_time: وقت النهاية (milliseconds)
            
        Returns:
            DataFrame بالبيانات
        """
        url = f"{self.BASE_URL}/klines"
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        }
        
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time
        
        try:
            async with self._session.get(url, params=params) as response:
                data = await response.json()
                
                if isinstance(data, list):
                    df = pd.DataFrame(data, columns=[
                        "open_time", "open", "high", "low", "close", "volume",
                        "close_time", "quote_volume", "trades", "taker_buy_base",
                        "taker_buy_quote", "ignore"
                    ])
                    
                    # تحويل الأنواع
                    for col in ["open", "high", "low", "close", "volume", "quote_volume"]:
                        df[col] = df[col].astype(float)
                    
                    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
                    df.set_index("open_time", inplace=True)
                    
                    return df
                else:
                    self.logger.error(f"خطأ في البيانات: {data}")
                    return pd.DataFrame()
                    
        except Exception as e:
            self.logger.error(f"خطأ في جلب البيانات: {e}")
            return pd.DataFrame()
    
    async def get_historical_klines(self, symbol: str, interval: str,
                                   days: int = 365) -> pd.DataFrame:
        """جلب بيانات تاريخية."""
        all_data = []
        
        # حساب الأوقات
        end_time = int(datetime.utcnow().timestamp() * 1000)
        start_time = int((datetime.utcnow() - timedelta(days=days)).timestamp() * 1000)
        
        current_start = start_time
        
        while current_start < end_time:
            df = await self.get_klines(
                symbol, interval,
                limit=1000,
                start_time=current_start,
                end_time=end_time
            )
            
            if df.empty:
                break
            
            all_data.append(df)
            
            # تحديث وقت البداية
            last_time = int(df.index[-1].timestamp() * 1000)
            current_start = last_time + 1
            
            # تأخير لتجنب حدود API
            await asyncio.sleep(0.1)
        
        if all_data:
            return pd.concat(all_data).drop_duplicates()
        return pd.DataFrame()
    
    async def get_ticker_24h(self, symbol: str = None) -> Dict:
        """جلب بيانات 24 ساعة."""
        url = f"{self.BASE_URL}/ticker/24hr"
        params = {}
        if symbol:
            params["symbol"] = symbol
        
        try:
            async with self._session.get(url, params=params) as response:
                return await response.json()
        except Exception as e:
            self.logger.error(f"خطأ في جلب ticker: {e}")
            return {}
    
    async def get_top_symbols(self, quote: str = "USDT", 
                             limit: int = 100) -> List[str]:
        """الحصول على أهم العملات."""
        tickers = await self.get_ticker_24h()
        
        if isinstance(tickers, list):
            # تصفية بالعملة الأساسية
            filtered = [
                t for t in tickers
                if t["symbol"].endswith(quote)
            ]
            
            # ترتيب بحجم التداول
            sorted_tickers = sorted(
                filtered,
                key=lambda x: float(x.get("quoteVolume", 0)),
                reverse=True
            )
            
            return [t["symbol"] for t in sorted_tickers[:limit]]
        
        return []


class FeatureEngineer:
    """هندسة الميزات للتدريب."""
    
    def __init__(self):
        self.logger = logging.getLogger("FeatureEngineer")
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        إنشاء ميزات للتدريب.
        
        Args:
            df: DataFrame بالبيانات الخام
            
        Returns:
            DataFrame بالميزات
        """
        features = pd.DataFrame(index=df.index)
        
        # الأسعار الأساسية
        features["price"] = df["close"]
        features["returns"] = df["close"].pct_change()
        features["log_returns"] = np.log(df["close"] / df["close"].shift(1))
        
        # المتوسطات المتحركة
        for period in [7, 14, 21, 50, 100, 200]:
            features[f"sma_{period}"] = df["close"].rolling(period).mean()
            features[f"ema_{period}"] = df["close"].ewm(span=period).mean()
        
        # نسب المتوسطات
        features["price_sma_7_ratio"] = df["close"] / features["sma_7"]
        features["price_sma_21_ratio"] = df["close"] / features["sma_21"]
        features["sma_7_21_ratio"] = features["sma_7"] / features["sma_21"]
        
        # RSI
        features["rsi_14"] = self._calculate_rsi(df["close"], 14)
        features["rsi_7"] = self._calculate_rsi(df["close"], 7)
        
        # MACD
        macd, signal, hist = self._calculate_macd(df["close"])
        features["macd"] = macd
        features["macd_signal"] = signal
        features["macd_hist"] = hist
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = self._calculate_bollinger(df["close"])
        features["bb_upper"] = bb_upper
        features["bb_middle"] = bb_middle
        features["bb_lower"] = bb_lower
        features["bb_width"] = (bb_upper - bb_lower) / bb_middle
        features["bb_position"] = (df["close"] - bb_lower) / (bb_upper - bb_lower)
        
        # ATR
        features["atr_14"] = self._calculate_atr(df, 14)
        features["atr_ratio"] = features["atr_14"] / df["close"]
        
        # الحجم
        features["volume"] = df["volume"]
        features["volume_sma_20"] = df["volume"].rolling(20).mean()
        features["volume_ratio"] = df["volume"] / features["volume_sma_20"]
        
        # التقلب
        features["volatility_20"] = features["returns"].rolling(20).std()
        features["volatility_50"] = features["returns"].rolling(50).std()
        
        # الزخم
        for period in [5, 10, 20]:
            features[f"momentum_{period}"] = df["close"] / df["close"].shift(period) - 1
        
        # أنماط الشموع
        features["body_size"] = abs(df["close"] - df["open"]) / df["open"]
        features["upper_shadow"] = (df["high"] - df[["open", "close"]].max(axis=1)) / df["open"]
        features["lower_shadow"] = (df[["open", "close"]].min(axis=1) - df["low"]) / df["open"]
        features["is_bullish"] = (df["close"] > df["open"]).astype(int)
        
        # High/Low
        for period in [5, 10, 20, 50]:
            features[f"high_{period}"] = df["high"].rolling(period).max()
            features[f"low_{period}"] = df["low"].rolling(period).min()
            features[f"price_high_ratio_{period}"] = df["close"] / features[f"high_{period}"]
            features[f"price_low_ratio_{period}"] = df["close"] / features[f"low_{period}"]
        
        # تنظيف القيم المفقودة
        features = features.fillna(method="ffill").fillna(0)
        
        return features
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """حساب RSI."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices: pd.Series,
                       fast: int = 12, slow: int = 26,
                       signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """حساب MACD."""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal).mean()
        histogram = macd - signal_line
        return macd, signal_line, histogram
    
    def _calculate_bollinger(self, prices: pd.Series,
                            period: int = 20,
                            std_dev: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """حساب Bollinger Bands."""
        middle = prices.rolling(period).mean()
        std = prices.rolling(period).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        return upper, middle, lower
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """حساب ATR."""
        high_low = df["high"] - df["low"]
        high_close = abs(df["high"] - df["close"].shift())
        low_close = abs(df["low"] - df["close"].shift())
        
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        return atr
    
    def normalize_features(self, features: pd.DataFrame,
                          method: str = "zscore") -> pd.DataFrame:
        """
        تطبيع الميزات.
        
        Args:
            features: DataFrame بالميزات
            method: طريقة التطبيع (zscore, minmax)
            
        Returns:
            DataFrame بالميزات المطبعة
        """
        if method == "zscore":
            return (features - features.mean()) / features.std()
        elif method == "minmax":
            return (features - features.min()) / (features.max() - features.min())
        else:
            return features


class DataPipeline:
    """
    خط أنابيب البيانات الكامل.
    
    يدير:
    - جلب البيانات
    - التخزين المؤقت
    - هندسة الميزات
    - تحضير البيانات للتدريب
    """
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.logger = logging.getLogger("DataPipeline")
        
        self.fetcher = BinanceDataFetcher()
        self.engineer = FeatureEngineer()
        
        self.cache_dir = Path(config.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self._data_cache: Dict[str, pd.DataFrame] = {}
    
    async def initialize(self):
        """تهيئة خط الأنابيب."""
        await self.fetcher.initialize()
        self.logger.info("تم تهيئة خط أنابيب البيانات")
    
    async def close(self):
        """إغلاق خط الأنابيب."""
        await self.fetcher.close()
    
    async def get_training_data(self, symbol: str,
                               timeframe: str = "1h") -> np.ndarray:
        """
        الحصول على بيانات التدريب.
        
        Args:
            symbol: رمز العملة
            timeframe: الإطار الزمني
            
        Returns:
            مصفوفة numpy بالميزات
        """
        # التحقق من التخزين المؤقت
        cache_key = f"{symbol}_{timeframe}"
        cache_file = self.cache_dir / f"{cache_key}.parquet"
        
        if cache_file.exists():
            # التحقق من حداثة البيانات
            mtime = datetime.fromtimestamp(cache_file.stat().st_mtime)
            if datetime.utcnow() - mtime < timedelta(hours=1):
                self.logger.info(f"استخدام البيانات المخزنة لـ {cache_key}")
                df = pd.read_parquet(cache_file)
                return self._prepare_for_training(df)
        
        # جلب بيانات جديدة
        self.logger.info(f"جلب بيانات جديدة لـ {symbol}")
        raw_data = await self.fetcher.get_historical_klines(
            symbol, timeframe, self.config.lookback_days
        )
        
        if raw_data.empty:
            self.logger.error(f"فشل جلب البيانات لـ {symbol}")
            return np.array([])
        
        # هندسة الميزات
        features = self.engineer.create_features(raw_data)
        
        # التطبيع
        normalized = self.engineer.normalize_features(features)
        
        # حفظ في التخزين المؤقت
        normalized.to_parquet(cache_file)
        
        return self._prepare_for_training(normalized)
    
    async def get_recent_data(self, symbol: str,
                             timeframe: str = "1h",
                             periods: int = 100) -> np.ndarray:
        """الحصول على بيانات حديثة."""
        raw_data = await self.fetcher.get_klines(symbol, timeframe, limit=periods)
        
        if raw_data.empty:
            return np.array([])
        
        features = self.engineer.create_features(raw_data)
        normalized = self.engineer.normalize_features(features)
        
        return self._prepare_for_training(normalized)
    
    async def get_multi_symbol_data(self, symbols: List[str] = None,
                                   timeframe: str = "1h") -> Dict[str, np.ndarray]:
        """الحصول على بيانات لعدة عملات."""
        if symbols is None:
            symbols = self.config.symbols
        
        data = {}
        for symbol in symbols:
            data[symbol] = await self.get_training_data(symbol, timeframe)
            await asyncio.sleep(0.1)  # تجنب حدود API
        
        return data
    
    def _prepare_for_training(self, df: pd.DataFrame) -> np.ndarray:
        """تحضير البيانات للتدريب."""
        # إزالة القيم المفقودة
        df = df.dropna()
        
        # تحويل إلى numpy
        return df.values.astype(np.float32)
    
    def split_data(self, data: np.ndarray,
                  train_ratio: float = 0.8,
                  val_ratio: float = 0.1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        تقسيم البيانات.
        
        Args:
            data: البيانات
            train_ratio: نسبة التدريب
            val_ratio: نسبة التحقق
            
        Returns:
            بيانات التدريب، التحقق، الاختبار
        """
        n = len(data)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        train_data = data[:train_end]
        val_data = data[train_end:val_end]
        test_data = data[val_end:]
        
        return train_data, val_data, test_data
    
    async def update_cache(self):
        """تحديث التخزين المؤقت."""
        self.logger.info("تحديث التخزين المؤقت...")
        
        for symbol in self.config.symbols:
            for timeframe in self.config.timeframes:
                await self.get_training_data(symbol, timeframe)
                await asyncio.sleep(0.5)
        
        self.logger.info("تم تحديث التخزين المؤقت")
