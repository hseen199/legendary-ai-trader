"""
═══════════════════════════════════════════════════════════════
LEGENDARY AGENT - Data Collector
جلب البيانات من Hugging Face و Binance API
═══════════════════════════════════════════════════════════════
"""

import os
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from loguru import logger
from datasets import load_dataset
from tqdm import tqdm


class DataCollector:
    """
    جامع البيانات من مصادر متعددة
    - Hugging Face (duonlabs/apogee)
    - Binance Public API
    """
    
    # أفضل 100 عملة للتداول
    TOP_100_SYMBOLS = [
        "BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT",
        "DOGEUSDT", "SOLUSDT", "DOTUSDT", "MATICUSDT", "LTCUSDT",
        "SHIBUSDT", "TRXUSDT", "AVAXUSDT", "LINKUSDT", "ATOMUSDT",
        "UNIUSDT", "ETCUSDT", "XLMUSDT", "BCHUSDT", "FILUSDT",
        "LDOUSDT", "APTUSDT", "ARBUSDT", "OPUSDT", "NEARUSDT",
        "ICPUSDT", "VETUSDT", "HBARUSDT", "QNTUSDT", "AAVEUSDT",
        "GRTUSDT", "ALGOUSDT", "FTMUSDT", "SANDUSDT", "MANAUSDT",
        "AXSUSDT", "THETAUSDT", "EGLDUSDT", "EOSUSDT", "XTZUSDT",
        "FLOWUSDT", "CHZUSDT", "MKRUSDT", "SNXUSDT", "NEOUSDT",
        "RNDRUSDT", "KAVAUSDT", "MINAUSDT", "XMRUSDT", "BTCUSDT",
        "RUNEUSDT", "ZILUSDT", "ENJUSDT", "BATUSDT", "CRVUSDT",
        "LRCUSDT", "COMPUSDT", "YFIUSDT", "1INCHUSDT", "ANKRUSDT",
        "KSMUSDT", "DASHUSDT", "ZECUSDT", "WAVESUSDT", "IOSTUSDT",
        "ONTUSDT", "HOTUSDT", "ZENUSDT", "COTIUSDT", "SCUSDT",
        "DGBUSDT", "ICXUSDT", "RVNUSDT", "STXUSDT", "IOTAUSDT",
        "CELRUSDT", "CKBUSDT", "SXPUSDT", "RENUSDT", "OCEANUSDT",
        "RSRUSDT", "BLZUSDT", "CVCUSDT", "STMXUSDT", "DUSKUSDT",
        "ARUSDT", "CTSIUSDT", "MTLUSDT", "OGNUSDT", "NKNUSDT",
        "REEFUSDT", "LITUSDT", "SFPUSDT", "TLMUSDT", "ALICEUSDT",
        "LINAUSDT", "PERPUSDT", "RAREUSDT", "HIGHUSDT", "WLDUSDT"
    ]
    
    BINANCE_BASE_URL = "https://api.binance.com"
    
    def __init__(self, data_dir: str = "data/raw"):
        """
        تهيئة جامع البيانات
        
        Args:
            data_dir: مجلد حفظ البيانات
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"📊 DataCollector initialized. Data dir: {self.data_dir}")
    
    # ═══════════════════════════════════════════════════════════════
    # HUGGING FACE DATA
    # ═══════════════════════════════════════════════════════════════
    
    def load_from_huggingface(
        self, 
        symbols: Optional[List[str]] = None,
        max_symbols: int = 100
    ) -> Dict[str, pd.DataFrame]:
        """
        تحميل البيانات من Hugging Face
        
        Args:
            symbols: قائمة العملات (اختياري)
            max_symbols: الحد الأقصى للعملات
            
        Returns:
            قاموس بالبيانات لكل عملة
        """
        logger.info("📥 Loading data from Hugging Face (duonlabs/apogee)...")
        
        try:
            # تحميل مجموعة البيانات
            dataset = load_dataset("duonlabs/apogee", "binance", split="train")
            logger.info(f"✅ Dataset loaded: {len(dataset)} records")
            
            # تحويل إلى DataFrame
            df = dataset.to_pandas()
            logger.info(f"📊 Columns: {df.columns.tolist()}")
            
            # تحديد العملات المتاحة
            if 'symbol' in df.columns:
                available_symbols = df['symbol'].unique().tolist()
            else:
                # محاولة استخراج الرمز من اسم الملف أو عمود آخر
                available_symbols = self.TOP_100_SYMBOLS[:max_symbols]
            
            logger.info(f"📊 Available symbols: {len(available_symbols)}")
            
            # تصفية العملات المطلوبة
            if symbols:
                target_symbols = [s for s in symbols if s in available_symbols]
            else:
                target_symbols = available_symbols[:max_symbols]
            
            # تنظيم البيانات حسب العملة
            data_dict = {}
            
            if 'symbol' in df.columns:
                for symbol in tqdm(target_symbols, desc="Processing symbols"):
                    symbol_df = df[df['symbol'] == symbol].copy()
                    if len(symbol_df) > 0:
                        symbol_df = self._standardize_columns(symbol_df)
                        data_dict[symbol] = symbol_df
            else:
                # إذا كانت البيانات بتنسيق مختلف
                df = self._standardize_columns(df)
                data_dict['COMBINED'] = df
            
            logger.info(f"✅ Loaded data for {len(data_dict)} symbols")
            return data_dict
            
        except Exception as e:
            logger.error(f"❌ Error loading from HuggingFace: {e}")
            logger.info("📥 Falling back to Binance API...")
            return self.fetch_from_binance(symbols or self.TOP_100_SYMBOLS[:max_symbols])
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """توحيد أسماء الأعمدة"""
        column_mapping = {
            'Open': 'open', 'High': 'high', 'Low': 'low', 
            'Close': 'close', 'Volume': 'volume',
            'open_time': 'timestamp', 'close_time': 'close_timestamp',
            'Timestamp': 'timestamp', 'Date': 'timestamp',
            'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'
        }
        
        df = df.rename(columns=column_mapping)
        
        # التأكد من وجود الأعمدة الأساسية
        required = ['open', 'high', 'low', 'close', 'volume']
        for col in required:
            if col not in df.columns:
                logger.warning(f"⚠️ Missing column: {col}")
        
        # تحويل timestamp إلى datetime
        if 'timestamp' in df.columns:
            if df['timestamp'].dtype in ['int64', 'float64']:
                # تحويل من milliseconds
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            else:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            df = df.set_index('timestamp').sort_index()
        
        # تحويل الأعمدة الرقمية
        for col in required:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    
    # ═══════════════════════════════════════════════════════════════
    # BINANCE API DATA
    # ═══════════════════════════════════════════════════════════════
    
    def fetch_from_binance(
        self,
        symbols: List[str],
        interval: str = "1h",
        days: int = 180
    ) -> Dict[str, pd.DataFrame]:
        """
        جلب البيانات من Binance API
        
        Args:
            symbols: قائمة العملات
            interval: الإطار الزمني
            days: عدد الأيام
            
        Returns:
            قاموس بالبيانات لكل عملة
        """
        logger.info(f"📥 Fetching data from Binance API for {len(symbols)} symbols...")
        
        data_dict = {}
        
        for symbol in tqdm(symbols, desc="Fetching from Binance"):
            try:
                df = self._fetch_klines(symbol, interval, days)
                if df is not None and len(df) > 0:
                    data_dict[symbol] = df
                    time.sleep(0.1)  # تجنب تجاوز حد الطلبات
            except Exception as e:
                logger.warning(f"⚠️ Failed to fetch {symbol}: {e}")
                continue
        
        logger.info(f"✅ Fetched data for {len(data_dict)} symbols")
        return data_dict
    
    def _fetch_klines(
        self, 
        symbol: str, 
        interval: str = "1h",
        days: int = 180
    ) -> Optional[pd.DataFrame]:
        """
        جلب بيانات الشموع لعملة واحدة
        
        Args:
            symbol: رمز العملة
            interval: الإطار الزمني
            days: عدد الأيام
            
        Returns:
            DataFrame بالبيانات
        """
        endpoint = f"{self.BINANCE_BASE_URL}/api/v3/klines"
        
        # حساب الوقت
        end_time = int(datetime.now().timestamp() * 1000)
        start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
        
        all_data = []
        current_start = start_time
        
        while current_start < end_time:
            params = {
                "symbol": symbol,
                "interval": interval,
                "startTime": current_start,
                "endTime": end_time,
                "limit": 1000
            }
            
            try:
                response = requests.get(endpoint, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()
                
                if not data:
                    break
                
                all_data.extend(data)
                current_start = data[-1][0] + 1
                
            except Exception as e:
                logger.warning(f"⚠️ Error fetching {symbol}: {e}")
                break
        
        if not all_data:
            return None
        
        # تحويل إلى DataFrame
        df = pd.DataFrame(all_data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        
        # تنظيف البيانات
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.set_index('timestamp')
        
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # الاحتفاظ بالأعمدة الأساسية فقط
        df = df[['open', 'high', 'low', 'close', 'volume']]
        
        return df
    
    # ═══════════════════════════════════════════════════════════════
    # MULTI-TIMEFRAME DATA
    # ═══════════════════════════════════════════════════════════════
    
    def fetch_multi_timeframe(
        self,
        symbols: List[str],
        timeframes: List[str] = ["1m", "5m", "15m", "1h", "4h"],
        days: int = 30
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        جلب بيانات متعددة الإطارات الزمنية
        
        Args:
            symbols: قائمة العملات
            timeframes: الإطارات الزمنية
            days: عدد الأيام
            
        Returns:
            قاموس متداخل {symbol: {timeframe: DataFrame}}
        """
        logger.info(f"📥 Fetching multi-timeframe data for {len(symbols)} symbols...")
        
        result = {}
        
        for symbol in tqdm(symbols, desc="Fetching symbols"):
            result[symbol] = {}
            
            for tf in timeframes:
                try:
                    df = self._fetch_klines(symbol, tf, days)
                    if df is not None:
                        result[symbol][tf] = df
                    time.sleep(0.05)
                except Exception as e:
                    logger.warning(f"⚠️ Failed {symbol} {tf}: {e}")
        
        return result
    
    # ═══════════════════════════════════════════════════════════════
    # DATA SAVING & LOADING
    # ═══════════════════════════════════════════════════════════════
    
    def save_data(
        self, 
        data_dict: Dict[str, pd.DataFrame],
        prefix: str = "ohlcv"
    ) -> None:
        """
        حفظ البيانات إلى ملفات
        
        Args:
            data_dict: قاموس البيانات
            prefix: بادئة اسم الملف
        """
        save_dir = self.data_dir / prefix
        save_dir.mkdir(parents=True, exist_ok=True)
        
        for symbol, df in data_dict.items():
            filepath = save_dir / f"{symbol}.parquet"
            df.to_parquet(filepath)
        
        logger.info(f"✅ Saved {len(data_dict)} files to {save_dir}")
    
    def load_data(
        self, 
        symbols: Optional[List[str]] = None,
        prefix: str = "ohlcv"
    ) -> Dict[str, pd.DataFrame]:
        """
        تحميل البيانات من الملفات
        
        Args:
            symbols: قائمة العملات (اختياري)
            prefix: بادئة اسم الملف
            
        Returns:
            قاموس البيانات
        """
        load_dir = self.data_dir / prefix
        
        if not load_dir.exists():
            logger.warning(f"⚠️ Directory not found: {load_dir}")
            return {}
        
        data_dict = {}
        files = list(load_dir.glob("*.parquet"))
        
        for filepath in files:
            symbol = filepath.stem
            if symbols is None or symbol in symbols:
                df = pd.read_parquet(filepath)
                data_dict[symbol] = df
        
        logger.info(f"✅ Loaded {len(data_dict)} files from {load_dir}")
        return data_dict
    
    # ═══════════════════════════════════════════════════════════════
    # TOP SYMBOLS BY VOLUME
    # ═══════════════════════════════════════════════════════════════
    
    def get_top_symbols_by_volume(
        self, 
        limit: int = 100,
        quote_asset: str = "USDT"
    ) -> List[str]:
        """
        الحصول على أفضل العملات حسب الحجم
        
        Args:
            limit: عدد العملات
            quote_asset: عملة التسعير
            
        Returns:
            قائمة رموز العملات
        """
        endpoint = f"{self.BINANCE_BASE_URL}/api/v3/ticker/24hr"
        
        try:
            response = requests.get(endpoint, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            # تصفية العملات حسب عملة التسعير
            filtered = [
                d for d in data 
                if d['symbol'].endswith(quote_asset)
            ]
            
            # ترتيب حسب الحجم
            sorted_data = sorted(
                filtered, 
                key=lambda x: float(x['quoteVolume']), 
                reverse=True
            )
            
            symbols = [d['symbol'] for d in sorted_data[:limit]]
            logger.info(f"✅ Found top {len(symbols)} symbols by volume")
            
            return symbols
            
        except Exception as e:
            logger.error(f"❌ Error getting top symbols: {e}")
            return self.TOP_100_SYMBOLS[:limit]
    
    # ═══════════════════════════════════════════════════════════════
    # DATA VALIDATION
    # ═══════════════════════════════════════════════════════════════
    
    def validate_data(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        التحقق من صحة البيانات
        
        Args:
            df: DataFrame للتحقق
            
        Returns:
            (صالح, قائمة الأخطاء)
        """
        errors = []
        
        # التحقق من الأعمدة
        required = ['open', 'high', 'low', 'close', 'volume']
        missing = [c for c in required if c not in df.columns]
        if missing:
            errors.append(f"Missing columns: {missing}")
        
        # التحقق من القيم
        if (df['high'] < df['low']).any():
            errors.append("High < Low detected")
        
        if (df['volume'] < 0).any():
            errors.append("Negative volume detected")
        
        if df.isnull().any().any():
            null_counts = df.isnull().sum()
            errors.append(f"Null values: {null_counts[null_counts > 0].to_dict()}")
        
        return len(errors) == 0, errors


# ═══════════════════════════════════════════════════════════════
# STANDALONE EXECUTION
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # اختبار الوحدة
    collector = DataCollector()
    
    # جلب البيانات
    data = collector.load_from_huggingface(max_symbols=10)
    
    if data:
        for symbol, df in list(data.items())[:3]:
            print(f"\n{symbol}:")
            print(f"  Shape: {df.shape}")
            print(f"  Columns: {df.columns.tolist()}")
            print(f"  Date range: {df.index.min()} to {df.index.max()}")
