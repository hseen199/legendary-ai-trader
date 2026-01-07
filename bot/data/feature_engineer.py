"""
═══════════════════════════════════════════════════════════════
LEGENDARY AGENT - Feature Engineer
هندسة الميزات واستخراج المؤشرات التقنية
═══════════════════════════════════════════════════════════════
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from loguru import logger
import ta
from ta.trend import SMAIndicator, EMAIndicator, MACD, ADXIndicator, IchimokuIndicator
from ta.momentum import RSIIndicator, StochasticOscillator, WilliamsRIndicator
from ta.volatility import BollingerBands, AverageTrueRange, KeltnerChannel
from ta.volume import OnBalanceVolumeIndicator, VolumeWeightedAveragePrice, MFIIndicator
import warnings
warnings.filterwarnings('ignore')


class FeatureEngineer:
    """
    مهندس الميزات
    - استخراج 50+ مؤشر تقني
    - ميزات السعر والحجم
    - ميزات الوقت
    - ميزات السوق
    """
    
    def __init__(self):
        """تهيئة مهندس الميزات"""
        self.feature_names: List[str] = []
        logger.info("🔧 FeatureEngineer initialized")
    
    # ═══════════════════════════════════════════════════════════════
    # MAIN FEATURE EXTRACTION
    # ═══════════════════════════════════════════════════════════════
    
    def extract_all_features(
        self, 
        df: pd.DataFrame,
        include_time: bool = True
    ) -> pd.DataFrame:
        """
        استخراج جميع الميزات
        
        Args:
            df: DataFrame مع OHLCV
            include_time: تضمين ميزات الوقت
            
        Returns:
            DataFrame مع جميع الميزات
        """
        logger.info(f"🔧 Extracting features from {len(df)} rows...")
        
        df = df.copy()
        
        # 1. المؤشرات التقنية
        df = self._add_trend_indicators(df)
        df = self._add_momentum_indicators(df)
        df = self._add_volatility_indicators(df)
        df = self._add_volume_indicators(df)
        
        # 2. ميزات السعر
        df = self._add_price_features(df)
        
        # 3. ميزات الوقت
        if include_time:
            df = self._add_time_features(df)
        
        # 4. ميزات مشتقة
        df = self._add_derived_features(df)
        
        # 5. تنظيف
        df = self._clean_features(df)
        
        # حفظ أسماء الميزات
        self.feature_names = df.columns.tolist()
        
        logger.info(f"✅ Extracted {len(self.feature_names)} features")
        return df
    
    # ═══════════════════════════════════════════════════════════════
    # TREND INDICATORS
    # ═══════════════════════════════════════════════════════════════
    
    def _add_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """إضافة مؤشرات الاتجاه"""
        
        close = df['close']
        high = df['high']
        low = df['low']
        
        # SMA - المتوسط المتحرك البسيط
        for period in [5, 10, 20, 50, 100, 200]:
            df[f'sma_{period}'] = SMAIndicator(close, window=period).sma_indicator()
        
        # EMA - المتوسط المتحرك الأسي
        for period in [9, 12, 21, 26, 50]:
            df[f'ema_{period}'] = EMAIndicator(close, window=period).ema_indicator()
        
        # MACD
        macd = MACD(close)
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_histogram'] = macd.macd_diff()
        
        # ADX - مؤشر الاتجاه المتوسط
        adx = ADXIndicator(high, low, close)
        df['adx'] = adx.adx()
        df['adx_pos'] = adx.adx_pos()
        df['adx_neg'] = adx.adx_neg()
        
        # Ichimoku Cloud
        try:
            ichimoku = IchimokuIndicator(high, low)
            df['ichimoku_a'] = ichimoku.ichimoku_a()
            df['ichimoku_b'] = ichimoku.ichimoku_b()
            df['ichimoku_base'] = ichimoku.ichimoku_base_line()
            df['ichimoku_conv'] = ichimoku.ichimoku_conversion_line()
        except Exception:
            pass
        
        # تقاطعات المتوسطات
        df['sma_cross_20_50'] = (df['sma_20'] > df['sma_50']).astype(int)
        df['ema_cross_9_21'] = (df['ema_9'] > df['ema_21']).astype(int)
        
        # المسافة من المتوسطات
        df['dist_from_sma_20'] = (close - df['sma_20']) / df['sma_20'] * 100
        df['dist_from_sma_50'] = (close - df['sma_50']) / df['sma_50'] * 100
        df['dist_from_sma_200'] = (close - df['sma_200']) / df['sma_200'] * 100
        
        return df
    
    # ═══════════════════════════════════════════════════════════════
    # MOMENTUM INDICATORS
    # ═══════════════════════════════════════════════════════════════
    
    def _add_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """إضافة مؤشرات الزخم"""
        
        close = df['close']
        high = df['high']
        low = df['low']
        
        # RSI - مؤشر القوة النسبية
        for period in [7, 14, 21]:
            df[f'rsi_{period}'] = RSIIndicator(close, window=period).rsi()
        
        # Stochastic Oscillator
        stoch = StochasticOscillator(high, low, close)
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()
        
        # Williams %R
        df['williams_r'] = WilliamsRIndicator(high, low, close).williams_r()
        
        # Rate of Change (ROC)
        for period in [5, 10, 20]:
            df[f'roc_{period}'] = close.pct_change(periods=period) * 100
        
        # Momentum
        for period in [5, 10, 20]:
            df[f'momentum_{period}'] = close - close.shift(period)
        
        # CCI - Commodity Channel Index
        typical_price = (high + low + close) / 3
        sma_tp = typical_price.rolling(window=20).mean()
        mad = typical_price.rolling(window=20).apply(lambda x: np.abs(x - x.mean()).mean())
        df['cci'] = (typical_price - sma_tp) / (0.015 * mad)
        
        return df
    
    # ═══════════════════════════════════════════════════════════════
    # VOLATILITY INDICATORS
    # ═══════════════════════════════════════════════════════════════
    
    def _add_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """إضافة مؤشرات التقلب"""
        
        close = df['close']
        high = df['high']
        low = df['low']
        
        # Bollinger Bands
        bb = BollingerBands(close)
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_middle'] = bb.bollinger_mavg()
        df['bb_lower'] = bb.bollinger_lband()
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle'] * 100
        df['bb_percent'] = (close - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # ATR - متوسط المدى الحقيقي
        for period in [7, 14, 21]:
            df[f'atr_{period}'] = AverageTrueRange(high, low, close, window=period).average_true_range()
        
        # ATR Percent
        df['atr_percent'] = df['atr_14'] / close * 100
        
        # Keltner Channel
        kc = KeltnerChannel(high, low, close)
        df['kc_upper'] = kc.keltner_channel_hband()
        df['kc_middle'] = kc.keltner_channel_mband()
        df['kc_lower'] = kc.keltner_channel_lband()
        
        # Historical Volatility
        for period in [10, 20, 30]:
            returns = close.pct_change()
            df[f'volatility_{period}'] = returns.rolling(window=period).std() * np.sqrt(252) * 100
        
        # True Range
        df['true_range'] = np.maximum(
            high - low,
            np.maximum(
                np.abs(high - close.shift(1)),
                np.abs(low - close.shift(1))
            )
        )
        
        return df
    
    # ═══════════════════════════════════════════════════════════════
    # VOLUME INDICATORS
    # ═══════════════════════════════════════════════════════════════
    
    def _add_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """إضافة مؤشرات الحجم"""
        
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']
        
        # OBV - On Balance Volume
        df['obv'] = OnBalanceVolumeIndicator(close, volume).on_balance_volume()
        
        # VWAP - Volume Weighted Average Price
        try:
            df['vwap'] = VolumeWeightedAveragePrice(high, low, close, volume).volume_weighted_average_price()
        except Exception:
            df['vwap'] = (high + low + close) / 3
        
        # MFI - Money Flow Index
        df['mfi'] = MFIIndicator(high, low, close, volume).money_flow_index()
        
        # Volume SMA
        for period in [5, 10, 20]:
            df[f'volume_sma_{period}'] = volume.rolling(window=period).mean()
        
        # Volume Ratio
        df['volume_ratio'] = volume / df['volume_sma_20']
        
        # Volume Change
        df['volume_change'] = volume.pct_change()
        
        # Price-Volume Trend
        df['pvt'] = ((close - close.shift(1)) / close.shift(1) * volume).cumsum()
        
        # Accumulation/Distribution
        clv = ((close - low) - (high - close)) / (high - low + 1e-10)
        df['ad'] = (clv * volume).cumsum()
        
        # Chaikin Money Flow
        mf_multiplier = ((close - low) - (high - close)) / (high - low + 1e-10)
        mf_volume = mf_multiplier * volume
        df['cmf'] = mf_volume.rolling(window=20).sum() / volume.rolling(window=20).sum()
        
        return df
    
    # ═══════════════════════════════════════════════════════════════
    # PRICE FEATURES
    # ═══════════════════════════════════════════════════════════════
    
    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """إضافة ميزات السعر"""
        
        open_price = df['open']
        high = df['high']
        low = df['low']
        close = df['close']
        
        # العوائد
        df['returns'] = close.pct_change()
        df['log_returns'] = np.log(close / close.shift(1))
        
        # نطاق الشمعة
        df['candle_range'] = high - low
        df['candle_body'] = np.abs(close - open_price)
        df['body_ratio'] = df['candle_body'] / (df['candle_range'] + 1e-10)
        
        # الظلال
        df['upper_shadow'] = high - np.maximum(open_price, close)
        df['lower_shadow'] = np.minimum(open_price, close) - low
        df['shadow_ratio'] = (df['upper_shadow'] + df['lower_shadow']) / (df['candle_range'] + 1e-10)
        
        # اتجاه الشمعة
        df['candle_direction'] = np.where(close >= open_price, 1, -1)
        
        # سلسلة الشموع
        df['consecutive_up'] = (df['candle_direction'] == 1).astype(int)
        df['consecutive_up'] = df['consecutive_up'].groupby(
            (df['consecutive_up'] != df['consecutive_up'].shift()).cumsum()
        ).cumsum() * df['consecutive_up']
        
        df['consecutive_down'] = (df['candle_direction'] == -1).astype(int)
        df['consecutive_down'] = df['consecutive_down'].groupby(
            (df['consecutive_down'] != df['consecutive_down'].shift()).cumsum()
        ).cumsum() * df['consecutive_down']
        
        # أعلى/أدنى سعر
        for period in [5, 10, 20, 50]:
            df[f'highest_{period}'] = high.rolling(window=period).max()
            df[f'lowest_{period}'] = low.rolling(window=period).min()
            df[f'price_position_{period}'] = (close - df[f'lowest_{period}']) / (df[f'highest_{period}'] - df[f'lowest_{period}'] + 1e-10)
        
        # Gap
        df['gap'] = open_price - close.shift(1)
        df['gap_percent'] = df['gap'] / close.shift(1) * 100
        
        return df
    
    # ═══════════════════════════════════════════════════════════════
    # TIME FEATURES
    # ═══════════════════════════════════════════════════════════════
    
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """إضافة ميزات الوقت"""
        
        if not isinstance(df.index, pd.DatetimeIndex):
            return df
        
        # الساعة
        df['hour'] = df.index.hour
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        # يوم الأسبوع
        df['day_of_week'] = df.index.dayofweek
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # يوم الشهر
        df['day_of_month'] = df.index.day
        df['dom_sin'] = np.sin(2 * np.pi * df['day_of_month'] / 31)
        df['dom_cos'] = np.cos(2 * np.pi * df['day_of_month'] / 31)
        
        # الشهر
        df['month'] = df.index.month
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # عطلة نهاية الأسبوع
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # جلسات التداول
        df['is_asian_session'] = ((df['hour'] >= 0) & (df['hour'] < 8)).astype(int)
        df['is_european_session'] = ((df['hour'] >= 8) & (df['hour'] < 16)).astype(int)
        df['is_american_session'] = ((df['hour'] >= 16) & (df['hour'] < 24)).astype(int)
        
        return df
    
    # ═══════════════════════════════════════════════════════════════
    # DERIVED FEATURES
    # ═══════════════════════════════════════════════════════════════
    
    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """إضافة ميزات مشتقة"""
        
        # قوة الاتجاه
        df['trend_strength'] = np.abs(df['adx']) / 100
        
        # حالة RSI
        df['rsi_oversold'] = (df['rsi_14'] < 30).astype(int)
        df['rsi_overbought'] = (df['rsi_14'] > 70).astype(int)
        
        # حالة Bollinger
        df['bb_squeeze'] = (df['bb_width'] < df['bb_width'].rolling(50).mean()).astype(int)
        
        # تقارب/تباعد MACD
        df['macd_bullish'] = (df['macd'] > df['macd_signal']).astype(int)
        df['macd_bearish'] = (df['macd'] < df['macd_signal']).astype(int)
        
        # قوة الحجم
        df['volume_spike'] = (df['volume_ratio'] > 2).astype(int)
        
        # تغيرات متعددة الفترات
        for period in [1, 3, 5, 10]:
            df[f'price_change_{period}'] = df['close'].pct_change(period)
            df[f'volume_change_{period}'] = df['volume'].pct_change(period)
        
        # الزخم المركب
        df['composite_momentum'] = (
            df['rsi_14'] / 100 * 0.3 +
            (df['macd'] / df['close'] * 100).clip(-1, 1) * 0.3 +
            df['stoch_k'] / 100 * 0.2 +
            (50 + df['williams_r']) / 100 * 0.2
        )
        
        return df
    
    # ═══════════════════════════════════════════════════════════════
    # CLEANING
    # ═══════════════════════════════════════════════════════════════
    
    def _clean_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """تنظيف الميزات"""
        
        # استبدال اللانهايات
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # ملء القيم المفقودة
        df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # قص القيم المتطرفة
        for col in df.columns:
            if df[col].dtype in ['float64', 'float32']:
                q1 = df[col].quantile(0.01)
                q99 = df[col].quantile(0.99)
                df[col] = df[col].clip(lower=q1, upper=q99)
        
        return df
    
    # ═══════════════════════════════════════════════════════════════
    # FEATURE SELECTION
    # ═══════════════════════════════════════════════════════════════
    
    def get_feature_names(self) -> List[str]:
        """الحصول على أسماء الميزات"""
        return self.feature_names
    
    def select_features(
        self, 
        df: pd.DataFrame, 
        features: List[str]
    ) -> pd.DataFrame:
        """اختيار ميزات محددة"""
        available = [f for f in features if f in df.columns]
        return df[available]
    
    def get_feature_groups(self) -> Dict[str, List[str]]:
        """الحصول على مجموعات الميزات"""
        return {
            'trend': [f for f in self.feature_names if any(x in f for x in ['sma', 'ema', 'macd', 'adx', 'ichimoku'])],
            'momentum': [f for f in self.feature_names if any(x in f for x in ['rsi', 'stoch', 'williams', 'roc', 'momentum', 'cci'])],
            'volatility': [f for f in self.feature_names if any(x in f for x in ['bb', 'atr', 'kc', 'volatility', 'true_range'])],
            'volume': [f for f in self.feature_names if any(x in f for x in ['volume', 'obv', 'vwap', 'mfi', 'pvt', 'ad', 'cmf'])],
            'price': [f for f in self.feature_names if any(x in f for x in ['returns', 'candle', 'shadow', 'gap', 'highest', 'lowest', 'position'])],
            'time': [f for f in self.feature_names if any(x in f for x in ['hour', 'day', 'month', 'weekend', 'session'])]
        }


# ═══════════════════════════════════════════════════════════════
# STANDALONE EXECUTION
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # اختبار الوحدة
    engineer = FeatureEngineer()
    
    # إنشاء بيانات تجريبية
    dates = pd.date_range(start='2024-01-01', periods=500, freq='1H')
    np.random.seed(42)
    
    df = pd.DataFrame({
        'open': np.random.uniform(40000, 50000, 500),
        'high': np.random.uniform(40000, 51000, 500),
        'low': np.random.uniform(39000, 50000, 500),
        'close': np.random.uniform(40000, 50000, 500),
        'volume': np.random.uniform(100, 10000, 500)
    }, index=dates)
    
    # إصلاح high/low
    df['high'] = df[['open', 'high', 'close']].max(axis=1)
    df['low'] = df[['open', 'low', 'close']].min(axis=1)
    
    # استخراج الميزات
    features_df = engineer.extract_all_features(df)
    
    print(f"Original shape: {df.shape}")
    print(f"Features shape: {features_df.shape}")
    print(f"\nFeature groups:")
    for group, features in engineer.get_feature_groups().items():
        print(f"  {group}: {len(features)} features")
