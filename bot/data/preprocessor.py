"""
═══════════════════════════════════════════════════════════════
LEGENDARY AGENT - Data Preprocessor
معالجة وتنظيف البيانات
═══════════════════════════════════════════════════════════════
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from loguru import logger
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import warnings
warnings.filterwarnings('ignore')


class DataPreprocessor:
    """
    معالج البيانات
    - تنظيف البيانات
    - معالجة القيم المفقودة
    - تطبيع البيانات
    - إزالة القيم الشاذة
    """
    
    def __init__(self, scaling_method: str = "robust"):
        """
        تهيئة المعالج
        
        Args:
            scaling_method: طريقة التطبيع (standard, minmax, robust)
        """
        self.scaling_method = scaling_method
        self.scalers: Dict[str, Any] = {}
        self.feature_stats: Dict[str, Dict] = {}
        
        logger.info(f"🔧 DataPreprocessor initialized with {scaling_method} scaling")
    
    # ═══════════════════════════════════════════════════════════════
    # MAIN PROCESSING PIPELINE
    # ═══════════════════════════════════════════════════════════════
    
    def process(
        self, 
        df: pd.DataFrame,
        symbol: str = "UNKNOWN",
        remove_outliers: bool = True,
        fill_missing: bool = True,
        scale: bool = True
    ) -> pd.DataFrame:
        """
        معالجة البيانات الكاملة
        
        Args:
            df: DataFrame الخام
            symbol: رمز العملة
            remove_outliers: إزالة القيم الشاذة
            fill_missing: ملء القيم المفقودة
            scale: تطبيع البيانات
            
        Returns:
            DataFrame معالج
        """
        logger.info(f"🔧 Processing {symbol}: {len(df)} rows")
        
        # نسخة للعمل عليها
        df = df.copy()
        
        # 1. التحقق الأولي
        df = self._validate_and_clean(df)
        
        # 2. معالجة القيم المفقودة
        if fill_missing:
            df = self._fill_missing_values(df)
        
        # 3. إزالة القيم الشاذة
        if remove_outliers:
            df = self._remove_outliers(df)
        
        # 4. التطبيع
        if scale:
            df = self._scale_data(df, symbol)
        
        # 5. حفظ الإحصائيات
        self._save_stats(df, symbol)
        
        logger.info(f"✅ Processed {symbol}: {len(df)} rows remaining")
        return df
    
    def process_batch(
        self, 
        data_dict: Dict[str, pd.DataFrame],
        **kwargs
    ) -> Dict[str, pd.DataFrame]:
        """
        معالجة مجموعة من البيانات
        
        Args:
            data_dict: قاموس البيانات
            **kwargs: معاملات إضافية
            
        Returns:
            قاموس البيانات المعالجة
        """
        processed = {}
        
        for symbol, df in data_dict.items():
            try:
                processed[symbol] = self.process(df, symbol, **kwargs)
            except Exception as e:
                logger.error(f"❌ Error processing {symbol}: {e}")
                continue
        
        return processed
    
    # ═══════════════════════════════════════════════════════════════
    # VALIDATION & CLEANING
    # ═══════════════════════════════════════════════════════════════
    
    def _validate_and_clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """التحقق والتنظيف الأولي"""
        
        # التأكد من وجود الأعمدة الأساسية
        required = ['open', 'high', 'low', 'close', 'volume']
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # تحويل إلى أرقام
        for col in required:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # إزالة الصفوف الفارغة تماماً
        df = df.dropna(how='all')
        
        # إصلاح القيم غير المنطقية
        # high يجب أن يكون >= open, close, low
        df['high'] = df[['open', 'high', 'low', 'close']].max(axis=1)
        # low يجب أن يكون <= open, close, high
        df['low'] = df[['open', 'high', 'low', 'close']].min(axis=1)
        
        # إزالة القيم السالبة
        for col in required:
            df = df[df[col] >= 0]
        
        # إزالة الصفوف المكررة
        df = df[~df.index.duplicated(keep='first')]
        
        # ترتيب حسب الوقت
        df = df.sort_index()
        
        return df
    
    # ═══════════════════════════════════════════════════════════════
    # MISSING VALUES
    # ═══════════════════════════════════════════════════════════════
    
    def _fill_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """ملء القيم المفقودة"""
        
        # حساب نسبة القيم المفقودة
        missing_pct = df.isnull().sum() / len(df) * 100
        
        for col in df.columns:
            if df[col].isnull().any():
                if missing_pct[col] > 50:
                    # إذا كانت النسبة عالية جداً، استخدم الوسيط
                    df[col] = df[col].fillna(df[col].median())
                else:
                    # استخدام الاستيفاء الخطي
                    df[col] = df[col].interpolate(method='linear')
                    # ملء أي قيم متبقية في البداية/النهاية
                    df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
        
        return df
    
    # ═══════════════════════════════════════════════════════════════
    # OUTLIER REMOVAL
    # ═══════════════════════════════════════════════════════════════
    
    def _remove_outliers(
        self, 
        df: pd.DataFrame,
        method: str = "iqr",
        threshold: float = 3.0
    ) -> pd.DataFrame:
        """
        إزالة القيم الشاذة
        
        Args:
            df: DataFrame
            method: طريقة الكشف (iqr, zscore)
            threshold: عتبة الكشف
        """
        original_len = len(df)
        
        if method == "iqr":
            df = self._remove_outliers_iqr(df, threshold)
        elif method == "zscore":
            df = self._remove_outliers_zscore(df, threshold)
        
        removed = original_len - len(df)
        if removed > 0:
            logger.debug(f"  Removed {removed} outliers ({removed/original_len*100:.2f}%)")
        
        return df
    
    def _remove_outliers_iqr(self, df: pd.DataFrame, multiplier: float = 1.5) -> pd.DataFrame:
        """إزالة القيم الشاذة باستخدام IQR"""
        
        # تطبيق على أعمدة OHLCV فقط
        columns = ['open', 'high', 'low', 'close', 'volume']
        
        for col in columns:
            if col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower = Q1 - multiplier * IQR
                upper = Q3 + multiplier * IQR
                
                # بدلاً من الإزالة، نقوم بالقص
                df[col] = df[col].clip(lower=lower, upper=upper)
        
        return df
    
    def _remove_outliers_zscore(self, df: pd.DataFrame, threshold: float = 3.0) -> pd.DataFrame:
        """إزالة القيم الشاذة باستخدام Z-score"""
        
        columns = ['open', 'high', 'low', 'close', 'volume']
        
        for col in columns:
            if col in df.columns:
                mean = df[col].mean()
                std = df[col].std()
                
                if std > 0:
                    z_scores = np.abs((df[col] - mean) / std)
                    df = df[z_scores < threshold]
        
        return df
    
    # ═══════════════════════════════════════════════════════════════
    # SCALING
    # ═══════════════════════════════════════════════════════════════
    
    def _scale_data(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """تطبيع البيانات"""
        
        # اختيار المُطبّع
        if self.scaling_method == "standard":
            scaler = StandardScaler()
        elif self.scaling_method == "minmax":
            scaler = MinMaxScaler()
        else:
            scaler = RobustScaler()
        
        # تطبيع أعمدة OHLCV
        columns_to_scale = ['open', 'high', 'low', 'close', 'volume']
        columns_present = [c for c in columns_to_scale if c in df.columns]
        
        # حفظ البيانات الأصلية للعكس لاحقاً
        self.scalers[symbol] = {
            'scaler': scaler,
            'columns': columns_present,
            'original_values': {
                col: {'min': df[col].min(), 'max': df[col].max()}
                for col in columns_present
            }
        }
        
        # تطبيق التطبيع
        df[columns_present] = scaler.fit_transform(df[columns_present])
        
        return df
    
    def inverse_scale(
        self, 
        df: pd.DataFrame, 
        symbol: str
    ) -> pd.DataFrame:
        """عكس التطبيع"""
        
        if symbol not in self.scalers:
            logger.warning(f"⚠️ No scaler found for {symbol}")
            return df
        
        scaler_info = self.scalers[symbol]
        columns = scaler_info['columns']
        scaler = scaler_info['scaler']
        
        df[columns] = scaler.inverse_transform(df[columns])
        
        return df
    
    # ═══════════════════════════════════════════════════════════════
    # STATISTICS
    # ═══════════════════════════════════════════════════════════════
    
    def _save_stats(self, df: pd.DataFrame, symbol: str) -> None:
        """حفظ إحصائيات البيانات"""
        
        self.feature_stats[symbol] = {
            'count': len(df),
            'date_range': {
                'start': str(df.index.min()),
                'end': str(df.index.max())
            },
            'columns': {}
        }
        
        for col in df.columns:
            self.feature_stats[symbol]['columns'][col] = {
                'mean': float(df[col].mean()),
                'std': float(df[col].std()),
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                'median': float(df[col].median())
            }
    
    def get_stats(self, symbol: str = None) -> Dict:
        """الحصول على الإحصائيات"""
        if symbol:
            return self.feature_stats.get(symbol, {})
        return self.feature_stats
    
    # ═══════════════════════════════════════════════════════════════
    # SPECIAL TRANSFORMATIONS
    # ═══════════════════════════════════════════════════════════════
    
    def add_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """إضافة العوائد"""
        df = df.copy()
        
        # العوائد البسيطة
        df['returns'] = df['close'].pct_change()
        
        # العوائد اللوغاريتمية
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # ملء القيم الأولى
        df['returns'] = df['returns'].fillna(0)
        df['log_returns'] = df['log_returns'].fillna(0)
        
        return df
    
    def add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """إضافة ميزات السعر"""
        df = df.copy()
        
        # نطاق الشمعة
        df['candle_range'] = df['high'] - df['low']
        df['candle_body'] = abs(df['close'] - df['open'])
        
        # نسبة الجسم للنطاق
        df['body_ratio'] = df['candle_body'] / (df['candle_range'] + 1e-10)
        
        # الظل العلوي والسفلي
        df['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']
        
        # اتجاه الشمعة
        df['candle_direction'] = np.where(df['close'] >= df['open'], 1, -1)
        
        return df
    
    def add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """إضافة ميزات الحجم"""
        df = df.copy()
        
        # متوسط الحجم
        df['volume_ma_20'] = df['volume'].rolling(window=20).mean()
        
        # نسبة الحجم للمتوسط
        df['volume_ratio'] = df['volume'] / (df['volume_ma_20'] + 1e-10)
        
        # تغير الحجم
        df['volume_change'] = df['volume'].pct_change()
        
        # ملء القيم المفقودة
        df = df.fillna(method='bfill').fillna(0)
        
        return df
    
    # ═══════════════════════════════════════════════════════════════
    # SEQUENCE CREATION
    # ═══════════════════════════════════════════════════════════════
    
    def create_sequences(
        self,
        df: pd.DataFrame,
        sequence_length: int = 60,
        target_column: str = 'close',
        prediction_horizon: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        إنشاء تسلسلات للتدريب
        
        Args:
            df: DataFrame
            sequence_length: طول التسلسل
            target_column: عمود الهدف
            prediction_horizon: أفق التنبؤ
            
        Returns:
            (X, y) مصفوفات التدريب
        """
        data = df.values
        target_idx = df.columns.get_loc(target_column)
        
        X, y = [], []
        
        for i in range(len(data) - sequence_length - prediction_horizon + 1):
            X.append(data[i:i+sequence_length])
            
            # الهدف: تغير السعر
            current_price = data[i+sequence_length-1, target_idx]
            future_price = data[i+sequence_length+prediction_horizon-1, target_idx]
            
            if current_price != 0:
                change = (future_price - current_price) / current_price
            else:
                change = 0
            
            y.append(change)
        
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)
    
    def create_classification_labels(
        self,
        y: np.ndarray,
        threshold: float = 0.01
    ) -> np.ndarray:
        """
        تحويل التغيرات إلى تصنيفات
        
        Args:
            y: مصفوفة التغيرات
            threshold: عتبة التصنيف
            
        Returns:
            مصفوفة التصنيفات (0=BUY, 1=SELL, 2=HOLD)
        """
        labels = np.zeros(len(y), dtype=np.int64)
        
        labels[y > threshold] = 0   # BUY
        labels[y < -threshold] = 1  # SELL
        labels[(y >= -threshold) & (y <= threshold)] = 2  # HOLD
        
        return labels


# ═══════════════════════════════════════════════════════════════
# STANDALONE EXECUTION
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # اختبار الوحدة
    preprocessor = DataPreprocessor()
    
    # إنشاء بيانات تجريبية
    dates = pd.date_range(start='2024-01-01', periods=1000, freq='1H')
    df = pd.DataFrame({
        'open': np.random.uniform(40000, 50000, 1000),
        'high': np.random.uniform(40000, 51000, 1000),
        'low': np.random.uniform(39000, 50000, 1000),
        'close': np.random.uniform(40000, 50000, 1000),
        'volume': np.random.uniform(100, 10000, 1000)
    }, index=dates)
    
    # إصلاح high/low
    df['high'] = df[['open', 'high', 'close']].max(axis=1)
    df['low'] = df[['open', 'low', 'close']].min(axis=1)
    
    # معالجة
    processed = preprocessor.process(df, "BTCUSDT")
    
    print(f"Original shape: {df.shape}")
    print(f"Processed shape: {processed.shape}")
    print(f"\nStats: {preprocessor.get_stats('BTCUSDT')}")
