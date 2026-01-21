"""
Legendary Trading System - Technical Analyst Agent
نظام التداول الخارق - وكيل المحلل الفني

يحلل المؤشرات الفنية والأنماط السعرية لتوليد إشارات التداول.
"""

import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging

from ...core.base_agent import AnalystAgent
from ...core.types import (
    AnalysisResult, SignalType, AnalystType,
    TechnicalIndicators, OHLCV, MarketRegime
)


class TechnicalAnalystAgent(AnalystAgent):
    """
    وكيل المحلل الفني.
    
    يستخدم مجموعة شاملة من المؤشرات الفنية:
    - المتوسطات المتحركة (SMA, EMA)
    - مؤشرات الزخم (RSI, MACD, Stochastic)
    - مؤشرات التقلب (Bollinger Bands, ATR)
    - مؤشرات الحجم (OBV, VWAP)
    - مؤشرات الاتجاه (ADX, Parabolic SAR)
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(
            name="TechnicalAnalyst",
            config=config,
            analyst_type="technical"
        )
        self.weight = config.get("analyst_weights", {}).get("technical", 0.30)
        
        # إعدادات المؤشرات
        self.rsi_period = config.get("rsi_period", 14)
        self.macd_fast = config.get("macd_fast", 12)
        self.macd_slow = config.get("macd_slow", 26)
        self.macd_signal = config.get("macd_signal", 9)
        self.bb_period = config.get("bb_period", 20)
        self.bb_std = config.get("bb_std", 2)
        self.atr_period = config.get("atr_period", 14)
        
        # عتبات الإشارات
        self.rsi_oversold = config.get("rsi_oversold", 30)
        self.rsi_overbought = config.get("rsi_overbought", 70)
        self.adx_trend_threshold = config.get("adx_trend_threshold", 25)
    
    async def initialize(self) -> bool:
        """تهيئة المحلل الفني."""
        self.logger.info("تهيئة المحلل الفني...")
        return True
    
    async def process(self, data: Any) -> Any:
        """معالجة البيانات."""
        return await self.analyze(data.get("symbol"), data)
    
    async def shutdown(self) -> None:
        """إيقاف المحلل."""
        self.logger.info("إيقاف المحلل الفني")
    
    async def analyze(self, symbol: str, data: Dict[str, Any]) -> AnalysisResult:
        """
        تحليل فني شامل للرمز.
        
        Args:
            symbol: رمز العملة
            data: بيانات السوق (OHLCV)
            
        Returns:
            نتيجة التحليل الفني
        """
        self._update_activity()
        
        try:
            # استخراج بيانات الشموع
            candles = data.get("ohlcv", {}).get("15m", [])
            if not candles or len(candles) < 50:
                return self._create_neutral_result(symbol, "بيانات غير كافية")
            
            # تحويل إلى مصفوفات
            closes = np.array([c.close if isinstance(c, OHLCV) else c["close"] for c in candles])
            highs = np.array([c.high if isinstance(c, OHLCV) else c["high"] for c in candles])
            lows = np.array([c.low if isinstance(c, OHLCV) else c["low"] for c in candles])
            volumes = np.array([c.volume if isinstance(c, OHLCV) else c["volume"] for c in candles])
            
            # حساب المؤشرات
            indicators = self._calculate_all_indicators(closes, highs, lows, volumes)
            
            # تحليل الإشارات
            signals = self._analyze_signals(indicators, closes)
            
            # حساب الإشارة النهائية
            final_signal, confidence, reasoning = self._calculate_final_signal(signals)
            
            return AnalysisResult(
                analyst_type=AnalystType.TECHNICAL,
                symbol=symbol,
                timestamp=datetime.utcnow(),
                signal=final_signal,
                confidence=confidence,
                reasoning=reasoning,
                data={
                    "indicators": indicators,
                    "signals": signals,
                    "market_regime": self._detect_market_regime(indicators).value
                }
            )
            
        except Exception as e:
            self.logger.error(f"خطأ في التحليل الفني: {e}")
            self._handle_error(e)
            return self._create_neutral_result(symbol, f"خطأ: {str(e)}")
    
    def _calculate_all_indicators(self, closes: np.ndarray, highs: np.ndarray,
                                  lows: np.ndarray, volumes: np.ndarray) -> Dict[str, float]:
        """حساب جميع المؤشرات الفنية."""
        indicators = {}
        
        # المتوسطات المتحركة
        indicators["sma_20"] = self._sma(closes, 20)
        indicators["sma_50"] = self._sma(closes, 50)
        indicators["sma_200"] = self._sma(closes, 200) if len(closes) >= 200 else None
        indicators["ema_12"] = self._ema(closes, 12)
        indicators["ema_26"] = self._ema(closes, 26)
        
        # RSI
        indicators["rsi"] = self._rsi(closes, self.rsi_period)
        
        # MACD
        macd, signal, histogram = self._macd(closes)
        indicators["macd"] = macd
        indicators["macd_signal"] = signal
        indicators["macd_histogram"] = histogram
        
        # Stochastic
        k, d = self._stochastic(closes, highs, lows)
        indicators["stoch_k"] = k
        indicators["stoch_d"] = d
        
        # Bollinger Bands
        upper, middle, lower = self._bollinger_bands(closes)
        indicators["bb_upper"] = upper
        indicators["bb_middle"] = middle
        indicators["bb_lower"] = lower
        indicators["bb_width"] = (upper - lower) / middle if middle else 0
        
        # ATR
        indicators["atr"] = self._atr(closes, highs, lows, self.atr_period)
        
        # ADX
        indicators["adx"] = self._adx(closes, highs, lows)
        
        # OBV
        indicators["obv"] = self._obv(closes, volumes)
        
        # السعر الحالي
        indicators["current_price"] = closes[-1]
        
        return indicators
    
    def _analyze_signals(self, indicators: Dict[str, float],
                        closes: np.ndarray) -> Dict[str, Dict]:
        """تحليل الإشارات من المؤشرات."""
        signals = {}
        current_price = closes[-1]
        
        # إشارة المتوسطات المتحركة
        ma_signal = self._analyze_ma_signal(indicators, current_price)
        signals["moving_averages"] = ma_signal
        
        # إشارة RSI
        rsi_signal = self._analyze_rsi_signal(indicators["rsi"])
        signals["rsi"] = rsi_signal
        
        # إشارة MACD
        macd_signal = self._analyze_macd_signal(indicators)
        signals["macd"] = macd_signal
        
        # إشارة Stochastic
        stoch_signal = self._analyze_stochastic_signal(
            indicators["stoch_k"], indicators["stoch_d"]
        )
        signals["stochastic"] = stoch_signal
        
        # إشارة Bollinger Bands
        bb_signal = self._analyze_bb_signal(indicators, current_price)
        signals["bollinger"] = bb_signal
        
        # إشارة الاتجاه (ADX)
        trend_signal = self._analyze_trend_signal(indicators)
        signals["trend"] = trend_signal
        
        return signals
    
    def _analyze_ma_signal(self, indicators: Dict, price: float) -> Dict:
        """تحليل إشارة المتوسطات المتحركة."""
        sma_20 = indicators.get("sma_20")
        sma_50 = indicators.get("sma_50")
        ema_12 = indicators.get("ema_12")
        ema_26 = indicators.get("ema_26")
        
        signals = []
        
        # السعر فوق/تحت المتوسطات
        if sma_20 and price > sma_20:
            signals.append(0.3)
        elif sma_20:
            signals.append(-0.3)
        
        if sma_50 and price > sma_50:
            signals.append(0.3)
        elif sma_50:
            signals.append(-0.3)
        
        # تقاطع EMA
        if ema_12 and ema_26:
            if ema_12 > ema_26:
                signals.append(0.4)
            else:
                signals.append(-0.4)
        
        avg_signal = np.mean(signals) if signals else 0
        
        return {
            "value": avg_signal,
            "strength": abs(avg_signal),
            "description": "صعودي" if avg_signal > 0 else "هبوطي" if avg_signal < 0 else "محايد"
        }
    
    def _analyze_rsi_signal(self, rsi: float) -> Dict:
        """تحليل إشارة RSI."""
        if rsi is None:
            return {"value": 0, "strength": 0, "description": "غير متاح"}
        
        if rsi < self.rsi_oversold:
            # ذروة بيع - إشارة شراء
            signal = 0.5 + (self.rsi_oversold - rsi) / 100
            return {
                "value": min(signal, 1.0),
                "strength": signal,
                "description": f"ذروة بيع (RSI={rsi:.1f})"
            }
        elif rsi > self.rsi_overbought:
            # ذروة شراء - إشارة بيع
            signal = -0.5 - (rsi - self.rsi_overbought) / 100
            return {
                "value": max(signal, -1.0),
                "strength": abs(signal),
                "description": f"ذروة شراء (RSI={rsi:.1f})"
            }
        else:
            # منطقة محايدة
            normalized = (rsi - 50) / 50
            return {
                "value": normalized * 0.3,
                "strength": abs(normalized) * 0.3,
                "description": f"محايد (RSI={rsi:.1f})"
            }
    
    def _analyze_macd_signal(self, indicators: Dict) -> Dict:
        """تحليل إشارة MACD."""
        macd = indicators.get("macd")
        signal = indicators.get("macd_signal")
        histogram = indicators.get("macd_histogram")
        
        if macd is None or signal is None:
            return {"value": 0, "strength": 0, "description": "غير متاح"}
        
        # تقاطع MACD مع خط الإشارة
        if macd > signal:
            base_signal = 0.4
            description = "MACD فوق خط الإشارة"
        else:
            base_signal = -0.4
            description = "MACD تحت خط الإشارة"
        
        # قوة الإشارة من الهيستوغرام
        if histogram:
            strength_modifier = min(abs(histogram) / 0.001, 0.3)
            if histogram > 0:
                base_signal += strength_modifier
            else:
                base_signal -= strength_modifier
        
        return {
            "value": np.clip(base_signal, -1, 1),
            "strength": abs(base_signal),
            "description": description
        }
    
    def _analyze_stochastic_signal(self, k: float, d: float) -> Dict:
        """تحليل إشارة Stochastic."""
        if k is None or d is None:
            return {"value": 0, "strength": 0, "description": "غير متاح"}
        
        if k < 20 and d < 20:
            signal = 0.5
            description = "ذروة بيع"
        elif k > 80 and d > 80:
            signal = -0.5
            description = "ذروة شراء"
        elif k > d:
            signal = 0.2
            description = "تقاطع صعودي"
        elif k < d:
            signal = -0.2
            description = "تقاطع هبوطي"
        else:
            signal = 0
            description = "محايد"
        
        return {
            "value": signal,
            "strength": abs(signal),
            "description": f"{description} (K={k:.1f}, D={d:.1f})"
        }
    
    def _analyze_bb_signal(self, indicators: Dict, price: float) -> Dict:
        """تحليل إشارة Bollinger Bands."""
        upper = indicators.get("bb_upper")
        lower = indicators.get("bb_lower")
        middle = indicators.get("bb_middle")
        
        if not all([upper, lower, middle]):
            return {"value": 0, "strength": 0, "description": "غير متاح"}
        
        # موقع السعر بالنسبة للنطاقات
        bb_position = (price - lower) / (upper - lower) if (upper - lower) > 0 else 0.5
        
        if bb_position < 0.1:
            signal = 0.6
            description = "السعر عند الحد السفلي"
        elif bb_position > 0.9:
            signal = -0.6
            description = "السعر عند الحد العلوي"
        elif bb_position < 0.3:
            signal = 0.3
            description = "السعر قريب من الحد السفلي"
        elif bb_position > 0.7:
            signal = -0.3
            description = "السعر قريب من الحد العلوي"
        else:
            signal = 0
            description = "السعر في المنتصف"
        
        return {
            "value": signal,
            "strength": abs(signal),
            "description": description
        }
    
    def _analyze_trend_signal(self, indicators: Dict) -> Dict:
        """تحليل قوة الاتجاه."""
        adx = indicators.get("adx")
        
        if adx is None:
            return {"value": 0, "strength": 0, "description": "غير متاح"}
        
        if adx > self.adx_trend_threshold:
            strength = min(adx / 50, 1.0)
            description = f"اتجاه قوي (ADX={adx:.1f})"
        else:
            strength = adx / self.adx_trend_threshold
            description = f"اتجاه ضعيف (ADX={adx:.1f})"
        
        return {
            "value": 0,  # ADX لا يحدد الاتجاه، فقط قوته
            "strength": strength,
            "description": description,
            "is_trending": adx > self.adx_trend_threshold
        }
    
    def _calculate_final_signal(self, signals: Dict) -> tuple:
        """حساب الإشارة النهائية من جميع الإشارات."""
        # أوزان المؤشرات
        weights = {
            "moving_averages": 0.25,
            "rsi": 0.20,
            "macd": 0.20,
            "stochastic": 0.15,
            "bollinger": 0.15,
            "trend": 0.05
        }
        
        weighted_sum = 0
        total_weight = 0
        reasoning_parts = []
        
        for indicator, weight in weights.items():
            if indicator in signals:
                signal_data = signals[indicator]
                value = signal_data.get("value", 0)
                strength = signal_data.get("strength", 0)
                
                weighted_sum += value * weight * strength
                total_weight += weight * strength
                
                if strength > 0.3:
                    reasoning_parts.append(signal_data.get("description", ""))
        
        # حساب المتوسط المرجح
        if total_weight > 0:
            avg_signal = weighted_sum / total_weight
        else:
            avg_signal = 0
        
        # تحديد نوع الإشارة
        final_signal = self._signals_to_signal_type(avg_signal)
        
        # حساب الثقة
        confidence = self._calculate_confidence([
            s.get("value", 0) for s in signals.values()
        ])
        
        # بناء التفسير
        reasoning = "التحليل الفني: " + " | ".join(reasoning_parts[:3])
        
        return final_signal, confidence, reasoning
    
    def _detect_market_regime(self, indicators: Dict) -> MarketRegime:
        """تحديد حالة السوق."""
        adx = indicators.get("adx", 0)
        bb_width = indicators.get("bb_width", 0)
        rsi = indicators.get("rsi", 50)
        
        # تقلب عالي
        if bb_width > 0.1:
            return MarketRegime.HIGH_VOLATILITY
        
        # تقلب منخفض
        if bb_width < 0.02:
            return MarketRegime.LOW_VOLATILITY
        
        # اتجاه قوي
        if adx > 40:
            if rsi > 60:
                return MarketRegime.STRONG_BULLISH
            elif rsi < 40:
                return MarketRegime.STRONG_BEARISH
        
        # اتجاه معتدل
        if adx > 25:
            if rsi > 55:
                return MarketRegime.BULLISH
            elif rsi < 45:
                return MarketRegime.BEARISH
        
        return MarketRegime.NEUTRAL
    
    def _create_neutral_result(self, symbol: str, reason: str) -> AnalysisResult:
        """إنشاء نتيجة محايدة."""
        return AnalysisResult(
            analyst_type=AnalystType.TECHNICAL,
            symbol=symbol,
            timestamp=datetime.utcnow(),
            signal=SignalType.NEUTRAL,
            confidence=0.0,
            reasoning=reason,
            data={}
        )
    
    # ==========================================
    # دوال حساب المؤشرات
    # ==========================================
    
    def _sma(self, data: np.ndarray, period: int) -> Optional[float]:
        """حساب المتوسط المتحرك البسيط."""
        if len(data) < period:
            return None
        return float(np.mean(data[-period:]))
    
    def _ema(self, data: np.ndarray, period: int) -> Optional[float]:
        """حساب المتوسط المتحرك الأسي."""
        if len(data) < period:
            return None
        
        multiplier = 2 / (period + 1)
        ema = data[0]
        
        for price in data[1:]:
            ema = (price - ema) * multiplier + ema
        
        return float(ema)
    
    def _rsi(self, data: np.ndarray, period: int) -> Optional[float]:
        """حساب مؤشر القوة النسبية."""
        if len(data) < period + 1:
            return None
        
        deltas = np.diff(data)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return float(rsi)
    
    def _macd(self, data: np.ndarray) -> tuple:
        """حساب MACD."""
        if len(data) < self.macd_slow:
            return None, None, None
        
        ema_fast = self._ema(data, self.macd_fast)
        ema_slow = self._ema(data, self.macd_slow)
        
        if ema_fast is None or ema_slow is None:
            return None, None, None
        
        macd_line = ema_fast - ema_slow
        
        # حساب خط الإشارة (EMA للـ MACD)
        # نحتاج لحساب MACD لفترة كافية أولاً
        macd_values = []
        for i in range(self.macd_slow, len(data) + 1):
            ema_f = self._ema(data[:i], self.macd_fast)
            ema_s = self._ema(data[:i], self.macd_slow)
            if ema_f and ema_s:
                macd_values.append(ema_f - ema_s)
        
        if len(macd_values) >= self.macd_signal:
            signal_line = self._ema(np.array(macd_values), self.macd_signal)
        else:
            signal_line = macd_line
        
        histogram = macd_line - signal_line if signal_line else 0
        
        return macd_line, signal_line, histogram
    
    def _stochastic(self, closes: np.ndarray, highs: np.ndarray,
                   lows: np.ndarray, period: int = 14) -> tuple:
        """حساب Stochastic Oscillator."""
        if len(closes) < period:
            return None, None
        
        lowest_low = np.min(lows[-period:])
        highest_high = np.max(highs[-period:])
        
        if highest_high == lowest_low:
            k = 50.0
        else:
            k = ((closes[-1] - lowest_low) / (highest_high - lowest_low)) * 100
        
        # D هو SMA للـ K
        d = k  # مبسط
        
        return float(k), float(d)
    
    def _bollinger_bands(self, data: np.ndarray) -> tuple:
        """حساب Bollinger Bands."""
        if len(data) < self.bb_period:
            return None, None, None
        
        middle = self._sma(data, self.bb_period)
        std = np.std(data[-self.bb_period:])
        
        upper = middle + (self.bb_std * std)
        lower = middle - (self.bb_std * std)
        
        return float(upper), float(middle), float(lower)
    
    def _atr(self, closes: np.ndarray, highs: np.ndarray,
            lows: np.ndarray, period: int) -> Optional[float]:
        """حساب Average True Range."""
        if len(closes) < period + 1:
            return None
        
        tr_values = []
        for i in range(1, len(closes)):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i-1]),
                abs(lows[i] - closes[i-1])
            )
            tr_values.append(tr)
        
        return float(np.mean(tr_values[-period:]))
    
    def _adx(self, closes: np.ndarray, highs: np.ndarray,
            lows: np.ndarray, period: int = 14) -> Optional[float]:
        """حساب Average Directional Index."""
        if len(closes) < period + 1:
            return None
        
        # حساب مبسط للـ ADX
        tr_sum = 0
        plus_dm_sum = 0
        minus_dm_sum = 0
        
        for i in range(1, min(period + 1, len(closes))):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i-1]),
                abs(lows[i] - closes[i-1])
            )
            tr_sum += tr
            
            plus_dm = max(highs[i] - highs[i-1], 0)
            minus_dm = max(lows[i-1] - lows[i], 0)
            
            if plus_dm > minus_dm:
                plus_dm_sum += plus_dm
                minus_dm_sum += 0
            else:
                plus_dm_sum += 0
                minus_dm_sum += minus_dm
        
        if tr_sum == 0:
            return 0.0
        
        plus_di = (plus_dm_sum / tr_sum) * 100
        minus_di = (minus_dm_sum / tr_sum) * 100
        
        di_sum = plus_di + minus_di
        if di_sum == 0:
            return 0.0
        
        dx = abs(plus_di - minus_di) / di_sum * 100
        
        return float(dx)
    
    def _obv(self, closes: np.ndarray, volumes: np.ndarray) -> Optional[float]:
        """حساب On-Balance Volume."""
        if len(closes) < 2:
            return None
        
        obv = 0
        for i in range(1, len(closes)):
            if closes[i] > closes[i-1]:
                obv += volumes[i]
            elif closes[i] < closes[i-1]:
                obv -= volumes[i]
        
        return float(obv)
