"""
═══════════════════════════════════════════════════════════════
LEGENDARY AGENT - Perception Layer
طبقة الإدراك - جمع وتجميع البيانات من مصادر متعددة
═══════════════════════════════════════════════════════════════
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from loguru import logger


class DataSource(Enum):
    """مصادر البيانات"""
    PRICE = "price_data"
    VOLUME = "volume_data"
    ORDERBOOK = "orderbook"
    TRADES = "recent_trades"
    FUNDING = "funding_rate"
    OPEN_INTEREST = "open_interest"
    LIQUIDATIONS = "liquidations"
    SENTIMENT = "sentiment"
    NEWS = "news"
    ONCHAIN = "onchain"


@dataclass
class MarketData:
    """بيانات السوق"""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    quote_volume: float = 0.0
    trades_count: int = 0


@dataclass
class OrderBookSnapshot:
    """لقطة دفتر الأوامر"""
    symbol: str
    timestamp: datetime
    bids: List[Tuple[float, float]]  # (price, quantity)
    asks: List[Tuple[float, float]]
    bid_depth: float = 0.0
    ask_depth: float = 0.0
    spread: float = 0.0
    imbalance: float = 0.0


@dataclass
class SentimentData:
    """بيانات المشاعر"""
    symbol: str
    timestamp: datetime
    fear_greed_index: float
    social_volume: float
    social_sentiment: float  # -1 to 1
    news_sentiment: float  # -1 to 1
    whale_activity: str  # "accumulating", "distributing", "neutral"


@dataclass
class PerceptionState:
    """حالة الإدراك"""
    symbol: str
    timestamp: datetime
    price_data: Optional[MarketData] = None
    orderbook: Optional[OrderBookSnapshot] = None
    sentiment: Optional[SentimentData] = None
    features: Dict[str, float] = field(default_factory=dict)
    data_quality: float = 1.0
    sources_available: List[str] = field(default_factory=list)


class PerceptionLayer:
    """
    طبقة الإدراك
    
    مسؤولة عن:
    - جمع البيانات من مصادر متعددة
    - تنظيف وتوحيد البيانات
    - استخراج الميزات
    - تقييم جودة البيانات
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        تهيئة طبقة الإدراك
        
        Args:
            config: إعدادات الطبقة
        """
        self.config = config or {}
        
        # مصادر البيانات المتاحة
        self.available_sources: List[DataSource] = [
            DataSource.PRICE,
            DataSource.VOLUME
        ]
        
        # ذاكرة البيانات
        self.data_cache: Dict[str, List[MarketData]] = {}
        self.cache_size = self.config.get('cache_size', 1000)
        
        # إحصائيات
        self.stats = {
            'total_perceptions': 0,
            'data_quality_avg': 1.0,
            'sources_used': {}
        }
        
        logger.info("👁️ PerceptionLayer initialized")
    
    # ═══════════════════════════════════════════════════════════════
    # DATA COLLECTION
    # ═══════════════════════════════════════════════════════════════
    
    def perceive(
        self,
        symbol: str,
        raw_data: Dict[str, Any]
    ) -> PerceptionState:
        """
        إدراك البيانات
        
        Args:
            symbol: رمز العملة
            raw_data: البيانات الخام
            
        Returns:
            حالة الإدراك
        """
        timestamp = datetime.now()
        
        # جمع بيانات السعر
        price_data = self._extract_price_data(symbol, raw_data)
        
        # جمع بيانات دفتر الأوامر
        orderbook = self._extract_orderbook(symbol, raw_data)
        
        # جمع بيانات المشاعر
        sentiment = self._extract_sentiment(symbol, raw_data)
        
        # استخراج الميزات
        features = self._extract_features(price_data, orderbook, sentiment, raw_data)
        
        # تقييم جودة البيانات
        data_quality = self._assess_data_quality(price_data, orderbook, sentiment)
        
        # تحديد المصادر المتاحة
        sources_available = self._get_available_sources(raw_data)
        
        # إنشاء حالة الإدراك
        state = PerceptionState(
            symbol=symbol,
            timestamp=timestamp,
            price_data=price_data,
            orderbook=orderbook,
            sentiment=sentiment,
            features=features,
            data_quality=data_quality,
            sources_available=sources_available
        )
        
        # تحديث الذاكرة
        self._update_cache(symbol, price_data)
        
        # تحديث الإحصائيات
        self.stats['total_perceptions'] += 1
        self.stats['data_quality_avg'] = (
            self.stats['data_quality_avg'] * 0.99 + data_quality * 0.01
        )
        
        return state
    
    def _extract_price_data(
        self,
        symbol: str,
        raw_data: Dict
    ) -> Optional[MarketData]:
        """استخراج بيانات السعر"""
        try:
            ohlcv = raw_data.get('ohlcv', raw_data.get('price', {}))
            
            if isinstance(ohlcv, dict):
                return MarketData(
                    symbol=symbol,
                    timestamp=datetime.now(),
                    open=float(ohlcv.get('open', 0)),
                    high=float(ohlcv.get('high', 0)),
                    low=float(ohlcv.get('low', 0)),
                    close=float(ohlcv.get('close', 0)),
                    volume=float(ohlcv.get('volume', 0)),
                    quote_volume=float(ohlcv.get('quote_volume', 0)),
                    trades_count=int(ohlcv.get('trades_count', 0))
                )
            elif isinstance(ohlcv, (list, np.ndarray)) and len(ohlcv) >= 5:
                return MarketData(
                    symbol=symbol,
                    timestamp=datetime.now(),
                    open=float(ohlcv[0]),
                    high=float(ohlcv[1]),
                    low=float(ohlcv[2]),
                    close=float(ohlcv[3]),
                    volume=float(ohlcv[4])
                )
        except Exception as e:
            logger.warning(f"Failed to extract price data: {e}")
        
        return None
    
    def _extract_orderbook(
        self,
        symbol: str,
        raw_data: Dict
    ) -> Optional[OrderBookSnapshot]:
        """استخراج بيانات دفتر الأوامر"""
        try:
            ob_data = raw_data.get('orderbook', {})
            
            if not ob_data:
                return None
            
            bids = ob_data.get('bids', [])
            asks = ob_data.get('asks', [])
            
            if not bids or not asks:
                return None
            
            # حساب العمق
            bid_depth = sum(b[1] for b in bids[:10]) if bids else 0
            ask_depth = sum(a[1] for a in asks[:10]) if asks else 0
            
            # حساب الفارق
            best_bid = bids[0][0] if bids else 0
            best_ask = asks[0][0] if asks else 0
            spread = (best_ask - best_bid) / best_bid if best_bid > 0 else 0
            
            # حساب عدم التوازن
            total_depth = bid_depth + ask_depth
            imbalance = (bid_depth - ask_depth) / total_depth if total_depth > 0 else 0
            
            return OrderBookSnapshot(
                symbol=symbol,
                timestamp=datetime.now(),
                bids=bids[:20],
                asks=asks[:20],
                bid_depth=bid_depth,
                ask_depth=ask_depth,
                spread=spread,
                imbalance=imbalance
            )
        except Exception as e:
            logger.warning(f"Failed to extract orderbook: {e}")
        
        return None
    
    def _extract_sentiment(
        self,
        symbol: str,
        raw_data: Dict
    ) -> Optional[SentimentData]:
        """استخراج بيانات المشاعر"""
        try:
            sent_data = raw_data.get('sentiment', {})
            
            if not sent_data:
                return None
            
            return SentimentData(
                symbol=symbol,
                timestamp=datetime.now(),
                fear_greed_index=float(sent_data.get('fear_greed', 50)),
                social_volume=float(sent_data.get('social_volume', 0)),
                social_sentiment=float(sent_data.get('social_sentiment', 0)),
                news_sentiment=float(sent_data.get('news_sentiment', 0)),
                whale_activity=sent_data.get('whale_activity', 'neutral')
            )
        except Exception as e:
            logger.warning(f"Failed to extract sentiment: {e}")
        
        return None
    
    # ═══════════════════════════════════════════════════════════════
    # FEATURE EXTRACTION
    # ═══════════════════════════════════════════════════════════════
    
    def _extract_features(
        self,
        price_data: Optional[MarketData],
        orderbook: Optional[OrderBookSnapshot],
        sentiment: Optional[SentimentData],
        raw_data: Dict
    ) -> Dict[str, float]:
        """استخراج الميزات"""
        features = {}
        
        # ميزات السعر
        if price_data:
            features['open'] = price_data.open
            features['high'] = price_data.high
            features['low'] = price_data.low
            features['close'] = price_data.close
            features['volume'] = price_data.volume
            
            # نطاق الشمعة
            if price_data.high > 0:
                features['candle_range'] = (price_data.high - price_data.low) / price_data.high
            
            # جسم الشمعة
            if price_data.high > 0:
                features['candle_body'] = abs(price_data.close - price_data.open) / price_data.high
            
            # اتجاه الشمعة
            features['candle_direction'] = 1 if price_data.close > price_data.open else -1
        
        # ميزات دفتر الأوامر
        if orderbook:
            features['orderbook_spread'] = orderbook.spread
            features['orderbook_imbalance'] = orderbook.imbalance
            features['bid_depth'] = orderbook.bid_depth
            features['ask_depth'] = orderbook.ask_depth
        
        # ميزات المشاعر
        if sentiment:
            features['fear_greed'] = sentiment.fear_greed_index
            features['social_sentiment'] = sentiment.social_sentiment
            features['news_sentiment'] = sentiment.news_sentiment
            features['whale_accumulating'] = 1 if sentiment.whale_activity == 'accumulating' else 0
            features['whale_distributing'] = 1 if sentiment.whale_activity == 'distributing' else 0
        
        # ميزات إضافية من البيانات الخام
        if 'features' in raw_data:
            features.update(raw_data['features'])
        
        return features
    
    # ═══════════════════════════════════════════════════════════════
    # DATA QUALITY
    # ═══════════════════════════════════════════════════════════════
    
    def _assess_data_quality(
        self,
        price_data: Optional[MarketData],
        orderbook: Optional[OrderBookSnapshot],
        sentiment: Optional[SentimentData]
    ) -> float:
        """تقييم جودة البيانات"""
        quality_score = 0.0
        max_score = 0.0
        
        # جودة بيانات السعر (40%)
        max_score += 0.4
        if price_data:
            if price_data.close > 0:
                quality_score += 0.2
            if price_data.volume > 0:
                quality_score += 0.1
            if price_data.high >= price_data.low:
                quality_score += 0.1
        
        # جودة دفتر الأوامر (30%)
        max_score += 0.3
        if orderbook:
            if len(orderbook.bids) >= 10:
                quality_score += 0.15
            if len(orderbook.asks) >= 10:
                quality_score += 0.15
        
        # جودة المشاعر (30%)
        max_score += 0.3
        if sentiment:
            if 0 <= sentiment.fear_greed_index <= 100:
                quality_score += 0.15
            if -1 <= sentiment.social_sentiment <= 1:
                quality_score += 0.15
        
        return quality_score / max_score if max_score > 0 else 0.0
    
    def _get_available_sources(self, raw_data: Dict) -> List[str]:
        """تحديد المصادر المتاحة"""
        sources = []
        
        if 'ohlcv' in raw_data or 'price' in raw_data:
            sources.append('price')
        if 'orderbook' in raw_data:
            sources.append('orderbook')
        if 'sentiment' in raw_data:
            sources.append('sentiment')
        if 'funding_rate' in raw_data:
            sources.append('funding')
        if 'open_interest' in raw_data:
            sources.append('open_interest')
        if 'liquidations' in raw_data:
            sources.append('liquidations')
        
        return sources
    
    # ═══════════════════════════════════════════════════════════════
    # CACHE MANAGEMENT
    # ═══════════════════════════════════════════════════════════════
    
    def _update_cache(
        self,
        symbol: str,
        price_data: Optional[MarketData]
    ) -> None:
        """تحديث الذاكرة"""
        if price_data is None:
            return
        
        if symbol not in self.data_cache:
            self.data_cache[symbol] = []
        
        self.data_cache[symbol].append(price_data)
        
        # الحفاظ على حجم الذاكرة
        if len(self.data_cache[symbol]) > self.cache_size:
            self.data_cache[symbol] = self.data_cache[symbol][-self.cache_size:]
    
    def get_historical_data(
        self,
        symbol: str,
        periods: int = 100
    ) -> List[MarketData]:
        """الحصول على البيانات التاريخية"""
        if symbol not in self.data_cache:
            return []
        
        return self.data_cache[symbol][-periods:]
    
    # ═══════════════════════════════════════════════════════════════
    # MULTI-SYMBOL PERCEPTION
    # ═══════════════════════════════════════════════════════════════
    
    def perceive_multiple(
        self,
        symbols_data: Dict[str, Dict[str, Any]]
    ) -> Dict[str, PerceptionState]:
        """
        إدراك عدة عملات
        
        Args:
            symbols_data: بيانات العملات
            
        Returns:
            حالات الإدراك
        """
        states = {}
        
        for symbol, raw_data in symbols_data.items():
            states[symbol] = self.perceive(symbol, raw_data)
        
        return states
    
    def get_market_overview(
        self,
        states: Dict[str, PerceptionState]
    ) -> Dict[str, Any]:
        """
        الحصول على نظرة عامة على السوق
        
        Args:
            states: حالات الإدراك
            
        Returns:
            النظرة العامة
        """
        if not states:
            return {}
        
        # حساب المتوسطات
        avg_quality = np.mean([s.data_quality for s in states.values()])
        
        # حساب اتجاه السوق العام
        bullish_count = 0
        bearish_count = 0
        
        for state in states.values():
            if state.features.get('candle_direction', 0) > 0:
                bullish_count += 1
            else:
                bearish_count += 1
        
        market_direction = 'BULLISH' if bullish_count > bearish_count else 'BEARISH'
        
        # حساب متوسط المشاعر
        sentiments = [
            s.sentiment.fear_greed_index
            for s in states.values()
            if s.sentiment
        ]
        avg_sentiment = np.mean(sentiments) if sentiments else 50
        
        return {
            'total_symbols': len(states),
            'avg_data_quality': avg_quality,
            'market_direction': market_direction,
            'bullish_ratio': bullish_count / len(states),
            'avg_fear_greed': avg_sentiment,
            'timestamp': datetime.now().isoformat()
        }
    
    # ═══════════════════════════════════════════════════════════════
    # STATUS
    # ═══════════════════════════════════════════════════════════════
    
    def get_status(self) -> Dict[str, Any]:
        """الحصول على حالة الطبقة"""
        return {
            'total_perceptions': self.stats['total_perceptions'],
            'avg_data_quality': self.stats['data_quality_avg'],
            'cached_symbols': list(self.data_cache.keys()),
            'cache_sizes': {
                symbol: len(data)
                for symbol, data in self.data_cache.items()
            }
        }


# ═══════════════════════════════════════════════════════════════
# STANDALONE EXECUTION
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # اختبار طبقة الإدراك
    perception = PerceptionLayer()
    
    raw_data = {
        'ohlcv': {
            'open': 50000,
            'high': 51000,
            'low': 49500,
            'close': 50500,
            'volume': 1000000
        },
        'orderbook': {
            'bids': [[50400, 10], [50300, 20], [50200, 30]],
            'asks': [[50600, 15], [50700, 25], [50800, 35]]
        },
        'sentiment': {
            'fear_greed': 65,
            'social_sentiment': 0.3,
            'news_sentiment': 0.2,
            'whale_activity': 'accumulating'
        },
        'features': {
            'rsi_14': 55,
            'macd': 0.5,
            'macd_signal': 0.3
        }
    }
    
    state = perception.perceive('BTCUSDT', raw_data)
    
    print("👁️ Perception State:")
    print(f"Symbol: {state.symbol}")
    print(f"Data Quality: {state.data_quality:.2%}")
    print(f"Sources: {state.sources_available}")
    print(f"\nFeatures: {list(state.features.keys())}")
    print(f"\nStatus: {perception.get_status()}")
