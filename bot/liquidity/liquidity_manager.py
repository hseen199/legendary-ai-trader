"""
Legendary Trading System - Liquidity Management System
نظام التداول الخارق - نظام إدارة السيولة

نظام متقدم لتحليل وإدارة السيولة.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
from enum import Enum
import logging
import numpy as np


class LiquidityLevel(Enum):
    """مستويات السيولة"""
    VERY_HIGH = "very_high"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"
    VERY_LOW = "very_low"
    CRITICAL = "critical"


class ExecutionStrategy(Enum):
    """استراتيجيات التنفيذ"""
    MARKET = "market"           # سوق فوري
    LIMIT = "limit"             # محدد
    TWAP = "twap"               # متوسط مرجح بالوقت
    VWAP = "vwap"               # متوسط مرجح بالحجم
    ICEBERG = "iceberg"         # جبل الجليد
    ADAPTIVE = "adaptive"       # تكيفي


@dataclass
class OrderBookSnapshot:
    """لقطة دفتر الأوامر"""
    timestamp: datetime
    symbol: str
    
    # العروض (Bids)
    bids: List[Tuple[float, float]]  # (سعر، كمية)
    
    # الطلبات (Asks)
    asks: List[Tuple[float, float]]  # (سعر، كمية)
    
    # مقاييس محسوبة
    spread: float = 0.0
    spread_percentage: float = 0.0
    mid_price: float = 0.0
    imbalance: float = 0.0  # -1 إلى 1


@dataclass
class LiquidityAnalysis:
    """تحليل السيولة"""
    timestamp: datetime
    symbol: str
    level: LiquidityLevel
    
    # المقاييس
    spread: float
    spread_percentage: float
    depth_bid: float  # عمق العروض
    depth_ask: float  # عمق الطلبات
    imbalance: float
    
    # تقديرات الانزلاق
    slippage_1k: float    # انزلاق لـ 1000$
    slippage_10k: float   # انزلاق لـ 10000$
    slippage_100k: float  # انزلاق لـ 100000$
    
    # التوصيات
    max_order_size: float
    recommended_strategy: ExecutionStrategy
    optimal_execution_time: str


@dataclass
class ExecutionPlan:
    """خطة التنفيذ"""
    id: str
    symbol: str
    side: str  # buy/sell
    total_quantity: float
    
    # التقسيم
    chunks: List[Dict[str, Any]]
    
    # التوقيت
    start_time: datetime
    estimated_duration: timedelta
    
    # التقديرات
    estimated_avg_price: float
    estimated_slippage: float
    estimated_cost: float
    
    # الاستراتيجية
    strategy: ExecutionStrategy


class LiquidityManager:
    """
    نظام إدارة السيولة.
    
    يوفر:
    - تحليل عمق السوق (Order Book)
    - تجنب الانزلاق السعري
    - تقسيم الأوامر الكبيرة
    - اختيار أفضل وقت للتنفيذ
    """
    
    def __init__(self, exchange_client=None, config: Dict[str, Any] = None):
        self.logger = logging.getLogger("LiquidityManager")
        self.exchange = exchange_client
        self.config = config or {}
        
        # تاريخ دفتر الأوامر
        self.orderbook_history: Dict[str, deque] = {}
        
        # تحليلات السيولة
        self.liquidity_cache: Dict[str, LiquidityAnalysis] = {}
        
        # خطط التنفيذ النشطة
        self.active_plans: Dict[str, ExecutionPlan] = {}
        
        # عتبات السيولة
        self.thresholds = {
            "min_spread_pct": 0.001,      # 0.1%
            "max_spread_pct": 0.01,       # 1%
            "min_depth": 10000,           # 10k$
            "critical_depth": 1000,       # 1k$
            "max_slippage": 0.005,        # 0.5%
            "chunk_percentage": 0.02      # 2% من العمق
        }
        
        # إحصائيات
        self.stats = {
            "analyses_performed": 0,
            "orders_split": 0,
            "slippage_saved": 0.0,
            "execution_plans_created": 0
        }
    
    async def analyze_liquidity(self, 
                               symbol: str,
                               orderbook: Dict[str, Any] = None) -> LiquidityAnalysis:
        """
        تحليل السيولة لرمز معين.
        
        Args:
            symbol: الرمز
            orderbook: دفتر الأوامر (اختياري)
            
        Returns:
            تحليل السيولة
        """
        # الحصول على دفتر الأوامر
        if orderbook is None and self.exchange:
            orderbook = await self._fetch_orderbook(symbol)
        
        if not orderbook:
            return self._default_analysis(symbol)
        
        # إنشاء لقطة
        snapshot = self._create_snapshot(symbol, orderbook)
        
        # حفظ في التاريخ
        if symbol not in self.orderbook_history:
            self.orderbook_history[symbol] = deque(maxlen=100)
        self.orderbook_history[symbol].append(snapshot)
        
        # حساب المقاييس
        depth_bid = sum(price * qty for price, qty in snapshot.bids[:20])
        depth_ask = sum(price * qty for price, qty in snapshot.asks[:20])
        
        # حساب الانزلاق
        slippage_1k = self._calculate_slippage(snapshot, 1000, "buy")
        slippage_10k = self._calculate_slippage(snapshot, 10000, "buy")
        slippage_100k = self._calculate_slippage(snapshot, 100000, "buy")
        
        # تحديد مستوى السيولة
        level = self._classify_liquidity(
            snapshot.spread_percentage,
            depth_bid,
            depth_ask,
            slippage_10k
        )
        
        # تحديد الاستراتيجية الموصى بها
        strategy = self._recommend_strategy(level, slippage_10k)
        
        # حساب الحجم الأقصى
        max_order = min(depth_bid, depth_ask) * self.thresholds["chunk_percentage"]
        
        # تحديد أفضل وقت
        optimal_time = self._find_optimal_time(symbol)
        
        analysis = LiquidityAnalysis(
            timestamp=datetime.utcnow(),
            symbol=symbol,
            level=level,
            spread=snapshot.spread,
            spread_percentage=snapshot.spread_percentage,
            depth_bid=depth_bid,
            depth_ask=depth_ask,
            imbalance=snapshot.imbalance,
            slippage_1k=slippage_1k,
            slippage_10k=slippage_10k,
            slippage_100k=slippage_100k,
            max_order_size=max_order,
            recommended_strategy=strategy,
            optimal_execution_time=optimal_time
        )
        
        # تخزين مؤقت
        self.liquidity_cache[symbol] = analysis
        self.stats["analyses_performed"] += 1
        
        self.logger.debug(f"سيولة {symbol}: {level.value}")
        
        return analysis
    
    async def _fetch_orderbook(self, symbol: str) -> Optional[Dict[str, Any]]:
        """جلب دفتر الأوامر."""
        try:
            if self.exchange:
                return await self.exchange.fetch_order_book(symbol, limit=50)
        except Exception as e:
            self.logger.error(f"خطأ في جلب دفتر الأوامر: {e}")
        return None
    
    def _create_snapshot(self, 
                        symbol: str,
                        orderbook: Dict[str, Any]) -> OrderBookSnapshot:
        """إنشاء لقطة دفتر الأوامر."""
        bids = [(float(b[0]), float(b[1])) for b in orderbook.get("bids", [])]
        asks = [(float(a[0]), float(a[1])) for a in orderbook.get("asks", [])]
        
        if bids and asks:
            best_bid = bids[0][0]
            best_ask = asks[0][0]
            spread = best_ask - best_bid
            mid_price = (best_bid + best_ask) / 2
            spread_percentage = spread / mid_price
            
            # حساب عدم التوازن
            bid_volume = sum(qty for _, qty in bids[:10])
            ask_volume = sum(qty for _, qty in asks[:10])
            total_volume = bid_volume + ask_volume
            imbalance = (bid_volume - ask_volume) / total_volume if total_volume > 0 else 0
        else:
            spread = 0
            spread_percentage = 0
            mid_price = 0
            imbalance = 0
        
        return OrderBookSnapshot(
            timestamp=datetime.utcnow(),
            symbol=symbol,
            bids=bids,
            asks=asks,
            spread=spread,
            spread_percentage=spread_percentage,
            mid_price=mid_price,
            imbalance=imbalance
        )
    
    def _calculate_slippage(self,
                           snapshot: OrderBookSnapshot,
                           amount_usd: float,
                           side: str) -> float:
        """حساب الانزلاق المتوقع."""
        if side == "buy":
            orders = snapshot.asks
        else:
            orders = snapshot.bids
        
        if not orders or snapshot.mid_price == 0:
            return 0.01  # 1% افتراضي
        
        remaining = amount_usd
        total_cost = 0
        total_qty = 0
        
        for price, qty in orders:
            order_value = price * qty
            
            if remaining <= order_value:
                # نستهلك جزء من هذا المستوى
                qty_needed = remaining / price
                total_cost += remaining
                total_qty += qty_needed
                remaining = 0
                break
            else:
                # نستهلك كل هذا المستوى
                total_cost += order_value
                total_qty += qty
                remaining -= order_value
        
        if total_qty == 0:
            return 0.01
        
        avg_price = total_cost / total_qty
        slippage = abs(avg_price - snapshot.mid_price) / snapshot.mid_price
        
        return slippage
    
    def _classify_liquidity(self,
                           spread_pct: float,
                           depth_bid: float,
                           depth_ask: float,
                           slippage: float) -> LiquidityLevel:
        """تصنيف مستوى السيولة."""
        min_depth = min(depth_bid, depth_ask)
        
        # حالة حرجة
        if min_depth < self.thresholds["critical_depth"] or slippage > 0.02:
            return LiquidityLevel.CRITICAL
        
        # سيولة منخفضة جداً
        if spread_pct > self.thresholds["max_spread_pct"] or slippage > 0.01:
            return LiquidityLevel.VERY_LOW
        
        # سيولة منخفضة
        if spread_pct > 0.005 or min_depth < self.thresholds["min_depth"]:
            return LiquidityLevel.LOW
        
        # سيولة عالية جداً
        if spread_pct < 0.0005 and min_depth > 100000 and slippage < 0.001:
            return LiquidityLevel.VERY_HIGH
        
        # سيولة عالية
        if spread_pct < 0.001 and min_depth > 50000:
            return LiquidityLevel.HIGH
        
        return LiquidityLevel.NORMAL
    
    def _recommend_strategy(self,
                           level: LiquidityLevel,
                           slippage: float) -> ExecutionStrategy:
        """تحديد استراتيجية التنفيذ الموصى بها."""
        if level in [LiquidityLevel.VERY_HIGH, LiquidityLevel.HIGH]:
            return ExecutionStrategy.MARKET
        
        if level == LiquidityLevel.NORMAL:
            if slippage < 0.002:
                return ExecutionStrategy.LIMIT
            else:
                return ExecutionStrategy.TWAP
        
        if level == LiquidityLevel.LOW:
            return ExecutionStrategy.TWAP
        
        if level in [LiquidityLevel.VERY_LOW, LiquidityLevel.CRITICAL]:
            return ExecutionStrategy.ICEBERG
        
        return ExecutionStrategy.ADAPTIVE
    
    def _find_optimal_time(self, symbol: str) -> str:
        """تحديد أفضل وقت للتنفيذ."""
        # تحليل التاريخ
        if symbol in self.orderbook_history:
            history = list(self.orderbook_history[symbol])
            if len(history) >= 10:
                # البحث عن أفضل سبريد
                best_spread_time = min(history, key=lambda s: s.spread_percentage)
                return best_spread_time.timestamp.strftime("%H:%M")
        
        # أوقات افتراضية (ساعات التداول النشطة)
        hour = datetime.utcnow().hour
        if 8 <= hour <= 16:
            return "الآن (ساعات نشطة)"
        elif 20 <= hour <= 23:
            return "الآن (جلسة آسيا)"
        else:
            return "انتظر حتى 08:00 UTC"
    
    async def create_execution_plan(self,
                                   symbol: str,
                                   side: str,
                                   quantity: float,
                                   max_slippage: float = None) -> ExecutionPlan:
        """
        إنشاء خطة تنفيذ.
        
        Args:
            symbol: الرمز
            side: الجانب (buy/sell)
            quantity: الكمية
            max_slippage: الانزلاق الأقصى المقبول
            
        Returns:
            خطة التنفيذ
        """
        # تحليل السيولة
        analysis = await self.analyze_liquidity(symbol)
        
        max_slippage = max_slippage or self.thresholds["max_slippage"]
        
        # تحديد عدد الأجزاء
        if quantity <= analysis.max_order_size:
            num_chunks = 1
        else:
            num_chunks = int(np.ceil(quantity / analysis.max_order_size))
            num_chunks = min(num_chunks, 20)  # حد أقصى 20 جزء
        
        chunk_size = quantity / num_chunks
        
        # تحديد الفترة الزمنية
        if analysis.recommended_strategy == ExecutionStrategy.TWAP:
            interval_seconds = 60  # دقيقة بين كل جزء
        elif analysis.recommended_strategy == ExecutionStrategy.VWAP:
            interval_seconds = 30
        else:
            interval_seconds = 10
        
        # إنشاء الأجزاء
        chunks = []
        current_time = datetime.utcnow()
        
        for i in range(num_chunks):
            chunk = {
                "index": i,
                "quantity": chunk_size,
                "scheduled_time": current_time + timedelta(seconds=i * interval_seconds),
                "status": "pending",
                "strategy": analysis.recommended_strategy.value
            }
            chunks.append(chunk)
        
        # تقدير السعر المتوسط
        if analysis.level in [LiquidityLevel.VERY_HIGH, LiquidityLevel.HIGH]:
            estimated_slippage = analysis.slippage_1k
        elif analysis.level == LiquidityLevel.NORMAL:
            estimated_slippage = analysis.slippage_10k
        else:
            estimated_slippage = analysis.slippage_100k
        
        # إنشاء الخطة
        plan = ExecutionPlan(
            id=f"{symbol}_{datetime.utcnow().timestamp()}",
            symbol=symbol,
            side=side,
            total_quantity=quantity,
            chunks=chunks,
            start_time=current_time,
            estimated_duration=timedelta(seconds=num_chunks * interval_seconds),
            estimated_avg_price=0,  # يُحسب لاحقاً
            estimated_slippage=estimated_slippage,
            estimated_cost=quantity * estimated_slippage,
            strategy=analysis.recommended_strategy
        )
        
        self.active_plans[plan.id] = plan
        self.stats["execution_plans_created"] += 1
        
        if num_chunks > 1:
            self.stats["orders_split"] += 1
        
        self.logger.info(
            f"خطة تنفيذ: {symbol} {side} {quantity} "
            f"({num_chunks} أجزاء، {analysis.recommended_strategy.value})"
        )
        
        return plan
    
    async def execute_chunk(self,
                           plan_id: str,
                           chunk_index: int) -> Dict[str, Any]:
        """
        تنفيذ جزء من الخطة.
        
        Args:
            plan_id: معرف الخطة
            chunk_index: فهرس الجزء
            
        Returns:
            نتيجة التنفيذ
        """
        if plan_id not in self.active_plans:
            return {"error": "خطة غير موجودة"}
        
        plan = self.active_plans[plan_id]
        
        if chunk_index >= len(plan.chunks):
            return {"error": "فهرس غير صالح"}
        
        chunk = plan.chunks[chunk_index]
        
        # تحديث السيولة قبل التنفيذ
        analysis = await self.analyze_liquidity(plan.symbol)
        
        # فحص إذا كانت السيولة مناسبة
        if analysis.level == LiquidityLevel.CRITICAL:
            chunk["status"] = "delayed"
            return {
                "status": "delayed",
                "reason": "سيولة حرجة",
                "retry_after": 60
            }
        
        # محاكاة التنفيذ (في الواقع يتم التنفيذ عبر البورصة)
        result = {
            "chunk_index": chunk_index,
            "quantity": chunk["quantity"],
            "status": "executed",
            "executed_at": datetime.utcnow().isoformat(),
            "slippage": analysis.slippage_1k
        }
        
        chunk["status"] = "executed"
        chunk["result"] = result
        
        return result
    
    def should_execute_now(self, symbol: str) -> Tuple[bool, str]:
        """
        تحديد إذا يجب التنفيذ الآن.
        
        Args:
            symbol: الرمز
            
        Returns:
            (يجب التنفيذ؟, السبب)
        """
        if symbol not in self.liquidity_cache:
            return True, "لا توجد بيانات سيولة"
        
        analysis = self.liquidity_cache[symbol]
        
        # فحص العمر
        age = (datetime.utcnow() - analysis.timestamp).total_seconds()
        if age > 60:
            return True, "بيانات قديمة"
        
        # فحص السيولة
        if analysis.level == LiquidityLevel.CRITICAL:
            return False, "سيولة حرجة - انتظر"
        
        if analysis.level == LiquidityLevel.VERY_LOW:
            return False, "سيولة منخفضة جداً - انتظر"
        
        # فحص عدم التوازن
        if abs(analysis.imbalance) > 0.7:
            return False, f"عدم توازن كبير ({analysis.imbalance:.1%})"
        
        return True, "السيولة مناسبة"
    
    def get_slippage_estimate(self,
                             symbol: str,
                             amount_usd: float) -> float:
        """
        تقدير الانزلاق.
        
        Args:
            symbol: الرمز
            amount_usd: المبلغ بالدولار
            
        Returns:
            الانزلاق المتوقع
        """
        if symbol not in self.liquidity_cache:
            return 0.005  # 0.5% افتراضي
        
        analysis = self.liquidity_cache[symbol]
        
        # تقدير بناءً على المبلغ
        if amount_usd <= 1000:
            return analysis.slippage_1k
        elif amount_usd <= 10000:
            # تداخل خطي
            ratio = (amount_usd - 1000) / 9000
            return analysis.slippage_1k + ratio * (analysis.slippage_10k - analysis.slippage_1k)
        elif amount_usd <= 100000:
            ratio = (amount_usd - 10000) / 90000
            return analysis.slippage_10k + ratio * (analysis.slippage_100k - analysis.slippage_10k)
        else:
            # تقدير خطي للمبالغ الكبيرة
            return analysis.slippage_100k * (amount_usd / 100000)
    
    def _default_analysis(self, symbol: str) -> LiquidityAnalysis:
        """تحليل افتراضي."""
        return LiquidityAnalysis(
            timestamp=datetime.utcnow(),
            symbol=symbol,
            level=LiquidityLevel.NORMAL,
            spread=0,
            spread_percentage=0.001,
            depth_bid=50000,
            depth_ask=50000,
            imbalance=0,
            slippage_1k=0.001,
            slippage_10k=0.003,
            slippage_100k=0.01,
            max_order_size=5000,
            recommended_strategy=ExecutionStrategy.LIMIT,
            optimal_execution_time="غير محدد"
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """الحصول على الإحصائيات."""
        return {
            **self.stats,
            "cached_symbols": len(self.liquidity_cache),
            "active_plans": len(self.active_plans)
        }
