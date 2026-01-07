"""
Legendary Trading System - Trader Agent
نظام التداول الخارق - وكيل التداول

ينفذ قرارات التداول بناءً على تحليلات المحللين ونتائج المناظرات.
"""

import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging
import hmac
import hashlib
import time
import aiohttp
from urllib.parse import urlencode

from ...core.base_agent import TradingAgent
from ...core.types import (
    AnalysisResult, SignalType, TradingDecision, Order,
    OrderType, OrderSide, OrderStatus
)


class TraderAgent(TradingAgent):
    """
    وكيل التداول الرئيسي.
    
    المسؤوليات:
    - اتخاذ قرارات التداول
    - تنفيذ الأوامر على Binance
    - إدارة الأوامر المفتوحة
    - تتبع الصفقات
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(name="Trader", config=config)
        
        # إعدادات Binance
        self.api_key = config.get("binance", {}).get("api_key", "")
        self.api_secret = config.get("binance", {}).get("api_secret", "")
        self.base_url = config.get("binance", {}).get("base_url", "https://api.binance.com")
        
        # إعدادات التداول
        self.min_confidence = config.get("trading", {}).get("min_confidence", 0.6)
        self.min_trade_amount = config.get("trading", {}).get("min_trade_amount_usdt", 10)
        self.max_trade_amount = config.get("trading", {}).get("max_trade_amount_usdt", 1000)
        
        # حالة التداول
        self._session: Optional[aiohttp.ClientSession] = None
        self._open_orders: Dict[str, Order] = {}
        self._trade_history: List[Dict] = []
    
    async def initialize(self) -> bool:
        """تهيئة وكيل التداول."""
        self.logger.info("تهيئة وكيل التداول...")
        
        # التحقق من المفاتيح
        if not self.api_key or not self.api_secret:
            self.logger.warning("مفاتيح Binance غير مكتملة - وضع المحاكاة")
        
        # إنشاء جلسة HTTP
        self._session = aiohttp.ClientSession()
        
        # اختبار الاتصال
        try:
            await self._test_connection()
            self.logger.info("تم الاتصال بـ Binance بنجاح")
            return True
        except Exception as e:
            self.logger.error(f"فشل الاتصال بـ Binance: {e}")
            return False
    
    async def process(self, data: Any) -> Any:
        """معالجة البيانات."""
        return await self.make_decision(
            data.get("symbol"),
            data.get("analyses", []),
            data.get("debate_result", {})
        )
    
    async def shutdown(self) -> None:
        """إيقاف وكيل التداول."""
        self.logger.info("إيقاف وكيل التداول...")
        
        if self._session:
            await self._session.close()
    
    async def make_decision(self, symbol: str,
                           analysis_results: List[AnalysisResult],
                           debate_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        اتخاذ قرار التداول.
        
        Args:
            symbol: رمز العملة
            analysis_results: نتائج التحليل
            debate_result: نتيجة المناظرة
            
        Returns:
            قرار التداول
        """
        self._update_activity()
        
        try:
            # حساب الإشارة المجمعة
            aggregated = self._aggregate_signals(analysis_results)
            
            # دمج مع نتيجة المناظرة
            final_signal = self._incorporate_debate(aggregated, debate_result)
            
            # تحديد الإجراء
            action = self._determine_action(final_signal)
            
            if action == "hold":
                return {
                    "symbol": symbol,
                    "action": "hold",
                    "reasoning": "لا توجد فرصة تداول واضحة",
                    "signal_strength": final_signal["strength"],
                    "confidence": final_signal["confidence"]
                }
            
            # حساب مستويات الدخول والخروج
            levels = self._calculate_levels(symbol, action, analysis_results)
            
            return {
                "symbol": symbol,
                "action": action,
                "order_type": "market",
                "entry_price": levels.get("entry"),
                "stop_loss": levels.get("stop_loss"),
                "take_profit": levels.get("take_profit"),
                "confidence": final_signal["confidence"],
                "reasoning": self._build_reasoning(analysis_results, debate_result),
                "analysis_summary": {
                    "signal_strength": final_signal["strength"],
                    "bullish_count": aggregated["bullish_count"],
                    "bearish_count": aggregated["bearish_count"],
                    "debate_verdict": debate_result.get("final_verdict", "neutral")
                }
            }
            
        except Exception as e:
            self.logger.error(f"خطأ في اتخاذ القرار: {e}")
            self._handle_error(e)
            return {"symbol": symbol, "action": "hold", "reasoning": f"خطأ: {str(e)}"}
    
    async def execute_trade(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """
        تنفيذ الصفقة.
        
        Args:
            decision: قرار التداول
            
        Returns:
            نتيجة التنفيذ
        """
        symbol = decision.get("symbol")
        action = decision.get("action")
        quantity = decision.get("quantity", 0)
        
        if action == "hold" or quantity <= 0:
            return {"status": "skipped", "reason": "لا يوجد إجراء"}
        
        try:
            # تحويل الإجراء إلى جهة الأمر
            side = "BUY" if action == "buy" else "SELL"
            
            # تنفيذ الأمر
            if decision.get("order_type") == "limit":
                result = await self._place_limit_order(
                    symbol, side, quantity, decision.get("entry_price")
                )
            else:
                result = await self._place_market_order(symbol, side, quantity)
            
            # تسجيل الصفقة
            self._trade_history.append({
                "timestamp": datetime.utcnow().isoformat(),
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "result": result
            })
            
            # وضع أوامر وقف الخسارة وجني الأرباح
            if result.get("status") == "filled":
                await self._place_exit_orders(
                    symbol, side, quantity,
                    decision.get("stop_loss"),
                    decision.get("take_profit")
                )
            
            return result
            
        except Exception as e:
            self.logger.error(f"خطأ في تنفيذ الصفقة: {e}")
            return {"status": "error", "error": str(e)}
    
    def _aggregate_signals(self, analyses: List[AnalysisResult]) -> Dict[str, Any]:
        """تجميع الإشارات من جميع المحللين."""
        if not analyses:
            return {
                "signal": 0,
                "strength": 0,
                "confidence": 0,
                "bullish_count": 0,
                "bearish_count": 0
            }
        
        signal_values = {
            SignalType.STRONG_BUY: 1.0,
            SignalType.BUY: 0.6,
            SignalType.WEAK_BUY: 0.3,
            SignalType.NEUTRAL: 0.0,
            SignalType.WEAK_SELL: -0.3,
            SignalType.SELL: -0.6,
            SignalType.STRONG_SELL: -1.0
        }
        
        # أوزان المحللين
        analyst_weights = {
            "technical": 0.30,
            "fundamental": 0.20,
            "sentiment": 0.15,
            "news": 0.15,
            "onchain": 0.20
        }
        
        weighted_sum = 0
        total_weight = 0
        total_confidence = 0
        bullish_count = 0
        bearish_count = 0
        
        for analysis in analyses:
            signal_value = signal_values.get(analysis.signal, 0)
            weight = analyst_weights.get(analysis.analyst_type.value, 0.2)
            
            # تعديل الوزن بالثقة
            effective_weight = weight * analysis.confidence
            
            weighted_sum += signal_value * effective_weight
            total_weight += effective_weight
            total_confidence += analysis.confidence
            
            if signal_value > 0.1:
                bullish_count += 1
            elif signal_value < -0.1:
                bearish_count += 1
        
        if total_weight > 0:
            avg_signal = weighted_sum / total_weight
        else:
            avg_signal = 0
        
        avg_confidence = total_confidence / len(analyses) if analyses else 0
        
        return {
            "signal": avg_signal,
            "strength": abs(avg_signal),
            "confidence": avg_confidence,
            "bullish_count": bullish_count,
            "bearish_count": bearish_count
        }
    
    def _incorporate_debate(self, aggregated: Dict, debate: Dict) -> Dict[str, Any]:
        """دمج نتيجة المناظرة مع الإشارات المجمعة."""
        if not debate:
            return aggregated
        
        debate_signal = 0
        verdict = debate.get("final_verdict")
        
        if verdict:
            if hasattr(verdict, 'value'):
                verdict = verdict.value
            
            verdict_map = {
                "strong_buy": 0.8,
                "buy": 0.5,
                "weak_buy": 0.2,
                "neutral": 0,
                "weak_sell": -0.2,
                "sell": -0.5,
                "strong_sell": -0.8
            }
            debate_signal = verdict_map.get(verdict, 0)
        
        consensus_score = debate.get("consensus_score", 0.5)
        
        # دمج الإشارات (60% تحليلات، 40% مناظرة)
        final_signal = aggregated["signal"] * 0.6 + debate_signal * 0.4
        
        # تعديل الثقة بناءً على الإجماع
        final_confidence = aggregated["confidence"] * (0.7 + consensus_score * 0.3)
        
        return {
            "signal": final_signal,
            "strength": abs(final_signal),
            "confidence": min(1.0, final_confidence),
            "bullish_count": aggregated["bullish_count"],
            "bearish_count": aggregated["bearish_count"]
        }
    
    def _determine_action(self, signal: Dict) -> str:
        """تحديد الإجراء بناءً على الإشارة."""
        strength = signal["strength"]
        confidence = signal["confidence"]
        signal_value = signal["signal"]
        
        # التحقق من الحد الأدنى للثقة
        if confidence < self.min_confidence:
            return "hold"
        
        # التحقق من قوة الإشارة
        if strength < 0.3:
            return "hold"
        
        # تحديد الاتجاه
        if signal_value > 0.3 and confidence >= self.min_confidence:
            return "buy"
        elif signal_value < -0.3 and confidence >= self.min_confidence:
            return "sell"
        else:
            return "hold"
    
    def _calculate_levels(self, symbol: str, action: str,
                         analyses: List[AnalysisResult]) -> Dict[str, float]:
        """حساب مستويات الدخول والخروج."""
        # استخراج السعر الحالي من التحليلات
        current_price = None
        atr = None
        
        for analysis in analyses:
            if analysis.data:
                indicators = analysis.data.get("indicators", {})
                if indicators.get("current_price"):
                    current_price = indicators["current_price"]
                if indicators.get("atr"):
                    atr = indicators["atr"]
        
        if not current_price:
            return {}
        
        # حساب ATR افتراضي إذا لم يكن متاحاً
        if not atr:
            atr = current_price * 0.02  # 2% افتراضي
        
        # حساب المستويات
        if action == "buy":
            stop_loss = current_price - (atr * 2)
            take_profit = current_price + (atr * 3)
        else:  # sell
            stop_loss = current_price + (atr * 2)
            take_profit = current_price - (atr * 3)
        
        return {
            "entry": current_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "atr": atr
        }
    
    def _build_reasoning(self, analyses: List[AnalysisResult],
                        debate: Dict) -> str:
        """بناء تفسير القرار."""
        reasons = []
        
        # إضافة أسباب من التحليلات
        for analysis in analyses:
            if analysis.confidence > 0.5:
                reasons.append(f"{analysis.analyst_type.value}: {analysis.reasoning[:50]}")
        
        # إضافة نتيجة المناظرة
        if debate:
            key_points = debate.get("key_points", [])
            if key_points:
                reasons.append(f"المناظرة: {key_points[0][:50]}")
        
        return " | ".join(reasons[:3])
    
    # ==========================================
    # Binance API Methods
    # ==========================================
    
    async def _test_connection(self) -> bool:
        """اختبار الاتصال بـ Binance."""
        url = f"{self.base_url}/api/v3/ping"
        async with self._session.get(url) as response:
            return response.status == 200
    
    def _sign_request(self, params: Dict) -> str:
        """توقيع الطلب."""
        query_string = urlencode(params)
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return signature
    
    async def _place_market_order(self, symbol: str, side: str,
                                  quantity: float) -> Dict[str, Any]:
        """وضع أمر سوق."""
        if not self.api_key or not self.api_secret:
            # وضع المحاكاة
            return {
                "status": "simulated",
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "price": 0,
                "orderId": f"SIM_{int(time.time())}"
            }
        
        url = f"{self.base_url}/api/v3/order"
        
        params = {
            "symbol": symbol,
            "side": side,
            "type": "MARKET",
            "quantity": quantity,
            "timestamp": int(time.time() * 1000)
        }
        
        params["signature"] = self._sign_request(params)
        
        headers = {"X-MBX-APIKEY": self.api_key}
        
        async with self._session.post(url, params=params, headers=headers) as response:
            result = await response.json()
            
            if response.status == 200:
                return {
                    "status": "filled",
                    "orderId": result.get("orderId"),
                    "symbol": symbol,
                    "side": side,
                    "quantity": float(result.get("executedQty", 0)),
                    "price": float(result.get("fills", [{}])[0].get("price", 0))
                }
            else:
                return {
                    "status": "error",
                    "error": result.get("msg", "Unknown error")
                }
    
    async def _place_limit_order(self, symbol: str, side: str,
                                 quantity: float, price: float) -> Dict[str, Any]:
        """وضع أمر محدد."""
        if not self.api_key or not self.api_secret:
            return {
                "status": "simulated",
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "price": price,
                "orderId": f"SIM_{int(time.time())}"
            }
        
        url = f"{self.base_url}/api/v3/order"
        
        params = {
            "symbol": symbol,
            "side": side,
            "type": "LIMIT",
            "timeInForce": "GTC",
            "quantity": quantity,
            "price": price,
            "timestamp": int(time.time() * 1000)
        }
        
        params["signature"] = self._sign_request(params)
        
        headers = {"X-MBX-APIKEY": self.api_key}
        
        async with self._session.post(url, params=params, headers=headers) as response:
            result = await response.json()
            
            if response.status == 200:
                return {
                    "status": "open",
                    "orderId": result.get("orderId"),
                    "symbol": symbol,
                    "side": side,
                    "quantity": quantity,
                    "price": price
                }
            else:
                return {
                    "status": "error",
                    "error": result.get("msg", "Unknown error")
                }
    
    async def _place_exit_orders(self, symbol: str, entry_side: str,
                                 quantity: float, stop_loss: float,
                                 take_profit: float) -> None:
        """وضع أوامر الخروج (وقف الخسارة وجني الأرباح)."""
        exit_side = "SELL" if entry_side == "BUY" else "BUY"
        
        # أمر وقف الخسارة
        if stop_loss:
            await self._place_stop_loss_order(symbol, exit_side, quantity, stop_loss)
        
        # أمر جني الأرباح
        if take_profit:
            await self._place_take_profit_order(symbol, exit_side, quantity, take_profit)
    
    async def _place_stop_loss_order(self, symbol: str, side: str,
                                     quantity: float, stop_price: float) -> Dict:
        """وضع أمر وقف الخسارة."""
        if not self.api_key or not self.api_secret:
            return {"status": "simulated", "type": "stop_loss"}
        
        url = f"{self.base_url}/api/v3/order"
        
        params = {
            "symbol": symbol,
            "side": side,
            "type": "STOP_LOSS_LIMIT",
            "timeInForce": "GTC",
            "quantity": quantity,
            "price": stop_price,
            "stopPrice": stop_price,
            "timestamp": int(time.time() * 1000)
        }
        
        params["signature"] = self._sign_request(params)
        headers = {"X-MBX-APIKEY": self.api_key}
        
        async with self._session.post(url, params=params, headers=headers) as response:
            return await response.json()
    
    async def _place_take_profit_order(self, symbol: str, side: str,
                                       quantity: float, price: float) -> Dict:
        """وضع أمر جني الأرباح."""
        if not self.api_key or not self.api_secret:
            return {"status": "simulated", "type": "take_profit"}
        
        url = f"{self.base_url}/api/v3/order"
        
        params = {
            "symbol": symbol,
            "side": side,
            "type": "TAKE_PROFIT_LIMIT",
            "timeInForce": "GTC",
            "quantity": quantity,
            "price": price,
            "stopPrice": price,
            "timestamp": int(time.time() * 1000)
        }
        
        params["signature"] = self._sign_request(params)
        headers = {"X-MBX-APIKEY": self.api_key}
        
        async with self._session.post(url, params=params, headers=headers) as response:
            return await response.json()
    
    async def get_account_balance(self) -> Dict[str, float]:
        """الحصول على رصيد الحساب."""
        if not self.api_key or not self.api_secret:
            return {"USDT": 10000.0}  # رصيد محاكاة
        
        url = f"{self.base_url}/api/v3/account"
        
        params = {"timestamp": int(time.time() * 1000)}
        params["signature"] = self._sign_request(params)
        
        headers = {"X-MBX-APIKEY": self.api_key}
        
        async with self._session.get(url, params=params, headers=headers) as response:
            result = await response.json()
            
            balances = {}
            for balance in result.get("balances", []):
                free = float(balance.get("free", 0))
                if free > 0:
                    balances[balance["asset"]] = free
            
            return balances
    
    async def get_open_orders(self, symbol: str = None) -> List[Dict]:
        """الحصول على الأوامر المفتوحة."""
        if not self.api_key or not self.api_secret:
            return []
        
        url = f"{self.base_url}/api/v3/openOrders"
        
        params = {"timestamp": int(time.time() * 1000)}
        if symbol:
            params["symbol"] = symbol
        
        params["signature"] = self._sign_request(params)
        headers = {"X-MBX-APIKEY": self.api_key}
        
        async with self._session.get(url, params=params, headers=headers) as response:
            return await response.json()
