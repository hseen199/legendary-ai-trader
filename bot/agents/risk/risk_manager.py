"""
Legendary Trading System - Risk Manager Agent
نظام التداول الخارق - وكيل إدارة المخاطر

يدير المخاطر ويحمي رأس المال من الخسائر الكبيرة.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging
import math

from ...core.base_agent import RiskManagerAgent
from ...core.types import (
    RiskAssessment, Portfolio, Position, SignalType
)


class RiskManagerAgent(RiskManagerAgent):
    """
    وكيل إدارة المخاطر.
    
    المسؤوليات:
    - تقييم مخاطر الصفقات
    - حساب حجم المركز الأمثل
    - مراقبة الحدود والقيود
    - حماية رأس المال
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(name="RiskManager", config=config)
        
        # إعدادات المخاطر
        risk_config = config.get("risk", {})
        self.max_portfolio_risk = risk_config.get("max_portfolio_risk", 0.02)
        self.max_single_trade_risk = risk_config.get("max_single_trade_risk", 0.01)
        self.max_daily_loss = risk_config.get("max_daily_loss", 0.05)
        self.max_drawdown = risk_config.get("max_drawdown", 0.15)
        self.default_stop_loss = risk_config.get("default_stop_loss", 0.02)
        self.use_kelly = risk_config.get("use_kelly_criterion", True)
        self.kelly_fraction = risk_config.get("kelly_fraction", 0.25)
        
        # إعدادات التداول
        trading_config = config.get("trading", {})
        self.max_open_trades = trading_config.get("max_open_trades", 10)
        self.min_trade_amount = trading_config.get("min_trade_amount_usdt", 10)
        self.max_trade_amount = trading_config.get("max_trade_amount_usdt", 1000)
        
        # حالة المخاطر
        self._daily_pnl = 0.0
        self._daily_start = datetime.utcnow().date()
        self._current_drawdown = 0.0
        self._peak_balance = 0.0
        self._trade_history: List[Dict] = []
        self._blocked_symbols: Dict[str, datetime] = {}
    
    async def initialize(self) -> bool:
        """تهيئة وكيل إدارة المخاطر."""
        self.logger.info("تهيئة وكيل إدارة المخاطر...")
        return True
    
    async def process(self, data: Any) -> Any:
        """معالجة البيانات."""
        return await self.assess_risk(data.get("symbol"), data.get("decision", {}))
    
    async def shutdown(self) -> None:
        """إيقاف الوكيل."""
        self.logger.info("إيقاف وكيل إدارة المخاطر")
    
    async def assess_risk(self, symbol: str, 
                         decision: Dict[str, Any]) -> Dict[str, Any]:
        """
        تقييم مخاطر الصفقة.
        
        Args:
            symbol: رمز العملة
            decision: قرار التداول المقترح
            
        Returns:
            تقييم المخاطر
        """
        self._update_activity()
        
        try:
            warnings = []
            
            # التحقق من الحظر
            if self._is_symbol_blocked(symbol):
                return {
                    "approved": False,
                    "reason": "الرمز محظور مؤقتاً",
                    "risk_score": 1.0,
                    "warnings": ["الرمز محظور بسبب خسائر سابقة"]
                }
            
            # التحقق من الحد اليومي
            if self._check_daily_limit():
                return {
                    "approved": False,
                    "reason": "تم الوصول للحد اليومي للخسائر",
                    "risk_score": 1.0,
                    "warnings": ["تم إيقاف التداول لهذا اليوم"]
                }
            
            # التحقق من السحب الأقصى
            if self._current_drawdown >= self.max_drawdown:
                return {
                    "approved": False,
                    "reason": "تم الوصول للحد الأقصى للسحب",
                    "risk_score": 1.0,
                    "warnings": ["يجب مراجعة الاستراتيجية"]
                }
            
            # حساب مقاييس المخاطر
            volatility = self._calculate_volatility(decision)
            liquidity_score = self._assess_liquidity(decision)
            correlation_risk = self._assess_correlation(symbol)
            
            # حساب نتيجة المخاطر الإجمالية
            risk_score = self._calculate_risk_score(
                volatility, liquidity_score, correlation_risk, decision
            )
            
            # تحديد مستويات الخروج
            stop_loss = decision.get("stop_loss") or self._calculate_stop_loss(decision)
            take_profit = decision.get("take_profit") or self._calculate_take_profit(decision)
            
            # حساب الحد الأقصى لحجم المركز
            max_position = self._calculate_max_position(
                decision, risk_score, stop_loss
            )
            
            # جمع التحذيرات
            if volatility > 0.05:
                warnings.append(f"تقلب عالي ({volatility*100:.1f}%)")
            if liquidity_score < 0.5:
                warnings.append("سيولة منخفضة")
            if correlation_risk > 0.7:
                warnings.append("ارتباط عالي مع مراكز موجودة")
            if risk_score > 0.7:
                warnings.append("مخاطر إجمالية مرتفعة")
            
            return {
                "approved": risk_score < 0.8,
                "risk_score": risk_score,
                "volatility": volatility,
                "liquidity_score": liquidity_score,
                "correlation_risk": correlation_risk,
                "max_position_size": max_position,
                "recommended_stop_loss": stop_loss,
                "recommended_take_profit": take_profit,
                "warnings": warnings,
                "risk_reward_ratio": self._calculate_risk_reward(
                    decision.get("entry_price", 0), stop_loss, take_profit
                )
            }
            
        except Exception as e:
            self.logger.error(f"خطأ في تقييم المخاطر: {e}")
            self._handle_error(e)
            return {
                "approved": False,
                "reason": f"خطأ: {str(e)}",
                "risk_score": 1.0,
                "warnings": ["فشل تقييم المخاطر"]
            }
    
    async def approve_trade(self, decision: Dict[str, Any],
                           risk_assessment: Dict[str, Any]) -> bool:
        """
        الموافقة على الصفقة أو رفضها.
        
        Args:
            decision: قرار التداول
            risk_assessment: تقييم المخاطر
            
        Returns:
            True للموافقة، False للرفض
        """
        # التحقق من الموافقة الأساسية
        if not risk_assessment.get("approved", False):
            self.logger.info(
                f"رفض الصفقة: {risk_assessment.get('reason', 'غير محدد')}"
            )
            return False
        
        # التحقق من نسبة المخاطرة للعائد
        risk_reward = risk_assessment.get("risk_reward_ratio", 0)
        if risk_reward < 1.5:
            self.logger.info(f"رفض الصفقة: نسبة مخاطرة/عائد ضعيفة ({risk_reward:.2f})")
            return False
        
        # التحقق من الثقة
        confidence = decision.get("confidence", 0)
        if confidence < 0.5:
            self.logger.info(f"رفض الصفقة: ثقة منخفضة ({confidence:.2f})")
            return False
        
        # التحقق من عدد المراكز المفتوحة
        # (يحتاج للتكامل مع نظام المحفظة)
        
        self.logger.info(
            f"الموافقة على الصفقة: {decision.get('symbol')} "
            f"(المخاطر: {risk_assessment['risk_score']:.2f}, "
            f"R/R: {risk_reward:.2f})"
        )
        
        return True
    
    async def calculate_position_size(self, symbol: str,
                                     risk_assessment: Dict[str, Any]) -> float:
        """
        حساب حجم المركز الأمثل.
        
        Args:
            symbol: رمز العملة
            risk_assessment: تقييم المخاطر
            
        Returns:
            حجم المركز بالعملة الأساسية
        """
        max_position = risk_assessment.get("max_position_size", 0)
        
        # تطبيق Kelly Criterion إذا مفعل
        if self.use_kelly:
            kelly_size = self._kelly_criterion(risk_assessment)
            max_position = min(max_position, kelly_size)
        
        # التأكد من الحدود
        position_size = max(self.min_trade_amount, min(max_position, self.max_trade_amount))
        
        self.logger.info(f"حجم المركز المحسوب لـ {symbol}: {position_size:.2f} USDT")
        
        return position_size
    
    def _is_symbol_blocked(self, symbol: str) -> bool:
        """التحقق من حظر الرمز."""
        if symbol in self._blocked_symbols:
            block_time = self._blocked_symbols[symbol]
            if datetime.utcnow() < block_time:
                return True
            else:
                del self._blocked_symbols[symbol]
        return False
    
    def _check_daily_limit(self) -> bool:
        """التحقق من الحد اليومي للخسائر."""
        # إعادة تعيين إذا يوم جديد
        if datetime.utcnow().date() != self._daily_start:
            self._daily_pnl = 0.0
            self._daily_start = datetime.utcnow().date()
        
        return self._daily_pnl <= -self.max_daily_loss
    
    def _calculate_volatility(self, decision: Dict) -> float:
        """حساب التقلب."""
        # استخدام ATR إذا متاح
        analysis_summary = decision.get("analysis_summary", {})
        atr = analysis_summary.get("atr")
        entry_price = decision.get("entry_price", 1)
        
        if atr and entry_price:
            return atr / entry_price
        
        # قيمة افتراضية
        return 0.03
    
    def _assess_liquidity(self, decision: Dict) -> float:
        """تقييم السيولة."""
        # يحتاج للتكامل مع بيانات السوق
        # قيمة افتراضية للعملات الرئيسية
        return 0.8
    
    def _assess_correlation(self, symbol: str) -> float:
        """تقييم الارتباط مع المراكز الموجودة."""
        # يحتاج للتكامل مع نظام المحفظة
        return 0.3
    
    def _calculate_risk_score(self, volatility: float, liquidity: float,
                             correlation: float, decision: Dict) -> float:
        """حساب نتيجة المخاطر الإجمالية."""
        # أوزان العوامل
        weights = {
            "volatility": 0.30,
            "liquidity": 0.25,
            "correlation": 0.20,
            "confidence": 0.25
        }
        
        # تطبيع العوامل
        vol_score = min(1.0, volatility / 0.1)  # 10% = مخاطر قصوى
        liq_score = 1 - liquidity  # سيولة منخفضة = مخاطر عالية
        corr_score = correlation
        conf_score = 1 - decision.get("confidence", 0.5)  # ثقة منخفضة = مخاطر عالية
        
        risk_score = (
            vol_score * weights["volatility"] +
            liq_score * weights["liquidity"] +
            corr_score * weights["correlation"] +
            conf_score * weights["confidence"]
        )
        
        return min(1.0, max(0.0, risk_score))
    
    def _calculate_stop_loss(self, decision: Dict) -> float:
        """حساب مستوى وقف الخسارة."""
        entry_price = decision.get("entry_price", 0)
        action = decision.get("action", "buy")
        
        if not entry_price:
            return 0
        
        # استخدام ATR إذا متاح
        atr = decision.get("analysis_summary", {}).get("atr")
        
        if atr:
            stop_distance = atr * 2
        else:
            stop_distance = entry_price * self.default_stop_loss
        
        if action == "buy":
            return entry_price - stop_distance
        else:
            return entry_price + stop_distance
    
    def _calculate_take_profit(self, decision: Dict) -> float:
        """حساب مستوى جني الأرباح."""
        entry_price = decision.get("entry_price", 0)
        stop_loss = decision.get("stop_loss") or self._calculate_stop_loss(decision)
        action = decision.get("action", "buy")
        
        if not entry_price or not stop_loss:
            return 0
        
        # نسبة مخاطرة/عائد 1:2 على الأقل
        risk = abs(entry_price - stop_loss)
        reward = risk * 2
        
        if action == "buy":
            return entry_price + reward
        else:
            return entry_price - reward
    
    def _calculate_max_position(self, decision: Dict, risk_score: float,
                               stop_loss: float) -> float:
        """حساب الحد الأقصى لحجم المركز."""
        entry_price = decision.get("entry_price", 0)
        
        if not entry_price or not stop_loss:
            return self.min_trade_amount
        
        # حساب المخاطرة لكل وحدة
        risk_per_unit = abs(entry_price - stop_loss) / entry_price
        
        if risk_per_unit == 0:
            return self.min_trade_amount
        
        # الحد الأقصى بناءً على مخاطرة الصفقة الواحدة
        # (يحتاج للتكامل مع رصيد المحفظة)
        portfolio_value = 10000  # قيمة افتراضية
        max_risk_amount = portfolio_value * self.max_single_trade_risk
        
        max_position = max_risk_amount / risk_per_unit
        
        # تعديل بناءً على نتيجة المخاطر
        max_position *= (1 - risk_score * 0.5)
        
        return max(self.min_trade_amount, min(max_position, self.max_trade_amount))
    
    def _calculate_risk_reward(self, entry: float, stop_loss: float,
                              take_profit: float) -> float:
        """حساب نسبة المخاطرة للعائد."""
        if not all([entry, stop_loss, take_profit]):
            return 0
        
        risk = abs(entry - stop_loss)
        reward = abs(take_profit - entry)
        
        if risk == 0:
            return 0
        
        return reward / risk
    
    def _kelly_criterion(self, risk_assessment: Dict) -> float:
        """حساب حجم المركز باستخدام Kelly Criterion."""
        # نحتاج لإحصائيات الصفقات السابقة
        # قيم افتراضية
        win_rate = 0.55
        avg_win = 0.03
        avg_loss = 0.02
        
        if avg_loss == 0:
            return self.max_trade_amount
        
        # Kelly = W - (1-W)/R
        # حيث W = معدل الفوز، R = نسبة الربح/الخسارة
        win_loss_ratio = avg_win / avg_loss
        kelly = win_rate - ((1 - win_rate) / win_loss_ratio)
        
        # استخدام جزء من Kelly للحذر
        kelly *= self.kelly_fraction
        
        # تطبيق على رأس المال
        portfolio_value = 10000  # قيمة افتراضية
        kelly_size = portfolio_value * max(0, kelly)
        
        return min(kelly_size, self.max_trade_amount)
    
    def update_trade_result(self, symbol: str, pnl: float,
                           pnl_percentage: float) -> None:
        """تحديث نتيجة الصفقة."""
        self._daily_pnl += pnl_percentage
        
        self._trade_history.append({
            "timestamp": datetime.utcnow().isoformat(),
            "symbol": symbol,
            "pnl": pnl,
            "pnl_percentage": pnl_percentage
        })
        
        # حظر الرمز إذا خسارة كبيرة
        if pnl_percentage < -0.05:  # خسارة أكثر من 5%
            self._blocked_symbols[symbol] = datetime.utcnow() + timedelta(hours=24)
            self.logger.warning(f"تم حظر {symbol} لمدة 24 ساعة بسبب خسارة كبيرة")
    
    def update_drawdown(self, current_balance: float) -> None:
        """تحديث السحب."""
        if current_balance > self._peak_balance:
            self._peak_balance = current_balance
        
        if self._peak_balance > 0:
            self._current_drawdown = (self._peak_balance - current_balance) / self._peak_balance
    
    def get_risk_status(self) -> Dict[str, Any]:
        """الحصول على حالة المخاطر."""
        return {
            "daily_pnl": self._daily_pnl,
            "current_drawdown": self._current_drawdown,
            "peak_balance": self._peak_balance,
            "blocked_symbols": list(self._blocked_symbols.keys()),
            "daily_limit_reached": self._check_daily_limit(),
            "drawdown_limit_reached": self._current_drawdown >= self.max_drawdown,
            "recent_trades": self._trade_history[-10:]
        }
