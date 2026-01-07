"""
Legendary Trading System - On-Chain Analyst Agent
نظام التداول الخارق - وكيل محلل البيانات على السلسلة

يحلل بيانات البلوك تشين لتتبع تحركات الحيتان والتدفقات.
"""

from datetime import datetime
from typing import Dict, List, Any, Optional
import logging

from ...core.base_agent import AnalystAgent
from ...core.types import (
    AnalysisResult, SignalType, AnalystType, OnChainData
)


class OnChainAnalystAgent(AnalystAgent):
    """
    وكيل محلل بيانات On-Chain.
    
    يحلل:
    - تحركات الحيتان (المعاملات الكبيرة)
    - تدفقات المنصات (الإيداع/السحب)
    - العناوين النشطة
    - مؤشرات السلسلة (MVRV, SOPR, etc.)
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(
            name="OnChainAnalyst",
            config=config,
            analyst_type="onchain"
        )
        self.weight = config.get("analyst_weights", {}).get("onchain", 0.20)
        
        # عتبات التحليل
        self.whale_threshold_usd = config.get("whale_threshold", 1_000_000)
        self.significant_flow_ratio = config.get("significant_flow_ratio", 0.1)
    
    async def initialize(self) -> bool:
        """تهيئة محلل On-Chain."""
        self.logger.info("تهيئة محلل On-Chain...")
        return True
    
    async def process(self, data: Any) -> Any:
        """معالجة البيانات."""
        return await self.analyze(data.get("symbol"), data)
    
    async def shutdown(self) -> None:
        """إيقاف المحلل."""
        self.logger.info("إيقاف محلل On-Chain")
    
    async def analyze(self, symbol: str, data: Dict[str, Any]) -> AnalysisResult:
        """
        تحليل بيانات On-Chain.
        
        Args:
            symbol: رمز العملة
            data: بيانات السلسلة
            
        Returns:
            نتيجة التحليل
        """
        self._update_activity()
        
        try:
            # جمع بيانات On-Chain
            onchain_data = self._extract_onchain_data(symbol, data)
            
            # تحليل تحركات الحيتان
            whale_analysis = self._analyze_whale_movements(onchain_data)
            
            # تحليل تدفقات المنصات
            flow_analysis = self._analyze_exchange_flows(onchain_data)
            
            # تحليل نشاط الشبكة
            network_analysis = self._analyze_network_activity(onchain_data)
            
            # دمج التحليلات
            final_analysis = self._combine_analyses(
                whale_analysis, flow_analysis, network_analysis
            )
            
            return AnalysisResult(
                analyst_type=AnalystType.ONCHAIN,
                symbol=symbol,
                timestamp=datetime.utcnow(),
                signal=final_analysis["signal"],
                confidence=final_analysis["confidence"],
                reasoning=final_analysis["reasoning"],
                data={
                    "whale_score": whale_analysis["score"],
                    "flow_score": flow_analysis["score"],
                    "network_score": network_analysis["score"],
                    "whale_transactions": onchain_data.get("whale_transactions", []),
                    "net_flow": onchain_data.get("net_flow", 0)
                }
            )
            
        except Exception as e:
            self.logger.error(f"خطأ في تحليل On-Chain: {e}")
            self._handle_error(e)
            return self._create_neutral_result(symbol, f"خطأ: {str(e)}")
    
    def _extract_onchain_data(self, symbol: str, data: Dict) -> Dict[str, Any]:
        """استخراج بيانات On-Chain من البيانات الخام."""
        onchain = data.get("onchain", {})
        
        return {
            "symbol": symbol,
            "whale_transactions": onchain.get("whale_transactions", []),
            "exchange_inflow": onchain.get("exchange_inflow", 0),
            "exchange_outflow": onchain.get("exchange_outflow", 0),
            "net_flow": onchain.get("net_flow", 
                onchain.get("exchange_inflow", 0) - onchain.get("exchange_outflow", 0)),
            "active_addresses": onchain.get("active_addresses", 0),
            "active_addresses_change": onchain.get("active_addresses_change", 0),
            "transaction_count": onchain.get("transaction_count", 0),
            "transaction_count_change": onchain.get("transaction_count_change", 0),
            "large_transactions": onchain.get("large_transactions", 0),
            "mvrv_ratio": onchain.get("mvrv_ratio"),
            "sopr": onchain.get("sopr"),
            "nupl": onchain.get("nupl"),
            "supply_on_exchanges": onchain.get("supply_on_exchanges"),
            "supply_on_exchanges_change": onchain.get("supply_on_exchanges_change", 0)
        }
    
    def _analyze_whale_movements(self, data: Dict) -> Dict[str, Any]:
        """تحليل تحركات الحيتان."""
        whale_txs = data.get("whale_transactions", [])
        
        if not whale_txs:
            return {
                "score": 0,
                "signal": SignalType.NEUTRAL,
                "description": "لا توجد تحركات حيتان ملحوظة"
            }
        
        # تحليل اتجاه التحركات
        buy_volume = 0
        sell_volume = 0
        exchange_deposits = 0
        exchange_withdrawals = 0
        
        for tx in whale_txs:
            amount = tx.get("amount_usd", 0)
            tx_type = tx.get("type", "")
            
            if tx_type == "exchange_deposit":
                exchange_deposits += amount
            elif tx_type == "exchange_withdrawal":
                exchange_withdrawals += amount
            elif tx_type == "buy" or tx.get("to_exchange", False) == False:
                buy_volume += amount
            elif tx_type == "sell" or tx.get("to_exchange", True):
                sell_volume += amount
        
        # حساب النتيجة
        total_volume = buy_volume + sell_volume
        if total_volume > 0:
            whale_score = (buy_volume - sell_volume) / total_volume
        else:
            whale_score = 0
        
        # تعديل بناءً على تدفقات المنصات
        net_exchange_flow = exchange_deposits - exchange_withdrawals
        if net_exchange_flow > self.whale_threshold_usd:
            whale_score -= 0.2  # إيداعات كبيرة = ضغط بيع محتمل
        elif net_exchange_flow < -self.whale_threshold_usd:
            whale_score += 0.2  # سحوبات كبيرة = تجميع
        
        whale_score = max(-1, min(1, whale_score))
        
        # تحديد الإشارة
        if whale_score > 0.3:
            signal = SignalType.BUY
            description = "الحيتان تتجمع"
        elif whale_score < -0.3:
            signal = SignalType.SELL
            description = "الحيتان تبيع"
        else:
            signal = SignalType.NEUTRAL
            description = "نشاط حيتان متوازن"
        
        return {
            "score": whale_score,
            "signal": signal,
            "description": description,
            "buy_volume": buy_volume,
            "sell_volume": sell_volume,
            "exchange_deposits": exchange_deposits,
            "exchange_withdrawals": exchange_withdrawals
        }
    
    def _analyze_exchange_flows(self, data: Dict) -> Dict[str, Any]:
        """تحليل تدفقات المنصات."""
        inflow = data.get("exchange_inflow", 0)
        outflow = data.get("exchange_outflow", 0)
        net_flow = data.get("net_flow", inflow - outflow)
        supply_change = data.get("supply_on_exchanges_change", 0)
        
        # حساب نسبة التدفق
        total_flow = inflow + outflow
        if total_flow > 0:
            flow_ratio = net_flow / total_flow
        else:
            flow_ratio = 0
        
        # تحليل التدفق
        if net_flow < 0 and abs(flow_ratio) > self.significant_flow_ratio:
            # تدفق خارجي كبير = صعودي
            score = min(1, abs(flow_ratio) * 2)
            signal = SignalType.BUY
            description = f"تدفق خارجي قوي من المنصات ({abs(net_flow):,.0f})"
        elif net_flow > 0 and abs(flow_ratio) > self.significant_flow_ratio:
            # تدفق داخلي كبير = هبوطي
            score = max(-1, -abs(flow_ratio) * 2)
            signal = SignalType.SELL
            description = f"تدفق داخلي قوي للمنصات ({net_flow:,.0f})"
        else:
            score = 0
            signal = SignalType.NEUTRAL
            description = "تدفقات متوازنة"
        
        # تعديل بناءً على تغير العرض على المنصات
        if supply_change < -0.05:  # انخفاض 5%
            score += 0.2
            description += " | انخفاض العرض على المنصات"
        elif supply_change > 0.05:  # ارتفاع 5%
            score -= 0.2
            description += " | ارتفاع العرض على المنصات"
        
        return {
            "score": max(-1, min(1, score)),
            "signal": signal,
            "description": description,
            "net_flow": net_flow,
            "inflow": inflow,
            "outflow": outflow
        }
    
    def _analyze_network_activity(self, data: Dict) -> Dict[str, Any]:
        """تحليل نشاط الشبكة."""
        active_addresses = data.get("active_addresses", 0)
        address_change = data.get("active_addresses_change", 0)
        tx_count = data.get("transaction_count", 0)
        tx_change = data.get("transaction_count_change", 0)
        
        # تحليل النمو
        signals = []
        descriptions = []
        
        # نمو العناوين النشطة
        if address_change > 0.1:  # نمو 10%
            signals.append(0.4)
            descriptions.append("نمو قوي في العناوين النشطة")
        elif address_change > 0.05:
            signals.append(0.2)
            descriptions.append("نمو معتدل في العناوين النشطة")
        elif address_change < -0.1:
            signals.append(-0.4)
            descriptions.append("انخفاض حاد في العناوين النشطة")
        elif address_change < -0.05:
            signals.append(-0.2)
            descriptions.append("انخفاض معتدل في العناوين النشطة")
        
        # نمو المعاملات
        if tx_change > 0.15:
            signals.append(0.3)
            descriptions.append("نشاط معاملات مرتفع")
        elif tx_change < -0.15:
            signals.append(-0.3)
            descriptions.append("نشاط معاملات منخفض")
        
        # حساب النتيجة
        if signals:
            score = sum(signals) / len(signals)
        else:
            score = 0
        
        # تحديد الإشارة
        if score > 0.2:
            signal = SignalType.BUY
        elif score < -0.2:
            signal = SignalType.SELL
        else:
            signal = SignalType.NEUTRAL
        
        return {
            "score": score,
            "signal": signal,
            "description": " | ".join(descriptions) if descriptions else "نشاط شبكة عادي",
            "active_addresses": active_addresses,
            "transaction_count": tx_count
        }
    
    def _analyze_onchain_indicators(self, data: Dict) -> Dict[str, Any]:
        """تحليل مؤشرات On-Chain المتقدمة."""
        mvrv = data.get("mvrv_ratio")
        sopr = data.get("sopr")
        nupl = data.get("nupl")
        
        signals = []
        descriptions = []
        
        # MVRV (Market Value to Realized Value)
        if mvrv is not None:
            if mvrv < 1:
                signals.append(0.5)
                descriptions.append(f"MVRV منخفض ({mvrv:.2f}) - منطقة تجميع")
            elif mvrv > 3.5:
                signals.append(-0.5)
                descriptions.append(f"MVRV مرتفع ({mvrv:.2f}) - منطقة توزيع")
            elif mvrv > 2.5:
                signals.append(-0.3)
                descriptions.append(f"MVRV مرتفع نسبياً ({mvrv:.2f})")
        
        # SOPR (Spent Output Profit Ratio)
        if sopr is not None:
            if sopr < 0.95:
                signals.append(0.4)
                descriptions.append("SOPR يشير لاستسلام البائعين")
            elif sopr > 1.05:
                signals.append(-0.3)
                descriptions.append("SOPR يشير لجني الأرباح")
        
        # NUPL (Net Unrealized Profit/Loss)
        if nupl is not None:
            if nupl < 0:
                signals.append(0.4)
                descriptions.append("NUPL سلبي - خسائر غير محققة")
            elif nupl > 0.75:
                signals.append(-0.5)
                descriptions.append("NUPL مرتفع جداً - طمع شديد")
            elif nupl > 0.5:
                signals.append(-0.3)
                descriptions.append("NUPL مرتفع - منطقة حذر")
        
        if signals:
            score = sum(signals) / len(signals)
        else:
            score = 0
        
        return {
            "score": score,
            "description": " | ".join(descriptions) if descriptions else "مؤشرات On-Chain محايدة"
        }
    
    def _combine_analyses(self, whale: Dict, flow: Dict, network: Dict) -> Dict[str, Any]:
        """دمج جميع التحليلات."""
        # أوزان التحليلات
        weights = {
            "whale": 0.40,
            "flow": 0.35,
            "network": 0.25
        }
        
        # حساب النتيجة المرجحة
        weighted_score = (
            whale["score"] * weights["whale"] +
            flow["score"] * weights["flow"] +
            network["score"] * weights["network"]
        )
        
        # تحديد الإشارة النهائية
        if weighted_score >= 0.5:
            signal = SignalType.STRONG_BUY
        elif weighted_score >= 0.25:
            signal = SignalType.BUY
        elif weighted_score >= 0.1:
            signal = SignalType.WEAK_BUY
        elif weighted_score <= -0.5:
            signal = SignalType.STRONG_SELL
        elif weighted_score <= -0.25:
            signal = SignalType.SELL
        elif weighted_score <= -0.1:
            signal = SignalType.WEAK_SELL
        else:
            signal = SignalType.NEUTRAL
        
        # حساب الثقة
        scores = [whale["score"], flow["score"], network["score"]]
        avg_score = sum(scores) / len(scores)
        variance = sum((s - avg_score) ** 2 for s in scores) / len(scores)
        consistency = max(0, 1 - variance * 2)
        confidence = abs(weighted_score) * consistency
        
        # بناء التفسير
        reasoning_parts = []
        if abs(whale["score"]) > 0.2:
            reasoning_parts.append(whale["description"])
        if abs(flow["score"]) > 0.2:
            reasoning_parts.append(flow["description"])
        if abs(network["score"]) > 0.2:
            reasoning_parts.append(network["description"])
        
        reasoning = "تحليل On-Chain: " + " | ".join(reasoning_parts) if reasoning_parts else "تحليل On-Chain محايد"
        
        return {
            "signal": signal,
            "confidence": min(1.0, confidence),
            "score": weighted_score,
            "reasoning": reasoning
        }
    
    def _create_neutral_result(self, symbol: str, reason: str) -> AnalysisResult:
        """إنشاء نتيجة محايدة."""
        return AnalysisResult(
            analyst_type=AnalystType.ONCHAIN,
            symbol=symbol,
            timestamp=datetime.utcnow(),
            signal=SignalType.NEUTRAL,
            confidence=0.0,
            reasoning=reason,
            data={}
        )
