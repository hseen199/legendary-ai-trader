"""
Legendary Trading System - Agent Orchestrator
نظام التداول الخارق - منسق الوكلاء

هذا هو قلب النظام الذي ينسق العمل بين جميع الوكلاء.
يعمل كخلية النحل - كل وكيل له دوره ويتعاونون معاً.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Type
from collections import defaultdict

from ..core.base_agent import (
    BaseAgent, AgentMessage, AnalystAgent, 
    ResearcherAgent, TradingAgent, RiskManagerAgent
)
from ..core.types import (
    AnalysisResult, DebateResult, TradingDecision,
    SignalType, SystemState, Portfolio
)


class AgentOrchestrator:
    """
    منسق الوكلاء - يدير التواصل والتنسيق بين جميع الوكلاء.
    
    المسؤوليات:
    1. تسجيل وإدارة الوكلاء
    2. توجيه الرسائل بين الوكلاء
    3. تنسيق تدفق العمل (Pipeline)
    4. مراقبة صحة الوكلاء
    5. إدارة دورة حياة الوكلاء
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        تهيئة المنسق.
        
        Args:
            config: إعدادات النظام
        """
        self.config = config
        self.logger = logging.getLogger("Orchestrator")
        
        # سجلات الوكلاء
        self._agents: Dict[str, BaseAgent] = {}
        self._analysts: Dict[str, AnalystAgent] = {}
        self._researchers: Dict[str, ResearcherAgent] = {}
        self._trader: Optional[TradingAgent] = None
        self._risk_manager: Optional[RiskManagerAgent] = None
        
        # قوائم الرسائل
        self._message_queue: asyncio.Queue = asyncio.Queue()
        self._broadcast_subscribers: Dict[str, List[str]] = defaultdict(list)
        
        # حالة النظام
        self._is_running = False
        self._last_cycle_time: Optional[datetime] = None
        self._cycle_count = 0
        self._errors: List[str] = []
        
        # إعدادات التنسيق
        self._analysis_timeout = config.get("analysis_timeout", 30)
        self._debate_rounds = config.get("max_debate_rounds", 3)
        self._consensus_threshold = config.get("consensus_threshold", 0.7)
    
    # ==========================================
    # إدارة الوكلاء
    # ==========================================
    
    def register_agent(self, agent: BaseAgent) -> None:
        """
        تسجيل وكيل جديد في النظام.
        
        Args:
            agent: الوكيل المراد تسجيله
        """
        if agent.name in self._agents:
            self.logger.warning(f"الوكيل {agent.name} مسجل بالفعل، سيتم استبداله")
        
        self._agents[agent.name] = agent
        
        # تصنيف الوكيل حسب نوعه
        if isinstance(agent, AnalystAgent):
            self._analysts[agent.name] = agent
        elif isinstance(agent, ResearcherAgent):
            self._researchers[agent.name] = agent
        elif isinstance(agent, TradingAgent):
            self._trader = agent
        elif isinstance(agent, RiskManagerAgent):
            self._risk_manager = agent
        
        self.logger.info(f"تم تسجيل الوكيل: {agent.name}")
    
    def unregister_agent(self, agent_name: str) -> None:
        """إلغاء تسجيل وكيل."""
        if agent_name in self._agents:
            agent = self._agents.pop(agent_name)
            
            if agent_name in self._analysts:
                del self._analysts[agent_name]
            elif agent_name in self._researchers:
                del self._researchers[agent_name]
            elif self._trader and self._trader.name == agent_name:
                self._trader = None
            elif self._risk_manager and self._risk_manager.name == agent_name:
                self._risk_manager = None
            
            self.logger.info(f"تم إلغاء تسجيل الوكيل: {agent_name}")
    
    def get_agent(self, agent_name: str) -> Optional[BaseAgent]:
        """الحصول على وكيل بالاسم."""
        return self._agents.get(agent_name)
    
    # ==========================================
    # التواصل بين الوكلاء
    # ==========================================
    
    async def route_message(self, message: AgentMessage) -> None:
        """
        توجيه رسالة إلى الوكيل المستهدف.
        
        Args:
            message: الرسالة المراد توجيهها
        """
        receiver = self._agents.get(message.receiver)
        if receiver:
            await receiver.receive_message(message)
            self.logger.debug(
                f"تم توجيه رسالة من {message.sender} إلى {message.receiver}"
            )
        else:
            self.logger.warning(f"الوكيل المستهدف غير موجود: {message.receiver}")
    
    async def broadcast_message(self, sender: str, message_type: str,
                               content: Any, target_group: str = "all") -> None:
        """
        بث رسالة لمجموعة من الوكلاء.
        
        Args:
            sender: اسم المرسل
            message_type: نوع الرسالة
            content: محتوى الرسالة
            target_group: المجموعة المستهدفة (all, analysts, researchers)
        """
        targets = []
        
        if target_group == "all":
            targets = list(self._agents.keys())
        elif target_group == "analysts":
            targets = list(self._analysts.keys())
        elif target_group == "researchers":
            targets = list(self._researchers.keys())
        
        for target in targets:
            if target != sender:
                message = AgentMessage(
                    sender=sender,
                    receiver=target,
                    message_type=message_type,
                    content=content,
                    timestamp=datetime.utcnow()
                )
                await self.route_message(message)
    
    # ==========================================
    # تدفق العمل الرئيسي
    # ==========================================
    
    async def run_analysis_cycle(self, symbol: str, 
                                market_data: Dict[str, Any]) -> Optional[TradingDecision]:
        """
        تشغيل دورة تحليل كاملة لرمز معين.
        
        هذا هو التدفق الرئيسي:
        1. جمع التحليلات من جميع المحللين
        2. إجراء المناظرة بين الباحثين
        3. اتخاذ قرار التداول
        4. تقييم المخاطر
        5. تنفيذ الصفقة (إذا تمت الموافقة)
        
        Args:
            symbol: رمز العملة
            market_data: بيانات السوق
            
        Returns:
            قرار التداول أو None
        """
        self._cycle_count += 1
        self._last_cycle_time = datetime.utcnow()
        
        self.logger.info(f"بدء دورة التحليل #{self._cycle_count} للرمز {symbol}")
        
        try:
            # المرحلة 1: جمع التحليلات
            analysis_results = await self._collect_analyses(symbol, market_data)
            if not analysis_results:
                self.logger.warning(f"لم يتم الحصول على تحليلات للرمز {symbol}")
                return None
            
            # المرحلة 2: المناظرة
            debate_result = await self._run_debate(symbol, analysis_results)
            
            # المرحلة 3: قرار التداول
            if not self._trader:
                self.logger.error("وكيل التداول غير مسجل")
                return None
            
            decision = await self._trader.make_decision(
                symbol, analysis_results, debate_result
            )
            
            if not decision or decision.get("action") == "hold":
                self.logger.info(f"قرار الاحتفاظ للرمز {symbol}")
                return None
            
            # المرحلة 4: تقييم المخاطر
            if not self._risk_manager:
                self.logger.error("وكيل إدارة المخاطر غير مسجل")
                return None
            
            risk_assessment = await self._risk_manager.assess_risk(symbol, decision)
            
            # المرحلة 5: الموافقة والتنفيذ
            approved = await self._risk_manager.approve_trade(decision, risk_assessment)
            
            if approved:
                # حساب حجم المركز
                position_size = await self._risk_manager.calculate_position_size(
                    symbol, risk_assessment
                )
                decision["quantity"] = position_size
                
                # تنفيذ الصفقة
                execution_result = await self._trader.execute_trade(decision)
                
                self.logger.info(
                    f"تم تنفيذ صفقة {decision.get('action')} على {symbol}"
                )
                
                return self._create_trading_decision(decision, execution_result)
            else:
                self.logger.info(
                    f"تم رفض الصفقة على {symbol} بسبب المخاطر"
                )
                return None
                
        except Exception as e:
            self.logger.error(f"خطأ في دورة التحليل: {e}")
            self._errors.append(f"{datetime.utcnow()}: {str(e)}")
            return None
    
    async def _collect_analyses(self, symbol: str,
                               market_data: Dict[str, Any]) -> List[AnalysisResult]:
        """
        جمع التحليلات من جميع المحللين بشكل متوازي.
        
        Args:
            symbol: رمز العملة
            market_data: بيانات السوق
            
        Returns:
            قائمة نتائج التحليل
        """
        if not self._analysts:
            return []
        
        # تشغيل جميع المحللين بشكل متوازي
        tasks = []
        for analyst in self._analysts.values():
            if analyst.is_active:
                task = asyncio.create_task(
                    self._run_analyst_with_timeout(analyst, symbol, market_data)
                )
                tasks.append(task)
        
        # انتظار جميع النتائج
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # تصفية النتائج الصالحة
        valid_results = []
        for result in results:
            if isinstance(result, AnalysisResult):
                valid_results.append(result)
            elif isinstance(result, Exception):
                self.logger.error(f"خطأ في التحليل: {result}")
        
        self.logger.info(
            f"تم جمع {len(valid_results)} تحليل من {len(self._analysts)} محلل"
        )
        
        return valid_results
    
    async def _run_analyst_with_timeout(self, analyst: AnalystAgent,
                                       symbol: str,
                                       market_data: Dict[str, Any]) -> AnalysisResult:
        """تشغيل محلل مع حد زمني."""
        try:
            return await asyncio.wait_for(
                analyst.analyze(symbol, market_data),
                timeout=self._analysis_timeout
            )
        except asyncio.TimeoutError:
            self.logger.warning(f"انتهت مهلة المحلل {analyst.name}")
            raise
    
    async def _run_debate(self, symbol: str,
                         analysis_results: List[AnalysisResult]) -> DebateResult:
        """
        إجراء المناظرة بين الباحثين.
        
        Args:
            symbol: رمز العملة
            analysis_results: نتائج التحليل
            
        Returns:
            نتيجة المناظرة
        """
        bullish_researcher = None
        bearish_researcher = None
        
        for researcher in self._researchers.values():
            if researcher.stance == "bullish":
                bullish_researcher = researcher
            elif researcher.stance == "bearish":
                bearish_researcher = researcher
        
        if not bullish_researcher or not bearish_researcher:
            # إذا لم يكن هناك باحثين، نستخدم نتائج التحليل مباشرة
            return self._create_default_debate_result(symbol, analysis_results)
        
        # جمع البحث الأولي
        bullish_report = await bullish_researcher.research(symbol, analysis_results)
        bearish_report = await bearish_researcher.research(symbol, analysis_results)
        
        # جولات المناظرة
        debate_history = []
        for round_num in range(self._debate_rounds):
            # رد المتفائل على المتشائم
            bullish_response = await bullish_researcher.debate(
                bearish_report.get("thesis", "")
            )
            
            # رد المتشائم على المتفائل
            bearish_response = await bearish_researcher.debate(
                bullish_report.get("thesis", "")
            )
            
            debate_history.append({
                "round": round_num + 1,
                "bullish": bullish_response,
                "bearish": bearish_response
            })
        
        # حساب النتيجة النهائية
        return self._calculate_debate_result(
            symbol, bullish_report, bearish_report, debate_history
        )
    
    def _create_default_debate_result(self, symbol: str,
                                     analysis_results: List[AnalysisResult]) -> DebateResult:
        """إنشاء نتيجة مناظرة افتراضية من نتائج التحليل."""
        # حساب متوسط الإشارات
        signal_values = {
            SignalType.STRONG_BUY: 1.0,
            SignalType.BUY: 0.6,
            SignalType.WEAK_BUY: 0.3,
            SignalType.NEUTRAL: 0.0,
            SignalType.WEAK_SELL: -0.3,
            SignalType.SELL: -0.6,
            SignalType.STRONG_SELL: -1.0
        }
        
        total_weight = 0
        weighted_signal = 0
        
        for result in analysis_results:
            weight = result.confidence
            signal_value = signal_values.get(result.signal, 0)
            weighted_signal += signal_value * weight
            total_weight += weight
        
        if total_weight > 0:
            avg_signal = weighted_signal / total_weight
        else:
            avg_signal = 0
        
        # تحديد الإشارة النهائية
        if avg_signal >= 0.5:
            final_verdict = SignalType.BUY
        elif avg_signal <= -0.5:
            final_verdict = SignalType.SELL
        else:
            final_verdict = SignalType.NEUTRAL
        
        return DebateResult(
            symbol=symbol,
            timestamp=datetime.utcnow(),
            bullish_report=None,
            bearish_report=None,
            rounds=0,
            final_verdict=final_verdict,
            consensus_score=abs(avg_signal),
            key_points=[r.reasoning for r in analysis_results[:3]]
        )
    
    def _calculate_debate_result(self, symbol: str,
                                bullish_report: Dict,
                                bearish_report: Dict,
                                debate_history: List[Dict]) -> DebateResult:
        """حساب نتيجة المناظرة النهائية."""
        # تقييم قوة الحجج
        bullish_score = bullish_report.get("confidence", 0.5)
        bearish_score = bearish_report.get("confidence", 0.5)
        
        # تحديد الفائز
        if bullish_score > bearish_score + 0.2:
            final_verdict = SignalType.BUY
        elif bearish_score > bullish_score + 0.2:
            final_verdict = SignalType.SELL
        else:
            final_verdict = SignalType.NEUTRAL
        
        consensus_score = abs(bullish_score - bearish_score)
        
        return DebateResult(
            symbol=symbol,
            timestamp=datetime.utcnow(),
            bullish_report=bullish_report,
            bearish_report=bearish_report,
            rounds=len(debate_history),
            final_verdict=final_verdict,
            consensus_score=consensus_score,
            key_points=self._extract_key_points(debate_history)
        )
    
    def _extract_key_points(self, debate_history: List[Dict]) -> List[str]:
        """استخراج النقاط الرئيسية من المناظرة."""
        key_points = []
        for round_data in debate_history:
            if round_data.get("bullish"):
                key_points.append(f"متفائل: {round_data['bullish'][:100]}")
            if round_data.get("bearish"):
                key_points.append(f"متشائم: {round_data['bearish'][:100]}")
        return key_points[:5]  # أهم 5 نقاط
    
    def _create_trading_decision(self, decision: Dict,
                                execution_result: Dict) -> TradingDecision:
        """إنشاء كائن قرار التداول."""
        from ..core.types import OrderSide, OrderType
        
        return TradingDecision(
            symbol=decision.get("symbol", ""),
            timestamp=datetime.utcnow(),
            action=OrderSide.BUY if decision.get("action") == "buy" else OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=decision.get("quantity", 0),
            price=execution_result.get("price"),
            stop_loss=decision.get("stop_loss"),
            take_profit=decision.get("take_profit"),
            confidence=decision.get("confidence", 0),
            reasoning=decision.get("reasoning", ""),
            analysis_summary=decision.get("analysis_summary", {})
        )
    
    # ==========================================
    # إدارة دورة الحياة
    # ==========================================
    
    async def start_all_agents(self) -> None:
        """تشغيل جميع الوكلاء."""
        self.logger.info("بدء تشغيل جميع الوكلاء...")
        
        tasks = [agent.start() for agent in self._agents.values()]
        await asyncio.gather(*tasks, return_exceptions=True)
        
        self._is_running = True
        self.logger.info(f"تم تشغيل {len(self._agents)} وكيل")
    
    async def stop_all_agents(self) -> None:
        """إيقاف جميع الوكلاء."""
        self.logger.info("إيقاف جميع الوكلاء...")
        
        tasks = [agent.stop() for agent in self._agents.values()]
        await asyncio.gather(*tasks, return_exceptions=True)
        
        self._is_running = False
        self.logger.info("تم إيقاف جميع الوكلاء")
    
    def get_system_state(self) -> Dict[str, Any]:
        """الحصول على حالة النظام."""
        return {
            "is_running": self._is_running,
            "cycle_count": self._cycle_count,
            "last_cycle_time": self._last_cycle_time.isoformat() if self._last_cycle_time else None,
            "agents": {
                name: agent.get_status() 
                for name, agent in self._agents.items()
            },
            "analysts_count": len(self._analysts),
            "researchers_count": len(self._researchers),
            "has_trader": self._trader is not None,
            "has_risk_manager": self._risk_manager is not None,
            "recent_errors": self._errors[-10:]
        }
    
    def health_check(self) -> Dict[str, Any]:
        """فحص صحة النظام."""
        healthy_agents = sum(
            1 for agent in self._agents.values() 
            if agent.is_active and agent._error_count < agent._max_errors
        )
        
        return {
            "status": "healthy" if healthy_agents == len(self._agents) else "degraded",
            "total_agents": len(self._agents),
            "healthy_agents": healthy_agents,
            "unhealthy_agents": len(self._agents) - healthy_agents,
            "error_count": len(self._errors)
        }
