"""
Legendary Trading System - Full Integration
نظام التداول الخارق - التكامل الكامل

الملف الرئيسي الذي يجمع ويدمج جميع مكونات النظام القديمة والجديدة.
"""

import asyncio
import os
import sys
import signal
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List

# إضافة المسار للاستيراد
sys.path.insert(0, str(Path(__file__).parent))

# استيراد الإعدادات
from config.settings import Settings, load_settings

# استيراد المكونات الأصلية
from coordination.orchestrator import AgentOrchestrator
from memory.memory_system import MemorySystem
from protection.protection_system import ProtectionSystem, ProtectionConfig
from llm.llm_integration import LLMManager, LLMConfig
from training.auto_trainer import AutoTrainer, TrainingConfig
from training.data_pipeline import DataPipeline, DataConfig

# استيراد الأنظمة الجديدة المتقدمة
from awareness.self_awareness import SelfAwarenessSystem
from learning_from_mistakes.mistake_learner import MistakeLearningSystem
from market_regime.regime_detector import MarketRegimeDetector
from intuition.ai_intuition import AIIntuitionSystem
from communication.agent_protocol import AgentCommunicationProtocol, AgentRole
from liquidity.liquidity_manager import LiquidityManager
from events.event_system import EventSystem, EventType
from emergency.emergency_system import EmergencySystem, EmergencyLevel

# استيراد العقل المحسن
from mind.inner_dialogue_enhanced import EnhancedInnerDialogue
from mind.reasoning_engine_enhanced import EnhancedReasoningEngine
from mind.strategy_inventor_enhanced import EnhancedStrategyInventor


class LegendaryTradingSystemFull:
    """
    نظام التداول الخارق المتكامل - الإصدار الكامل.
    
    يجمع بين:
    - النظام الأصلي (منسق الوكلاء، الذاكرة، الحماية، التدريب)
    - الأنظمة الجديدة (الوعي الذاتي، التعلم من الأخطاء، الحدس، إلخ)
    - العقل المحسن (حوار داخلي، تفكير، اختراع استراتيجيات)
    
    يعمل مثل خلية النحل - كل مكون له دوره ويتناغم مع الآخرين.
    """
    
    VERSION = "3.0.0"
    
    def __init__(self, config_path: str = None):
        self.logger = self._setup_logging()
        self.logger.info("=" * 70)
        self.logger.info(f"🚀 نظام التداول الخارق V{self.VERSION}")
        self.logger.info("=" * 70)
        
        # تحميل الإعدادات
        self.settings = load_settings(config_path) if config_path else Settings()
        
        # ===== المكونات الأصلية =====
        self.orchestrator: Optional[AgentOrchestrator] = None
        self.memory: Optional[MemorySystem] = None
        self.protection: Optional[ProtectionSystem] = None
        self.llm_manager: Optional[LLMManager] = None
        self.trainer: Optional[AutoTrainer] = None
        self.data_pipeline: Optional[DataPipeline] = None
        
        # ===== الأنظمة الجديدة =====
        self.awareness: Optional[SelfAwarenessSystem] = None
        self.mistake_learner: Optional[MistakeLearningSystem] = None
        self.regime_detector: Optional[MarketRegimeDetector] = None
        self.intuition: Optional[AIIntuitionSystem] = None
        self.communication: Optional[AgentCommunicationProtocol] = None
        self.liquidity: Optional[LiquidityManager] = None
        self.events: Optional[EventSystem] = None
        self.emergency: Optional[EmergencySystem] = None
        
        # ===== العقل المحسن =====
        self.inner_dialogue: Optional[EnhancedInnerDialogue] = None
        self.reasoning: Optional[EnhancedReasoningEngine] = None
        self.strategy_inventor: Optional[EnhancedStrategyInventor] = None
        
        # حالة النظام
        self._running = False
        self._initialized = False
        self._shutdown_event = asyncio.Event()
        self._start_time = None
    
    def _setup_logging(self) -> logging.Logger:
        """إعداد نظام السجلات."""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        logger = logging.getLogger("LegendaryTradingSystem")
        logger.setLevel(logging.DEBUG)
        
        # معالج الملف
        file_handler = logging.FileHandler(
            log_dir / f"system_{datetime.utcnow().strftime('%Y%m%d')}.log",
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        
        # معالج الكونسول
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        if not logger.handlers:
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)
        
        return logger
    
    async def initialize(self) -> bool:
        """
        تهيئة جميع مكونات النظام.
        
        Returns:
            True إذا نجحت التهيئة
        """
        self.logger.info("📦 بدء تهيئة مكونات النظام...")
        
        try:
            # ===== تهيئة المكونات الأصلية =====
            await self._init_core_components()
            
            # ===== تهيئة الأنظمة الجديدة =====
            await self._init_advanced_systems()
            
            # ===== تهيئة العقل المحسن =====
            await self._init_enhanced_mind()
            
            # ===== ربط المكونات =====
            await self._connect_components()
            
            self._initialized = True
            self.logger.info("✅ تم تهيئة جميع مكونات النظام بنجاح")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ فشل في تهيئة النظام: {e}", exc_info=True)
            return False
    
    async def _init_core_components(self):
        """تهيئة المكونات الأساسية."""
        self.logger.info("  [1/3] تهيئة المكونات الأساسية...")
        
        # نظام الذاكرة
        self.logger.info("      - نظام الذاكرة")
        self.memory = MemorySystem({
            "episodic_capacity": self.settings.memory.episodic_capacity,
            "semantic_capacity": self.settings.memory.semantic_capacity,
            "semantic_db": str(Path(self.settings.data_dir) / "semantic_memory.db")
        })
        await self.memory.initialize()
        
        # نظام الحماية
        self.logger.info("      - نظام الحماية")
        protection_config = ProtectionConfig(
            max_daily_loss_percent=self.settings.risk.max_daily_loss,
            max_drawdown_percent=self.settings.risk.max_drawdown,
            max_position_size_percent=self.settings.risk.max_position_size,
            max_open_positions=self.settings.risk.max_open_positions
        )
        self.protection = ProtectionSystem(protection_config)
        await self.protection.initialize(self.settings.trading.initial_capital)
        
        # نظام LLM
        self.logger.info("      - نظام LLM")
        llm_config = LLMConfig(
            model=self.settings.llm.model,
            temperature=self.settings.llm.temperature,
            api_key=os.getenv("OPENAI_API_KEY", "")
        )
        self.llm_manager = LLMManager([llm_config])
        
        # خط أنابيب البيانات
        self.logger.info("      - خط أنابيب البيانات")
        data_config = DataConfig(
            symbols=self.settings.trading.symbols[:100],
            timeframes=self.settings.trading.timeframes,
            lookback_days=self.settings.training.lookback_days,
            cache_dir=str(Path(self.settings.data_dir) / "cache")
        )
        self.data_pipeline = DataPipeline(data_config)
        await self.data_pipeline.initialize()
        
        # منسق الوكلاء
        self.logger.info("      - منسق الوكلاء")
        self.orchestrator = AgentOrchestrator(
            settings=self.settings,
            memory=self.memory,
            protection=self.protection,
            llm_manager=self.llm_manager,
            data_pipeline=self.data_pipeline
        )
        await self.orchestrator.initialize()
        
        # المدرب التلقائي
        self.logger.info("      - المدرب التلقائي")
        training_config = TrainingConfig(
            model_type=self.settings.training.model_type,
            max_episodes=self.settings.training.max_episodes,
            optimize_hyperparams=self.settings.training.optimize_hyperparams,
            checkpoint_dir=str(Path(self.settings.data_dir) / "checkpoints")
        )
        self.trainer = AutoTrainer(training_config)
    
    async def _init_advanced_systems(self):
        """تهيئة الأنظمة المتقدمة الجديدة."""
        self.logger.info("  [2/3] تهيئة الأنظمة المتقدمة...")
        
        # نظام الوعي الذاتي
        self.logger.info("      - نظام الوعي الذاتي")
        self.awareness = SelfAwarenessSystem({
            "confidence_threshold": 0.6,
            "performance_window": 100
        })
        
        # نظام التعلم من الأخطاء
        self.logger.info("      - نظام التعلم من الأخطاء")
        self.mistake_learner = MistakeLearningSystem(self.memory)
        
        # كاشف الأنظمة السوقية
        self.logger.info("      - كاشف الأنظمة السوقية")
        self.regime_detector = MarketRegimeDetector({
            "lookback_period": 50,
            "volatility_threshold": 0.02
        })
        
        # نظام الحدس الاصطناعي
        self.logger.info("      - نظام الحدس الاصطناعي")
        self.intuition = AIIntuitionSystem({
            "pattern_memory_size": 10000,
            "intuition_threshold": 0.7
        })
        
        # بروتوكول التواصل
        self.logger.info("      - بروتوكول التواصل بين الوكلاء")
        self.communication = AgentCommunicationProtocol({
            "max_message_queue": 1000,
            "consensus_threshold": 0.6
        })
        
        # مدير السيولة
        self.logger.info("      - مدير السيولة")
        self.liquidity = LiquidityManager(config={
            "min_depth": 10000,
            "max_slippage": 0.005
        })
        
        # نظام الأحداث
        self.logger.info("      - نظام الأحداث")
        self.events = EventSystem({
            "event_history_size": 1000
        })
        
        # نظام الطوارئ
        self.logger.info("      - نظام الطوارئ")
        self.emergency = EmergencySystem(
            trading_system=self,
            config={
                "flash_crash_threshold": -0.10,
                "max_drawdown_threshold": -0.20
            }
        )
    
    async def _init_enhanced_mind(self):
        """تهيئة العقل المحسن."""
        self.logger.info("  [3/3] تهيئة العقل المحسن...")
        
        mind_config = {
            "llm_manager": self.llm_manager,
            "max_dialogue_turns": 10,
            "reasoning_depth": 5
        }
        
        # الحوار الداخلي المحسن
        self.logger.info("      - الحوار الداخلي المحسن")
        self.inner_dialogue = EnhancedInnerDialogue(
            memory_system=self.memory,
            config=mind_config
        )
        
        # محرك التفكير المحسن
        self.logger.info("      - محرك التفكير المحسن")
        self.reasoning = EnhancedReasoningEngine(
            memory_system=self.memory,
            config=mind_config
        )
        
        # مخترع الاستراتيجيات المحسن
        self.logger.info("      - مخترع الاستراتيجيات المحسن")
        self.strategy_inventor = EnhancedStrategyInventor(
            memory_system=self.memory,
            config=mind_config
        )
    
    async def _connect_components(self):
        """ربط المكونات ببعضها."""
        self.logger.info("  🔗 ربط المكونات...")
        
        # تسجيل الوكلاء في نظام التواصل
        agents_to_register = [
            ("orchestrator", AgentRole.COORDINATOR),
            ("awareness", AgentRole.ANALYST),
            ("regime_detector", AgentRole.ANALYST),
            ("intuition", AgentRole.ANALYST),
            ("mistake_learner", AgentRole.ANALYST),
            ("liquidity", AgentRole.ANALYST),
            ("events", AgentRole.ANALYST),
            ("emergency", AgentRole.RISK_MANAGER),
            ("inner_dialogue", AgentRole.COORDINATOR),
            ("reasoning", AgentRole.RESEARCHER),
            ("strategy_inventor", AgentRole.RESEARCHER),
        ]
        
        for agent_id, role in agents_to_register:
            await self.communication.register_agent(agent_id, role)
        
        # ربط نظام الأحداث بالطوارئ
        self.events.register_handler(
            EventType.PRICE_CRASH,
            self._handle_price_crash
        )
        
        self.logger.info("  ✅ تم ربط المكونات")
    
    async def _handle_price_crash(self, event):
        """معالجة حدث انهيار السعر."""
        self.logger.warning(f"🚨 انهيار سعر: {event.title}")
        # تفعيل نظام الطوارئ
        await self.emergency.emergency_exit(
            reason=event.description,
            symbols=event.affected_symbols
        )
    
    async def start(self):
        """بدء تشغيل النظام."""
        if not self._initialized:
            success = await self.initialize()
            if not success:
                raise RuntimeError("فشل في تهيئة النظام")
        
        self._running = True
        self._start_time = datetime.utcnow()
        self.logger.info("🚀 بدء تشغيل نظام التداول الخارق")
        
        # إعداد معالجات الإشارات
        self._setup_signal_handlers()
        
        try:
            # تشغيل المهام المتوازية
            await asyncio.gather(
                self._run_main_trading_loop(),
                self._run_monitoring_loop(),
                self._run_learning_loop(),
                self._run_event_loop(),
                self._run_awareness_loop(),
                return_exceptions=True
            )
        except asyncio.CancelledError:
            self.logger.info("تم إلغاء المهام")
        finally:
            await self.shutdown()
    
    async def _run_main_trading_loop(self):
        """حلقة التداول الرئيسية."""
        self.logger.info("▶️ بدء حلقة التداول")
        
        while self._running and not self._shutdown_event.is_set():
            try:
                # فحص إمكانية التداول
                can_trade, reason = self.emergency.can_trade()
                if not can_trade:
                    self.logger.warning(f"التداول متوقف: {reason}")
                    await asyncio.sleep(60)
                    continue
                
                # فحص الوعي الذاتي
                awareness_ok, awareness_reason = await self.awareness.should_trade()
                if not awareness_ok:
                    self.logger.info(f"الوعي الذاتي: {awareness_reason}")
                    await asyncio.sleep(30)
                    continue
                
                # جمع البيانات
                market_data = await self._collect_comprehensive_data()
                
                # كشف النظام السوقي
                regime = await self.regime_detector.detect_regime(market_data)
                
                # قراءة الحدس
                intuition_signal = await self.intuition.get_intuition(market_data)
                
                # التفكير العميق
                reasoning_result = await self.reasoning.reason({
                    "market_data": market_data,
                    "regime": regime,
                    "intuition": intuition_signal
                })
                
                # الحوار الداخلي
                dialogue_result = await self.inner_dialogue.deliberate({
                    "reasoning": reasoning_result,
                    "context": market_data
                })
                
                # فحص الأخطاء المحتملة
                if dialogue_result.get("proposed_trade"):
                    warnings = await self.mistake_learner.check_for_repeat_mistake(
                        dialogue_result["proposed_trade"]
                    )
                    if warnings:
                        self.logger.warning(f"تحذيرات من نظام الأخطاء: {warnings}")
                        continue
                
                # تنفيذ دورة التداول
                await self.orchestrator.execute_trading_cycle()
                
                await asyncio.sleep(self.settings.trading.cycle_interval)
                
            except Exception as e:
                self.logger.error(f"خطأ في حلقة التداول: {e}")
                await self.protection.record_trade_result(False, str(e))
                await asyncio.sleep(60)
    
    async def _run_monitoring_loop(self):
        """حلقة المراقبة."""
        self.logger.info("▶️ بدء حلقة المراقبة")
        
        while self._running and not self._shutdown_event.is_set():
            try:
                # جمع الإحصائيات
                stats = await self._collect_full_stats()
                
                # مراقبة الطوارئ
                emergency_level = await self.emergency.monitor(stats)
                
                if emergency_level.value >= EmergencyLevel.ORANGE.value:
                    self.logger.warning(f"⚠️ مستوى الطوارئ: {emergency_level.name}")
                
                # تسجيل الحالة
                self.logger.info(f"📊 حالة النظام: {self._format_stats(stats)}")
                
                # حفظ الحالة
                await self._save_state()
                
                await asyncio.sleep(300)
                
            except Exception as e:
                self.logger.error(f"خطأ في حلقة المراقبة: {e}")
                await asyncio.sleep(60)
    
    async def _run_learning_loop(self):
        """حلقة التعلم المستمر."""
        self.logger.info("▶️ بدء حلقة التعلم")
        
        while self._running and not self._shutdown_event.is_set():
            try:
                # تحليل الأخطاء
                mistake_report = await self.mistake_learner.get_mistake_report()
                
                if mistake_report["summary"]["total_mistakes"] > 0:
                    self.logger.info(f"📚 تعلم من {mistake_report['summary']['total_mistakes']} أخطاء")
                    
                    # تطوير الاستراتيجيات
                    await self.strategy_inventor.evolve_strategies()
                
                # التحقق من الحاجة للتدريب
                if await self._should_train():
                    self.logger.info("🎓 بدء جلسة تدريب...")
                    await self._run_training_session()
                
                await asyncio.sleep(3600)
                
            except Exception as e:
                self.logger.error(f"خطأ في حلقة التعلم: {e}")
                await asyncio.sleep(300)
    
    async def _run_event_loop(self):
        """حلقة معالجة الأحداث."""
        self.logger.info("▶️ بدء حلقة الأحداث")
        
        while self._running and not self._shutdown_event.is_set():
            try:
                # فحص الأحداث المجدولة
                upcoming = await self.events.check_scheduled_events()
                for event in upcoming:
                    self.logger.info(f"📅 حدث قادم: {event.title} في {event.scheduled_time}")
                
                # ملخص الأحداث
                summary = self.events.get_event_summary()
                if summary["total_24h"] > 0:
                    self.logger.debug(f"أحداث آخر 24 ساعة: {summary['total_24h']}")
                
                await asyncio.sleep(60)
                
            except Exception as e:
                self.logger.error(f"خطأ في حلقة الأحداث: {e}")
                await asyncio.sleep(60)
    
    async def _run_awareness_loop(self):
        """حلقة الوعي الذاتي."""
        self.logger.info("▶️ بدء حلقة الوعي الذاتي")
        
        while self._running and not self._shutdown_event.is_set():
            try:
                # تقييم الأداء
                performance = await self._evaluate_performance()
                
                # تحديث الوعي الذاتي
                await self.awareness.update_performance(performance)
                
                # الحصول على تقرير
                report = await self.awareness.get_awareness_report()
                
                if report.get("needs_attention"):
                    self.logger.warning(f"⚠️ تنبيه الوعي الذاتي: {report.get('attention_reason')}")
                
                await asyncio.sleep(120)
                
            except Exception as e:
                self.logger.error(f"خطأ في حلقة الوعي: {e}")
                await asyncio.sleep(120)
    
    async def _collect_comprehensive_data(self) -> Dict[str, Any]:
        """جمع بيانات شاملة."""
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "market_data": await self.data_pipeline.get_latest_data() if self.data_pipeline else {},
            "regime": self.regime_detector.current_state if self.regime_detector else None,
            "intuition": self.intuition.get_market_mood() if self.intuition else {},
            "liquidity": self.liquidity.liquidity_cache if self.liquidity else {},
            "emergency_level": self.emergency.current_level.value if self.emergency else 1
        }
    
    async def _collect_full_stats(self) -> Dict[str, Any]:
        """جمع إحصائيات كاملة."""
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "uptime_hours": (datetime.utcnow() - self._start_time).total_seconds() / 3600 if self._start_time else 0,
            "protection": self.protection.get_status() if self.protection else {},
            "memory": await self.memory.get_stats() if self.memory else {},
            "orchestrator": self.orchestrator.get_status() if self.orchestrator else {},
            "emergency": self.emergency.get_status() if self.emergency else {},
            "communication": self.communication.get_stats() if self.communication else {},
            "events": self.events.get_event_summary() if self.events else {},
            "regime": self.regime_detector.current_state.regime.value if self.regime_detector and self.regime_detector.current_state else "unknown",
            "intuition": self.intuition.get_market_mood() if self.intuition else {}
        }
    
    def _format_stats(self, stats: Dict[str, Any]) -> str:
        """تنسيق الإحصائيات للعرض."""
        return (
            f"Uptime: {stats.get('uptime_hours', 0):.1f}h | "
            f"Regime: {stats.get('regime', 'N/A')} | "
            f"Emergency: {stats.get('emergency', {}).get('level_name', 'N/A')}"
        )
    
    async def _evaluate_performance(self) -> Dict[str, Any]:
        """تقييم الأداء."""
        if self.protection:
            status = self.protection.get_status()
            return {
                "win_rate": status.get("win_rate", 0),
                "profit_factor": status.get("profit_factor", 0),
                "drawdown": status.get("current_drawdown", 0),
                "total_trades": status.get("total_trades", 0)
            }
        return {}
    
    async def _should_train(self) -> bool:
        """التحقق من الحاجة للتدريب."""
        if not self.trainer:
            return False
        
        status = self.trainer.get_training_status()
        
        if status.get("episodes_completed", 0) == 0:
            return True
        
        if status.get("best_sharpe", 0) < 0.5:
            return True
        
        return False
    
    async def _run_training_session(self):
        """تشغيل جلسة تدريب."""
        try:
            symbol = self.settings.trading.symbols[0]
            data = await self.data_pipeline.get_training_data(symbol)
            
            if len(data) == 0:
                self.logger.warning("لا توجد بيانات للتدريب")
                return
            
            train_data, val_data, _ = self.data_pipeline.split_data(data)
            
            from models.drl.ppo_agent import PPOAgent, PPOConfig
            
            config = PPOConfig(state_dim=data.shape[1])
            await self.trainer.initialize(PPOAgent, config)
            
            metrics = await self.trainer.train(train_data, val_data)
            
            self.logger.info(f"✅ انتهى التدريب: Sharpe={metrics.sharpe_ratio:.2f}")
            
        except Exception as e:
            self.logger.error(f"خطأ في التدريب: {e}")
    
    async def _save_state(self):
        """حفظ حالة النظام."""
        try:
            state_dir = Path(self.settings.data_dir) / "state"
            state_dir.mkdir(parents=True, exist_ok=True)
            
            await self.memory.consolidate_all()
            
            if self.orchestrator:
                await self.orchestrator.save_state(state_dir / "orchestrator_state.json")
            
            self.logger.debug("تم حفظ حالة النظام")
            
        except Exception as e:
            self.logger.error(f"خطأ في حفظ الحالة: {e}")
    
    def _setup_signal_handlers(self):
        """إعداد معالجات الإشارات."""
        def signal_handler(sig, frame):
            self.logger.info(f"استلام إشارة {sig}")
            self._running = False
            self._shutdown_event.set()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    # ===== واجهات نظام الطوارئ =====
    
    async def close_all_positions(self) -> List[str]:
        """إغلاق جميع الصفقات."""
        self.logger.warning("🚨 إغلاق جميع الصفقات...")
        if self.orchestrator:
            return await self.orchestrator.close_all_positions()
        return []
    
    async def close_losing_positions(self) -> List[str]:
        """إغلاق الصفقات الخاسرة."""
        self.logger.warning("إغلاق الصفقات الخاسرة...")
        if self.orchestrator:
            return await self.orchestrator.close_losing_positions()
        return []
    
    async def tighten_all_stops(self, multiplier: float):
        """تضييق وقف الخسارة."""
        self.logger.info(f"تضييق وقف الخسارة بمعامل {multiplier}")
        if self.orchestrator:
            await self.orchestrator.tighten_stops(multiplier)
    
    async def reduce_all_positions(self, percentage: float):
        """تقليل جميع الصفقات."""
        self.logger.info(f"تقليل الصفقات بنسبة {percentage}%")
        if self.orchestrator:
            await self.orchestrator.reduce_positions(percentage)
    
    async def shutdown(self):
        """إيقاف النظام."""
        self.logger.info("🛑 بدء إيقاف النظام...")
        
        self._running = False
        
        try:
            await self._save_state()
            
            if self.data_pipeline:
                await self.data_pipeline.close()
            
            if self.orchestrator:
                await self.orchestrator.shutdown()
            
            self.logger.info("✅ تم إيقاف النظام بنجاح")
            
        except Exception as e:
            self.logger.error(f"خطأ أثناء الإيقاف: {e}")
    
    def get_full_status(self) -> Dict[str, Any]:
        """الحصول على الحالة الكاملة."""
        return {
            "version": self.VERSION,
            "is_running": self._running,
            "initialized": self._initialized,
            "start_time": self._start_time.isoformat() if self._start_time else None,
            "components": {
                "core": ["orchestrator", "memory", "protection", "llm", "trainer", "data_pipeline"],
                "advanced": ["awareness", "mistake_learner", "regime_detector", "intuition", 
                           "communication", "liquidity", "events", "emergency"],
                "mind": ["inner_dialogue", "reasoning", "strategy_inventor"]
            }
        }


async def main():
    """نقطة الدخول الرئيسية."""
    config_path = os.getenv("CONFIG_PATH", "config/config.yaml")
    
    system = LegendaryTradingSystemFull(
        config_path if Path(config_path).exists() else None
    )
    
    await system.start()


if __name__ == "__main__":
    asyncio.run(main())
