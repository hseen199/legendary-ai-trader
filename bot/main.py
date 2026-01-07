"""
Legendary Trading System V2 - Main Entry Point
نظام التداول الخارق الإصدار الثاني - نقطة الدخول الرئيسية

النظام المتكامل للتداول بالذكاء الاصطناعي على Binance Spot.
"""

import asyncio
import os
import sys
import signal
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

# إضافة المسار للاستيراد
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import Settings, load_settings
from coordination.orchestrator import AgentOrchestrator
from memory.memory_system import MemorySystem
from protection.protection_system import ProtectionSystem, ProtectionConfig
from llm.llm_integration import LLMManager, LLMConfig
from training.auto_trainer import AutoTrainer, TrainingConfig
from training.data_pipeline import DataPipeline, DataConfig


class LegendaryTradingSystem:
    """
    نظام التداول الخارق - النظام الرئيسي.
    
    يدمج جميع المكونات ويدير دورة حياة النظام.
    """
    
    VERSION = "2.0.0"
    
    def __init__(self, config_path: str = None):
        self.logger = self._setup_logging()
        self.logger.info(f"تهيئة نظام التداول الخارق V{self.VERSION}")
        
        # تحميل الإعدادات
        self.settings = load_settings(config_path) if config_path else Settings()
        
        # المكونات الرئيسية
        self.orchestrator: Optional[AgentOrchestrator] = None
        self.memory: Optional[MemorySystem] = None
        self.protection: Optional[ProtectionSystem] = None
        self.llm_manager: Optional[LLMManager] = None
        self.trainer: Optional[AutoTrainer] = None
        self.data_pipeline: Optional[DataPipeline] = None
        
        # حالة النظام
        self._running = False
        self._initialized = False
        self._shutdown_event = asyncio.Event()
    
    def _setup_logging(self) -> logging.Logger:
        """إعداد نظام السجلات."""
        # إنشاء مجلد السجلات
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # إعداد المسجل
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
        
        # التنسيق
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    async def initialize(self) -> bool:
        """
        تهيئة جميع مكونات النظام.
        
        Returns:
            True إذا نجحت التهيئة
        """
        self.logger.info("بدء تهيئة مكونات النظام...")
        
        try:
            # 1. تهيئة نظام الذاكرة
            self.logger.info("تهيئة نظام الذاكرة...")
            self.memory = MemorySystem({
                "episodic_capacity": self.settings.memory.episodic_capacity,
                "semantic_capacity": self.settings.memory.semantic_capacity,
                "semantic_db": str(Path(self.settings.data_dir) / "semantic_memory.db")
            })
            await self.memory.initialize()
            
            # 2. تهيئة نظام الحماية
            self.logger.info("تهيئة نظام الحماية...")
            protection_config = ProtectionConfig(
                max_daily_loss_percent=self.settings.risk.max_daily_loss,
                max_drawdown_percent=self.settings.risk.max_drawdown,
                max_position_size_percent=self.settings.risk.max_position_size,
                max_open_positions=self.settings.risk.max_open_positions
            )
            self.protection = ProtectionSystem(protection_config)
            await self.protection.initialize(self.settings.trading.initial_capital)
            
            # 3. تهيئة LLM
            self.logger.info("تهيئة نظام LLM...")
            llm_config = LLMConfig(
                model=self.settings.llm.model,
                temperature=self.settings.llm.temperature,
                api_key=os.getenv("OPENAI_API_KEY", "")
            )
            self.llm_manager = LLMManager([llm_config])
            
            # 4. تهيئة خط أنابيب البيانات
            self.logger.info("تهيئة خط أنابيب البيانات...")
            data_config = DataConfig(
                symbols=self.settings.trading.symbols[:100],  # أهم 100 عملة
                timeframes=self.settings.trading.timeframes,
                lookback_days=self.settings.training.lookback_days,
                cache_dir=str(Path(self.settings.data_dir) / "cache")
            )
            self.data_pipeline = DataPipeline(data_config)
            await self.data_pipeline.initialize()
            
            # 5. تهيئة منسق الوكلاء
            self.logger.info("تهيئة منسق الوكلاء...")
            self.orchestrator = AgentOrchestrator(
                settings=self.settings,
                memory=self.memory,
                protection=self.protection,
                llm_manager=self.llm_manager,
                data_pipeline=self.data_pipeline
            )
            await self.orchestrator.initialize()
            
            # 6. تهيئة المدرب التلقائي
            self.logger.info("تهيئة نظام التدريب التلقائي...")
            training_config = TrainingConfig(
                model_type=self.settings.training.model_type,
                max_episodes=self.settings.training.max_episodes,
                optimize_hyperparams=self.settings.training.optimize_hyperparams,
                checkpoint_dir=str(Path(self.settings.data_dir) / "checkpoints")
            )
            self.trainer = AutoTrainer(training_config)
            
            self._initialized = True
            self.logger.info("✅ تم تهيئة جميع مكونات النظام بنجاح")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ فشل في تهيئة النظام: {e}", exc_info=True)
            return False
    
    async def start(self):
        """بدء تشغيل النظام."""
        if not self._initialized:
            success = await self.initialize()
            if not success:
                raise RuntimeError("فشل في تهيئة النظام")
        
        self._running = True
        self.logger.info("🚀 بدء تشغيل نظام التداول الخارق")
        
        # إعداد معالجات الإشارات
        self._setup_signal_handlers()
        
        try:
            # تشغيل المهام المتوازية
            await asyncio.gather(
                self._run_trading_loop(),
                self._run_monitoring_loop(),
                self._run_learning_loop(),
                return_exceptions=True
            )
        except asyncio.CancelledError:
            self.logger.info("تم إلغاء المهام")
        finally:
            await self.shutdown()
    
    async def _run_trading_loop(self):
        """حلقة التداول الرئيسية."""
        self.logger.info("بدء حلقة التداول")
        
        while self._running and not self._shutdown_event.is_set():
            try:
                # تنفيذ دورة تداول واحدة
                await self.orchestrator.execute_trading_cycle()
                
                # انتظار قبل الدورة التالية
                await asyncio.sleep(self.settings.trading.cycle_interval)
                
            except Exception as e:
                self.logger.error(f"خطأ في حلقة التداول: {e}")
                await self.protection.record_trade_result(False, str(e))
                await asyncio.sleep(60)  # انتظار قبل المحاولة مرة أخرى
    
    async def _run_monitoring_loop(self):
        """حلقة المراقبة."""
        self.logger.info("بدء حلقة المراقبة")
        
        while self._running and not self._shutdown_event.is_set():
            try:
                # جمع الإحصائيات
                stats = await self._collect_stats()
                
                # تسجيل الحالة
                self.logger.info(f"حالة النظام: {stats}")
                
                # حفظ الحالة
                await self._save_state()
                
                # انتظار
                await asyncio.sleep(300)  # كل 5 دقائق
                
            except Exception as e:
                self.logger.error(f"خطأ في حلقة المراقبة: {e}")
                await asyncio.sleep(60)
    
    async def _run_learning_loop(self):
        """حلقة التعلم المستمر."""
        self.logger.info("بدء حلقة التعلم")
        
        while self._running and not self._shutdown_event.is_set():
            try:
                # التحقق من الحاجة للتدريب
                if await self._should_train():
                    self.logger.info("بدء جلسة تدريب...")
                    await self._run_training_session()
                
                # انتظار
                await asyncio.sleep(3600)  # كل ساعة
                
            except Exception as e:
                self.logger.error(f"خطأ في حلقة التعلم: {e}")
                await asyncio.sleep(300)
    
    async def _collect_stats(self) -> Dict[str, Any]:
        """جمع إحصائيات النظام."""
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "protection": self.protection.get_status(),
            "memory": await self.memory.get_stats(),
            "orchestrator": self.orchestrator.get_status() if self.orchestrator else {},
            "llm": self.llm_manager.clients[0].get_stats() if self.llm_manager else {}
        }
    
    async def _save_state(self):
        """حفظ حالة النظام."""
        try:
            state_dir = Path(self.settings.data_dir) / "state"
            state_dir.mkdir(parents=True, exist_ok=True)
            
            # حفظ الذاكرة
            await self.memory.consolidate_all()
            
            # حفظ حالة المنسق
            if self.orchestrator:
                await self.orchestrator.save_state(state_dir / "orchestrator_state.json")
            
            self.logger.debug("تم حفظ حالة النظام")
            
        except Exception as e:
            self.logger.error(f"خطأ في حفظ الحالة: {e}")
    
    async def _should_train(self) -> bool:
        """التحقق من الحاجة للتدريب."""
        if not self.trainer:
            return False
        
        status = self.trainer.get_training_status()
        
        # التدريب إذا لم يكن هناك نموذج
        if status.get("episodes_completed", 0) == 0:
            return True
        
        # التدريب إذا انخفض الأداء
        if status.get("best_sharpe", 0) < 0.5:
            return True
        
        return False
    
    async def _run_training_session(self):
        """تشغيل جلسة تدريب."""
        try:
            # جلب البيانات
            symbol = self.settings.trading.symbols[0]
            data = await self.data_pipeline.get_training_data(symbol)
            
            if len(data) == 0:
                self.logger.warning("لا توجد بيانات للتدريب")
                return
            
            # تقسيم البيانات
            train_data, val_data, _ = self.data_pipeline.split_data(data)
            
            # التدريب
            from models.drl.ppo_agent import PPOAgent, PPOConfig
            
            config = PPOConfig(state_dim=data.shape[1])
            await self.trainer.initialize(PPOAgent, config)
            
            metrics = await self.trainer.train(train_data, val_data)
            
            self.logger.info(f"انتهى التدريب: Sharpe={metrics.sharpe_ratio:.2f}")
            
        except Exception as e:
            self.logger.error(f"خطأ في التدريب: {e}")
    
    def _setup_signal_handlers(self):
        """إعداد معالجات الإشارات."""
        def signal_handler(sig, frame):
            self.logger.info(f"استلام إشارة {sig}")
            self._running = False
            self._shutdown_event.set()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def shutdown(self):
        """إيقاف النظام."""
        self.logger.info("بدء إيقاف النظام...")
        
        self._running = False
        
        try:
            # حفظ الحالة النهائية
            await self._save_state()
            
            # إغلاق المكونات
            if self.data_pipeline:
                await self.data_pipeline.close()
            
            if self.orchestrator:
                await self.orchestrator.shutdown()
            
            self.logger.info("✅ تم إيقاف النظام بنجاح")
            
        except Exception as e:
            self.logger.error(f"خطأ أثناء الإيقاف: {e}")


async def main():
    """نقطة الدخول الرئيسية."""
    # تحميل الإعدادات من متغيرات البيئة أو ملف
    config_path = os.getenv("CONFIG_PATH", "config/config.yaml")
    
    # إنشاء النظام
    system = LegendaryTradingSystem(config_path if Path(config_path).exists() else None)
    
    # تشغيل النظام
    await system.start()


if __name__ == "__main__":
    asyncio.run(main())
