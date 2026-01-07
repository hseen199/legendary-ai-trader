"""
خدمة تكامل البوت - Bot Integration Service
تدير التواصل بين الخادم الخلفي والبوت التداول الذكي
"""

import asyncio
import logging
import json
from typing import Dict, Any, Optional, List
from datetime import datetime
from enum import Enum
import subprocess
import os
import signal

logger = logging.getLogger(__name__)


class BotStatus(str, Enum):
    """حالات البوت الممكنة"""
    IDLE = "idle"
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


class BotCommand(str, Enum):
    """الأوامر التي يمكن إرسالها للبوت"""
    START = "start"
    STOP = "stop"
    PAUSE = "pause"
    RESUME = "resume"
    STATUS = "status"
    GET_PERFORMANCE = "get_performance"
    GET_TRADES = "get_trades"
    GET_PORTFOLIO = "get_portfolio"


class BotIntegrationService:
    """
    خدمة تكامل البوت مع الخادم الخلفي.
    
    المسؤوليات:
    - إدارة دورة حياة البوت (بدء، إيقاف، إعادة تشغيل)
    - إرسال الأوامر للبوت
    - استقبال البيانات من البوت
    - مراقبة صحة البوت
    - معالجة الأخطاء
    """
    
    def __init__(self):
        self.bot_process: Optional[subprocess.Popen] = None
        self.status: BotStatus = BotStatus.IDLE
        self.last_heartbeat: Optional[datetime] = None
        self.error_count: int = 0
        self.max_retries: int = 3
        self.bot_data_file = "bot_data.json"
        self.command_queue: List[Dict[str, Any]] = []
        self.response_cache: Dict[str, Any] = {}
        
    async def initialize(self) -> bool:
        """
        تهيئة خدمة التكامل.
        
        Returns:
            True إذا نجحت التهيئة
        """
        logger.info("🤖 تهيئة خدمة تكامل البوت...")
        try:
            # التحقق من وجود ملف البوت الرئيسي
            bot_main_path = os.path.join(os.path.dirname(__file__), "../../bot/main_integrated.py")
            if not os.path.exists(bot_main_path):
                logger.error(f"❌ ملف البوت غير موجود: {bot_main_path}")
                self.status = BotStatus.ERROR
                return False
            
            logger.info("✅ تم تهيئة خدمة التكامل بنجاح")
            self.status = BotStatus.IDLE
            return True
            
        except Exception as e:
            logger.error(f"❌ خطأ في تهيئة خدمة التكامل: {e}")
            self.status = BotStatus.ERROR
            return False
    
    async def start_bot(self) -> bool:
        """
        بدء البوت.
        
        Returns:
            True إذا بدأ البوت بنجاح
        """
        logger.info("🚀 بدء البوت...")
        
        if self.status == BotStatus.RUNNING:
            logger.warning("⚠️ البوت يعمل بالفعل")
            return True
        
        try:
            self.status = BotStatus.INITIALIZING
            
            # بدء عملية البوت
            bot_main_path = os.path.join(os.path.dirname(__file__), "../../bot/main_integrated.py")
            
            self.bot_process = subprocess.Popen(
                ["python", bot_main_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=os.path.dirname(bot_main_path)
            )
            
            # انتظر قليلاً للتأكد من أن البوت بدأ
            await asyncio.sleep(2)
            
            if self.bot_process.poll() is None:  # العملية تعمل
                self.status = BotStatus.RUNNING
                self.error_count = 0
                self.last_heartbeat = datetime.utcnow()
                logger.info(f"✅ تم بدء البوت بنجاح (PID: {self.bot_process.pid})")
                return True
            else:
                logger.error("❌ فشل البوت في البدء")
                self.status = BotStatus.ERROR
                return False
                
        except Exception as e:
            logger.error(f"❌ خطأ في بدء البوت: {e}")
            self.status = BotStatus.ERROR
            return False
    
    async def stop_bot(self) -> bool:
        """
        إيقاف البوت بشكل آمن.
        
        Returns:
            True إذا تم إيقاف البوت بنجاح
        """
        logger.info("🛑 إيقاف البوت...")
        
        if self.status == BotStatus.STOPPED or self.status == BotStatus.IDLE:
            logger.warning("⚠️ البوت غير مشغل")
            return True
        
        try:
            self.status = BotStatus.STOPPING
            
            if self.bot_process:
                # إرسال إشارة SIGTERM للإيقاف الآمن
                self.bot_process.send_signal(signal.SIGTERM)
                
                # انتظر 10 ثواني للإيقاف الآمن
                try:
                    self.bot_process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    # إذا لم يتوقف، استخدم SIGKILL
                    logger.warning("⚠️ البوت لم يتوقف بشكل آمن، استخدام SIGKILL...")
                    self.bot_process.kill()
                    self.bot_process.wait()
            
            self.status = BotStatus.STOPPED
            self.bot_process = None
            logger.info("✅ تم إيقاف البوت بنجاح")
            return True
            
        except Exception as e:
            logger.error(f"❌ خطأ في إيقاف البوت: {e}")
            self.status = BotStatus.ERROR
            return False
    
    async def pause_bot(self) -> bool:
        """إيقاف البوت مؤقتاً (بدون إيقاف العملية)"""
        logger.info("⏸️ إيقاف البوت مؤقتاً...")
        
        if self.status != BotStatus.RUNNING:
            logger.warning("⚠️ البوت غير مشغل")
            return False
        
        self.status = BotStatus.PAUSED
        logger.info("✅ تم إيقاف البوت مؤقتاً")
        return True
    
    async def resume_bot(self) -> bool:
        """استئناف البوت المتوقف مؤقتاً"""
        logger.info("▶️ استئناف البوت...")
        
        if self.status != BotStatus.PAUSED:
            logger.warning("⚠️ البوت لم يكن موقوفاً مؤقتاً")
            return False
        
        self.status = BotStatus.RUNNING
        logger.info("✅ تم استئناف البوت")
        return True
    
    async def get_bot_status(self) -> Dict[str, Any]:
        """
        الحصول على حالة البوت الحالية.
        
        Returns:
            قاموس يحتوي على معلومات حالة البوت
        """
        return {
            "status": self.status.value,
            "is_running": self.status == BotStatus.RUNNING,
            "last_heartbeat": self.last_heartbeat.isoformat() if self.last_heartbeat else None,
            "error_count": self.error_count,
            "process_id": self.bot_process.pid if self.bot_process else None,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def get_bot_performance(self) -> Dict[str, Any]:
        """
        الحصول على بيانات أداء البوت.
        
        Returns:
            قاموس يحتوي على معلومات الأداء
        """
        try:
            # في الإنتاج، سيتم قراءة هذه البيانات من قاعدة البيانات
            # أو من ملف يكتبه البوت
            if os.path.exists(self.bot_data_file):
                with open(self.bot_data_file, 'r') as f:
                    return json.load(f)
            else:
                return {
                    "total_trades": 0,
                    "winning_trades": 0,
                    "losing_trades": 0,
                    "win_rate": 0,
                    "total_profit": 0,
                    "total_loss": 0,
                    "net_profit": 0,
                    "roi_percent": 0
                }
        except Exception as e:
            logger.error(f"❌ خطأ في الحصول على بيانات الأداء: {e}")
            return {}
    
    async def get_bot_trades(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        الحصول على قائمة الصفقات الأخيرة.
        
        Args:
            limit: عدد الصفقات المطلوبة
            
        Returns:
            قائمة الصفقات
        """
        try:
            # في الإنتاج، سيتم قراءة هذه البيانات من قاعدة البيانات
            return []
        except Exception as e:
            logger.error(f"❌ خطأ في الحصول على الصفقات: {e}")
            return []
    
    async def get_bot_portfolio(self) -> Dict[str, Any]:
        """
        الحصول على تفاصيل محفظة البوت.
        
        Returns:
            قاموس يحتوي على تفاصيل المحفظة
        """
        try:
            # في الإنتاج، سيتم قراءة هذه البيانات من Binance API
            return {
                "total_balance": 0,
                "usdc_balance": 0,
                "positions": [],
                "last_updated": datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error(f"❌ خطأ في الحصول على بيانات المحفظة: {e}")
            return {}
    
    async def health_check(self) -> bool:
        """
        فحص صحة البوت.
        
        Returns:
            True إذا كان البوت بصحة جيدة
        """
        try:
            if self.status != BotStatus.RUNNING:
                return False
            
            # التحقق من أن العملية تعمل
            if self.bot_process and self.bot_process.poll() is None:
                self.last_heartbeat = datetime.utcnow()
                return True
            else:
                logger.warning("⚠️ البوت توقف بشكل غير متوقع")
                self.status = BotStatus.ERROR
                self.error_count += 1
                
                # محاولة إعادة تشغيل البوت
                if self.error_count < self.max_retries:
                    logger.info(f"🔄 محاولة إعادة تشغيل البوت ({self.error_count}/{self.max_retries})...")
                    await self.start_bot()
                
                return False
                
        except Exception as e:
            logger.error(f"❌ خطأ في فحص صحة البوت: {e}")
            return False
    
    async def shutdown(self):
        """إيقاف الخدمة بشكل آمن"""
        logger.info("🛑 إيقاف خدمة التكامل...")
        await self.stop_bot()
        logger.info("✅ تم إيقاف خدمة التكامل")


# إنشاء instance واحد من الخدمة
bot_service = BotIntegrationService()
