"""
Deposit Monitor Service
خدمة مراقبة الإيداعات - تتتبع الإيداعات الجديدة وتضيفها تلقائياً
مدمج من نسخة المستخدم (crowdfund) مع تحسينات
"""

import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, Set, Optional, List
import logging
import os

from .binance_subaccount_service import BinanceSubAccountService, get_binance_service

logger = logging.getLogger(__name__)


class DepositMonitorService:
    """
    خدمة مراقبة الإيداعات
    تتحقق بشكل دوري من الإيداعات الجديدة وتعالجها تلقائياً
    
    الميزات:
    - مراقبة مستمرة لجميع الحسابات الفرعية
    - معالجة الإيداعات الجديدة وتحويلها للحساب الرئيسي
    - حساب الوحدات الجديدة بناءً على NAV الحالي
    - إرسال إشعارات للمستثمرين
    """
    
    # الفاصل الزمني بين عمليات الفحص (بالثواني)
    CHECK_INTERVAL = int(os.getenv('DEPOSIT_CHECK_INTERVAL', '60'))
    
    # عدد التأكيدات المطلوبة للإيداع
    CONFIRMATIONS_REQUIRED = int(os.getenv('DEPOSIT_CONFIRMATIONS', '1'))
    
    # العملة الافتراضية
    DEFAULT_COIN = os.getenv('DEFAULT_COIN', 'USDC')
    
    def __init__(
        self,
        binance_service: Optional[BinanceSubAccountService] = None,
        nav_service = None,
        email_service = None,
        db_session_factory = None
    ):
        """
        تهيئة خدمة مراقبة الإيداعات
        
        Args:
            binance_service: خدمة Binance للحسابات الفرعية
            nav_service: خدمة حساب NAV
            email_service: خدمة البريد الإلكتروني
            db_session_factory: مصنع جلسات قاعدة البيانات
        """
        self.binance = binance_service or get_binance_service()
        self.nav_service = nav_service
        self.email_service = email_service
        self.db_session_factory = db_session_factory
        
        # مجموعة الإيداعات المعالجة (لتجنب المعالجة المكررة)
        self.processed_deposits: Set[str] = set()
        
        # حالة التشغيل
        self.running = False
        self._task: Optional[asyncio.Task] = None
        
    async def start(self):
        """بدء خدمة المراقبة"""
        if self.running:
            logger.warning("Deposit monitor is already running")
            return
            
        self.running = True
        self._task = asyncio.create_task(self._monitor_loop())
        logger.info(f"Deposit monitor started (interval: {self.CHECK_INTERVAL}s)")
        
    async def stop(self):
        """إيقاف خدمة المراقبة"""
        self.running = False
        
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
            
        logger.info("Deposit monitor stopped")
        
    async def _monitor_loop(self):
        """حلقة المراقبة الرئيسية"""
        while self.running:
            try:
                await self.check_all_deposits()
            except Exception as e:
                logger.error(f"Error in deposit monitor loop: {e}")
                
            await asyncio.sleep(self.CHECK_INTERVAL)
            
    async def check_all_deposits(self) -> Dict[str, int]:
        """
        التحقق من إيداعات جميع المستثمرين
        
        Returns:
            Dict: إحصائيات المعالجة {checked, processed, errors}
        """
        stats = {'checked': 0, 'processed': 0, 'errors': 0}
        
        if not self.db_session_factory:
            logger.warning("No database session factory configured")
            return stats
            
        session = self.db_session_factory()
        
        try:
            # جلب جميع المستثمرين الذين لديهم حسابات فرعية
            from ..models.investor import Investor
            
            investors = session.query(Investor).filter(
                Investor.binance_sub_email.isnot(None),
                Investor.status == 'active'
            ).all()
            
            for investor in investors:
                stats['checked'] += 1
                try:
                    processed = await self._check_investor_deposits(investor, session)
                    stats['processed'] += processed
                except Exception as e:
                    stats['errors'] += 1
                    logger.error(f"Error checking deposits for investor {investor.id}: {e}")
                    
            session.commit()
            
            if stats['processed'] > 0:
                logger.info(f"Deposit check complete: {stats}")
                
        except Exception as e:
            logger.error(f"Error in check_all_deposits: {e}")
            session.rollback()
        finally:
            session.close()
            
        return stats
        
    async def _check_investor_deposits(self, investor, session) -> int:
        """
        التحقق من إيداعات مستثمر معين
        
        Args:
            investor: كائن المستثمر
            session: جلسة قاعدة البيانات
            
        Returns:
            int: عدد الإيداعات المعالجة
        """
        processed_count = 0
        
        try:
            # جلب سجل الإيداعات من Binance
            deposits = await self.binance.get_deposit_history(
                investor.binance_sub_email,
                coin=self.DEFAULT_COIN
            )
            
            for deposit_data in deposits:
                tx_hash = deposit_data.get('txId', '')
                
                # تخطي الإيداعات المعالجة مسبقاً
                if tx_hash in self.processed_deposits:
                    continue
                    
                # التحقق من وجود الإيداع في قاعدة البيانات
                from ..models.deposit import Deposit
                
                existing = session.query(Deposit).filter(
                    Deposit.tx_hash == tx_hash
                ).first()
                
                if existing:
                    self.processed_deposits.add(tx_hash)
                    continue
                    
                # التحقق من عدد التأكيدات
                status = deposit_data.get('status', 0)
                if status < self.CONFIRMATIONS_REQUIRED:
                    continue
                    
                # معالجة الإيداع الجديد
                await self._process_new_deposit(investor, deposit_data, session)
                self.processed_deposits.add(tx_hash)
                processed_count += 1
                
        except Exception as e:
            logger.error(f"Error checking deposits for {investor.binance_sub_email}: {e}")
            raise
            
        return processed_count
        
    async def _process_new_deposit(self, investor, deposit_data: Dict, session):
        """
        معالجة إيداع جديد
        
        الخطوات:
        1. تحويل الأموال من الحساب الفرعي إلى الحساب الرئيسي
        2. حساب الوحدات الجديدة بناءً على NAV الحالي
        3. تسجيل الإيداع في قاعدة البيانات
        4. تحديث رصيد المستثمر
        5. إرسال إشعار بالبريد الإلكتروني
        """
        amount = Decimal(str(deposit_data.get('amount', 0)))
        tx_hash = deposit_data.get('txId', '')
        network = deposit_data.get('network', 'TRX')
        
        logger.info(f"Processing new deposit: {amount} {self.DEFAULT_COIN} for investor {investor.id}")
        
        # الخطوة 1: تحويل إلى الحساب الرئيسي
        try:
            await self.binance.transfer_to_master(
                investor.binance_sub_email,
                self.DEFAULT_COIN,
                float(amount)
            )
            logger.info(f"Transferred {amount} {self.DEFAULT_COIN} to master account")
        except Exception as e:
            logger.error(f"Failed to transfer deposit to master: {e}")
            # نستمر في المعالجة حتى لو فشل التحويل (قد يكون تم تحويله مسبقاً)
        
        # الخطوة 2: حساب الوحدات الجديدة
        current_nav = Decimal('1.0')  # القيمة الافتراضية
        units_credited = amount  # افتراضياً: 1 USDC = 1 وحدة
        
        if self.nav_service:
            try:
                portfolio_value = Decimal(str(await self.binance.get_total_portfolio_value()))
                current_nav = self.nav_service.get_current_nav(portfolio_value)
                units_credited = amount / current_nav if current_nav > 0 else amount
            except Exception as e:
                logger.error(f"Error calculating NAV: {e}")
        
        # الخطوة 3: تسجيل الإيداع
        from ..models.deposit import Deposit, DepositStatus
        
        lock_days = int(os.getenv('LOCK_PERIOD_DAYS', '30'))
        lock_until = datetime.utcnow() + timedelta(days=lock_days)
        
        deposit = Deposit(
            investor_id=investor.id,
            tx_hash=tx_hash,
            amount=amount,
            coin=self.DEFAULT_COIN,
            network=network,
            units_credited=units_credited,
            nav_at_deposit=current_nav,
            status=DepositStatus.CREDITED,
            lock_until=lock_until,
            confirmed_at=datetime.utcnow()
        )
        session.add(deposit)
        
        # الخطوة 4: تحديث رصيد المستثمر
        investor.total_units = (investor.total_units or Decimal('0')) + units_credited
        investor.total_deposited = (investor.total_deposited or Decimal('0')) + amount
        
        # الخطوة 5: إرسال إشعار
        if self.email_service and hasattr(investor, 'user') and investor.user:
            try:
                await self.email_service.send_deposit_confirmed(
                    to_email=investor.user.email,
                    name=investor.user.full_name,
                    amount=float(amount),
                    units=float(units_credited),
                    nav=float(current_nav)
                )
            except Exception as e:
                logger.error(f"Failed to send deposit confirmation email: {e}")
                
        logger.info(
            f"Deposit processed: {amount} {self.DEFAULT_COIN} = {units_credited} units "
            f"at NAV {current_nav} (locked until {lock_until})"
        )
        
    async def manual_check(self, investor_id: int) -> Dict:
        """
        فحص يدوي لإيداعات مستثمر معين
        
        Args:
            investor_id: معرف المستثمر
            
        Returns:
            Dict: نتيجة الفحص
        """
        if not self.db_session_factory:
            return {'error': 'No database configured'}
            
        session = self.db_session_factory()
        
        try:
            from ..models.investor import Investor
            
            investor = session.query(Investor).filter(
                Investor.id == investor_id
            ).first()
            
            if not investor:
                return {'error': 'Investor not found'}
                
            if not investor.binance_sub_email:
                return {'error': 'Investor has no Binance sub-account'}
                
            processed = await self._check_investor_deposits(investor, session)
            session.commit()
            
            return {
                'investor_id': investor_id,
                'processed': processed,
                'status': 'success'
            }
            
        except Exception as e:
            session.rollback()
            return {'error': str(e)}
        finally:
            session.close()
            
    def get_status(self) -> Dict:
        """الحصول على حالة خدمة المراقبة"""
        return {
            'running': self.running,
            'check_interval': self.CHECK_INTERVAL,
            'processed_count': len(self.processed_deposits),
            'default_coin': self.DEFAULT_COIN,
            'confirmations_required': self.CONFIRMATIONS_REQUIRED
        }


# Singleton instance
_deposit_monitor: Optional[DepositMonitorService] = None


def get_deposit_monitor() -> DepositMonitorService:
    """الحصول على نسخة واحدة من الخدمة (Singleton)"""
    global _deposit_monitor
    if _deposit_monitor is None:
        _deposit_monitor = DepositMonitorService()
    return _deposit_monitor
