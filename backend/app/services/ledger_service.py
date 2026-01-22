"""
═══════════════════════════════════════════════════════════════════════════════
                    📒 LEDGER SERVICE
                    خدمة المحاسبة المزدوجة للصندوق
═══════════════════════════════════════════════════════════════════════════════

هذه الخدمة تُدير نظام المحاسبة المزدوجة (Double-Entry Accounting) للصندوق.
تضمن حساب NAV بدقة مثالية بدون ثغرات زمنية.

الاستخدام:
    ledger = LedgerService(db)
    
    # تسجيل إيداع
    await ledger.record_deposit(user_id=1, amount=100.0, transaction_id=123)
    
    # تسجيل سحب
    await ledger.record_withdrawal(user_id=1, amount=50.0, transaction_id=124)
    
    # تسجيل أرباح تداول
    await ledger.record_trading_pnl(pnl=15.50, trade_id="TRADE_001")
    
    # حساب NAV الحالي
    nav = await ledger.calculate_nav()
"""

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, desc
from typing import Optional, Dict, Any, Tuple
from datetime import datetime
import json
import logging

from app.models.fund_ledger import FundLedger, FundSnapshot, LedgerEntryType
from app.core.config import settings

logger = logging.getLogger(__name__)


class LedgerService:
    """خدمة إدارة سجل المحاسبة للصندوق"""
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def get_current_state(self) -> Tuple[float, float, float]:
        """
        الحصول على الحالة الحالية للصندوق
        
        Returns:
            Tuple[total_capital, total_units, nav]
        """
        # جلب آخر قيد محاسبي
        result = await self.db.execute(
            select(FundLedger)
            .order_by(desc(FundLedger.id))
            .limit(1)
        )
        last_entry = result.scalar_one_or_none()
        
        if last_entry:
            return (
                last_entry.cumulative_capital,
                last_entry.cumulative_units,
                last_entry.nav_after_entry
            )
        
        # إذا لم يكن هناك قيود، نُرجع القيم الافتراضية
        return (0.0, 0.0, settings.INITIAL_NAV)
    
    async def calculate_nav(self) -> float:
        """
        حساب قيمة NAV الحالية من السجلات
        
        Returns:
            قيمة NAV الحالية
        """
        total_capital, total_units, _ = await self.get_current_state()
        
        if total_units <= 0:
            return settings.INITIAL_NAV
        
        return total_capital / total_units
    
    async def get_fund_summary(self) -> Dict[str, Any]:
        """
        الحصول على ملخص شامل لحالة الصندوق
        
        Returns:
            قاموس يحتوي على جميع إحصائيات الصندوق
        """
        total_capital, total_units, nav = await self.get_current_state()
        
        # حساب إجمالي الإيداعات
        deposits_result = await self.db.execute(
            select(func.coalesce(func.sum(FundLedger.amount), 0.0))
            .where(FundLedger.entry_type == LedgerEntryType.DEPOSIT.value)
        )
        total_deposits = deposits_result.scalar() or 0.0
        
        # حساب إجمالي السحوبات
        withdrawals_result = await self.db.execute(
            select(func.coalesce(func.sum(func.abs(FundLedger.amount)), 0.0))
            .where(FundLedger.entry_type == LedgerEntryType.WITHDRAWAL.value)
        )
        total_withdrawals = withdrawals_result.scalar() or 0.0
        
        # حساب إجمالي أرباح التداول
        pnl_result = await self.db.execute(
            select(func.coalesce(func.sum(FundLedger.amount), 0.0))
            .where(FundLedger.entry_type.in_([
                LedgerEntryType.TRADE_PNL.value,
                LedgerEntryType.REALIZED_PNL.value,
                LedgerEntryType.UNREALIZED_PNL.value
            ]))
        )
        total_trading_pnl = pnl_result.scalar() or 0.0
        
        # حساب إجمالي الرسوم
        fees_result = await self.db.execute(
            select(func.coalesce(func.sum(func.abs(FundLedger.amount)), 0.0))
            .where(FundLedger.entry_type.in_([
                LedgerEntryType.PERFORMANCE_FEE.value,
                LedgerEntryType.MANAGEMENT_FEE.value
            ]))
        )
        total_fees = fees_result.scalar() or 0.0
        
        # عدد القيود
        entries_count_result = await self.db.execute(
            select(func.count(FundLedger.id))
        )
        entries_count = entries_count_result.scalar() or 0
        
        return {
            "total_capital": round(total_capital, 2),
            "total_units": round(total_units, 6),
            "nav": round(nav, 6),
            "total_deposits": round(total_deposits, 2),
            "total_withdrawals": round(total_withdrawals, 2),
            "net_deposits": round(total_deposits - total_withdrawals, 2),
            "total_trading_pnl": round(total_trading_pnl, 2),
            "total_fees": round(total_fees, 2),
            "entries_count": entries_count,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def _create_entry(
        self,
        entry_type: str,
        amount: float,
        units_delta: float,
        user_id: Optional[int] = None,
        transaction_id: Optional[int] = None,
        trade_id: Optional[str] = None,
        description: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> FundLedger:
        """
        إنشاء قيد محاسبي جديد
        
        هذه الدالة الداخلية تضمن حساب القيم التراكمية بشكل صحيح.
        """
        # الحصول على الحالة الحالية
        current_capital, current_units, current_nav = await self.get_current_state()
        
        # حساب القيم الجديدة
        new_capital = current_capital + amount
        new_units = current_units + units_delta
        
        # حساب NAV الجديد
        if new_units > 0:
            new_nav = new_capital / new_units
        else:
            new_nav = settings.INITIAL_NAV
        
        # إنشاء القيد
        entry = FundLedger(
            entry_type=entry_type,
            amount=amount,
            units_delta=units_delta,
            nav_at_entry=current_nav if current_units > 0 else settings.INITIAL_NAV,
            cumulative_capital=new_capital,
            cumulative_units=new_units,
            nav_after_entry=new_nav,
            user_id=user_id,
            transaction_id=transaction_id,
            trade_id=trade_id,
            description=description,
            metadata=json.dumps(metadata) if metadata else None
        )
        
        self.db.add(entry)
        await self.db.flush()
        
        logger.info(
            f"Ledger entry created: {entry_type} | "
            f"Amount: ${amount:+.2f} | Units: {units_delta:+.4f} | "
            f"NAV: ${new_nav:.6f}"
        )
        
        return entry
    
    async def initialize_fund(
        self,
        initial_capital: float,
        description: str = "Initial fund capital"
    ) -> FundLedger:
        """
        تهيئة الصندوق برأس مال أولي
        
        يُستخدم مرة واحدة فقط عند إنشاء الصندوق.
        
        Args:
            initial_capital: رأس المال الأولي
            description: وصف العملية
            
        Returns:
            قيد المحاسبة المُنشأ
        """
        # التحقق من عدم وجود قيود سابقة
        existing = await self.db.execute(
            select(func.count(FundLedger.id))
        )
        if existing.scalar() > 0:
            raise ValueError("Fund already initialized. Use record_deposit for new deposits.")
        
        # NAV الأولي = 1.0 دائماً
        initial_nav = settings.INITIAL_NAV
        initial_units = initial_capital / initial_nav
        
        return await self._create_entry(
            entry_type=LedgerEntryType.INITIAL.value,
            amount=initial_capital,
            units_delta=initial_units,
            description=description,
            metadata={"initial_nav": initial_nav}
        )
    
    async def record_deposit(
        self,
        user_id: int,
        amount: float,
        transaction_id: Optional[int] = None,
        description: Optional[str] = None
    ) -> FundLedger:
        """
        تسجيل إيداع مؤكد
        
        يُستدعى فقط بعد تأكيد الدفع من NOWPayments.
        
        Args:
            user_id: معرف المستخدم
            amount: مبلغ الإيداع (بعد خصم الرسوم)
            transaction_id: معرف المعاملة
            description: وصف العملية
            
        Returns:
            قيد المحاسبة المُنشأ
        """
        # الحصول على NAV الحالي
        current_nav = await self.calculate_nav()
        
        # حساب الوحدات الجديدة
        units_to_add = amount / current_nav
        
        return await self._create_entry(
            entry_type=LedgerEntryType.DEPOSIT.value,
            amount=amount,
            units_delta=units_to_add,
            user_id=user_id,
            transaction_id=transaction_id,
            description=description or f"Deposit from user {user_id}",
            metadata={
                "nav_at_deposit": current_nav,
                "units_purchased": units_to_add
            }
        )
    
    async def record_withdrawal(
        self,
        user_id: int,
        amount: float,
        units_to_withdraw: float,
        transaction_id: Optional[int] = None,
        description: Optional[str] = None
    ) -> FundLedger:
        """
        تسجيل سحب مؤكد
        
        يُستدعى فقط بعد إتمام تحويل الأموال للمستخدم.
        
        Args:
            user_id: معرف المستخدم
            amount: مبلغ السحب (سالب)
            units_to_withdraw: الوحدات المسحوبة
            transaction_id: معرف المعاملة
            description: وصف العملية
            
        Returns:
            قيد المحاسبة المُنشأ
        """
        # التأكد من أن المبلغ سالب
        if amount > 0:
            amount = -amount
        
        # التأكد من أن الوحدات سالبة
        if units_to_withdraw > 0:
            units_to_withdraw = -units_to_withdraw
        
        current_nav = await self.calculate_nav()
        
        return await self._create_entry(
            entry_type=LedgerEntryType.WITHDRAWAL.value,
            amount=amount,
            units_delta=units_to_withdraw,
            user_id=user_id,
            transaction_id=transaction_id,
            description=description or f"Withdrawal for user {user_id}",
            metadata={
                "nav_at_withdrawal": current_nav,
                "units_redeemed": abs(units_to_withdraw)
            }
        )
    
    async def record_trading_pnl(
        self,
        pnl: float,
        trade_id: Optional[str] = None,
        symbol: Optional[str] = None,
        is_realized: bool = True,
        description: Optional[str] = None
    ) -> FundLedger:
        """
        تسجيل أرباح/خسائر التداول
        
        يُستدعى من الوكيل عند إغلاق صفقة أو تحديث دوري.
        
        ملاحظة مهمة: أرباح التداول لا تُغير عدد الوحدات!
        
        Args:
            pnl: الربح (موجب) أو الخسارة (سالب)
            trade_id: معرف الصفقة
            symbol: رمز العملة
            is_realized: هل الربح محقق (صفقة مغلقة)؟
            description: وصف العملية
            
        Returns:
            قيد المحاسبة المُنشأ
        """
        entry_type = (
            LedgerEntryType.REALIZED_PNL.value 
            if is_realized 
            else LedgerEntryType.UNREALIZED_PNL.value
        )
        
        return await self._create_entry(
            entry_type=entry_type,
            amount=pnl,
            units_delta=0.0,  # أرباح التداول لا تُغير الوحدات!
            trade_id=trade_id,
            description=description or f"Trading PnL: {symbol or 'N/A'}",
            metadata={
                "symbol": symbol,
                "is_realized": is_realized,
                "pnl_type": "profit" if pnl >= 0 else "loss"
            }
        )
    
    async def record_performance_fee(
        self,
        fee_amount: float,
        high_water_mark: float,
        description: Optional[str] = None
    ) -> FundLedger:
        """
        تسجيل رسوم الأداء
        
        Args:
            fee_amount: مبلغ الرسوم (سالب)
            high_water_mark: أعلى قيمة NAV سابقة
            description: وصف العملية
            
        Returns:
            قيد المحاسبة المُنشأ
        """
        if fee_amount > 0:
            fee_amount = -fee_amount
        
        return await self._create_entry(
            entry_type=LedgerEntryType.PERFORMANCE_FEE.value,
            amount=fee_amount,
            units_delta=0.0,
            description=description or "Performance fee",
            metadata={
                "high_water_mark": high_water_mark,
                "fee_percentage": settings.PERFORMANCE_FEE_PERCENTAGE
            }
        )
    
    async def record_adjustment(
        self,
        amount: float,
        units_delta: float = 0.0,
        reason: str = "Manual adjustment",
        admin_id: Optional[int] = None
    ) -> FundLedger:
        """
        تسجيل تعديل يدوي
        
        يُستخدم للتصحيحات الضرورية فقط.
        
        Args:
            amount: مبلغ التعديل
            units_delta: تعديل الوحدات
            reason: سبب التعديل
            admin_id: معرف المسؤول
            
        Returns:
            قيد المحاسبة المُنشأ
        """
        return await self._create_entry(
            entry_type=LedgerEntryType.ADJUSTMENT.value,
            amount=amount,
            units_delta=units_delta,
            user_id=admin_id,
            description=reason,
            metadata={
                "adjustment_type": "manual",
                "admin_id": admin_id
            }
        )
    
    async def create_snapshot(
        self,
        binance_portfolio_value: Optional[float] = None,
        snapshot_type: str = "hourly"
    ) -> FundSnapshot:
        """
        إنشاء لقطة لحالة الصندوق
        
        Args:
            binance_portfolio_value: قيمة المحفظة على Binance (للمقارنة)
            snapshot_type: نوع اللقطة (hourly, daily, event)
            
        Returns:
            لقطة الصندوق
        """
        summary = await self.get_fund_summary()
        
        discrepancy = None
        if binance_portfolio_value is not None:
            discrepancy = binance_portfolio_value - summary["total_capital"]
        
        snapshot = FundSnapshot(
            total_capital=summary["total_capital"],
            total_units=summary["total_units"],
            nav_value=summary["nav"],
            binance_portfolio_value=binance_portfolio_value,
            discrepancy=discrepancy,
            total_deposits=summary["total_deposits"],
            total_withdrawals=summary["total_withdrawals"],
            total_trading_pnl=summary["total_trading_pnl"],
            total_fees_collected=summary["total_fees"],
            snapshot_type=snapshot_type
        )
        
        self.db.add(snapshot)
        await self.db.flush()
        
        logger.info(
            f"Fund snapshot created: NAV=${summary['nav']:.6f} | "
            f"Capital=${summary['total_capital']:.2f}"
        )
        
        return snapshot
    
    async def get_entries_history(
        self,
        entry_type: Optional[str] = None,
        user_id: Optional[int] = None,
        limit: int = 100,
        offset: int = 0
    ) -> list:
        """
        الحصول على سجل القيود المحاسبية
        
        Args:
            entry_type: تصفية حسب نوع القيد
            user_id: تصفية حسب المستخدم
            limit: الحد الأقصى للنتائج
            offset: الإزاحة
            
        Returns:
            قائمة القيود
        """
        query = select(FundLedger).order_by(desc(FundLedger.created_at))
        
        if entry_type:
            query = query.where(FundLedger.entry_type == entry_type)
        
        if user_id:
            query = query.where(FundLedger.user_id == user_id)
        
        query = query.limit(limit).offset(offset)
        
        result = await self.db.execute(query)
        return result.scalars().all()
