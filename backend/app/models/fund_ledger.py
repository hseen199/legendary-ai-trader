"""
═══════════════════════════════════════════════════════════════════════════════
                    📒 FUND LEDGER MODEL
                    نظام المحاسبة المزدوجة للصندوق
═══════════════════════════════════════════════════════════════════════════════

هذا النموذج يُسجل كل حركة مالية في الصندوق لحساب NAV بدقة مثالية.
يعمل بمبدأ المحاسبة المزدوجة (Double-Entry Accounting) المستخدم في صناديق الاستثمار.

أنواع العمليات:
- INITIAL: رأس المال الأولي
- DEPOSIT: إيداع مؤكد
- WITHDRAWAL: سحب مؤكد
- TRADE_PNL: أرباح/خسائر التداول
- PERFORMANCE_FEE: رسوم الأداء
- ADJUSTMENT: تعديل يدوي (للتصحيحات)
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Boolean, Enum as SQLEnum
from sqlalchemy.sql import func
from app.core.database import Base
import enum


class LedgerEntryType(str, enum.Enum):
    """أنواع القيود المحاسبية"""
    INITIAL = "initial"              # رأس المال الأولي
    DEPOSIT = "deposit"              # إيداع مؤكد
    WITHDRAWAL = "withdrawal"        # سحب مؤكد
    TRADE_PNL = "trade_pnl"          # أرباح/خسائر التداول
    REALIZED_PNL = "realized_pnl"    # أرباح محققة (صفقة مغلقة)
    UNREALIZED_PNL = "unrealized_pnl"  # أرباح غير محققة (تحديث دوري)
    PERFORMANCE_FEE = "performance_fee"  # رسوم الأداء
    MANAGEMENT_FEE = "management_fee"    # رسوم الإدارة
    ADJUSTMENT = "adjustment"        # تعديل يدوي


class FundLedger(Base):
    """
    سجل المحاسبة الرئيسي للصندوق
    
    كل صف يُمثل حركة مالية واحدة.
    NAV يُحسب من مجموع كل الصفوف.
    """
    __tablename__ = "fund_ledger"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # نوع العملية
    entry_type = Column(String(50), nullable=False, index=True)
    
    # التأثير على رأس المال (موجب للإضافة، سالب للخصم)
    amount = Column(Float, nullable=False, default=0.0)
    
    # التأثير على الوحدات (موجب للإضافة، سالب للخصم)
    # ملاحظة: أرباح التداول لا تُغير الوحدات، فقط الإيداعات والسحوبات
    units_delta = Column(Float, nullable=False, default=0.0)
    
    # قيمة NAV وقت تسجيل العملية
    nav_at_entry = Column(Float, nullable=False, default=1.0)
    
    # رأس المال التراكمي بعد هذه العملية
    cumulative_capital = Column(Float, nullable=False, default=0.0)
    
    # الوحدات التراكمية بعد هذه العملية
    cumulative_units = Column(Float, nullable=False, default=0.0)
    
    # NAV بعد هذه العملية
    nav_after_entry = Column(Float, nullable=False, default=1.0)
    
    # معرف المستخدم (للإيداعات والسحوبات)
    user_id = Column(Integer, nullable=True, index=True)
    
    # معرف المعاملة المرتبطة (للربط مع جدول transactions)
    transaction_id = Column(Integer, nullable=True, index=True)
    
    # معرف الصفقة (لأرباح التداول)
    trade_id = Column(String(100), nullable=True)
    
    # وصف العملية
    description = Column(Text, nullable=True)
    
    # بيانات إضافية (JSON)
    metadata = Column(Text, nullable=True)
    
    # هل تم التحقق من هذا القيد؟
    is_verified = Column(Boolean, default=True)
    
    # الطوابع الزمنية
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    
    def __repr__(self):
        return f"<FundLedger {self.id}: {self.entry_type} ${self.amount:.2f} ({self.units_delta:+.4f} units)>"


class FundSnapshot(Base):
    """
    لقطات دورية لحالة الصندوق
    
    تُستخدم للتحقق السريع وإنشاء التقارير.
    تُحفظ كل ساعة أو عند أحداث مهمة.
    """
    __tablename__ = "fund_snapshots"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # إجمالي رأس المال
    total_capital = Column(Float, nullable=False)
    
    # إجمالي الوحدات
    total_units = Column(Float, nullable=False)
    
    # قيمة NAV
    nav_value = Column(Float, nullable=False)
    
    # قيمة المحفظة الفعلية على Binance (للمقارنة)
    binance_portfolio_value = Column(Float, nullable=True)
    
    # الفرق بين القيمة المحسوبة والفعلية
    discrepancy = Column(Float, nullable=True)
    
    # عدد المستثمرين النشطين
    active_investors = Column(Integer, default=0)
    
    # إجمالي الإيداعات التراكمية
    total_deposits = Column(Float, default=0.0)
    
    # إجمالي السحوبات التراكمية
    total_withdrawals = Column(Float, default=0.0)
    
    # إجمالي أرباح التداول التراكمية
    total_trading_pnl = Column(Float, default=0.0)
    
    # إجمالي الرسوم المحصلة
    total_fees_collected = Column(Float, default=0.0)
    
    # نوع اللقطة
    snapshot_type = Column(String(50), default="hourly")  # hourly, daily, event
    
    # الطابع الزمني
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    
    def __repr__(self):
        return f"<FundSnapshot {self.id}: NAV=${self.nav_value:.6f} @ {self.created_at}>"
