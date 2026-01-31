"""
Enhanced Report Service - خدمة التقارير المحسّنة
تحسينات على /opt/asinax/backend/app/services/report_service.py
"""
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Optional, List, Dict, Any
from io import BytesIO
import logging

# ReportLab imports
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image, PageBreak, HRFlowable, ListFlowable, ListItem
)
from reportlab.lib.enums import TA_CENTER, TA_RIGHT, TA_LEFT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

logger = logging.getLogger(__name__)


class EnhancedReportService:
    """
    خدمة التقارير المحسّنة
    تدعم تقارير PDF تفصيلية مع رسوم بيانية وإحصائيات
    """
    
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
    
    def _setup_custom_styles(self):
        """إعداد الأنماط المخصصة"""
        # عنوان رئيسي
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#10b981'),
            alignment=TA_CENTER,
            spaceAfter=20
        ))
        
        # عنوان فرعي
        self.styles.add(ParagraphStyle(
            name='CustomSubtitle',
            parent=self.styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#ffffff'),
            alignment=TA_RIGHT,
            spaceAfter=10
        ))
        
        # نص عادي
        self.styles.add(ParagraphStyle(
            name='CustomBody',
            parent=self.styles['Normal'],
            fontSize=11,
            textColor=colors.HexColor('#e0e0e0'),
            alignment=TA_RIGHT,
            leading=16
        ))
        
        # نص مميز
        self.styles.add(ParagraphStyle(
            name='Highlight',
            parent=self.styles['Normal'],
            fontSize=12,
            textColor=colors.HexColor('#10b981'),
            alignment=TA_CENTER
        ))
        
        # نص الربح
        self.styles.add(ParagraphStyle(
            name='Profit',
            parent=self.styles['Normal'],
            fontSize=14,
            textColor=colors.HexColor('#22c55e'),
            alignment=TA_CENTER
        ))
        
        # نص الخسارة
        self.styles.add(ParagraphStyle(
            name='Loss',
            parent=self.styles['Normal'],
            fontSize=14,
            textColor=colors.HexColor('#ef4444'),
            alignment=TA_CENTER
        ))
    
    async def generate_detailed_performance_report(
        self,
        user_id: int,
        user_name: str,
        user_email: str,
        vip_level: str,
        start_date: datetime,
        end_date: datetime,
        portfolio_data: Dict[str, Any],
        trades: List[Dict],
        nav_history: List[Dict],
        language: str = 'ar'
    ) -> bytes:
        """
        إنشاء تقرير أداء تفصيلي
        """
        is_rtl = language == 'ar'
        
        buffer = BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            rightMargin=40,
            leftMargin=40,
            topMargin=60,
            bottomMargin=50
        )
        
        elements = []
        
        # ===== الصفحة الأولى: الملخص التنفيذي =====
        
        # الشعار والعنوان
        title = "تقرير الأداء التفصيلي" if is_rtl else "Detailed Performance Report"
        elements.append(Paragraph(title, self.styles['CustomTitle']))
        
        # الفترة
        date_range = f"{start_date.strftime('%Y-%m-%d')} - {end_date.strftime('%Y-%m-%d')}"
        elements.append(Paragraph(date_range, self.styles['Highlight']))
        elements.append(Spacer(1, 20))
        
        # معلومات المستخدم
        elements.append(self._create_user_info_section(user_name, user_email, vip_level, is_rtl))
        elements.append(Spacer(1, 30))
        
        # خط فاصل
        elements.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor('#10b981')))
        elements.append(Spacer(1, 20))
        
        # ملخص الأداء الرئيسي
        elements.append(self._create_performance_summary(portfolio_data, is_rtl))
        elements.append(Spacer(1, 30))
        
        # إحصائيات سريعة
        elements.append(self._create_quick_stats(portfolio_data, trades, is_rtl))
        elements.append(PageBreak())
        
        # ===== الصفحة الثانية: تحليل الصفقات =====
        
        trades_title = "تحليل الصفقات" if is_rtl else "Trade Analysis"
        elements.append(Paragraph(trades_title, self.styles['CustomTitle']))
        elements.append(Spacer(1, 20))
        
        # إحصائيات الصفقات
        elements.append(self._create_trade_statistics(trades, is_rtl))
        elements.append(Spacer(1, 30))
        
        # جدول الصفقات
        if trades:
            elements.append(self._create_trades_table(trades, is_rtl))
        elements.append(PageBreak())
        
        # ===== الصفحة الثالثة: تطور المحفظة =====
        
        portfolio_title = "تطور المحفظة" if is_rtl else "Portfolio Evolution"
        elements.append(Paragraph(portfolio_title, self.styles['CustomTitle']))
        elements.append(Spacer(1, 20))
        
        # جدول تطور NAV
        if nav_history:
            elements.append(self._create_nav_history_table(nav_history, is_rtl))
        elements.append(Spacer(1, 30))
        
        # ===== الصفحة الأخيرة: الملاحظات والتوصيات =====
        
        elements.append(self._create_notes_section(portfolio_data, is_rtl))
        
        # بناء PDF
        doc.build(
            elements,
            onFirstPage=lambda c, d: self._add_page_decorations(c, d),
            onLaterPages=lambda c, d: self._add_page_decorations(c, d)
        )
        
        buffer.seek(0)
        return buffer.getvalue()
    
    def _create_user_info_section(self, user_name: str, user_email: str, vip_level: str, is_rtl: bool) -> Table:
        """إنشاء قسم معلومات المستخدم"""
        vip_names = {
            "bronze": ("برونزي 🥉", "Bronze 🥉"),
            "silver": ("فضي 🥈", "Silver 🥈"),
            "gold": ("ذهبي 🥇", "Gold 🥇"),
            "platinum": ("بلاتيني 💎", "Platinum 💎"),
            "diamond": ("ماسي 💠", "Diamond 💠")
        }
        
        vip_display = vip_names.get(vip_level, ("برونزي", "Bronze"))[0 if is_rtl else 1]
        
        data = [
            [("المستخدم" if is_rtl else "User"), user_name],
            [("البريد" if is_rtl else "Email"), user_email],
            [("المستوى" if is_rtl else "Level"), vip_display],
            [("تاريخ التقرير" if is_rtl else "Report Date"), datetime.now().strftime('%Y-%m-%d %H:%M')]
        ]
        
        table = Table(data, colWidths=[150, 300])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#1a1a2e')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'RIGHT' if is_rtl else 'LEFT'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('PADDING', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#333333'))
        ]))
        
        return table
    
    def _create_performance_summary(self, portfolio_data: Dict, is_rtl: bool) -> Table:
        """إنشاء ملخص الأداء"""
        total_value = portfolio_data.get('total_value', 0)
        total_deposited = portfolio_data.get('total_deposited', 0)
        total_profit = portfolio_data.get('total_profit', 0)
        profit_percent = portfolio_data.get('profit_percent', 0)
        
        profit_color = colors.HexColor('#22c55e') if total_profit >= 0 else colors.HexColor('#ef4444')
        
        data = [
            [
                ("القيمة الحالية" if is_rtl else "Current Value"),
                f"${total_value:,.2f}"
            ],
            [
                ("إجمالي الإيداعات" if is_rtl else "Total Deposited"),
                f"${total_deposited:,.2f}"
            ],
            [
                ("الربح/الخسارة" if is_rtl else "Profit/Loss"),
                f"${total_profit:,.2f} ({profit_percent:+.2f}%)"
            ]
        ]
        
        table = Table(data, colWidths=[200, 250])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#0a0a0a')),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#888888')),
            ('TEXTCOLOR', (1, 0), (1, 1), colors.white),
            ('TEXTCOLOR', (1, 2), (1, 2), profit_color),
            ('ALIGN', (0, 0), (-1, -1), 'RIGHT' if is_rtl else 'LEFT'),
            ('FONTSIZE', (0, 0), (-1, -1), 14),
            ('FONTSIZE', (1, 0), (1, -1), 16),
            ('PADDING', (0, 0), (-1, -1), 15),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#222222'))
        ]))
        
        return table
    
    def _create_quick_stats(self, portfolio_data: Dict, trades: List[Dict], is_rtl: bool) -> Table:
        """إنشاء إحصائيات سريعة"""
        total_trades = len(trades)
        winning_trades = sum(1 for t in trades if t.get('pnl', 0) > 0)
        losing_trades = sum(1 for t in trades if t.get('pnl', 0) < 0)
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        avg_profit = sum(t.get('pnl', 0) for t in trades if t.get('pnl', 0) > 0) / winning_trades if winning_trades > 0 else 0
        avg_loss = sum(t.get('pnl', 0) for t in trades if t.get('pnl', 0) < 0) / losing_trades if losing_trades > 0 else 0
        
        data = [
            [
                ("إجمالي الصفقات" if is_rtl else "Total Trades"),
                str(total_trades),
                ("نسبة النجاح" if is_rtl else "Win Rate"),
                f"{win_rate:.1f}%"
            ],
            [
                ("صفقات رابحة" if is_rtl else "Winning Trades"),
                str(winning_trades),
                ("صفقات خاسرة" if is_rtl else "Losing Trades"),
                str(losing_trades)
            ],
            [
                ("متوسط الربح" if is_rtl else "Avg Profit"),
                f"${avg_profit:,.2f}",
                ("متوسط الخسارة" if is_rtl else "Avg Loss"),
                f"${abs(avg_loss):,.2f}"
            ]
        ]
        
        table = Table(data, colWidths=[120, 100, 120, 100])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#1a1a2e')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.white),
            ('TEXTCOLOR', (1, 1), (1, 1), colors.HexColor('#22c55e')),
            ('TEXTCOLOR', (3, 1), (3, 1), colors.HexColor('#ef4444')),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('PADDING', (0, 0), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#333333'))
        ]))
        
        return table
    
    def _create_trade_statistics(self, trades: List[Dict], is_rtl: bool) -> Table:
        """إنشاء إحصائيات الصفقات"""
        if not trades:
            return Paragraph("لا توجد صفقات" if is_rtl else "No trades", self.styles['CustomBody'])
        
        # تجميع حسب الرمز
        symbols = {}
        for trade in trades:
            symbol = trade.get('symbol', 'Unknown')
            if symbol not in symbols:
                symbols[symbol] = {'count': 0, 'pnl': 0, 'wins': 0}
            symbols[symbol]['count'] += 1
            symbols[symbol]['pnl'] += trade.get('pnl', 0)
            if trade.get('pnl', 0) > 0:
                symbols[symbol]['wins'] += 1
        
        # ترتيب حسب عدد الصفقات
        sorted_symbols = sorted(symbols.items(), key=lambda x: x[1]['count'], reverse=True)[:10]
        
        header = [
            ("الرمز" if is_rtl else "Symbol"),
            ("عدد الصفقات" if is_rtl else "Trades"),
            ("الربح/الخسارة" if is_rtl else "P/L"),
            ("نسبة النجاح" if is_rtl else "Win Rate")
        ]
        
        data = [header]
        for symbol, stats in sorted_symbols:
            win_rate = (stats['wins'] / stats['count'] * 100) if stats['count'] > 0 else 0
            pnl_text = f"${stats['pnl']:,.2f}"
            data.append([symbol, str(stats['count']), pnl_text, f"{win_rate:.1f}%"])
        
        table = Table(data, colWidths=[100, 80, 120, 100])
        
        style = [
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#10b981')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#0a0a0a')),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('PADDING', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#333333'))
        ]
        
        # تلوين الأرباح والخسائر
        for i, (symbol, stats) in enumerate(sorted_symbols, 1):
            if stats['pnl'] >= 0:
                style.append(('TEXTCOLOR', (2, i), (2, i), colors.HexColor('#22c55e')))
            else:
                style.append(('TEXTCOLOR', (2, i), (2, i), colors.HexColor('#ef4444')))
        
        table.setStyle(TableStyle(style))
        return table
    
    def _create_trades_table(self, trades: List[Dict], is_rtl: bool) -> Table:
        """إنشاء جدول الصفقات"""
        header = [
            ("التاريخ" if is_rtl else "Date"),
            ("الرمز" if is_rtl else "Symbol"),
            ("النوع" if is_rtl else "Side"),
            ("الدخول" if is_rtl else "Entry"),
            ("الخروج" if is_rtl else "Exit"),
            ("الربح/الخسارة" if is_rtl else "P/L")
        ]
        
        data = [header]
        
        # أخذ آخر 20 صفقة فقط
        recent_trades = sorted(trades, key=lambda x: x.get('closed_at', ''), reverse=True)[:20]
        
        for trade in recent_trades:
            date = trade.get('closed_at', '')
            if isinstance(date, datetime):
                date = date.strftime('%m/%d')
            elif isinstance(date, str) and len(date) > 10:
                date = date[:10]
            
            symbol = trade.get('symbol', '')
            side = "شراء" if trade.get('side', '').upper() == 'BUY' else "بيع"
            entry = f"${trade.get('entry_price', 0):,.2f}"
            exit_price = f"${trade.get('exit_price', 0):,.2f}"
            pnl = trade.get('pnl', 0)
            pnl_text = f"${pnl:,.2f}"
            
            data.append([date, symbol, side, entry, exit_price, pnl_text])
        
        table = Table(data, colWidths=[60, 70, 50, 80, 80, 80])
        
        style = [
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#10b981')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#0a0a0a')),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('PADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#333333'))
        ]
        
        # تلوين الأرباح والخسائر
        for i, trade in enumerate(recent_trades, 1):
            pnl = trade.get('pnl', 0)
            if pnl >= 0:
                style.append(('TEXTCOLOR', (5, i), (5, i), colors.HexColor('#22c55e')))
            else:
                style.append(('TEXTCOLOR', (5, i), (5, i), colors.HexColor('#ef4444')))
        
        table.setStyle(TableStyle(style))
        return table
    
    def _create_nav_history_table(self, nav_history: List[Dict], is_rtl: bool) -> Table:
        """إنشاء جدول تاريخ NAV"""
        header = [
            ("التاريخ" if is_rtl else "Date"),
            ("NAV" if is_rtl else "NAV"),
            ("التغير" if is_rtl else "Change"),
            ("النسبة" if is_rtl else "Percent")
        ]
        
        data = [header]
        
        # أخذ آخر 15 قراءة
        recent_nav = sorted(nav_history, key=lambda x: x.get('timestamp', ''), reverse=True)[:15]
        
        prev_nav = None
        for nav in reversed(recent_nav):
            date = nav.get('timestamp', '')
            if isinstance(date, datetime):
                date = date.strftime('%Y-%m-%d')
            
            nav_value = nav.get('nav_value', 1.0)
            
            if prev_nav:
                change = nav_value - prev_nav
                change_percent = ((nav_value - prev_nav) / prev_nav) * 100
            else:
                change = 0
                change_percent = 0
            
            data.append([
                date,
                f"${nav_value:,.4f}",
                f"${change:+,.4f}",
                f"{change_percent:+.2f}%"
            ])
            
            prev_nav = nav_value
        
        table = Table(data, colWidths=[100, 100, 100, 80])
        
        style = [
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#10b981')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#0a0a0a')),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('PADDING', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#333333'))
        ]
        
        table.setStyle(TableStyle(style))
        return table
    
    def _create_notes_section(self, portfolio_data: Dict, is_rtl: bool) -> List:
        """إنشاء قسم الملاحظات"""
        elements = []
        
        title = "ملاحظات وتوصيات" if is_rtl else "Notes & Recommendations"
        elements.append(Paragraph(title, self.styles['CustomSubtitle']))
        elements.append(Spacer(1, 15))
        
        notes = [
            "هذا التقرير يعكس أداء محفظتك خلال الفترة المحددة." if is_rtl else "This report reflects your portfolio performance during the specified period.",
            "الأداء السابق لا يضمن نتائج مستقبلية." if is_rtl else "Past performance does not guarantee future results.",
            "الوكيل الذكي يعمل على تحسين استراتيجيات التداول باستمرار." if is_rtl else "The AI agent continuously optimizes trading strategies.",
            "للاستفسارات، تواصل مع فريق الدعم عبر المنصة." if is_rtl else "For inquiries, contact support through the platform."
        ]
        
        for note in notes:
            elements.append(Paragraph(f"• {note}", self.styles['CustomBody']))
            elements.append(Spacer(1, 5))
        
        return elements
    
    def _add_page_decorations(self, canvas, doc):
        """إضافة زخارف الصفحة"""
        canvas.saveState()
        
        # الخلفية
        canvas.setFillColor(colors.HexColor('#0a0a0a'))
        canvas.rect(0, 0, A4[0], A4[1], fill=1)
        
        # الشريط العلوي
        canvas.setFillColor(colors.HexColor('#10b981'))
        canvas.rect(0, A4[1] - 40, A4[0], 40, fill=1)
        
        # اسم المنصة
        canvas.setFillColor(colors.white)
        canvas.setFont('Helvetica-Bold', 14)
        canvas.drawString(30, A4[1] - 28, "ASINAX")
        
        # رقم الصفحة
        canvas.setFont('Helvetica', 10)
        canvas.drawRightString(A4[0] - 30, 30, f"Page {doc.page}")
        
        canvas.restoreState()


# إنشاء instance عام
enhanced_report_service = EnhancedReportService()
