"""
Report Generation Service
خدمة إنشاء التقارير PDF
"""
import os
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from io import BytesIO
import logging
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image, PageBreak, HRFlowable
)
from reportlab.lib.enums import TA_CENTER, TA_RIGHT, TA_LEFT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

logger = logging.getLogger(__name__)


class ReportService:
    """خدمة إنشاء التقارير"""
    
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
            spaceAfter=30,
            textColor=colors.HexColor('#10b981'),
            alignment=TA_CENTER,
        ))
        
        # عنوان فرعي
        self.styles.add(ParagraphStyle(
            name='CustomSubtitle',
            parent=self.styles['Heading2'],
            fontSize=16,
            spaceAfter=12,
            textColor=colors.HexColor('#333333'),
        ))
        
        # نص عادي
        self.styles.add(ParagraphStyle(
            name='CustomBody',
            parent=self.styles['Normal'],
            fontSize=11,
            spaceAfter=8,
            textColor=colors.HexColor('#444444'),
        ))
        
        # نص مميز
        self.styles.add(ParagraphStyle(
            name='Highlight',
            parent=self.styles['Normal'],
            fontSize=14,
            textColor=colors.HexColor('#10b981'),
            alignment=TA_CENTER,
        ))

    def _create_header(self, canvas, doc):
        """إنشاء رأس الصفحة"""
        canvas.saveState()
        
        # شعار ورأس
        canvas.setFillColor(colors.HexColor('#10b981'))
        canvas.setFont('Helvetica-Bold', 20)
        canvas.drawString(50, A4[1] - 40, "ASINAX")
        
        # خط فاصل
        canvas.setStrokeColor(colors.HexColor('#10b981'))
        canvas.setLineWidth(2)
        canvas.line(50, A4[1] - 50, A4[0] - 50, A4[1] - 50)
        
        canvas.restoreState()

    def _create_footer(self, canvas, doc):
        """إنشاء تذييل الصفحة"""
        canvas.saveState()
        
        # رقم الصفحة
        canvas.setFont('Helvetica', 9)
        canvas.setFillColor(colors.HexColor('#666666'))
        canvas.drawCentredString(A4[0] / 2, 30, f"Page {doc.page}")
        
        # حقوق النشر
        canvas.drawCentredString(A4[0] / 2, 15, f"© {datetime.now().year} ASINAX - All Rights Reserved")
        
        canvas.restoreState()

    def _create_summary_table(self, data: Dict[str, Any], language: str = 'ar') -> Table:
        """إنشاء جدول الملخص"""
        is_rtl = language == 'ar'
        
        table_data = []
        
        labels = {
            'total_value': ('إجمالي القيمة', 'Total Value'),
            'total_profit': ('إجمالي الربح', 'Total Profit'),
            'profit_percentage': ('نسبة الربح', 'Profit %'),
            'total_trades': ('عدد الصفقات', 'Total Trades'),
            'winning_trades': ('الصفقات الرابحة', 'Winning Trades'),
            'losing_trades': ('الصفقات الخاسرة', 'Losing Trades'),
            'win_rate': ('نسبة النجاح', 'Win Rate'),
            'nav_start': ('NAV (البداية)', 'NAV (Start)'),
            'nav_end': ('NAV (النهاية)', 'NAV (End)'),
            'nav_change': ('تغير NAV', 'NAV Change'),
        }
        
        for key, (ar_label, en_label) in labels.items():
            if key in data:
                label = ar_label if is_rtl else en_label
                value = data[key]
                if isinstance(value, float):
                    if 'percentage' in key or 'rate' in key or 'change' in key:
                        value = f"{value:.2f}%"
                    else:
                        value = f"${value:,.2f}"
                table_data.append([label, str(value)])
        
        table = Table(table_data, colWidths=[200, 150])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f8f9fa')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#333333')),
            ('ALIGN', (0, 0), (0, -1), TA_RIGHT if is_rtl else TA_LEFT),
            ('ALIGN', (1, 0), (1, -1), TA_LEFT if is_rtl else TA_RIGHT),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
            ('TOPPADDING', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#dee2e6')),
            ('ROWBACKGROUNDS', (0, 0), (-1, -1), [colors.HexColor('#ffffff'), colors.HexColor('#f8f9fa')]),
        ]))
        
        return table

    def _create_trades_table(self, trades: List[Dict], language: str = 'ar') -> Table:
        """إنشاء جدول الصفقات"""
        is_rtl = language == 'ar'
        
        headers = [
            ('التاريخ', 'Date'),
            ('العملة', 'Symbol'),
            ('النوع', 'Type'),
            ('الكمية', 'Amount'),
            ('السعر', 'Price'),
            ('الربح/الخسارة', 'P/L'),
        ]
        
        header_row = [h[0] if is_rtl else h[1] for h in headers]
        table_data = [header_row]
        
        for trade in trades[:20]:  # أول 20 صفقة
            row = [
                trade.get('date', ''),
                trade.get('symbol', ''),
                trade.get('type', ''),
                f"{trade.get('amount', 0):.4f}",
                f"${trade.get('price', 0):.2f}",
                f"${trade.get('pnl', 0):+.2f}",
            ]
            table_data.append(row)
        
        table = Table(table_data, colWidths=[80, 60, 50, 70, 70, 80])
        table.setStyle(TableStyle([
            # Header
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#10b981')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            # Body
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.HexColor('#333333')),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('ALIGN', (0, 0), (-1, -1), TA_CENTER),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#dee2e6')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8f9fa')]),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
        ]))
        
        return table

    async def generate_monthly_report(
        self,
        user_id: int,
        user_name: str,
        user_email: str,
        month: int,
        year: int,
        portfolio_data: Dict[str, Any],
        trades: List[Dict],
        nav_history: List[Dict],
        language: str = 'ar'
    ) -> bytes:
        """إنشاء التقرير الشهري"""
        is_rtl = language == 'ar'
        
        buffer = BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            rightMargin=50,
            leftMargin=50,
            topMargin=70,
            bottomMargin=50
        )
        
        elements = []
        
        # العنوان
        month_names_ar = ['يناير', 'فبراير', 'مارس', 'أبريل', 'مايو', 'يونيو',
                         'يوليو', 'أغسطس', 'سبتمبر', 'أكتوبر', 'نوفمبر', 'ديسمبر']
        month_names_en = ['January', 'February', 'March', 'April', 'May', 'June',
                         'July', 'August', 'September', 'October', 'November', 'December']
        
        month_name = month_names_ar[month - 1] if is_rtl else month_names_en[month - 1]
        
        title = f"{'التقرير الشهري' if is_rtl else 'Monthly Report'}"
        elements.append(Paragraph(title, self.styles['CustomTitle']))
        
        subtitle = f"{month_name} {year}"
        elements.append(Paragraph(subtitle, self.styles['Highlight']))
        elements.append(Spacer(1, 20))
        
        # معلومات المستخدم
        user_info = f"{'المستخدم' if is_rtl else 'User'}: {user_name} ({user_email})"
        elements.append(Paragraph(user_info, self.styles['CustomBody']))
        
        report_date = f"{'تاريخ التقرير' if is_rtl else 'Report Date'}: {datetime.now().strftime('%Y-%m-%d')}"
        elements.append(Paragraph(report_date, self.styles['CustomBody']))
        elements.append(Spacer(1, 20))
        
        # خط فاصل
        elements.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#10b981')))
        elements.append(Spacer(1, 20))
        
        # ملخص الأداء
        summary_title = "ملخص الأداء" if is_rtl else "Performance Summary"
        elements.append(Paragraph(summary_title, self.styles['CustomSubtitle']))
        elements.append(Spacer(1, 10))
        
        elements.append(self._create_summary_table(portfolio_data, language))
        elements.append(Spacer(1, 30))
        
        # الصفقات
        if trades:
            trades_title = "سجل الصفقات" if is_rtl else "Trade History"
            elements.append(Paragraph(trades_title, self.styles['CustomSubtitle']))
            elements.append(Spacer(1, 10))
            
            elements.append(self._create_trades_table(trades, language))
            elements.append(Spacer(1, 30))
        
        # ملاحظات
        notes_title = "ملاحظات" if is_rtl else "Notes"
        elements.append(Paragraph(notes_title, self.styles['CustomSubtitle']))
        
        notes = [
            "هذا التقرير يعكس أداء محفظتك خلال الفترة المحددة." if is_rtl else "This report reflects your portfolio performance during the specified period.",
            "الأداء السابق لا يضمن نتائج مستقبلية." if is_rtl else "Past performance does not guarantee future results.",
            "للاستفسارات، تواصل مع فريق الدعم." if is_rtl else "For inquiries, contact the support team.",
        ]
        
        for note in notes:
            elements.append(Paragraph(f"• {note}", self.styles['CustomBody']))
        
        # بناء PDF
        doc.build(
            elements,
            onFirstPage=lambda c, d: (self._create_header(c, d), self._create_footer(c, d)),
            onLaterPages=lambda c, d: (self._create_header(c, d), self._create_footer(c, d))
        )
        
        buffer.seek(0)
        return buffer.getvalue()

    async def generate_weekly_report(
        self,
        user_id: int,
        user_name: str,
        user_email: str,
        start_date: datetime,
        end_date: datetime,
        portfolio_data: Dict[str, Any],
        trades: List[Dict],
        language: str = 'ar'
    ) -> bytes:
        """إنشاء التقرير الأسبوعي"""
        is_rtl = language == 'ar'
        
        buffer = BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            rightMargin=50,
            leftMargin=50,
            topMargin=70,
            bottomMargin=50
        )
        
        elements = []
        
        # العنوان
        title = f"{'التقرير الأسبوعي' if is_rtl else 'Weekly Report'}"
        elements.append(Paragraph(title, self.styles['CustomTitle']))
        
        date_range = f"{start_date.strftime('%Y-%m-%d')} - {end_date.strftime('%Y-%m-%d')}"
        elements.append(Paragraph(date_range, self.styles['Highlight']))
        elements.append(Spacer(1, 20))
        
        # معلومات المستخدم
        user_info = f"{'المستخدم' if is_rtl else 'User'}: {user_name}"
        elements.append(Paragraph(user_info, self.styles['CustomBody']))
        elements.append(Spacer(1, 20))
        
        # خط فاصل
        elements.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#10b981')))
        elements.append(Spacer(1, 20))
        
        # ملخص الأداء
        summary_title = "ملخص الأسبوع" if is_rtl else "Weekly Summary"
        elements.append(Paragraph(summary_title, self.styles['CustomSubtitle']))
        elements.append(Spacer(1, 10))
        
        elements.append(self._create_summary_table(portfolio_data, language))
        elements.append(Spacer(1, 30))
        
        # الصفقات
        if trades:
            trades_title = "صفقات الأسبوع" if is_rtl else "Weekly Trades"
            elements.append(Paragraph(trades_title, self.styles['CustomSubtitle']))
            elements.append(Spacer(1, 10))
            
            elements.append(self._create_trades_table(trades, language))
        
        # بناء PDF
        doc.build(
            elements,
            onFirstPage=lambda c, d: (self._create_header(c, d), self._create_footer(c, d)),
            onLaterPages=lambda c, d: (self._create_header(c, d), self._create_footer(c, d))
        )
        
        buffer.seek(0)
        return buffer.getvalue()


# إنشاء instance عام
report_service = ReportService()
