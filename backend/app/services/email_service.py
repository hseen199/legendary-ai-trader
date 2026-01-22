"""
Email Notification Service
خدمة إشعارات البريد الإلكتروني
"""
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from typing import Optional, List
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# Email Templates
EMAIL_TEMPLATES = {
    'welcome': {
        'subject_ar': 'مرحباً بك في ASINAX!',
        'subject_en': 'Welcome to ASINAX!',
        'template': 'welcome',
    },
    'login_alert': {
        'subject_ar': 'تنبيه: تسجيل دخول جديد لحسابك',
        'subject_en': 'Alert: New login to your account',
        'template': 'login_alert',
    },
    'new_device': {
        'subject_ar': 'تنبيه أمني: تسجيل دخول من جهاز جديد',
        'subject_en': 'Security Alert: Login from new device',
        'template': 'new_device',
    },
    'deposit_confirmed': {
        'subject_ar': 'تم تأكيد إيداعك بنجاح',
        'subject_en': 'Your deposit has been confirmed',
        'template': 'deposit_confirmed',
    },
    'withdrawal_requested': {
        'subject_ar': 'تم استلام طلب السحب',
        'subject_en': 'Withdrawal request received',
        'template': 'withdrawal_requested',
    },
    'withdrawal_completed': {
        'subject_ar': 'تم إتمام عملية السحب',
        'subject_en': 'Withdrawal completed',
        'template': 'withdrawal_completed',
    },
    'password_changed': {
        'subject_ar': 'تم تغيير كلمة المرور',
        'subject_en': 'Password changed',
        'template': 'password_changed',
    },
    '2fa_enabled': {
        'subject_ar': 'تم تفعيل المصادقة الثنائية',
        'subject_en': 'Two-factor authentication enabled',
        'template': '2fa_enabled',
    },
    'weekly_report': {
        'subject_ar': 'تقريرك الأسبوعي من ASINAX',
        'subject_en': 'Your weekly report from ASINAX',
        'template': 'weekly_report',
    },
    'monthly_report': {
        'subject_ar': 'تقريرك الشهري من ASINAX',
        'subject_en': 'Your monthly report from ASINAX',
        'template': 'monthly_report',
    },
}


class EmailService:
    def __init__(self):
        self.smtp_host = os.getenv('SMTP_HOST', 'smtp.gmail.com')
        self.smtp_port = int(os.getenv('SMTP_PORT', '587'))
        self.smtp_user = os.getenv('SMTP_USER', '')
        self.smtp_password = os.getenv('SMTP_PASSWORD', '')
        self.from_email = os.getenv('FROM_EMAIL', 'noreply@asinax.cloud')
        self.from_name = os.getenv('FROM_NAME', 'ASINAX')
        
    def _get_base_template(self, content: str, language: str = 'ar') -> str:
        """قالب HTML الأساسي للإيميلات"""
        direction = 'rtl' if language == 'ar' else 'ltr'
        font_family = 'Tajawal, Arial, sans-serif' if language == 'ar' else 'Arial, sans-serif'
        
        return f'''
        <!DOCTYPE html>
        <html dir="{direction}" lang="{language}">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <link href="https://fonts.googleapis.com/css2?family=Tajawal:wght@400;500;700&display=swap" rel="stylesheet">
            <style>
                body {{
                    font-family: {font_family};
                    margin: 0;
                    padding: 0;
                    background-color: #0a0a0a;
                    color: #ffffff;
                }}
                .container {{
                    max-width: 600px;
                    margin: 0 auto;
                    padding: 20px;
                }}
                .header {{
                    text-align: center;
                    padding: 30px 0;
                    background: linear-gradient(135deg, #10b981 0%, #059669 100%);
                    border-radius: 12px 12px 0 0;
                }}
                .logo {{
                    font-size: 28px;
                    font-weight: bold;
                    color: #ffffff;
                }}
                .content {{
                    background-color: #1a1a1a;
                    padding: 30px;
                    border-radius: 0 0 12px 12px;
                }}
                .button {{
                    display: inline-block;
                    padding: 12px 30px;
                    background: linear-gradient(135deg, #10b981 0%, #059669 100%);
                    color: #ffffff;
                    text-decoration: none;
                    border-radius: 8px;
                    font-weight: bold;
                    margin: 20px 0;
                }}
                .footer {{
                    text-align: center;
                    padding: 20px;
                    color: #666666;
                    font-size: 12px;
                }}
                .alert-box {{
                    background-color: #fef3c7;
                    border: 1px solid #f59e0b;
                    border-radius: 8px;
                    padding: 15px;
                    margin: 15px 0;
                    color: #92400e;
                }}
                .success-box {{
                    background-color: #d1fae5;
                    border: 1px solid #10b981;
                    border-radius: 8px;
                    padding: 15px;
                    margin: 15px 0;
                    color: #065f46;
                }}
                .info-row {{
                    display: flex;
                    justify-content: space-between;
                    padding: 10px 0;
                    border-bottom: 1px solid #333;
                }}
                .info-label {{
                    color: #888;
                }}
                .info-value {{
                    font-weight: bold;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <div class="logo">ASINAX</div>
                </div>
                <div class="content">
                    {content}
                </div>
                <div class="footer">
                    <p>© {datetime.now().year} ASINAX. {'جميع الحقوق محفوظة' if language == 'ar' else 'All rights reserved.'}</p>
                    <p>{'هذا البريد الإلكتروني تم إرساله تلقائياً، يرجى عدم الرد عليه.' if language == 'ar' else 'This email was sent automatically, please do not reply.'}</p>
                </div>
            </div>
        </body>
        </html>
        '''
    
    def _render_welcome_template(self, data: dict, language: str = 'ar') -> str:
        """قالب الترحيب"""
        if language == 'ar':
            content = f'''
            <h2>مرحباً {data.get('name', '')}! 👋</h2>
            <p>نحن سعداء بانضمامك إلى ASINAX - منصة التداول الذكي.</p>
            <p>مع ASINAX، يمكنك:</p>
            <ul>
                <li>الاستثمار بذكاء مع وكيل التداول الآلي</li>
                <li>متابعة أداء محفظتك على مدار الساعة</li>
                <li>سحب أرباحك بسهولة وأمان</li>
            </ul>
            <p>ابدأ الآن بإيداع أول مبلغ لك:</p>
            <a href="https://asinax.cloud/wallet" class="button">إيداع الآن</a>
            <p>إذا كان لديك أي أسئلة، لا تتردد في التواصل مع فريق الدعم.</p>
            '''
        else:
            content = f'''
            <h2>Welcome {data.get('name', '')}! 👋</h2>
            <p>We're happy to have you join ASINAX - the smart trading platform.</p>
            <p>With ASINAX, you can:</p>
            <ul>
                <li>Invest smartly with our AI trading agent</li>
                <li>Track your portfolio performance 24/7</li>
                <li>Withdraw your profits easily and securely</li>
            </ul>
            <p>Start now by making your first deposit:</p>
            <a href="https://asinax.cloud/wallet" class="button">Deposit Now</a>
            <p>If you have any questions, don't hesitate to contact our support team.</p>
            '''
        return self._get_base_template(content, language)
    
    def _render_login_alert_template(self, data: dict, language: str = 'ar') -> str:
        """قالب تنبيه تسجيل الدخول"""
        if language == 'ar':
            content = f'''
            <h2>تسجيل دخول جديد 🔐</h2>
            <p>تم تسجيل الدخول إلى حسابك:</p>
            <div class="info-row">
                <span class="info-label">الوقت:</span>
                <span class="info-value">{data.get('time', '')}</span>
            </div>
            <div class="info-row">
                <span class="info-label">الجهاز:</span>
                <span class="info-value">{data.get('device', 'غير معروف')}</span>
            </div>
            <div class="info-row">
                <span class="info-label">الموقع:</span>
                <span class="info-value">{data.get('location', 'غير معروف')}</span>
            </div>
            <div class="info-row">
                <span class="info-label">عنوان IP:</span>
                <span class="info-value">{data.get('ip', 'غير معروف')}</span>
            </div>
            <div class="alert-box">
                <strong>⚠️ تنبيه:</strong> إذا لم تكن أنت من قام بتسجيل الدخول، قم بتغيير كلمة المرور فوراً.
            </div>
            <a href="https://asinax.cloud/settings" class="button">إعدادات الأمان</a>
            '''
        else:
            content = f'''
            <h2>New Login 🔐</h2>
            <p>A new login was detected on your account:</p>
            <div class="info-row">
                <span class="info-label">Time:</span>
                <span class="info-value">{data.get('time', '')}</span>
            </div>
            <div class="info-row">
                <span class="info-label">Device:</span>
                <span class="info-value">{data.get('device', 'Unknown')}</span>
            </div>
            <div class="info-row">
                <span class="info-label">Location:</span>
                <span class="info-value">{data.get('location', 'Unknown')}</span>
            </div>
            <div class="info-row">
                <span class="info-label">IP Address:</span>
                <span class="info-value">{data.get('ip', 'Unknown')}</span>
            </div>
            <div class="alert-box">
                <strong>⚠️ Warning:</strong> If this wasn't you, change your password immediately.
            </div>
            <a href="https://asinax.cloud/settings" class="button">Security Settings</a>
            '''
        return self._get_base_template(content, language)
    
    def _render_deposit_confirmed_template(self, data: dict, language: str = 'ar') -> str:
        """قالب تأكيد الإيداع"""
        if language == 'ar':
            content = f'''
            <h2>تم تأكيد إيداعك ✅</h2>
            <div class="success-box">
                تم إضافة الإيداع إلى حسابك بنجاح!
            </div>
            <div class="info-row">
                <span class="info-label">المبلغ:</span>
                <span class="info-value">${data.get('amount', '0')} USDC</span>
            </div>
            <div class="info-row">
                <span class="info-label">الوحدات المضافة:</span>
                <span class="info-value">{data.get('units', '0')}</span>
            </div>
            <div class="info-row">
                <span class="info-label">سعر NAV:</span>
                <span class="info-value">${data.get('nav', '1.00')}</span>
            </div>
            <div class="info-row">
                <span class="info-label">رقم المعاملة:</span>
                <span class="info-value">{data.get('tx_id', '')}</span>
            </div>
            <p>الوكيل الذكي بدأ العمل على استثمار أموالك!</p>
            <a href="https://asinax.cloud/dashboard" class="button">عرض المحفظة</a>
            '''
        else:
            content = f'''
            <h2>Deposit Confirmed ✅</h2>
            <div class="success-box">
                Your deposit has been added to your account successfully!
            </div>
            <div class="info-row">
                <span class="info-label">Amount:</span>
                <span class="info-value">${data.get('amount', '0')} USDC</span>
            </div>
            <div class="info-row">
                <span class="info-label">Units Added:</span>
                <span class="info-value">{data.get('units', '0')}</span>
            </div>
            <div class="info-row">
                <span class="info-label">NAV Price:</span>
                <span class="info-value">${data.get('nav', '1.00')}</span>
            </div>
            <div class="info-row">
                <span class="info-label">Transaction ID:</span>
                <span class="info-value">{data.get('tx_id', '')}</span>
            </div>
            <p>The AI agent has started working on investing your funds!</p>
            <a href="https://asinax.cloud/dashboard" class="button">View Portfolio</a>
            '''
        return self._get_base_template(content, language)
    
    def _render_withdrawal_template(self, data: dict, template_type: str, language: str = 'ar') -> str:
        """قالب السحب"""
        if template_type == 'requested':
            if language == 'ar':
                content = f'''
                <h2>تم استلام طلب السحب 📤</h2>
                <p>تم استلام طلب السحب الخاص بك وهو قيد المراجعة.</p>
                <div class="info-row">
                    <span class="info-label">المبلغ:</span>
                    <span class="info-value">${data.get('amount', '0')} USDC</span>
                </div>
                <div class="info-row">
                    <span class="info-label">العنوان:</span>
                    <span class="info-value">{data.get('address', '')[:20]}...</span>
                </div>
                <div class="info-row">
                    <span class="info-label">الشبكة:</span>
                    <span class="info-value">{data.get('network', '')}</span>
                </div>
                <p>سيتم معالجة طلبك خلال 24-48 ساعة.</p>
                <a href="https://asinax.cloud/wallet" class="button">متابعة الطلب</a>
                '''
            else:
                content = f'''
                <h2>Withdrawal Request Received 📤</h2>
                <p>Your withdrawal request has been received and is under review.</p>
                <div class="info-row">
                    <span class="info-label">Amount:</span>
                    <span class="info-value">${data.get('amount', '0')} USDC</span>
                </div>
                <div class="info-row">
                    <span class="info-label">Address:</span>
                    <span class="info-value">{data.get('address', '')[:20]}...</span>
                </div>
                <div class="info-row">
                    <span class="info-label">Network:</span>
                    <span class="info-value">{data.get('network', '')}</span>
                </div>
                <p>Your request will be processed within 24-48 hours.</p>
                <a href="https://asinax.cloud/wallet" class="button">Track Request</a>
                '''
        else:  # completed
            if language == 'ar':
                content = f'''
                <h2>تم إتمام السحب ✅</h2>
                <div class="success-box">
                    تم إرسال المبلغ إلى محفظتك بنجاح!
                </div>
                <div class="info-row">
                    <span class="info-label">المبلغ:</span>
                    <span class="info-value">${data.get('amount', '0')} USDC</span>
                </div>
                <div class="info-row">
                    <span class="info-label">رقم المعاملة:</span>
                    <span class="info-value">{data.get('tx_hash', '')[:20]}...</span>
                </div>
                <a href="https://asinax.cloud/wallet" class="button">عرض المحفظة</a>
                '''
            else:
                content = f'''
                <h2>Withdrawal Completed ✅</h2>
                <div class="success-box">
                    The amount has been sent to your wallet successfully!
                </div>
                <div class="info-row">
                    <span class="info-label">Amount:</span>
                    <span class="info-value">${data.get('amount', '0')} USDC</span>
                </div>
                <div class="info-row">
                    <span class="info-label">Transaction Hash:</span>
                    <span class="info-value">{data.get('tx_hash', '')[:20]}...</span>
                </div>
                <a href="https://asinax.cloud/wallet" class="button">View Wallet</a>
                '''
        return self._get_base_template(content, language)

    async def send_email(
        self,
        to_email: str,
        template_name: str,
        data: dict,
        language: str = 'ar',
        attachments: Optional[List[str]] = None
    ) -> bool:
        """إرسال بريد إلكتروني"""
        try:
            template_info = EMAIL_TEMPLATES.get(template_name)
            if not template_info:
                logger.error(f"Template not found: {template_name}")
                return False
            
            # تحديد الموضوع
            subject = template_info[f'subject_{language}']
            
            # إنشاء محتوى الإيميل
            if template_name == 'welcome':
                html_content = self._render_welcome_template(data, language)
            elif template_name == 'login_alert':
                html_content = self._render_login_alert_template(data, language)
            elif template_name == 'deposit_confirmed':
                html_content = self._render_deposit_confirmed_template(data, language)
            elif template_name in ['withdrawal_requested', 'withdrawal_completed']:
                template_type = 'requested' if template_name == 'withdrawal_requested' else 'completed'
                html_content = self._render_withdrawal_template(data, template_type, language)
            else:
                # قالب افتراضي
                html_content = self._get_base_template(f"<p>{data.get('message', '')}</p>", language)
            
            # إنشاء الرسالة
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = f"{self.from_name} <{self.from_email}>"
            msg['To'] = to_email
            
            # إضافة المحتوى
            msg.attach(MIMEText(html_content, 'html', 'utf-8'))
            
            # إضافة المرفقات
            if attachments:
                for file_path in attachments:
                    if os.path.exists(file_path):
                        with open(file_path, 'rb') as f:
                            part = MIMEBase('application', 'octet-stream')
                            part.set_payload(f.read())
                            encoders.encode_base64(part)
                            part.add_header(
                                'Content-Disposition',
                                f'attachment; filename={os.path.basename(file_path)}'
                            )
                            msg.attach(part)
            
            # إرسال الإيميل
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                if self.smtp_user and self.smtp_password:
                    server.login(self.smtp_user, self.smtp_password)
                server.sendmail(self.from_email, to_email, msg.as_string())
            
            logger.info(f"Email sent successfully to {to_email}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email: {str(e)}")
            return False


# إنشاء instance عام
email_service = EmailService()
