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

    async def send_login_notification(
        self,
        email: str,
        ip_address: str = 'Unknown',
        device: str = 'Unknown',
        location: str = 'Unknown',
        login_time = None
    ) -> bool:
        """Send login notification email"""
        try:
            from datetime import datetime
            if login_time is None:
                login_time = datetime.utcnow()
            
            data = {
                'ip_address': ip_address,
                'device': device,
                'location': location,
                'login_time': login_time.strftime('%Y-%m-%d %H:%M:%S UTC')
            }
            return await self.send_email(email, 'login_alert', data, 'ar')
        except Exception as e:
            logger.error(f'Failed to send login notification: {str(e)}')
            return False

    async def send_welcome_email(
        self,
        email: str,
        name: str = 'مستخدم'
    ) -> bool:
        """Send welcome email to new users"""
        try:
            data = {'name': name}
            return await self.send_email(email, 'welcome', data, 'ar')
        except Exception as e:
            logger.error(f'Failed to send welcome email: {str(e)}')
            return False



    async def send_verification_otp(self, email: str, otp_code: str, name: str = "مستخدم") -> bool:
        """Send OTP verification code to user email"""
        try:
            html_content = self._get_base_template(f"""
                <div style="text-align: center; padding: 30px 0;">
                    <h2 style="color: #8B5CF6; margin-bottom: 20px;">رمز التحقق</h2>
                    <p style="color: #9CA3AF; margin-bottom: 30px;">مرحباً {name}،</p>
                    <p style="color: #9CA3AF; margin-bottom: 20px;">رمز التحقق الخاص بك هو:</p>
                    <div style="background: linear-gradient(135deg, #8B5CF6 0%, #6366F1 100%); padding: 20px 40px; border-radius: 12px; display: inline-block; margin: 20px 0;">
                        <span style="font-size: 32px; font-weight: bold; color: white; letter-spacing: 8px;">{otp_code}</span>
                    </div>
                    <p style="color: #9CA3AF; margin-top: 20px;">هذا الرمز صالح لمدة 10 دقائق فقط.</p>
                    <p style="color: #6B7280; font-size: 12px; margin-top: 30px;">إذا لم تطلب هذا الرمز، يرجى تجاهل هذه الرسالة.</p>
                </div>
            """, "ar")
            msg = MIMEMultipart("alternative")
            msg["Subject"] = "رمز التحقق - ASINAX"
            msg["From"] = f"{self.from_name} <{self.from_email}>"
            msg["To"] = email
            msg.attach(MIMEText(html_content, "html", "utf-8"))
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                if self.smtp_user and self.smtp_password:
                    server.login(self.smtp_user, self.smtp_password)
                server.sendmail(self.from_email, email, msg.as_string())
            logger.info(f"OTP verification email sent to {email}")
            return True
        except Exception as e:
            logger.error(f"Failed to send OTP email to {email}: {str(e)}")
            return False

    async def send_withdrawal_confirmation(
        self,
        email: str,
        name: str,
        amount: float,
        confirmation_token: str,
        withdrawal_id: int
    ) -> bool:
        """إرسال إيميل تأكيد الموافقة على السحب"""
        try:
            confirmation_link = f"https://asinax.cloud/api/v1/wallet/withdraw/confirm/{confirmation_token}"
            html_content = self._get_base_template(f'''
                <div style="text-align: center; padding: 30px 0;">
                    <h2 style="color: #10B981; margin-bottom: 20px;">تمت الموافقة على طلب السحب</h2>
                    <p style="color: #9CA3AF; margin-bottom: 20px;">مرحبا {name}،</p>
                    <p style="color: #9CA3AF; margin-bottom: 30px;">تمت الموافقة على طلب سحبك بمبلغ:</p>
                    <div style="background: linear-gradient(135deg, #10B981 0%, #059669 100%); padding: 20px 40px; border-radius: 12px; display: inline-block; margin: 20px 0;">
                        <span style="font-size: 32px; font-weight: bold; color: white;">${amount:.2f}</span>
                    </div>
                    <p style="color: #9CA3AF; margin-top: 20px;">يرجى تأكيد السحب بالضغط على الزر أدناه:</p>
                    <a href="{confirmation_link}" style="display: inline-block; background: #8B5CF6; color: white; padding: 15px 40px; border-radius: 8px; text-decoration: none; margin: 20px 0; font-weight: bold;">تأكيد السحب</a>
                    <p style="color: #6B7280; font-size: 12px; margin-top: 30px;">رقم الطلب: #{withdrawal_id}</p>
                </div>
            ''', "ar")
            
            msg = MIMEMultipart("alternative")
            msg["Subject"] = "تمت الموافقة على طلب السحب - ASINAX"
            msg["From"] = f"{self.from_name} <{self.from_email}>"
            msg["To"] = email
            msg.attach(MIMEText(html_content, "html", "utf-8"))
            
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                if self.smtp_user and self.smtp_password:
                    server.login(self.smtp_user, self.smtp_password)
                server.sendmail(self.from_email, email, msg.as_string())
            
            logger.info(f"Withdrawal confirmation email sent to {email}")
            return True
        except Exception as e:
            logger.error(f"Failed to send withdrawal confirmation: {str(e)}")
            return False

    async def send_withdrawal_rejected(
        self,
        email: str,
        amount: float,
        reason: str
    ) -> bool:
        """إرسال إيميل رفض السحب"""
        try:
            html_content = self._get_base_template(f'''
                <div style="text-align: center; padding: 30px 0;">
                    <h2 style="color: #EF4444; margin-bottom: 20px;">تم رفض طلب السحب</h2>
                    <p style="color: #9CA3AF; margin-bottom: 30px;">نأسف لإبلاغك أنه تم رفض طلب سحبك بمبلغ:</p>
                    <div style="background: linear-gradient(135deg, #EF4444 0%, #DC2626 100%); padding: 20px 40px; border-radius: 12px; display: inline-block; margin: 20px 0;">
                        <span style="font-size: 32px; font-weight: bold; color: white;">${amount:.2f}</span>
                    </div>
                    <div style="background: #1F2937; padding: 20px; border-radius: 8px; margin: 20px 0; text-align: right;">
                        <p style="color: #9CA3AF; margin: 0;"><strong>سبب الرفض:</strong></p>
                        <p style="color: #F87171; margin: 10px 0 0 0;">{reason}</p>
                    </div>
                    <p style="color: #6B7280; font-size: 12px; margin-top: 30px;">إذا كان لديك أي استفسار، يرجى التواصل مع الدعم الفني.</p>
                </div>
            ''', "ar")
            
            msg = MIMEMultipart("alternative")
            msg["Subject"] = "تم رفض طلب السحب - ASINAX"
            msg["From"] = f"{self.from_name} <{self.from_email}>"
            msg["To"] = email
            msg.attach(MIMEText(html_content, "html", "utf-8"))
            
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                if self.smtp_user and self.smtp_password:
                    server.login(self.smtp_user, self.smtp_password)
                server.sendmail(self.from_email, email, msg.as_string())
            
            logger.info(f"Withdrawal rejection email sent to {email}")
            return True
        except Exception as e:
            logger.error(f"Failed to send withdrawal rejection: {str(e)}")
            return False

    async def send_withdrawal_completed(
        self,
        email: str,
        amount: float,
        tx_hash: str,
        to_address: str
    ) -> bool:
        """إرسال إيميل إتمام السحب"""
        try:
            html_content = self._get_base_template(f'''
                <div style="text-align: center; padding: 30px 0;">
                    <h2 style="color: #10B981; margin-bottom: 20px;">تم إتمام عملية السحب</h2>
                    <p style="color: #9CA3AF; margin-bottom: 30px;">تم إرسال المبلغ التالي إلى محفظتك بنجاح:</p>
                    <div style="background: linear-gradient(135deg, #10B981 0%, #059669 100%); padding: 20px 40px; border-radius: 12px; display: inline-block; margin: 20px 0;">
                        <span style="font-size: 32px; font-weight: bold; color: white;">${amount:.2f}</span>
                    </div>
                    <div style="background: #1F2937; padding: 20px; border-radius: 8px; margin: 20px 0; text-align: right;">
                        <p style="color: #9CA3AF; margin: 0 0 10px 0;"><strong>عنوان المحفظة:</strong></p>
                        <p style="color: #60A5FA; font-family: monospace; font-size: 12px; word-break: break-all;">{to_address}</p>
                        <p style="color: #9CA3AF; margin: 15px 0 10px 0;"><strong>رقم المعاملة (TX Hash):</strong></p>
                        <p style="color: #60A5FA; font-family: monospace; font-size: 12px; word-break: break-all;">{tx_hash}</p>
                    </div>
                </div>
            ''', "ar")
            
            msg = MIMEMultipart("alternative")
            msg["Subject"] = "تم إتمام عملية السحب - ASINAX"
            msg["From"] = f"{self.from_name} <{self.from_email}>"
            msg["To"] = email
            msg.attach(MIMEText(html_content, "html", "utf-8"))
            
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                if self.smtp_user and self.smtp_password:
                    server.login(self.smtp_user, self.smtp_password)
                server.sendmail(self.from_email, email, msg.as_string())
            
            logger.info(f"Withdrawal completed email sent to {email}")
            return True
        except Exception as e:
            logger.error(f"Failed to send withdrawal completed: {str(e)}")
            return False

    async def send_balance_adjusted(
        self,
        email: str,
        name: str,
        amount: float,
        operation: str,
        reason: str,
        new_balance: float
    ) -> bool:
        """إرسال إيميل تعديل الرصيد"""
        try:
            is_add = operation == 'add'
            color = '#10B981' if is_add else '#EF4444'
            title = 'تم إضافة رصيد لحسابك' if is_add else 'تم خصم رصيد من حسابك'
            
            html_content = self._get_base_template(f'''
                <div style="text-align: center; padding: 30px 0;">
                    <h2 style="color: {color}; margin-bottom: 20px;">{title}</h2>
                    <p style="color: #9CA3AF; margin-bottom: 20px;">مرحبا {name}،</p>
                    <div style="background: #1F2937; padding: 20px; border-radius: 8px; margin: 20px 0; text-align: right;">
                        <p style="color: #9CA3AF; margin: 0 0 10px 0;"><strong>السبب:</strong> {reason}</p>
                        <p style="color: #9CA3AF; margin: 0;"><strong>الرصيد الجديد:</strong> ${new_balance:.2f}</p>
                    </div>
                </div>
            ''', "ar")
            
            msg = MIMEMultipart("alternative")
            msg["Subject"] = f"{title} - ASINAX"
            msg["From"] = f"{self.from_name} <{self.from_email}>"
            msg["To"] = email
            msg.attach(MIMEText(html_content, "html", "utf-8"))
            
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                if self.smtp_user and self.smtp_password:
                    server.login(self.smtp_user, self.smtp_password)
                server.sendmail(self.from_email, email, msg.as_string())
            
            logger.info(f"Balance adjusted email sent to {email}")
            return True
        except Exception as e:
            logger.error(f"Failed to send balance adjusted: {str(e)}")
            return False


    async def send_login_otp(
        self,
        email: str,
        otp_code: str,
        device: str = "Unknown",
        ip_address: str = "Unknown"
    ) -> bool:
        """إرسال رمز OTP لتسجيل الدخول"""
        try:
            html_content = self._get_base_template(f'''
                <div style="text-align: center; padding: 30px 0;">
                    <h2 style="color: #8B5CF6; margin-bottom: 20px;">رمز التحقق لتسجيل الدخول</h2>
                    <p style="color: #9CA3AF; margin-bottom: 30px;">استخدم الرمز التالي لإتمام تسجيل الدخول:</p>
                    <div style="background: linear-gradient(135deg, #8B5CF6 0%, #6366F1 100%); padding: 20px 40px; border-radius: 12px; display: inline-block; margin: 20px 0;">
                        <span style="font-size: 36px; font-weight: bold; color: white; letter-spacing: 8px;">{otp_code}</span>
                    </div>
                    <p style="color: #6B7280; font-size: 14px; margin-top: 20px;">هذا الرمز صالح لمدة 10 دقائق</p>
                    <div style="background: #1F2937; padding: 15px; border-radius: 8px; margin: 20px 0; text-align: right;">
                        <p style="color: #9CA3AF; margin: 5px 0;"><strong>الجهاز:</strong> {device}</p>
                        <p style="color: #9CA3AF; margin: 5px 0;"><strong>عنوان IP:</strong> {ip_address}</p>
                    </div>
                    <p style="color: #EF4444; font-size: 12px;">إذا لم تطلب هذا الرمز، يرجى تجاهل هذا البريد.</p>
                </div>
            ''', "ar")
            
            msg = MIMEMultipart("alternative")
            msg["Subject"] = f"رمز التحقق: {otp_code} - ASINAX"
            msg["From"] = f"{self.from_name} <{self.from_email}>"
            msg["To"] = email
            msg.attach(MIMEText(html_content, "html", "utf-8"))
            
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                if self.smtp_user and self.smtp_password:
                    server.login(self.smtp_user, self.smtp_password)
                server.sendmail(self.from_email, email, msg.as_string())
            
            logger.info(f"Login OTP email sent to {email}")
            return True
        except Exception as e:
            logger.error(f"Failed to send login OTP email: {str(e)}")
            return False


    async def send_deposit_approved(
        self,
        email: str,
        name: str,
        amount: float,
        units: float
    ) -> bool:
        """إرسال إيميل الموافقة على الإيداع"""
        try:
            html_content = self._get_base_template(f'''
                <div style="text-align: center; padding: 30px 0;">
                    <div style="background: linear-gradient(135deg, #10B981 0%, #059669 100%); width: 80px; height: 80px; border-radius: 50%; margin: 0 auto 20px; display: flex; align-items: center; justify-content: center;">
                        <span style="font-size: 40px; color: white;">✓</span>
                    </div>
                    <h2 style="color: #10B981; margin-bottom: 20px;">تمت الموافقة على إيداعك!</h2>
                    <p style="color: #9CA3AF; margin-bottom: 30px;">مرحباً {name}،</p>
                    <p style="color: #E5E7EB; margin-bottom: 20px;">تمت الموافقة على طلب إيداعك وتم إضافة الرصيد إلى حسابك.</p>
                    <div style="background: #1F2937; padding: 20px; border-radius: 12px; margin: 20px 0;">
                        <p style="color: #9CA3AF; margin: 10px 0;"><strong>المبلغ:</strong> <span style="color: #10B981; font-size: 24px;">${amount:.2f}</span></p>
                        <p style="color: #9CA3AF; margin: 10px 0;"><strong>الوحدات المضافة:</strong> <span style="color: #8B5CF6;">{units:.6f}</span></p>
                    </div>
                    <p style="color: #6B7280; font-size: 14px;">يمكنك الآن بدء الاستثمار من خلال لوحة التحكم.</p>
                    <a href="https://asinax.cloud/dashboard" style="display: inline-block; background: linear-gradient(135deg, #8B5CF6 0%, #6366F1 100%); color: white; padding: 15px 40px; border-radius: 8px; text-decoration: none; margin-top: 20px; font-weight: bold;">الذهاب للوحة التحكم</a>
                </div>
            ''', "ar")
            
            msg = MIMEMultipart("alternative")
            msg["Subject"] = f"✓ تمت الموافقة على إيداعك - ${amount:.2f} - ASINAX"
            msg["From"] = f"{self.from_name} <{self.from_email}>"
            msg["To"] = email
            msg.attach(MIMEText(html_content, "html", "utf-8"))
            
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                if self.smtp_user and self.smtp_password:
                    server.login(self.smtp_user, self.smtp_password)
                server.sendmail(self.from_email, email, msg.as_string())
            
            logger.info(f"Deposit approved email sent to {email}")
            return True
        except Exception as e:
            logger.error(f"Failed to send deposit approved email: {str(e)}")
            return False


    async def send_deposit_rejected(
        self,
        email: str,
        name: str,
        amount: float,
        reason: str
    ) -> bool:
        """إرسال إيميل رفض الإيداع"""
        try:
            html_content = self._get_base_template(f'''
                <div style="text-align: center; padding: 30px 0;">
                    <div style="background: linear-gradient(135deg, #EF4444 0%, #DC2626 100%); width: 80px; height: 80px; border-radius: 50%; margin: 0 auto 20px; display: flex; align-items: center; justify-content: center;">
                        <span style="font-size: 40px; color: white;">✕</span>
                    </div>
                    <h2 style="color: #EF4444; margin-bottom: 20px;">تم رفض طلب الإيداع</h2>
                    <p style="color: #9CA3AF; margin-bottom: 30px;">مرحباً {name}،</p>
                    <p style="color: #E5E7EB; margin-bottom: 20px;">نأسف لإبلاغك بأنه تم رفض طلب إيداعك.</p>
                    <div style="background: #1F2937; padding: 20px; border-radius: 12px; margin: 20px 0;">
                        <p style="color: #9CA3AF; margin: 10px 0;"><strong>المبلغ:</strong> <span style="color: #EF4444; font-size: 24px;">${amount:.2f}</span></p>
                        <p style="color: #9CA3AF; margin: 10px 0;"><strong>السبب:</strong> <span style="color: #F87171;">{reason}</span></p>
                    </div>
                    <p style="color: #6B7280; font-size: 14px;">إذا كان لديك أي استفسار، يرجى التواصل مع فريق الدعم.</p>
                    <a href="https://asinax.cloud/support" style="display: inline-block; background: linear-gradient(135deg, #8B5CF6 0%, #6366F1 100%); color: white; padding: 15px 40px; border-radius: 8px; text-decoration: none; margin-top: 20px; font-weight: bold;">تواصل مع الدعم</a>
                </div>
            ''', "ar")
            
            msg = MIMEMultipart("alternative")
            msg["Subject"] = f"✕ تم رفض طلب الإيداع - ASINAX"
            msg["From"] = f"{self.from_name} <{self.from_email}>"
            msg["To"] = email
            msg.attach(MIMEText(html_content, "html", "utf-8"))
            
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                if self.smtp_user and self.smtp_password:
                    server.login(self.smtp_user, self.smtp_password)
                server.sendmail(self.from_email, email, msg.as_string())
            
            logger.info(f"Deposit rejected email sent to {email}")
            return True
        except Exception as e:
            logger.error(f"Failed to send deposit rejected email: {str(e)}")
            return False


# إنشاء instance عام
email_service = EmailService()
