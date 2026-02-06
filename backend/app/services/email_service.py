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
    'deposit_pending': {
        'subject_ar': 'تم استلام طلب الإيداع',
        'subject_en': 'Deposit request received',
        'template': 'deposit_pending',
    },
    'deposit_failed': {
        'subject_ar': 'فشل عملية الإيداع',
        'subject_en': 'Deposit failed',
        'template': 'deposit_failed',
    },
    'withdrawal_requested': {
        'subject_ar': 'تم استلام طلب السحب',
        'subject_en': 'Withdrawal request received',
        'template': 'withdrawal_requested',
    },
    'withdrawal_approved': {
        'subject_ar': 'تمت الموافقة على طلب السحب',
        'subject_en': 'Withdrawal request approved',
        'template': 'withdrawal_approved',
    },
    'withdrawal_rejected': {
        'subject_ar': 'تم رفض طلب السحب',
        'subject_en': 'Withdrawal request rejected',
        'template': 'withdrawal_rejected',
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
    # قوالب جديدة
    'referral_bonus': {
        'subject_ar': '🎁 مكافأة إحالة جديدة!',
        'subject_en': '🎁 New Referral Bonus!',
        'template': 'referral_bonus',
    },
    'platform_announcement': {
        'subject_ar': '📢 إعلان هام من ASINAX',
        'subject_en': '📢 Important Announcement from ASINAX',
        'template': 'platform_announcement',
    },
    'admin_message': {
        'subject_ar': '💬 رسالة من إدارة ASINAX',
        'subject_en': '💬 Message from ASINAX Admin',
        'template': 'admin_message',
    },
    'promotion': {
        'subject_ar': '🌟 عرض خاص من ASINAX',
        'subject_en': '🌟 Special Offer from ASINAX',
        'template': 'promotion',
    },
    'vip_upgrade': {
        'subject_ar': '⭐ تهانينا! تمت ترقيتك إلى VIP',
        'subject_en': '⭐ Congratulations! You have been upgraded to VIP',
        'template': 'vip_upgrade',
    },
    'profit_notification': {
        'subject_ar': '💰 تحقيق أرباح جديدة!',
        'subject_en': '💰 New Profits Achieved!',
        'template': 'profit_notification',
    },
    'otp_verification': {
        'subject_ar': '🔐 رمز التحقق الخاص بك',
        'subject_en': '🔐 Your Verification Code',
        'template': 'otp_verification',
    },
}


class EmailService:
    def __init__(self):
        self.smtp_host = os.getenv('SMTP_HOST', 'smtp.gmail.com')
        self.smtp_port = int(os.getenv('SMTP_PORT', '587'))
        self.smtp_user = os.getenv('SMTP_USER', '')
        self.smtp_password = os.getenv('SMTP_PASSWORD', '')
        self.from_email = os.getenv('SMTP_FROM', os.getenv('FROM_EMAIL', 'noreply@asinax.cloud'))
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
                    background: linear-gradient(135deg, #8b5cf6 0%, #6366f1 50%, #3b82f6 100%);
                    border-radius: 12px 12px 0 0;
                }}
                .logo {{
                    font-size: 28px;
                    font-weight: bold;
                    color: #ffffff;
                }}
                .content {{
                    background-color: #1a1a2e;
                    padding: 30px;
                    border-radius: 0 0 12px 12px;
                }}
                .button {{
                    display: inline-block;
                    padding: 12px 30px;
                    background: linear-gradient(135deg, #8b5cf6 0%, #6366f1 100%);
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
                .info-box {{
                    background-color: #dbeafe;
                    border: 1px solid #3b82f6;
                    border-radius: 8px;
                    padding: 15px;
                    margin: 15px 0;
                    color: #1e40af;
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
                .highlight {{
                    color: #8b5cf6;
                    font-weight: bold;
                }}
                .amount {{
                    font-size: 24px;
                    color: #10b981;
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
        time_str = data.get('time', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        if language == 'ar':
            content = f'''
            <h2>تسجيل دخول جديد 🔐</h2>
            <p>تم تسجيل الدخول إلى حسابك:</p>
            <div class="info-row">
                <span class="info-label">الوقت:</span>
                <span class="info-value">{time_str}</span>
            </div>
            <div class="info-row">
                <span class="info-label">الجهاز:</span>
                <span class="info-value">{data.get('user_agent', data.get('device', 'غير معروف'))}</span>
            </div>
            <div class="info-row">
                <span class="info-label">عنوان IP:</span>
                <span class="info-value">{data.get('ip_address', data.get('ip', 'غير معروف'))}</span>
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
                <span class="info-value">{time_str}</span>
            </div>
            <div class="info-row">
                <span class="info-label">Device:</span>
                <span class="info-value">{data.get('user_agent', data.get('device', 'Unknown'))}</span>
            </div>
            <div class="info-row">
                <span class="info-label">IP Address:</span>
                <span class="info-value">{data.get('ip_address', data.get('ip', 'Unknown'))}</span>
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
            <h2>تم تأكيد إيداعك! ✅</h2>
            <div class="success-box">
                تم إضافة المبلغ إلى رصيدك بنجاح!
            </div>
            <div class="info-row">
                <span class="info-label">المبلغ:</span>
                <span class="amount">${data.get('amount', '0')} USDC</span>
            </div>
            <div class="info-row">
                <span class="info-label">الوحدات المضافة:</span>
                <span class="info-value">{data.get('units', '0')} وحدة</span>
            </div>
            <div class="info-row">
                <span class="info-label">الرصيد الجديد:</span>
                <span class="info-value">${data.get('new_balance', '0')}</span>
            </div>
            <a href="https://asinax.cloud/wallet" class="button">عرض المحفظة</a>
            '''
        else:
            content = f'''
            <h2>Deposit Confirmed! ✅</h2>
            <div class="success-box">
                The amount has been added to your balance successfully!
            </div>
            <div class="info-row">
                <span class="info-label">Amount:</span>
                <span class="amount">${data.get('amount', '0')} USDC</span>
            </div>
            <div class="info-row">
                <span class="info-label">Units Added:</span>
                <span class="info-value">{data.get('units', '0')} units</span>
            </div>
            <div class="info-row">
                <span class="info-label">New Balance:</span>
                <span class="info-value">${data.get('new_balance', '0')}</span>
            </div>
            <a href="https://asinax.cloud/wallet" class="button">View Wallet</a>
            '''
        return self._get_base_template(content, language)
    
    def _render_withdrawal_template(self, data: dict, template_type: str, language: str = 'ar') -> str:
        """قالب السحب"""
        if template_type == 'requested':
            if language == 'ar':
                content = f'''
                <h2>تم استلام طلب السحب 📤</h2>
                <div class="info-box">
                    طلب السحب الخاص بك قيد المراجعة
                </div>
                <div class="info-row">
                    <span class="info-label">المبلغ:</span>
                    <span class="info-value">${data.get('amount', '0')} USDC</span>
                </div>
                <div class="info-row">
                    <span class="info-label">رقم الطلب:</span>
                    <span class="info-value">#{data.get('withdrawal_id', '')}</span>
                </div>
                <div class="info-row">
                    <span class="info-label">العنوان:</span>
                    <span class="info-value">{data.get('address', '')[:20]}...</span>
                </div>
                <p>سيتم مراجعة طلبك خلال 24 ساعة.</p>
                <a href="https://asinax.cloud/wallet" class="button">متابعة الطلب</a>
                '''
            else:
                content = f'''
                <h2>Withdrawal Request Received 📤</h2>
                <div class="info-box">
                    Your withdrawal request is under review
                </div>
                <div class="info-row">
                    <span class="info-label">Amount:</span>
                    <span class="info-value">${data.get('amount', '0')} USDC</span>
                </div>
                <div class="info-row">
                    <span class="info-label">Request ID:</span>
                    <span class="info-value">#{data.get('withdrawal_id', '')}</span>
                </div>
                <div class="info-row">
                    <span class="info-label">Address:</span>
                    <span class="info-value">{data.get('address', '')[:20]}...</span>
                </div>
                <p>Your request will be reviewed within 24 hours.</p>
                <a href="https://asinax.cloud/wallet" class="button">Track Request</a>
                '''
        elif template_type == 'approved':
            if language == 'ar':
                content = f'''
                <h2>تمت الموافقة على السحب ✅</h2>
                <div class="success-box">
                    تمت الموافقة على طلب السحب الخاص بك!
                </div>
                <div class="info-row">
                    <span class="info-label">المبلغ:</span>
                    <span class="info-value">${data.get('amount', '0')} USDC</span>
                </div>
                <div class="info-row">
                    <span class="info-label">رقم الطلب:</span>
                    <span class="info-value">#{data.get('withdrawal_id', '')}</span>
                </div>
                <p>جاري معالجة التحويل...</p>
                <a href="https://asinax.cloud/wallet" class="button">عرض المحفظة</a>
                '''
            else:
                content = f'''
                <h2>Withdrawal Approved ✅</h2>
                <div class="success-box">
                    Your withdrawal request has been approved!
                </div>
                <div class="info-row">
                    <span class="info-label">Amount:</span>
                    <span class="info-value">${data.get('amount', '0')} USDC</span>
                </div>
                <div class="info-row">
                    <span class="info-label">Request ID:</span>
                    <span class="info-value">#{data.get('withdrawal_id', '')}</span>
                </div>
                <p>Processing the transfer...</p>
                <a href="https://asinax.cloud/wallet" class="button">View Wallet</a>
                '''
        elif template_type == 'rejected':
            if language == 'ar':
                content = f'''
                <h2>تم رفض طلب السحب ❌</h2>
                <div class="alert-box">
                    للأسف، تم رفض طلب السحب الخاص بك
                </div>
                <div class="info-row">
                    <span class="info-label">المبلغ:</span>
                    <span class="info-value">${data.get('amount', '0')} USDC</span>
                </div>
                <div class="info-row">
                    <span class="info-label">رقم الطلب:</span>
                    <span class="info-value">#{data.get('withdrawal_id', '')}</span>
                </div>
                <div class="info-row">
                    <span class="info-label">السبب:</span>
                    <span class="info-value">{data.get('reason', 'غير محدد')}</span>
                </div>
                <p>يرجى التواصل مع الدعم لمزيد من المعلومات.</p>
                <a href="https://asinax.cloud/support" class="button">تواصل مع الدعم</a>
                '''
            else:
                content = f'''
                <h2>Withdrawal Rejected ❌</h2>
                <div class="alert-box">
                    Unfortunately, your withdrawal request has been rejected
                </div>
                <div class="info-row">
                    <span class="info-label">Amount:</span>
                    <span class="info-value">${data.get('amount', '0')} USDC</span>
                </div>
                <div class="info-row">
                    <span class="info-label">Request ID:</span>
                    <span class="info-value">#{data.get('withdrawal_id', '')}</span>
                </div>
                <div class="info-row">
                    <span class="info-label">Reason:</span>
                    <span class="info-value">{data.get('reason', 'Not specified')}</span>
                </div>
                <p>Please contact support for more information.</p>
                <a href="https://asinax.cloud/support" class="button">Contact Support</a>
                '''
        else:  # completed
            if language == 'ar':
                content = f'''
                <h2>تم إتمام السحب بنجاح! 🎉</h2>
                <div class="success-box">
                    تم إرسال المبلغ إلى محفظتك بنجاح!
                </div>
                <div class="info-row">
                    <span class="info-label">المبلغ:</span>
                    <span class="amount">${data.get('amount', '0')} USDC</span>
                </div>
                <div class="info-row">
                    <span class="info-label">رقم المعاملة:</span>
                    <span class="info-value">{data.get('tx_hash', '')[:20]}...</span>
                </div>
                <a href="https://asinax.cloud/wallet" class="button">عرض المحفظة</a>
                '''
            else:
                content = f'''
                <h2>Withdrawal Completed! 🎉</h2>
                <div class="success-box">
                    The amount has been sent to your wallet successfully!
                </div>
                <div class="info-row">
                    <span class="info-label">Amount:</span>
                    <span class="amount">${data.get('amount', '0')} USDC</span>
                </div>
                <div class="info-row">
                    <span class="info-label">Transaction Hash:</span>
                    <span class="info-value">{data.get('tx_hash', '')[:20]}...</span>
                </div>
                <a href="https://asinax.cloud/wallet" class="button">View Wallet</a>
                '''
        return self._get_base_template(content, language)
    
    def _render_referral_bonus_template(self, data: dict, language: str = 'ar') -> str:
        """قالب مكافأة الإحالة"""
        if language == 'ar':
            content = f'''
            <h2>مكافأة إحالة جديدة! 🎁</h2>
            <div class="success-box">
                تهانينا! لقد حصلت على مكافأة إحالة!
            </div>
            <div class="info-row">
                <span class="info-label">المكافأة:</span>
                <span class="amount">${data.get('bonus', '0')}</span>
            </div>
            <div class="info-row">
                <span class="info-label">من المستخدم:</span>
                <span class="info-value">{data.get('referred_user', '')}</span>
            </div>
            <p>استمر في دعوة أصدقائك للحصول على المزيد من المكافآت!</p>
            <a href="https://asinax.cloud/referral" class="button">برنامج الإحالة</a>
            '''
        else:
            content = f'''
            <h2>New Referral Bonus! 🎁</h2>
            <div class="success-box">
                Congratulations! You've received a referral bonus!
            </div>
            <div class="info-row">
                <span class="info-label">Bonus:</span>
                <span class="amount">${data.get('bonus', '0')}</span>
            </div>
            <div class="info-row">
                <span class="info-label">From User:</span>
                <span class="info-value">{data.get('referred_user', '')}</span>
            </div>
            <p>Keep inviting friends to earn more bonuses!</p>
            <a href="https://asinax.cloud/referral" class="button">Referral Program</a>
            '''
        return self._get_base_template(content, language)
    
    def _render_platform_announcement_template(self, data: dict, language: str = 'ar') -> str:
        """قالب إعلان المنصة"""
        if language == 'ar':
            content = f'''
            <h2>📢 إعلان هام</h2>
            <h3>{data.get('title', '')}</h3>
            <div style="padding: 15px; background: rgba(139, 92, 246, 0.1); border-radius: 8px; margin: 15px 0;">
                {data.get('message', '')}
            </div>
            <a href="{data.get('action_url', 'https://asinax.cloud')}" class="button">اقرأ المزيد</a>
            '''
        else:
            content = f'''
            <h2>📢 Important Announcement</h2>
            <h3>{data.get('title', '')}</h3>
            <div style="padding: 15px; background: rgba(139, 92, 246, 0.1); border-radius: 8px; margin: 15px 0;">
                {data.get('message', '')}
            </div>
            <a href="{data.get('action_url', 'https://asinax.cloud')}" class="button">Read More</a>
            '''
        return self._get_base_template(content, language)
    
    def _render_admin_message_template(self, data: dict, language: str = 'ar') -> str:
        """قالب رسالة الأدمن"""
        if language == 'ar':
            content = f'''
            <h2>💬 رسالة من الإدارة</h2>
            <div style="padding: 20px; background: rgba(139, 92, 246, 0.1); border-radius: 8px; margin: 15px 0; border-right: 4px solid #8b5cf6;">
                <p style="font-size: 16px;">{data.get('message', '')}</p>
            </div>
            <p style="color: #888;">من: فريق ASINAX</p>
            <a href="https://asinax.cloud" class="button">زيارة المنصة</a>
            '''
        else:
            content = f'''
            <h2>💬 Message from Admin</h2>
            <div style="padding: 20px; background: rgba(139, 92, 246, 0.1); border-radius: 8px; margin: 15px 0; border-left: 4px solid #8b5cf6;">
                <p style="font-size: 16px;">{data.get('message', '')}</p>
            </div>
            <p style="color: #888;">From: ASINAX Team</p>
            <a href="https://asinax.cloud" class="button">Visit Platform</a>
            '''
        return self._get_base_template(content, language)
    
    def _render_promotion_template(self, data: dict, language: str = 'ar') -> str:
        """قالب العروض الترويجية"""
        if language == 'ar':
            content = f'''
            <h2>🌟 عرض خاص!</h2>
            <h3 style="color: #8b5cf6;">{data.get('title', '')}</h3>
            <div style="padding: 20px; background: linear-gradient(135deg, rgba(139, 92, 246, 0.2), rgba(99, 102, 241, 0.2)); border-radius: 12px; margin: 15px 0;">
                <p style="font-size: 16px;">{data.get('message', '')}</p>
                {f'<p class="amount">{data.get("discount", "")}</p>' if data.get("discount") else ''}
            </div>
            <p>⏰ ينتهي العرض: {data.get('expires_at', 'قريباً')}</p>
            <a href="{data.get('action_url', 'https://asinax.cloud')}" class="button">استفد الآن</a>
            '''
        else:
            content = f'''
            <h2>🌟 Special Offer!</h2>
            <h3 style="color: #8b5cf6;">{data.get('title', '')}</h3>
            <div style="padding: 20px; background: linear-gradient(135deg, rgba(139, 92, 246, 0.2), rgba(99, 102, 241, 0.2)); border-radius: 12px; margin: 15px 0;">
                <p style="font-size: 16px;">{data.get('message', '')}</p>
                {f'<p class="amount">{data.get("discount", "")}</p>' if data.get("discount") else ''}
            </div>
            <p>⏰ Offer ends: {data.get('expires_at', 'Soon')}</p>
            <a href="{data.get('action_url', 'https://asinax.cloud')}" class="button">Claim Now</a>
            '''
        return self._get_base_template(content, language)
    
    def _render_vip_upgrade_template(self, data: dict, language: str = 'ar') -> str:
        """قالب ترقية VIP"""
        if language == 'ar':
            content = f'''
            <h2>⭐ تهانينا! تمت ترقيتك!</h2>
            <div class="success-box">
                أنت الآن عضو {data.get('vip_level', 'VIP')}!
            </div>
            <h3>المزايا الجديدة:</h3>
            <ul>
                <li>رسوم أقل على المعاملات</li>
                <li>دعم فني أولوية</li>
                <li>عروض حصرية</li>
                <li>تقارير متقدمة</li>
            </ul>
            <a href="https://asinax.cloud/vip" class="button">استكشف مزاياك</a>
            '''
        else:
            content = f'''
            <h2>⭐ Congratulations! You've been upgraded!</h2>
            <div class="success-box">
                You are now a {data.get('vip_level', 'VIP')} member!
            </div>
            <h3>New Benefits:</h3>
            <ul>
                <li>Lower transaction fees</li>
                <li>Priority support</li>
                <li>Exclusive offers</li>
                <li>Advanced reports</li>
            </ul>
            <a href="https://asinax.cloud/vip" class="button">Explore Your Benefits</a>
            '''
        return self._get_base_template(content, language)
    
    def _render_profit_notification_template(self, data: dict, language: str = 'ar') -> str:
        """قالب إشعار الأرباح"""
        if language == 'ar':
            content = f'''
            <h2>💰 أرباح جديدة!</h2>
            <div class="success-box">
                تم تحقيق أرباح في محفظتك!
            </div>
            <div class="info-row">
                <span class="info-label">الربح:</span>
                <span class="amount">+${data.get('profit', '0')}</span>
            </div>
            <div class="info-row">
                <span class="info-label">نسبة الربح:</span>
                <span class="info-value" style="color: #10b981;">+{data.get('profit_percent', '0')}%</span>
            </div>
            <div class="info-row">
                <span class="info-label">الرصيد الحالي:</span>
                <span class="info-value">${data.get('current_balance', '0')}</span>
            </div>
            <a href="https://asinax.cloud/portfolio" class="button">عرض المحفظة</a>
            '''
        else:
            content = f'''
            <h2>💰 New Profits!</h2>
            <div class="success-box">
                Profits have been achieved in your portfolio!
            </div>
            <div class="info-row">
                <span class="info-label">Profit:</span>
                <span class="amount">+${data.get('profit', '0')}</span>
            </div>
            <div class="info-row">
                <span class="info-label">Profit Percentage:</span>
                <span class="info-value" style="color: #10b981;">+{data.get('profit_percent', '0')}%</span>
            </div>
            <div class="info-row">
                <span class="info-label">Current Balance:</span>
                <span class="info-value">${data.get('current_balance', '0')}</span>
            </div>
            <a href="https://asinax.cloud/portfolio" class="button">View Portfolio</a>
            '''
        return self._get_base_template(content, language)
    
    def _render_otp_template(self, data: dict, language: str = 'ar') -> str:
        """قالب رمز التحقق OTP"""
        if language == 'ar':
            content = f'''
            <h2>🔐 رمز التحقق</h2>
            <p>استخدم الرمز التالي للتحقق من حسابك:</p>
            <div style="text-align: center; padding: 30px; background: rgba(139, 92, 246, 0.1); border-radius: 12px; margin: 20px 0;">
                <span style="font-size: 36px; font-weight: bold; letter-spacing: 8px; color: #8b5cf6;">{data.get('otp_code', '')}</span>
            </div>
            <div class="alert-box">
                <strong>⚠️ تنبيه:</strong> هذا الرمز صالح لمدة {data.get('expires_in', '10')} دقائق فقط. لا تشاركه مع أي شخص.
            </div>
            '''
        else:
            content = f'''
            <h2>🔐 Verification Code</h2>
            <p>Use the following code to verify your account:</p>
            <div style="text-align: center; padding: 30px; background: rgba(139, 92, 246, 0.1); border-radius: 12px; margin: 20px 0;">
                <span style="font-size: 36px; font-weight: bold; letter-spacing: 8px; color: #8b5cf6;">{data.get('otp_code', '')}</span>
            </div>
            <div class="alert-box">
                <strong>⚠️ Warning:</strong> This code is valid for {data.get('expires_in', '10')} minutes only. Do not share it with anyone.
            </div>
            '''
        return self._get_base_template(content, language)
    
    async def send_email(
        self,
        to_email: str,
        template_name: str,
        data: dict = None,
        language: str = 'ar',
        attachments: Optional[List[str]] = None
    ) -> bool:
        """إرسال بريد إلكتروني"""
        try:
            if data is None:
                data = {}
            
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
            elif template_name == 'withdrawal_requested':
                html_content = self._render_withdrawal_template(data, 'requested', language)
            elif template_name == 'withdrawal_approved':
                html_content = self._render_withdrawal_template(data, 'approved', language)
            elif template_name == 'withdrawal_rejected':
                html_content = self._render_withdrawal_template(data, 'rejected', language)
            elif template_name == 'withdrawal_completed':
                html_content = self._render_withdrawal_template(data, 'completed', language)
            elif template_name == 'referral_bonus':
                html_content = self._render_referral_bonus_template(data, language)
            elif template_name == 'platform_announcement':
                html_content = self._render_platform_announcement_template(data, language)
            elif template_name == 'admin_message':
                html_content = self._render_admin_message_template(data, language)
            elif template_name == 'promotion':
                html_content = self._render_promotion_template(data, language)
            elif template_name == 'vip_upgrade':
                html_content = self._render_vip_upgrade_template(data, language)
            elif template_name == 'profit_notification':
                html_content = self._render_profit_notification_template(data, language)
            elif template_name == 'otp_verification':
                html_content = self._render_otp_template(data, language)
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
            
            logger.info(f"Email sent successfully to {to_email} (template: {template_name})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email: {str(e)}")
            return False
    
    async def send_direct_email(
        self,
        to_email: str,
        subject: str,
        html_content: str
    ) -> bool:
        """إرسال بريد إلكتروني مباشر بدون قالب"""
        try:
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = f"{self.from_name} <{self.from_email}>"
            msg['To'] = to_email
            
            msg.attach(MIMEText(html_content, 'html', 'utf-8'))
            
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                if self.smtp_user and self.smtp_password:
                    server.login(self.smtp_user, self.smtp_password)
                server.sendmail(self.from_email, to_email, msg.as_string())
            
            logger.info(f"Direct email sent successfully to {to_email}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send direct email: {str(e)}")
            return False

    async def send_login_notification(
        self,
        email: str,
        name: str,
        ip_address: str = "Unknown",
        user_agent: str = "Unknown",
        language: str = "ar"
    ) -> bool:
        """إرسال إشعار تسجيل دخول جديد"""
        try:
            data = {
                "name": name,
                "ip_address": ip_address,
                "user_agent": user_agent,
                "time": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            return await self.send_email(email, 'login_alert', data, language)
        except Exception as e:
            logger.error(f"Failed to send login notification: {e}")
            return False

    async def send_welcome_email(
        self,
        email: str,
        name: str,
        language: str = "ar"
    ) -> bool:
        """إرسال بريد ترحيبي للمستخدم الجديد"""
        try:
            data = {"name": name}
            return await self.send_email(email, 'welcome', data, language)
        except Exception as e:
            logger.error(f"Failed to send welcome email: {e}")
            return False
    
    async def send_deposit_confirmation(
        self,
        email: str,
        amount: float,
        units: float,
        new_balance: float,
        language: str = "ar"
    ) -> bool:
        """إرسال تأكيد الإيداع"""
        try:
            data = {
                "amount": f"{amount:.2f}",
                "units": f"{units:.4f}",
                "new_balance": f"{new_balance:.2f}"
            }
            return await self.send_email(email, 'deposit_confirmed', data, language)
        except Exception as e:
            logger.error(f"Failed to send deposit confirmation: {e}")
            return False
    
    async def send_withdrawal_notification(
        self,
        email: str,
        amount: float,
        withdrawal_id: int,
        status: str,
        address: str = "",
        tx_hash: str = "",
        reason: str = "",
        language: str = "ar"
    ) -> bool:
        """إرسال إشعار السحب"""
        try:
            data = {
                "amount": f"{amount:.2f}",
                "withdrawal_id": withdrawal_id,
                "address": address,
                "tx_hash": tx_hash,
                "reason": reason
            }
            template_map = {
                "pending": "withdrawal_requested",
                "approved": "withdrawal_approved",
                "rejected": "withdrawal_rejected",
                "completed": "withdrawal_completed"
            }
            template_name = template_map.get(status, "withdrawal_requested")
            return await self.send_email(email, template_name, data, language)
        except Exception as e:
            logger.error(f"Failed to send withdrawal notification: {e}")
            return False
    
    async def send_referral_bonus_notification(
        self,
        email: str,
        bonus: float,
        referred_user: str,
        language: str = "ar"
    ) -> bool:
        """إرسال إشعار مكافأة الإحالة"""
        try:
            data = {
                "bonus": f"{bonus:.2f}",
                "referred_user": referred_user
            }
            return await self.send_email(email, 'referral_bonus', data, language)
        except Exception as e:
            logger.error(f"Failed to send referral bonus notification: {e}")
            return False
    
    async def send_platform_announcement(
        self,
        email: str,
        title: str,
        message: str,
        action_url: str = "https://asinax.cloud",
        language: str = "ar"
    ) -> bool:
        """إرسال إعلان المنصة"""
        try:
            data = {
                "title": title,
                "message": message,
                "action_url": action_url
            }
            return await self.send_email(email, 'platform_announcement', data, language)
        except Exception as e:
            logger.error(f"Failed to send platform announcement: {e}")
            return False
    
    async def send_admin_message(
        self,
        email: str,
        message: str,
        language: str = "ar"
    ) -> bool:
        """إرسال رسالة من الأدمن"""
        try:
            data = {"message": message}
            return await self.send_email(email, 'admin_message', data, language)
        except Exception as e:
            logger.error(f"Failed to send admin message: {e}")
            return False
    
    async def send_promotion(
        self,
        email: str,
        title: str,
        message: str,
        discount: str = "",
        expires_at: str = "",
        action_url: str = "https://asinax.cloud",
        language: str = "ar"
    ) -> bool:
        """إرسال عرض ترويجي"""
        try:
            data = {
                "title": title,
                "message": message,
                "discount": discount,
                "expires_at": expires_at,
                "action_url": action_url
            }
            return await self.send_email(email, 'promotion', data, language)
        except Exception as e:
            logger.error(f"Failed to send promotion: {e}")
            return False
    
    async def send_vip_upgrade_notification(
        self,
        email: str,
        vip_level: str,
        language: str = "ar"
    ) -> bool:
        """إرسال إشعار ترقية VIP"""
        try:
            data = {"vip_level": vip_level}
            return await self.send_email(email, 'vip_upgrade', data, language)
        except Exception as e:
            logger.error(f"Failed to send VIP upgrade notification: {e}")
            return False
    
    async def send_profit_notification(
        self,
        email: str,
        profit: float,
        profit_percent: float,
        current_balance: float,
        language: str = "ar"
    ) -> bool:
        """إرسال إشعار الأرباح"""
        try:
            data = {
                "profit": f"{profit:.2f}",
                "profit_percent": f"{profit_percent:.2f}",
                "current_balance": f"{current_balance:.2f}"
            }
            return await self.send_email(email, 'profit_notification', data, language)
        except Exception as e:
            logger.error(f"Failed to send profit notification: {e}")
            return False
    
    async def send_otp(
        self,
        email: str,
        otp_code: str,
        expires_in: int = 10,
        language: str = "ar"
    ) -> bool:
        """إرسال رمز التحقق OTP"""
        try:
            data = {
                "otp_code": otp_code,
                "expires_in": str(expires_in)
            }
            return await self.send_email(email, 'otp_verification', data, language)
        except Exception as e:
            logger.error(f"Failed to send OTP: {e}")
            return False


# إنشاء instance عام
email_service = EmailService()
