import aiosmtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional
from app.core.config import settings
import logging
import secrets
from datetime import datetime

logger = logging.getLogger(__name__)

# Logo URL - hosted on CDN for email compatibility
LOGO_URL = "https://asinax.cloud/images/logo.jpg"


class EmailService:
    """Professional Email Service for ASINAX Platform"""
    
    def generate_confirmation_token(self) -> str:
        """Generate a secure confirmation token for withdrawals"""
        return secrets.token_urlsafe(32)
    
    def _get_professional_template(self, content: str, footer_note: str = "") -> str:
        """Get professional email template with ASINAX branding"""
        current_year = datetime.utcnow().year
        
        return f"""
<!DOCTYPE html>
<html dir="rtl" lang="ar">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <title>ASINAX</title>
    <!--[if mso]>
    <noscript>
        <xml>
            <o:OfficeDocumentSettings>
                <o:PixelsPerInch>96</o:PixelsPerInch>
            </o:OfficeDocumentSettings>
        </xml>
    </noscript>
    <![endif]-->
</head>
<body style="margin: 0; padding: 0; background-color: #000000; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;">
    <!-- Wrapper Table -->
    <table role="presentation" width="100%" cellspacing="0" cellpadding="0" border="0" style="background-color: #000000;">
        <tr>
            <td align="center" style="padding: 40px 20px;">
                <!-- Main Container -->
                <table role="presentation" width="600" cellspacing="0" cellpadding="0" border="0" style="max-width: 600px; width: 100%;">
                    
                    <!-- Header with Logo -->
                    <tr>
                        <td align="center" style="padding: 30px 40px; background: linear-gradient(135deg, #0a0a1a 0%, #1a1a3e 50%, #0a0a1a 100%); border-radius: 20px 20px 0 0; border: 1px solid #3b82f6; border-bottom: none;">
                            <img src="{LOGO_URL}" alt="ASINAX" width="120" height="120" style="display: block; border-radius: 50%; border: 3px solid #3b82f6; box-shadow: 0 0 30px rgba(59, 130, 246, 0.5);">
                            <h1 style="color: #3b82f6; font-size: 28px; margin: 20px 0 5px 0; letter-spacing: 3px; text-shadow: 0 0 20px rgba(59, 130, 246, 0.5);">ASINAX</h1>
                            <p style="color: #8b5cf6; font-size: 14px; margin: 0; letter-spacing: 2px;">CRYPTO AI TRADING</p>
                        </td>
                    </tr>
                    
                    <!-- Main Content -->
                    <tr>
                        <td style="background: linear-gradient(180deg, #0d0d1f 0%, #111127 100%); padding: 40px; border-left: 1px solid #3b82f6; border-right: 1px solid #3b82f6;">
                            {content}
                        </td>
                    </tr>
                    
                    <!-- Footer -->
                    <tr>
                        <td style="background: linear-gradient(135deg, #0a0a1a 0%, #1a1a3e 50%, #0a0a1a 100%); padding: 30px 40px; border-radius: 0 0 20px 20px; border: 1px solid #3b82f6; border-top: none;">
                            <!-- Social Links -->
                            <table role="presentation" width="100%" cellspacing="0" cellpadding="0" border="0">
                                <tr>
                                    <td align="center" style="padding-bottom: 20px;">
                                        <a href="https://asinax.cloud" style="color: #3b82f6; text-decoration: none; margin: 0 10px; font-size: 14px;">🌐 الموقع</a>
                                        <a href="https://t.me/asinax_support" style="color: #3b82f6; text-decoration: none; margin: 0 10px; font-size: 14px;">💬 تيليجرام</a>
                                        <a href="mailto:support@asinax.cloud" style="color: #3b82f6; text-decoration: none; margin: 0 10px; font-size: 14px;">📧 الدعم</a>
                                    </td>
                                </tr>
                            </table>
                            
                            <!-- Divider -->
                            <table role="presentation" width="100%" cellspacing="0" cellpadding="0" border="0">
                                <tr>
                                    <td style="border-top: 1px solid rgba(59, 130, 246, 0.3); padding-top: 20px;">
                                        <p style="color: #6b7280; font-size: 12px; margin: 0; text-align: center; line-height: 1.8;">
                                            {footer_note if footer_note else "هذا الإيميل تم إرساله تلقائياً من منصة ASINAX. يرجى عدم الرد على هذا الإيميل."}
                                        </p>
                                        <p style="color: #4b5563; font-size: 11px; margin: 15px 0 0 0; text-align: center;">
                                            © {current_year} ASINAX Crypto AI. جميع الحقوق محفوظة.
                                        </p>
                                        <p style="color: #374151; font-size: 10px; margin: 10px 0 0 0; text-align: center;">
                                            ASINAX هي منصة تداول بالذكاء الاصطناعي. التداول ينطوي على مخاطر.
                                        </p>
                                    </td>
                                </tr>
                            </table>
                        </td>
                    </tr>
                    
                </table>
            </td>
        </tr>
    </table>
</body>
</html>
"""
    
    async def send_email(
        self, 
        to_email: str, 
        subject: str, 
        html_content: str,
        text_content: Optional[str] = None
    ) -> bool:
        """Send an email"""
        try:
            message = MIMEMultipart("alternative")
            message["From"] = f"ASINAX <{settings.EMAIL_FROM}>"
            message["To"] = to_email
            message["Subject"] = subject
            
            if text_content:
                message.attach(MIMEText(text_content, "plain"))
            message.attach(MIMEText(html_content, "html"))
            
            await aiosmtplib.send(
                message,
                hostname=settings.SMTP_HOST,
                port=settings.SMTP_PORT,
                username=settings.SMTP_USER,
                password=settings.SMTP_PASSWORD,
                start_tls=True
            )
            
            logger.info(f"Email sent to {to_email}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email to {to_email}: {e}")
            return False
    
    # ============================================================
    # التسجيل والتحقق
    # ============================================================
    
    async def send_verification_otp(self, to_email: str, otp: str) -> bool:
        """Send OTP for email verification during registration"""
        content = f"""
        <h2 style="color: #ffffff; font-size: 24px; margin: 0 0 20px 0; text-align: center;">
            🔐 التحقق من البريد الإلكتروني
        </h2>
        
        <p style="color: #d1d5db; font-size: 16px; line-height: 1.8; text-align: center; margin: 0 0 30px 0;">
            مرحباً بك في <strong style="color: #3b82f6;">ASINAX</strong>!<br>
            لإكمال عملية التسجيل، يرجى استخدام رمز التحقق التالي:
        </p>
        
        <!-- OTP Box -->
        <table role="presentation" width="100%" cellspacing="0" cellpadding="0" border="0">
            <tr>
                <td align="center">
                    <table role="presentation" cellspacing="0" cellpadding="0" border="0" style="background: linear-gradient(135deg, rgba(59, 130, 246, 0.1) 0%, rgba(139, 92, 246, 0.1) 100%); border: 2px solid #3b82f6; border-radius: 16px; padding: 30px 50px;">
                        <tr>
                            <td align="center">
                                <p style="font-size: 48px; font-weight: bold; color: #3b82f6; letter-spacing: 12px; margin: 0; text-shadow: 0 0 20px rgba(59, 130, 246, 0.5);">{otp}</p>
                            </td>
                        </tr>
                    </table>
                </td>
            </tr>
        </table>
        
        <!-- Timer -->
        <p style="color: #9ca3af; font-size: 14px; text-align: center; margin: 25px 0;">
            ⏱️ هذا الرمز صالح لمدة <strong style="color: #f59e0b;">10 دقائق</strong> فقط
        </p>
        
        <!-- Security Warning -->
        <table role="presentation" width="100%" cellspacing="0" cellpadding="0" border="0" style="background: rgba(245, 158, 11, 0.1); border: 1px solid #f59e0b; border-radius: 12px; margin-top: 20px;">
            <tr>
                <td style="padding: 20px;">
                    <p style="color: #f59e0b; font-size: 14px; margin: 0; text-align: center;">
                        ⚠️ <strong>تنبيه أمني:</strong> لا تشارك هذا الرمز مع أي شخص.<br>
                        فريق ASINAX لن يطلب منك هذا الرمز أبداً عبر الهاتف أو الرسائل.
                    </p>
                </td>
            </tr>
        </table>
        """
        
        html = self._get_professional_template(content)
        return await self.send_email(to_email, "🔐 ASINAX - رمز التحقق من البريد الإلكتروني", html)
    
    async def send_welcome_email(self, to_email: str, full_name: str) -> bool:
        """Send welcome email after successful registration"""
        content = f"""
        <h2 style="color: #22c55e; font-size: 28px; margin: 0 0 20px 0; text-align: center;">
            🎉 مرحباً بك في عائلة ASINAX!
        </h2>
        
        <p style="color: #ffffff; font-size: 20px; text-align: center; margin: 0 0 10px 0;">
            أهلاً <strong style="color: #3b82f6;">{full_name}</strong>
        </p>
        
        <p style="color: #d1d5db; font-size: 16px; line-height: 1.8; text-align: center; margin: 0 0 30px 0;">
            تم إنشاء حسابك بنجاح! أنت الآن جزء من مجتمع المستثمرين الأذكياء في ASINAX.
        </p>
        
        <!-- Success Badge -->
        <table role="presentation" width="100%" cellspacing="0" cellpadding="0" border="0">
            <tr>
                <td align="center">
                    <table role="presentation" cellspacing="0" cellpadding="0" border="0" style="background: rgba(34, 197, 94, 0.1); border: 2px solid #22c55e; border-radius: 16px; padding: 20px 40px;">
                        <tr>
                            <td align="center">
                                <p style="color: #22c55e; font-size: 18px; margin: 0;">✅ حسابك جاهز للاستخدام</p>
                            </td>
                        </tr>
                    </table>
                </td>
            </tr>
        </table>
        
        <!-- Features -->
        <table role="presentation" width="100%" cellspacing="0" cellpadding="0" border="0" style="margin-top: 30px;">
            <tr>
                <td style="background: rgba(59, 130, 246, 0.05); border-radius: 12px; padding: 25px;">
                    <h3 style="color: #3b82f6; font-size: 18px; margin: 0 0 20px 0;">🚀 ماذا يمكنك فعله الآن؟</h3>
                    
                    <table role="presentation" width="100%" cellspacing="0" cellpadding="0" border="0">
                        <tr>
                            <td style="padding: 10px 0; border-bottom: 1px solid rgba(255,255,255,0.1);">
                                <span style="color: #22c55e; font-size: 16px;">💰</span>
                                <span style="color: #d1d5db; font-size: 14px; margin-right: 10px;">إيداع الأموال وبدء الاستثمار</span>
                            </td>
                        </tr>
                        <tr>
                            <td style="padding: 10px 0; border-bottom: 1px solid rgba(255,255,255,0.1);">
                                <span style="color: #22c55e; font-size: 16px;">🤖</span>
                                <span style="color: #d1d5db; font-size: 14px; margin-right: 10px;">متابعة أداء الوكيل الذكي في الوقت الفعلي</span>
                            </td>
                        </tr>
                        <tr>
                            <td style="padding: 10px 0; border-bottom: 1px solid rgba(255,255,255,0.1);">
                                <span style="color: #22c55e; font-size: 16px;">📊</span>
                                <span style="color: #d1d5db; font-size: 14px; margin-right: 10px;">مراقبة أرباحك وتحليلات محفظتك</span>
                            </td>
                        </tr>
                        <tr>
                            <td style="padding: 10px 0;">
                                <span style="color: #22c55e; font-size: 16px;">🔒</span>
                                <span style="color: #d1d5db; font-size: 14px; margin-right: 10px;">تأمين حسابك بالمصادقة الثنائية</span>
                            </td>
                        </tr>
                    </table>
                </td>
            </tr>
        </table>
        
        <!-- CTA Button -->
        <table role="presentation" width="100%" cellspacing="0" cellpadding="0" border="0" style="margin-top: 30px;">
            <tr>
                <td align="center">
                    <a href="https://asinax.cloud/dashboard" style="display: inline-block; background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%); color: #ffffff; padding: 18px 50px; text-decoration: none; border-radius: 12px; font-weight: bold; font-size: 16px; box-shadow: 0 10px 30px rgba(59, 130, 246, 0.3);">
                        🚀 الذهاب إلى لوحة التحكم
                    </a>
                </td>
            </tr>
        </table>
        """
        
        html = self._get_professional_template(content, "نحن سعداء بانضمامك إلينا! إذا كان لديك أي استفسار، فريق الدعم جاهز لمساعدتك على مدار الساعة.")
        return await self.send_email(to_email, "🎉 مرحباً بك في ASINAX - حسابك جاهز!", html)
    
    # ============================================================
    # تسجيل الدخول والأمان
    # ============================================================
    
    async def send_login_notification(
        self, 
        to_email: str, 
        ip_address: str = "غير معروف",
        device: str = "غير معروف",
        location: str = "غير معروف",
        login_time: Optional[datetime] = None
    ) -> bool:
        """Send notification when user logs in"""
        if login_time is None:
            login_time = datetime.utcnow()
        
        formatted_time = login_time.strftime("%Y-%m-%d %H:%M:%S UTC")
        
        content = f"""
        <h2 style="color: #3b82f6; font-size: 24px; margin: 0 0 20px 0; text-align: center;">
            🔔 تسجيل دخول جديد إلى حسابك
        </h2>
        
        <p style="color: #d1d5db; font-size: 16px; line-height: 1.8; text-align: center; margin: 0 0 30px 0;">
            تم تسجيل الدخول إلى حسابك في ASINAX بنجاح.<br>
            إليك تفاصيل الجلسة:
        </p>
        
        <!-- Login Details -->
        <table role="presentation" width="100%" cellspacing="0" cellpadding="0" border="0" style="background: rgba(59, 130, 246, 0.05); border-radius: 12px; border: 1px solid rgba(59, 130, 246, 0.2);">
            <tr>
                <td style="padding: 25px;">
                    <table role="presentation" width="100%" cellspacing="0" cellpadding="0" border="0">
                        <tr>
                            <td style="padding: 15px 0; border-bottom: 1px solid rgba(255,255,255,0.1);">
                                <span style="color: #9ca3af; font-size: 14px;">📅 التاريخ والوقت</span>
                                <span style="color: #ffffff; font-size: 14px; float: left; direction: ltr;">{formatted_time}</span>
                            </td>
                        </tr>
                        <tr>
                            <td style="padding: 15px 0; border-bottom: 1px solid rgba(255,255,255,0.1);">
                                <span style="color: #9ca3af; font-size: 14px;">🌐 عنوان IP</span>
                                <span style="color: #ffffff; font-size: 14px; float: left; direction: ltr;">{ip_address}</span>
                            </td>
                        </tr>
                        <tr>
                            <td style="padding: 15px 0; border-bottom: 1px solid rgba(255,255,255,0.1);">
                                <span style="color: #9ca3af; font-size: 14px;">💻 الجهاز</span>
                                <span style="color: #ffffff; font-size: 14px; float: left;">{device}</span>
                            </td>
                        </tr>
                        <tr>
                            <td style="padding: 15px 0;">
                                <span style="color: #9ca3af; font-size: 14px;">📍 الموقع التقريبي</span>
                                <span style="color: #ffffff; font-size: 14px; float: left;">{location}</span>
                            </td>
                        </tr>
                    </table>
                </td>
            </tr>
        </table>
        
        <!-- Security Warning -->
        <table role="presentation" width="100%" cellspacing="0" cellpadding="0" border="0" style="background: rgba(239, 68, 68, 0.1); border: 1px solid #ef4444; border-radius: 12px; margin-top: 25px;">
            <tr>
                <td style="padding: 20px;">
                    <p style="color: #ef4444; font-size: 14px; margin: 0 0 15px 0; text-align: center;">
                        ⚠️ <strong>هل هذا أنت؟</strong>
                    </p>
                    <p style="color: #fca5a5; font-size: 13px; margin: 0; text-align: center; line-height: 1.8;">
                        إذا لم تكن أنت من قام بتسجيل الدخول، يرجى اتخاذ الإجراءات التالية فوراً:<br>
                        1. تغيير كلمة المرور<br>
                        2. تفعيل المصادقة الثنائية<br>
                        3. التواصل مع فريق الدعم
                    </p>
                </td>
            </tr>
        </table>
        
        <!-- Security Button -->
        <table role="presentation" width="100%" cellspacing="0" cellpadding="0" border="0" style="margin-top: 25px;">
            <tr>
                <td align="center">
                    <a href="https://asinax.cloud/settings/security" style="display: inline-block; background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%); color: #ffffff; padding: 15px 40px; text-decoration: none; border-radius: 10px; font-weight: bold; font-size: 14px;">
                        🔒 تأمين حسابي الآن
                    </a>
                </td>
            </tr>
        </table>
        """
        
        html = self._get_professional_template(content, "نرسل لك هذا الإشعار للحفاظ على أمان حسابك. إذا كان هذا أنت، يمكنك تجاهل هذا الإيميل.")
        return await self.send_email(to_email, "🔔 ASINAX - تسجيل دخول جديد إلى حسابك", html)
    
    # ============================================================
    # استعادة كلمة السر
    # ============================================================
    
    async def send_password_reset_otp(self, to_email: str, otp: str) -> bool:
        """Send OTP for password reset"""
        content = f"""
        <h2 style="color: #f59e0b; font-size: 24px; margin: 0 0 20px 0; text-align: center;">
            🔑 إعادة تعيين كلمة المرور
        </h2>
        
        <p style="color: #d1d5db; font-size: 16px; line-height: 1.8; text-align: center; margin: 0 0 30px 0;">
            تلقينا طلباً لإعادة تعيين كلمة مرور حسابك في ASINAX.<br>
            استخدم الرمز التالي لإكمال العملية:
        </p>
        
        <!-- OTP Box -->
        <table role="presentation" width="100%" cellspacing="0" cellpadding="0" border="0">
            <tr>
                <td align="center">
                    <table role="presentation" cellspacing="0" cellpadding="0" border="0" style="background: linear-gradient(135deg, rgba(245, 158, 11, 0.1) 0%, rgba(234, 88, 12, 0.1) 100%); border: 2px solid #f59e0b; border-radius: 16px; padding: 30px 50px;">
                        <tr>
                            <td align="center">
                                <p style="font-size: 48px; font-weight: bold; color: #f59e0b; letter-spacing: 12px; margin: 0; text-shadow: 0 0 20px rgba(245, 158, 11, 0.5);">{otp}</p>
                            </td>
                        </tr>
                    </table>
                </td>
            </tr>
        </table>
        
        <!-- Timer -->
        <p style="color: #9ca3af; font-size: 14px; text-align: center; margin: 25px 0;">
            ⏱️ هذا الرمز صالح لمدة <strong style="color: #f59e0b;">10 دقائق</strong> فقط
        </p>
        
        <!-- Security Note -->
        <table role="presentation" width="100%" cellspacing="0" cellpadding="0" border="0" style="background: rgba(59, 130, 246, 0.05); border-radius: 12px; margin-top: 20px;">
            <tr>
                <td style="padding: 20px;">
                    <p style="color: #9ca3af; font-size: 13px; margin: 0; text-align: center; line-height: 1.8;">
                        💡 <strong style="color: #d1d5db;">نصيحة أمنية:</strong><br>
                        اختر كلمة مرور قوية تحتوي على أحرف كبيرة وصغيرة وأرقام ورموز.<br>
                        لا تستخدم نفس كلمة المرور في مواقع أخرى.
                    </p>
                </td>
            </tr>
        </table>
        
        <!-- Warning -->
        <table role="presentation" width="100%" cellspacing="0" cellpadding="0" border="0" style="background: rgba(245, 158, 11, 0.1); border: 1px solid #f59e0b; border-radius: 12px; margin-top: 20px;">
            <tr>
                <td style="padding: 20px;">
                    <p style="color: #f59e0b; font-size: 14px; margin: 0; text-align: center;">
                        ⚠️ إذا لم تطلب إعادة تعيين كلمة المرور، يرجى تجاهل هذا الإيميل.<br>
                        حسابك آمن ولم يتم إجراء أي تغييرات.
                    </p>
                </td>
            </tr>
        </table>
        """
        
        html = self._get_professional_template(content)
        return await self.send_email(to_email, "🔑 ASINAX - إعادة تعيين كلمة المرور", html)
    
    async def send_password_changed(self, to_email: str) -> bool:
        """Send notification when password is changed"""
        current_time = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        
        content = f"""
        <h2 style="color: #22c55e; font-size: 24px; margin: 0 0 20px 0; text-align: center;">
            ✅ تم تغيير كلمة المرور بنجاح
        </h2>
        
        <p style="color: #d1d5db; font-size: 16px; line-height: 1.8; text-align: center; margin: 0 0 30px 0;">
            تم تغيير كلمة مرور حسابك في ASINAX بنجاح.
        </p>
        
        <!-- Success Badge -->
        <table role="presentation" width="100%" cellspacing="0" cellpadding="0" border="0">
            <tr>
                <td align="center">
                    <table role="presentation" cellspacing="0" cellpadding="0" border="0" style="background: rgba(34, 197, 94, 0.1); border: 2px solid #22c55e; border-radius: 16px; padding: 20px 40px;">
                        <tr>
                            <td align="center">
                                <p style="color: #22c55e; font-size: 18px; margin: 0;">🔒 كلمة المرور الجديدة فعّالة الآن</p>
                            </td>
                        </tr>
                    </table>
                </td>
            </tr>
        </table>
        
        <!-- Details -->
        <table role="presentation" width="100%" cellspacing="0" cellpadding="0" border="0" style="background: rgba(59, 130, 246, 0.05); border-radius: 12px; margin-top: 25px;">
            <tr>
                <td style="padding: 20px;">
                    <p style="color: #9ca3af; font-size: 14px; margin: 0; text-align: center;">
                        📅 وقت التغيير: <span style="color: #ffffff; direction: ltr;">{current_time}</span>
                    </p>
                </td>
            </tr>
        </table>
        
        <!-- Warning -->
        <table role="presentation" width="100%" cellspacing="0" cellpadding="0" border="0" style="background: rgba(239, 68, 68, 0.1); border: 1px solid #ef4444; border-radius: 12px; margin-top: 25px;">
            <tr>
                <td style="padding: 20px;">
                    <p style="color: #ef4444; font-size: 14px; margin: 0 0 15px 0; text-align: center;">
                        ⚠️ <strong>لم تقم بهذا التغيير؟</strong>
                    </p>
                    <p style="color: #fca5a5; font-size: 13px; margin: 0; text-align: center;">
                        إذا لم تكن أنت من قام بتغيير كلمة المرور، يرجى التواصل مع فريق الدعم فوراً.
                    </p>
                </td>
            </tr>
        </table>
        
        <!-- Contact Support -->
        <table role="presentation" width="100%" cellspacing="0" cellpadding="0" border="0" style="margin-top: 25px;">
            <tr>
                <td align="center">
                    <a href="mailto:support@asinax.cloud" style="display: inline-block; background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%); color: #ffffff; padding: 15px 40px; text-decoration: none; border-radius: 10px; font-weight: bold; font-size: 14px;">
                        📧 التواصل مع الدعم
                    </a>
                </td>
            </tr>
        </table>
        """
        
        html = self._get_professional_template(content)
        return await self.send_email(to_email, "✅ ASINAX - تم تغيير كلمة المرور", html)
    
    # ============================================================
    # الإيداع
    # ============================================================
    
    async def send_deposit_pending(self, to_email: str, amount: float, address: str) -> bool:
        """Send notification when deposit is pending"""
        content = f"""
        <h2 style="color: #f59e0b; font-size: 24px; margin: 0 0 20px 0; text-align: center;">
            ⏳ في انتظار تأكيد الإيداع
        </h2>
        
        <p style="color: #d1d5db; font-size: 16px; line-height: 1.8; text-align: center; margin: 0 0 30px 0;">
            تم استلام طلب إيداع جديد. نحن بانتظار تأكيد المعاملة على شبكة البلوكتشين.
        </p>
        
        <!-- Amount Box -->
        <table role="presentation" width="100%" cellspacing="0" cellpadding="0" border="0">
            <tr>
                <td align="center">
                    <table role="presentation" cellspacing="0" cellpadding="0" border="0" style="background: linear-gradient(135deg, rgba(245, 158, 11, 0.1) 0%, rgba(234, 88, 12, 0.1) 100%); border: 2px solid #f59e0b; border-radius: 16px; padding: 25px 50px;">
                        <tr>
                            <td align="center">
                                <p style="color: #9ca3af; font-size: 14px; margin: 0 0 10px 0;">المبلغ المتوقع</p>
                                <p style="font-size: 36px; font-weight: bold; color: #f59e0b; margin: 0;">{amount} USDT</p>
                            </td>
                        </tr>
                    </table>
                </td>
            </tr>
        </table>
        
        <!-- Address -->
        <table role="presentation" width="100%" cellspacing="0" cellpadding="0" border="0" style="background: rgba(59, 130, 246, 0.05); border-radius: 12px; margin-top: 25px;">
            <tr>
                <td style="padding: 20px;">
                    <p style="color: #9ca3af; font-size: 14px; margin: 0 0 10px 0; text-align: center;">📍 عنوان الإيداع:</p>
                    <p style="color: #3b82f6; font-size: 12px; margin: 0; text-align: center; direction: ltr; word-break: break-all;">{address}</p>
                </td>
            </tr>
        </table>
        
        <!-- Info -->
        <table role="presentation" width="100%" cellspacing="0" cellpadding="0" border="0" style="margin-top: 25px;">
            <tr>
                <td style="padding: 20px; background: rgba(59, 130, 246, 0.05); border-radius: 12px;">
                    <p style="color: #9ca3af; font-size: 13px; margin: 0; text-align: center; line-height: 1.8;">
                        💡 عادةً ما يستغرق تأكيد المعاملة من 10 إلى 30 دقيقة حسب ازدحام الشبكة.<br>
                        سنرسل لك إشعاراً فور تأكيد الإيداع.
                    </p>
                </td>
            </tr>
        </table>
        """
        
        html = self._get_professional_template(content)
        return await self.send_email(to_email, "⏳ ASINAX - في انتظار تأكيد الإيداع", html)
    
    async def send_deposit_confirmed(self, to_email: str, amount: float, units: float) -> bool:
        """Send deposit confirmation notification"""
        content = f"""
        <h2 style="color: #22c55e; font-size: 24px; margin: 0 0 20px 0; text-align: center;">
            💰 تم تأكيد الإيداع بنجاح!
        </h2>
        
        <p style="color: #d1d5db; font-size: 16px; line-height: 1.8; text-align: center; margin: 0 0 30px 0;">
            تهانينا! تم إيداع الأموال في حسابك وبدأت العمل مع الوكيل الذكي.
        </p>
        
        <!-- Amount Box -->
        <table role="presentation" width="100%" cellspacing="0" cellpadding="0" border="0">
            <tr>
                <td align="center">
                    <table role="presentation" cellspacing="0" cellpadding="0" border="0" style="background: linear-gradient(135deg, rgba(34, 197, 94, 0.1) 0%, rgba(22, 163, 74, 0.1) 100%); border: 2px solid #22c55e; border-radius: 16px; padding: 25px 50px;">
                        <tr>
                            <td align="center">
                                <p style="color: #9ca3af; font-size: 14px; margin: 0 0 10px 0;">المبلغ المودع</p>
                                <p style="font-size: 42px; font-weight: bold; color: #22c55e; margin: 0; text-shadow: 0 0 20px rgba(34, 197, 94, 0.5);">{amount} USDT</p>
                            </td>
                        </tr>
                    </table>
                </td>
            </tr>
        </table>
        
        <!-- Units Info -->
        <table role="presentation" width="100%" cellspacing="0" cellpadding="0" border="0" style="background: rgba(59, 130, 246, 0.05); border-radius: 12px; margin-top: 25px;">
            <tr>
                <td style="padding: 25px;">
                    <table role="presentation" width="100%" cellspacing="0" cellpadding="0" border="0">
                        <tr>
                            <td style="padding: 10px 0; border-bottom: 1px solid rgba(255,255,255,0.1);">
                                <span style="color: #9ca3af; font-size: 14px;">📊 الوحدات الاستثمارية المكتسبة</span>
                                <span style="color: #3b82f6; font-size: 18px; font-weight: bold; float: left;">{units:.4f}</span>
                            </td>
                        </tr>
                        <tr>
                            <td style="padding: 10px 0;">
                                <span style="color: #9ca3af; font-size: 14px;">🤖 حالة الاستثمار</span>
                                <span style="color: #22c55e; font-size: 14px; float: left;">✅ نشط - يعمل مع الوكيل الذكي</span>
                            </td>
                        </tr>
                    </table>
                </td>
            </tr>
        </table>
        
        <!-- CTA -->
        <table role="presentation" width="100%" cellspacing="0" cellpadding="0" border="0" style="margin-top: 30px;">
            <tr>
                <td align="center">
                    <a href="https://asinax.cloud/dashboard" style="display: inline-block; background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%); color: #ffffff; padding: 18px 50px; text-decoration: none; border-radius: 12px; font-weight: bold; font-size: 16px; box-shadow: 0 10px 30px rgba(59, 130, 246, 0.3);">
                        📊 متابعة أداء استثمارك
                    </a>
                </td>
            </tr>
        </table>
        
        <!-- Note -->
        <p style="color: #6b7280; font-size: 12px; text-align: center; margin-top: 25px;">
            💡 يمكنك متابعة أداء استثمارك وأرباحك في الوقت الفعلي من لوحة التحكم.
        </p>
        """
        
        html = self._get_professional_template(content, "أموالك الآن تعمل مع الوكيل الذكي! تابع أداءك من لوحة التحكم.")
        return await self.send_email(to_email, "💰 ASINAX - تم تأكيد الإيداع بنجاح!", html)
    
    # ============================================================
    # السحب - متوافق مع الباك إند الحالي
    # ============================================================
    
    async def send_withdrawal_confirmation(
        self, 
        to_email: str, 
        user_name: str,
        amount: float, 
        email_token: str,
        withdrawal_id: int
    ) -> bool:
        """Send withdrawal confirmation email - Compatible with investor.py"""
        confirmation_link = f"https://asinax.cloud/api/v1/wallet/withdraw/confirm/{email_token}"
        
        content = f"""
        <h2 style="color: #f59e0b; font-size: 24px; margin: 0 0 20px 0; text-align: center;">
            💸 تأكيد طلب السحب
        </h2>
        
        <p style="color: #d1d5db; font-size: 16px; line-height: 1.8; text-align: center; margin: 0 0 10px 0;">
            مرحباً <strong style="color: #3b82f6;">{user_name}</strong>
        </p>
        
        <p style="color: #d1d5db; font-size: 16px; line-height: 1.8; text-align: center; margin: 0 0 30px 0;">
            تلقينا طلب سحب من حسابك. يرجى مراجعة التفاصيل وتأكيد الطلب.
        </p>
        
        <!-- Amount Box -->
        <table role="presentation" width="100%" cellspacing="0" cellpadding="0" border="0">
            <tr>
                <td align="center">
                    <table role="presentation" cellspacing="0" cellpadding="0" border="0" style="background: linear-gradient(135deg, rgba(245, 158, 11, 0.1) 0%, rgba(234, 88, 12, 0.1) 100%); border: 2px solid #f59e0b; border-radius: 16px; padding: 25px 50px;">
                        <tr>
                            <td align="center">
                                <p style="color: #9ca3af; font-size: 14px; margin: 0 0 10px 0;">صافي مبلغ السحب</p>
                                <p style="font-size: 36px; font-weight: bold; color: #f59e0b; margin: 0;">{amount:.2f} USDT</p>
                            </td>
                        </tr>
                    </table>
                </td>
            </tr>
        </table>
        
        <!-- Withdrawal Details -->
        <table role="presentation" width="100%" cellspacing="0" cellpadding="0" border="0" style="background: rgba(59, 130, 246, 0.05); border-radius: 12px; margin-top: 25px;">
            <tr>
                <td style="padding: 25px;">
                    <table role="presentation" width="100%" cellspacing="0" cellpadding="0" border="0">
                        <tr>
                            <td style="padding: 10px 0; border-bottom: 1px solid rgba(255,255,255,0.1);">
                                <span style="color: #9ca3af; font-size: 14px;">🔢 رقم الطلب</span>
                                <span style="color: #ffffff; font-size: 14px; float: left;">#{withdrawal_id}</span>
                            </td>
                        </tr>
                        <tr>
                            <td style="padding: 10px 0;">
                                <span style="color: #9ca3af; font-size: 14px;">📊 الحالة</span>
                                <span style="color: #f59e0b; font-size: 14px; float: left;">⏳ في انتظار التأكيد</span>
                            </td>
                        </tr>
                    </table>
                </td>
            </tr>
        </table>
        
        <!-- Confirm Button -->
        <table role="presentation" width="100%" cellspacing="0" cellpadding="0" border="0" style="margin-top: 30px;">
            <tr>
                <td align="center">
                    <a href="{confirmation_link}" style="display: inline-block; background: linear-gradient(135deg, #22c55e 0%, #16a34a 100%); color: #ffffff; padding: 18px 60px; text-decoration: none; border-radius: 12px; font-weight: bold; font-size: 18px; box-shadow: 0 10px 30px rgba(34, 197, 94, 0.3);">
                        ✅ تأكيد طلب السحب
                    </a>
                </td>
            </tr>
        </table>
        
        <!-- Timer -->
        <p style="color: #9ca3af; font-size: 14px; text-align: center; margin: 25px 0;">
            ⏱️ هذا الرابط صالح لمدة <strong style="color: #f59e0b;">24 ساعة</strong> فقط
        </p>
        
        <!-- Warning -->
        <table role="presentation" width="100%" cellspacing="0" cellpadding="0" border="0" style="background: rgba(239, 68, 68, 0.1); border: 1px solid #ef4444; border-radius: 12px; margin-top: 20px;">
            <tr>
                <td style="padding: 20px;">
                    <p style="color: #ef4444; font-size: 14px; margin: 0; text-align: center; line-height: 1.8;">
                        ⚠️ <strong>تحذير أمني:</strong><br>
                        إذا لم تقم بهذا الطلب، يرجى تجاهل هذا الإيميل وتغيير كلمة المرور فوراً.<br>
                        لا تشارك رابط التأكيد مع أي شخص.
                    </p>
                </td>
            </tr>
        </table>
        """
        
        html = self._get_professional_template(content)
        return await self.send_email(to_email, "💸 ASINAX - تأكيد طلب السحب", html)
    
    async def send_withdrawal_approved(self, to_email: str, amount: float) -> bool:
        """Send withdrawal approved notification"""
        content = f"""
        <h2 style="color: #22c55e; font-size: 24px; margin: 0 0 20px 0; text-align: center;">
            ✅ تمت الموافقة على طلب السحب
        </h2>
        
        <p style="color: #d1d5db; font-size: 16px; line-height: 1.8; text-align: center; margin: 0 0 30px 0;">
            تمت الموافقة على طلب السحب الخاص بك من قبل الإدارة.
        </p>
        
        <!-- Amount Box -->
        <table role="presentation" width="100%" cellspacing="0" cellpadding="0" border="0">
            <tr>
                <td align="center">
                    <table role="presentation" cellspacing="0" cellpadding="0" border="0" style="background: linear-gradient(135deg, rgba(34, 197, 94, 0.1) 0%, rgba(22, 163, 74, 0.1) 100%); border: 2px solid #22c55e; border-radius: 16px; padding: 25px 50px;">
                        <tr>
                            <td align="center">
                                <p style="color: #9ca3af; font-size: 14px; margin: 0 0 10px 0;">المبلغ الموافق عليه</p>
                                <p style="font-size: 36px; font-weight: bold; color: #22c55e; margin: 0;">{amount} USDT</p>
                            </td>
                        </tr>
                    </table>
                </td>
            </tr>
        </table>
        
        <!-- Status -->
        <table role="presentation" width="100%" cellspacing="0" cellpadding="0" border="0" style="background: rgba(59, 130, 246, 0.05); border-radius: 12px; margin-top: 25px;">
            <tr>
                <td style="padding: 20px;">
                    <p style="color: #3b82f6; font-size: 14px; margin: 0; text-align: center;">
                        📧 يرجى تأكيد الطلب عبر الإيميل الذي تم إرساله إليك سابقاً.
                    </p>
                </td>
            </tr>
        </table>
        
        <!-- Note -->
        <p style="color: #6b7280; font-size: 12px; text-align: center; margin-top: 25px;">
            💡 بعد التأكيد، سيتم معالجة السحب خلال 24-48 ساعة عمل.
        </p>
        """
        
        html = self._get_professional_template(content)
        return await self.send_email(to_email, "✅ ASINAX - تمت الموافقة على طلب السحب", html)
    
    async def send_withdrawal_rejected(
        self, 
        to_email: str, 
        amount: float,
        reason: str
    ) -> bool:
        """Send withdrawal rejected notification"""
        content = f"""
        <h2 style="color: #ef4444; font-size: 24px; margin: 0 0 20px 0; text-align: center;">
            ❌ تم رفض طلب السحب
        </h2>
        
        <p style="color: #d1d5db; font-size: 16px; line-height: 1.8; text-align: center; margin: 0 0 30px 0;">
            نأسف لإبلاغك بأن طلب السحب الخاص بك تم رفضه.
        </p>
        
        <!-- Amount Box -->
        <table role="presentation" width="100%" cellspacing="0" cellpadding="0" border="0">
            <tr>
                <td align="center">
                    <table role="presentation" cellspacing="0" cellpadding="0" border="0" style="background: linear-gradient(135deg, rgba(239, 68, 68, 0.1) 0%, rgba(220, 38, 38, 0.1) 100%); border: 2px solid #ef4444; border-radius: 16px; padding: 25px 50px;">
                        <tr>
                            <td align="center">
                                <p style="color: #9ca3af; font-size: 14px; margin: 0 0 10px 0;">المبلغ المرفوض</p>
                                <p style="font-size: 36px; font-weight: bold; color: #ef4444; margin: 0;">{amount} USDT</p>
                            </td>
                        </tr>
                    </table>
                </td>
            </tr>
        </table>
        
        <!-- Reason -->
        <table role="presentation" width="100%" cellspacing="0" cellpadding="0" border="0" style="background: rgba(239, 68, 68, 0.05); border-radius: 12px; margin-top: 25px; border: 1px solid rgba(239, 68, 68, 0.3);">
            <tr>
                <td style="padding: 25px;">
                    <p style="color: #9ca3af; font-size: 14px; margin: 0 0 10px 0;">📝 سبب الرفض:</p>
                    <p style="color: #fca5a5; font-size: 16px; margin: 0;">{reason}</p>
                </td>
            </tr>
        </table>
        
        <!-- Note -->
        <table role="presentation" width="100%" cellspacing="0" cellpadding="0" border="0" style="background: rgba(59, 130, 246, 0.05); border-radius: 12px; margin-top: 25px;">
            <tr>
                <td style="padding: 20px;">
                    <p style="color: #9ca3af; font-size: 13px; margin: 0; text-align: center; line-height: 1.8;">
                        💡 الأموال لا تزال في حسابك. يمكنك تقديم طلب سحب جديد بعد معالجة السبب أعلاه.<br>
                        إذا كان لديك أي استفسار، يرجى التواصل مع فريق الدعم.
                    </p>
                </td>
            </tr>
        </table>
        
        <!-- Support Button -->
        <table role="presentation" width="100%" cellspacing="0" cellpadding="0" border="0" style="margin-top: 25px;">
            <tr>
                <td align="center">
                    <a href="https://asinax.cloud/support" style="display: inline-block; background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%); color: #ffffff; padding: 15px 40px; text-decoration: none; border-radius: 10px; font-weight: bold; font-size: 14px;">
                        💬 التواصل مع الدعم
                    </a>
                </td>
            </tr>
        </table>
        """
        
        html = self._get_professional_template(content)
        return await self.send_email(to_email, "❌ ASINAX - تم رفض طلب السحب", html)
    
    async def send_withdrawal_completed(
        self, 
        to_email: str, 
        amount: float,
        to_address: str,
        tx_hash: str
    ) -> bool:
        """Send withdrawal completed notification - Compatible with admin.py"""
        content = f"""
        <h2 style="color: #22c55e; font-size: 24px; margin: 0 0 20px 0; text-align: center;">
            🎉 تم إتمام السحب بنجاح!
        </h2>
        
        <p style="color: #d1d5db; font-size: 16px; line-height: 1.8; text-align: center; margin: 0 0 30px 0;">
            تم تحويل الأموال إلى محفظتك بنجاح. يمكنك التحقق من المعاملة على شبكة البلوكتشين.
        </p>
        
        <!-- Amount Box -->
        <table role="presentation" width="100%" cellspacing="0" cellpadding="0" border="0">
            <tr>
                <td align="center">
                    <table role="presentation" cellspacing="0" cellpadding="0" border="0" style="background: linear-gradient(135deg, rgba(34, 197, 94, 0.1) 0%, rgba(22, 163, 74, 0.1) 100%); border: 2px solid #22c55e; border-radius: 16px; padding: 25px 50px;">
                        <tr>
                            <td align="center">
                                <p style="color: #9ca3af; font-size: 14px; margin: 0 0 10px 0;">المبلغ المحوّل</p>
                                <p style="font-size: 42px; font-weight: bold; color: #22c55e; margin: 0; text-shadow: 0 0 20px rgba(34, 197, 94, 0.5);">{amount} USDT</p>
                            </td>
                        </tr>
                    </table>
                </td>
            </tr>
        </table>
        
        <!-- Transaction Details -->
        <table role="presentation" width="100%" cellspacing="0" cellpadding="0" border="0" style="background: rgba(59, 130, 246, 0.05); border-radius: 12px; margin-top: 25px;">
            <tr>
                <td style="padding: 25px;">
                    <p style="color: #9ca3af; font-size: 14px; margin: 0 0 15px 0;">📍 عنوان المحفظة:</p>
                    <p style="color: #ffffff; font-size: 12px; margin: 0 0 20px 0; direction: ltr; word-break: break-all; background: rgba(0,0,0,0.3); padding: 15px; border-radius: 8px; font-family: monospace;">{to_address}</p>
                    
                    <p style="color: #9ca3af; font-size: 14px; margin: 0 0 15px 0;">🔗 معرف المعاملة (Transaction Hash):</p>
                    <p style="color: #3b82f6; font-size: 12px; margin: 0; direction: ltr; word-break: break-all; background: rgba(0,0,0,0.3); padding: 15px; border-radius: 8px; font-family: monospace;">{tx_hash}</p>
                </td>
            </tr>
        </table>
        
        <!-- Success Note -->
        <table role="presentation" width="100%" cellspacing="0" cellpadding="0" border="0" style="background: rgba(34, 197, 94, 0.1); border: 1px solid #22c55e; border-radius: 12px; margin-top: 25px;">
            <tr>
                <td style="padding: 20px;">
                    <p style="color: #22c55e; font-size: 14px; margin: 0; text-align: center;">
                        ✅ تم إتمام العملية بنجاح! يمكنك التحقق من المعاملة على مستكشف البلوكتشين.
                    </p>
                </td>
            </tr>
        </table>
        
        <!-- Thank You -->
        <p style="color: #6b7280; font-size: 14px; text-align: center; margin-top: 25px;">
            شكراً لاستخدامك ASINAX! نتطلع لخدمتك مرة أخرى. 🚀
        </p>
        """
        
        html = self._get_professional_template(content, "تم إتمام عملية السحب بنجاح. شكراً لثقتك في ASINAX!")
        return await self.send_email(to_email, "🎉 ASINAX - تم إتمام السحب بنجاح!", html)


# Singleton instance

    async def send_login_otp(self, to_email: str, otp: str, device: str, ip_address: str) -> bool:
        """Send login OTP email"""
        content = f"""
        <table role="presentation" width="100%" cellspacing="0" cellpadding="0" border="0">
            <tr>
                <td align="center">
                    <div style="width: 80px; height: 80px; background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%); border-radius: 20px; display: flex; align-items: center; justify-content: center; margin-bottom: 20px;">
                        <span style="font-size: 40px;">🔐</span>
                    </div>
                </td>
            </tr>
        </table>
        <h1 style="color: #ffffff; font-size: 28px; font-weight: bold; text-align: center; margin: 0 0 10px 0;">
            رمز تسجيل الدخول
        </h1>
        <p style="color: #9ca3af; font-size: 16px; text-align: center; margin: 0 0 30px 0;">
            استخدم هذا الرمز لإتمام تسجيل الدخول إلى حسابك
        </p>
        <table role="presentation" width="100%" cellspacing="0" cellpadding="0" border="0">
            <tr>
                <td align="center">
                    <table role="presentation" cellspacing="0" cellpadding="0" border="0" style="background: linear-gradient(135deg, rgba(59, 130, 246, 0.2) 0%, rgba(29, 78, 216, 0.2) 100%); border: 2px solid #3b82f6; border-radius: 16px; padding: 25px 50px;">
                        <tr>
                            <td align="center">
                                <p style="font-size: 48px; font-weight: bold; color: #3b82f6; margin: 0; letter-spacing: 15px;">{otp}</p>
                            </td>
                        </tr>
                    </table>
                </td>
            </tr>
        </table>
        <table role="presentation" width="100%" cellspacing="0" cellpadding="0" border="0" style="background: rgba(59, 130, 246, 0.05); border-radius: 12px; margin-top: 25px;">
            <tr>
                <td style="padding: 20px;">
                    <p style="color: #9ca3af; font-size: 14px; margin: 0 0 10px 0;">
                        <strong>📱 الجهاز:</strong> <span style="color: #ffffff;">{device}</span>
                    </p>
                    <p style="color: #9ca3af; font-size: 14px; margin: 0;">
                        <strong>🌐 عنوان IP:</strong> <span style="color: #ffffff;">{ip_address}</span>
                    </p>
                </td>
            </tr>
        </table>
        <table role="presentation" width="100%" cellspacing="0" cellpadding="0" border="0" style="background: rgba(239, 68, 68, 0.1); border: 1px solid #ef4444; border-radius: 12px; margin-top: 25px;">
            <tr>
                <td style="padding: 20px;">
                    <p style="color: #ef4444; font-size: 14px; margin: 0; text-align: center;">
                        ⚠️ إذا لم تكن أنت من يحاول تسجيل الدخول، يرجى تجاهل هذه الرسالة وتغيير كلمة المرور فوراً.
                    </p>
                </td>
            </tr>
        </table>
        <p style="color: #6b7280; font-size: 14px; text-align: center; margin-top: 25px;">
            ⏰ هذا الرمز صالح لمدة <strong style="color: #3b82f6;">10 دقائق</strong> فقط
        </p>
        """
        
        html = self._get_professional_template(content, "رمز تسجيل الدخول الخاص بك. لا تشاركه مع أي شخص")
        return await self.send_email(to_email, "🔐 ASINAX - رمز تسجيل الدخول", html)


# Singleton instance
email_service = EmailService()
