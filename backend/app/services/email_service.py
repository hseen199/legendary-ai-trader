import aiosmtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)


class EmailService:
    """Service for sending emails"""
    
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
            message["From"] = settings.EMAIL_FROM
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
    
    async def send_withdrawal_confirmation(
        self, 
        to_email: str, 
        amount: float, 
        address: str,
        confirmation_link: str
    ) -> bool:
        """Send withdrawal confirmation email"""
        subject = "تأكيد طلب السحب - Withdrawal Confirmation"
        
        html_content = f"""
        <html>
        <body dir="rtl" style="font-family: Arial, sans-serif;">
            <h2>تأكيد طلب السحب</h2>
            <p>لقد طلبت سحب <strong>{amount} USDT</strong> إلى العنوان:</p>
            <p style="background: #f5f5f5; padding: 10px; direction: ltr;">{address}</p>
            <p>لتأكيد هذا الطلب، يرجى الضغط على الرابط التالي:</p>
            <a href="{confirmation_link}" style="background: #4CAF50; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px;">
                تأكيد السحب
            </a>
            <p style="color: #888; margin-top: 20px;">
                هذا الرابط صالح لمدة 24 ساعة فقط.
                <br>
                إذا لم تقم بهذا الطلب، يرجى تجاهل هذا الإيميل وتغيير كلمة المرور.
            </p>
        </body>
        </html>
        """
        
        return await self.send_email(to_email, subject, html_content)
    
    async def send_withdrawal_approved(
        self, 
        to_email: str, 
        amount: float
    ) -> bool:
        """Send withdrawal approved notification"""
        subject = "تمت الموافقة على طلب السحب - Withdrawal Approved"
        
        html_content = f"""
        <html>
        <body dir="rtl" style="font-family: Arial, sans-serif;">
            <h2>تمت الموافقة على طلب السحب</h2>
            <p>تمت الموافقة على طلب سحب <strong>{amount} USDT</strong>.</p>
            <p>يرجى تأكيد الطلب عبر الإيميل الذي تم إرساله إليك.</p>
        </body>
        </html>
        """
        
        return await self.send_email(to_email, subject, html_content)
    
    async def send_withdrawal_rejected(
        self, 
        to_email: str, 
        amount: float,
        reason: str
    ) -> bool:
        """Send withdrawal rejected notification"""
        subject = "تم رفض طلب السحب - Withdrawal Rejected"
        
        html_content = f"""
        <html>
        <body dir="rtl" style="font-family: Arial, sans-serif;">
            <h2>تم رفض طلب السحب</h2>
            <p>نأسف لإبلاغك بأن طلب سحب <strong>{amount} USDT</strong> تم رفضه.</p>
            <p><strong>السبب:</strong> {reason}</p>
            <p>إذا كان لديك أي استفسار، يرجى التواصل مع الدعم.</p>
        </body>
        </html>
        """
        
        return await self.send_email(to_email, subject, html_content)
    
    async def send_withdrawal_completed(
        self, 
        to_email: str, 
        amount: float,
        tx_hash: str
    ) -> bool:
        """Send withdrawal completed notification"""
        subject = "تم إتمام السحب بنجاح - Withdrawal Completed"
        
        html_content = f"""
        <html>
        <body dir="rtl" style="font-family: Arial, sans-serif;">
            <h2>تم إتمام السحب بنجاح</h2>
            <p>تم سحب <strong>{amount} USDT</strong> بنجاح.</p>
            <p><strong>معرف العملية:</strong></p>
            <p style="background: #f5f5f5; padding: 10px; direction: ltr;">{tx_hash}</p>
        </body>
        </html>
        """
        
        return await self.send_email(to_email, subject, html_content)
    
    async def send_deposit_confirmed(
        self, 
        to_email: str, 
        amount: float,
        units: float
    ) -> bool:
        """Send deposit confirmation notification"""
        subject = "تم تأكيد الإيداع - Deposit Confirmed"
        
        html_content = f"""
        <html>
        <body dir="rtl" style="font-family: Arial, sans-serif;">
            <h2>تم تأكيد الإيداع</h2>
            <p>تم إيداع <strong>{amount} USDT</strong> بنجاح في حسابك.</p>
            <p>حصلت على <strong>{units:.4f}</strong> وحدة استثمارية.</p>
        </body>
        </html>
        """
        
        return await self.send_email(to_email, subject, html_content)
    
    async def send_password_changed(self, to_email: str) -> bool:
        """Send password changed notification"""
        subject = "تم تغيير كلمة المرور - Password Changed"
        
        html_content = """
        <html>
        <body dir="rtl" style="font-family: Arial, sans-serif;">
            <h2>تم تغيير كلمة المرور</h2>
            <p>تم تغيير كلمة مرور حسابك بنجاح.</p>
            <p style="color: #888;">
                إذا لم تقم بهذا التغيير، يرجى التواصل مع الدعم فوراً.
            </p>
        </body>
        </html>
        """
        
        return await self.send_email(to_email, subject, html_content)


# Singleton instance
email_service = EmailService()
