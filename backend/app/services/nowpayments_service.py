"""
NOWPayments Service - خدمة بوابة الدفع
للإيداع التلقائي عبر العملات الرقمية
"""

import httpx
import hmac
import hashlib
import json
from typing import Optional, Dict, Any, List
from datetime import datetime
from app.core.config import settings


class NOWPaymentsService:
    """خدمة NOWPayments للإيداع"""
    
    def __init__(self):
        self.api_key = settings.NOWPAYMENTS_API_KEY
        self.api_url = settings.NOWPAYMENTS_API_URL
        self.ipn_secret = settings.NOWPAYMENTS_IPN_SECRET
        self.headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json"
        }
    
    async def get_status(self) -> Dict[str, Any]:
        """التحقق من حالة API"""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.api_url}/status",
                headers=self.headers
            )
            return response.json()
    
    async def get_available_currencies(self) -> List[str]:
        """الحصول على العملات المتاحة"""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.api_url}/currencies",
                headers=self.headers
            )
            data = response.json()
            return data.get("currencies", [])
    
    async def get_minimum_amount(self, currency_from: str, currency_to: str = "usdcbsc") -> float:
        """الحصول على الحد الأدنى للإيداع"""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.api_url}/min-amount",
                params={
                    "currency_from": currency_from,
                    "currency_to": currency_to
                },
                headers=self.headers
            )
            data = response.json()
            return data.get("min_amount", 0)
    
    async def get_estimate(
        self, 
        amount: float, 
        currency_from: str, 
        currency_to: str = "usdcbsc"
    ) -> Dict[str, Any]:
        """تقدير المبلغ المستلم"""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.api_url}/estimate",
                params={
                    "amount": amount,
                    "currency_from": currency_from,
                    "currency_to": currency_to
                },
                headers=self.headers
            )
            return response.json()
    
    async def create_payment(
        self,
        price_amount: float,
        price_currency: str = "usd",
        pay_currency: str = "usdcbsc",
        order_id: str = None,
        order_description: str = None,
        ipn_callback_url: str = None,
        success_url: str = None,
        cancel_url: str = None,
    ) -> Dict[str, Any]:
        """
        إنشاء فاتورة دفع جديدة
        
        Args:
            price_amount: المبلغ بالدولار
            price_currency: عملة السعر (usd)
            pay_currency: عملة الدفع (usdcbsc, usdcsol, etc.)
            order_id: معرف الطلب (user_id + timestamp)
            order_description: وصف الطلب
            ipn_callback_url: رابط الـ webhook للإشعارات
            success_url: رابط النجاح
            cancel_url: رابط الإلغاء
        
        Returns:
            بيانات الفاتورة مع عنوان الدفع
        """
        payload = {
            "price_amount": price_amount,
            "price_currency": price_currency,
            "pay_currency": pay_currency,
            "order_id": order_id,
            "order_description": order_description or f"Deposit {price_amount} USD",
        }
        
        if ipn_callback_url:
            payload["ipn_callback_url"] = ipn_callback_url
        if success_url:
            payload["success_url"] = success_url
        if cancel_url:
            payload["cancel_url"] = cancel_url
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.api_url}/payment",
                json=payload,
                headers=self.headers
            )
            return response.json()
    
    async def create_invoice(
        self,
        price_amount: float,
        price_currency: str = "usd",
        order_id: str = None,
        order_description: str = None,
        ipn_callback_url: str = None,
        success_url: str = None,
        cancel_url: str = None,
    ) -> Dict[str, Any]:
        """
        إنشاء فاتورة (يختار المستخدم العملة)
        """
        payload = {
            "price_amount": price_amount,
            "price_currency": price_currency,
            "order_id": order_id,
            "order_description": order_description or f"Deposit {price_amount} USD",
        }
        
        if ipn_callback_url:
            payload["ipn_callback_url"] = ipn_callback_url
        if success_url:
            payload["success_url"] = success_url
        if cancel_url:
            payload["cancel_url"] = cancel_url
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.api_url}/invoice",
                json=payload,
                headers=self.headers
            )
            return response.json()
    
    async def get_payment_status(self, payment_id: int) -> Dict[str, Any]:
        """الحصول على حالة الدفع"""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.api_url}/payment/{payment_id}",
                headers=self.headers
            )
            return response.json()
    
    async def get_payments_list(
        self, 
        limit: int = 50, 
        page: int = 0,
        sort_by: str = "created_at",
        order_by: str = "desc",
        date_from: str = None,
        date_to: str = None,
    ) -> Dict[str, Any]:
        """الحصول على قائمة المدفوعات"""
        params = {
            "limit": limit,
            "page": page,
            "sortBy": sort_by,
            "orderBy": order_by,
        }
        if date_from:
            params["dateFrom"] = date_from
        if date_to:
            params["dateTo"] = date_to
        
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.api_url}/payment/",
                params=params,
                headers=self.headers
            )
            return response.json()
    
    def verify_ipn_signature(self, payload: Dict, signature: str) -> bool:
        """
        التحقق من توقيع IPN (Webhook)
        
        Args:
            payload: البيانات المستلمة
            signature: التوقيع من الـ header
        
        Returns:
            True إذا كان التوقيع صحيح
        """
        if not self.ipn_secret:
            return True  # Skip verification if no secret set
        
        # Sort payload and create string
        sorted_payload = json.dumps(payload, sort_keys=True, separators=(',', ':'))
        
        # Calculate HMAC
        calculated_signature = hmac.new(
            self.ipn_secret.encode(),
            sorted_payload.encode(),
            hashlib.sha512
        ).hexdigest()
        
        return hmac.compare_digest(calculated_signature, signature)
    
    def parse_ipn_status(self, status: str) -> str:
        """
        تحويل حالة IPN إلى حالة النظام
        
        NOWPayments statuses:
        - waiting: في انتظار الدفع
        - confirming: في انتظار التأكيدات
        - confirmed: تم التأكيد
        - sending: جاري الإرسال للمحفظة
        - partially_paid: دفع جزئي
        - finished: مكتمل
        - failed: فشل
        - refunded: تم الاسترداد
        - expired: منتهي الصلاحية
        """
        status_map = {
            "waiting": "pending",
            "confirming": "confirming",
            "confirmed": "confirmed",
            "sending": "processing",
            "partially_paid": "partial",
            "finished": "completed",
            "failed": "failed",
            "refunded": "refunded",
            "expired": "expired",
        }
        return status_map.get(status, "unknown")


# إنشاء instance واحد للاستخدام
nowpayments_service = NOWPaymentsService()
