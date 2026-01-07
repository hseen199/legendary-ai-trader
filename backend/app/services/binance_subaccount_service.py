"""
Binance Sub-Account Service
خدمة إدارة الحسابات الفرعية في Binance لكل مستثمر
مدمج من نسخة المستخدم (crowdfund) مع تحسينات
"""

import asyncio
import hashlib
import hmac
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import aiohttp
import logging
import os

logger = logging.getLogger(__name__)


@dataclass
class SubAccount:
    """نموذج بيانات الحساب الفرعي"""
    email: str
    sub_account_id: str
    user_id: str
    status: str
    created_at: datetime
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            'email': self.email,
            'sub_account_id': self.sub_account_id,
            'user_id': self.user_id,
            'status': self.status,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


@dataclass
class DepositAddress:
    """نموذج عنوان الإيداع"""
    coin: str
    address: str
    network: str
    tag: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class TransferResult:
    """نموذج نتيجة التحويل"""
    txn_id: str
    from_email: str
    to_email: str
    asset: str
    amount: float
    status: str
    timestamp: datetime
    
    def to_dict(self) -> Dict:
        return {
            'txn_id': self.txn_id,
            'from_email': self.from_email,
            'to_email': self.to_email,
            'asset': self.asset,
            'amount': self.amount,
            'status': self.status,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None
        }


class BinanceSubAccountService:
    """
    خدمة إدارة الحسابات الفرعية في Binance
    تتيح إنشاء حساب فرعي لكل مستثمر وإدارة أمواله
    
    الميزات:
    - إنشاء حسابات فرعية تلقائياً لكل مستثمر جديد
    - جلب عناوين إيداع فريدة لكل مستثمر
    - تحويل الأموال بين الحسابات الفرعية والحساب الرئيسي
    - مراقبة الإيداعات والأرصدة
    """
    
    BASE_URL = "https://api.binance.com"
    DEFAULT_COIN = "USDC"  # العملة الافتراضية للتداول
    DEFAULT_NETWORK = "TRX"  # شبكة TRC20 (أرخص رسوم)
    
    def __init__(
        self, 
        api_key: Optional[str] = None, 
        api_secret: Optional[str] = None
    ):
        self.api_key = api_key or os.getenv('BINANCE_API_KEY', '')
        self.api_secret = api_secret or os.getenv('BINANCE_API_SECRET', '')
        self.sub_accounts: Dict[str, SubAccount] = {}
        self._session: Optional[aiohttp.ClientSession] = None
        self._initialized = False
        
    async def initialize(self):
        """تهيئة الخدمة"""
        if not self._initialized:
            self._session = aiohttp.ClientSession()
            self._initialized = True
            logger.info("BinanceSubAccountService initialized")
        
    async def _get_session(self) -> aiohttp.ClientSession:
        """الحصول على جلسة HTTP"""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session
        
    async def close(self):
        """إغلاق الخدمة"""
        if self._session and not self._session.closed:
            await self._session.close()
            self._initialized = False
            logger.info("BinanceSubAccountService closed")
            
    def _generate_signature(self, params: Dict[str, Any]) -> str:
        """توليد التوقيع للطلبات الموقعة"""
        query_string = '&'.join([f"{k}={v}" for k, v in sorted(params.items())])
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return signature
        
    def _get_timestamp(self) -> int:
        """الحصول على الطابع الزمني"""
        return int(time.time() * 1000)
        
    async def _request(
        self, 
        method: str, 
        endpoint: str, 
        params: Optional[Dict] = None,
        signed: bool = True
    ) -> Dict:
        """إرسال طلب إلى Binance API"""
        session = await self._get_session()
        
        if params is None:
            params = {}
            
        if signed:
            params['timestamp'] = self._get_timestamp()
            params['signature'] = self._generate_signature(params)
            
        headers = {
            'X-MBX-APIKEY': self.api_key,
            'Content-Type': 'application/json'
        }
        
        url = f"{self.BASE_URL}{endpoint}"
        
        try:
            if method == 'GET':
                async with session.get(url, params=params, headers=headers) as resp:
                    data = await resp.json()
                    if resp.status != 200:
                        logger.error(f"Binance API error: {data}")
                    return data
            elif method == 'POST':
                async with session.post(url, params=params, headers=headers) as resp:
                    data = await resp.json()
                    if resp.status != 200:
                        logger.error(f"Binance API error: {data}")
                    return data
            elif method == 'DELETE':
                async with session.delete(url, params=params, headers=headers) as resp:
                    data = await resp.json()
                    if resp.status != 200:
                        logger.error(f"Binance API error: {data}")
                    return data
        except Exception as e:
            logger.error(f"Binance API request failed: {e}")
            raise
            
    # ==================== إدارة الحسابات الفرعية ====================
    
    async def create_sub_account(self, user_id: str) -> SubAccount:
        """
        إنشاء حساب فرعي جديد لمستثمر
        
        Args:
            user_id: معرف المستثمر في نظامنا
            
        Returns:
            SubAccount: بيانات الحساب الفرعي
        """
        sub_account_string = f"investor_{user_id}_{int(time.time())}"
        
        params = {
            'subAccountString': sub_account_string
        }
        
        result = await self._request('POST', '/sapi/v1/sub-account/virtualSubAccount', params)
        
        if 'email' in result:
            sub_account = SubAccount(
                email=result['email'],
                sub_account_id=result.get('subAccountId', ''),
                user_id=user_id,
                status='active',
                created_at=datetime.utcnow()
            )
            self.sub_accounts[user_id] = sub_account
            logger.info(f"Created sub-account for user {user_id}: {result['email']}")
            return sub_account
        else:
            error_msg = result.get('msg', 'Unknown error')
            logger.error(f"Failed to create sub-account: {error_msg}")
            raise Exception(f"Failed to create sub-account: {error_msg}")
            
    async def get_sub_account_list(self) -> List[Dict]:
        """جلب قائمة الحسابات الفرعية"""
        params = {'limit': 200}
        result = await self._request('GET', '/sapi/v1/sub-account/list', params)
        return result.get('subAccounts', [])
        
    async def get_sub_account_by_email(self, email: str) -> Optional[Dict]:
        """جلب حساب فرعي بالإيميل"""
        accounts = await self.get_sub_account_list()
        for account in accounts:
            if account.get('email') == email:
                return account
        return None
        
    # ==================== عناوين الإيداع ====================
    
    async def get_deposit_address(
        self, 
        sub_account_email: str, 
        coin: str = None,
        network: str = None
    ) -> DepositAddress:
        """
        جلب عنوان إيداع للحساب الفرعي
        
        Args:
            sub_account_email: إيميل الحساب الفرعي
            coin: العملة (USDC افتراضياً)
            network: الشبكة (TRX أرخص رسوم)
            
        Returns:
            DepositAddress: عنوان الإيداع
        """
        coin = coin or self.DEFAULT_COIN
        network = network or self.DEFAULT_NETWORK
        
        params = {
            'email': sub_account_email,
            'coin': coin,
            'network': network
        }
        
        result = await self._request('GET', '/sapi/v1/capital/deposit/subAddress', params)
        
        return DepositAddress(
            coin=coin,
            address=result.get('address', ''),
            network=network,
            tag=result.get('tag')
        )
        
    # ==================== الأرصدة والأصول ====================
    
    async def get_sub_account_assets(self, email: str) -> Dict[str, float]:
        """جلب أصول الحساب الفرعي"""
        params = {'email': email}
        result = await self._request('GET', '/sapi/v1/sub-account/assets', params)
        
        balances = {}
        for asset in result.get('balances', []):
            balance = float(asset.get('free', 0)) + float(asset.get('locked', 0))
            if balance > 0:
                balances[asset['asset']] = balance
                
        return balances
        
    async def get_master_account_balance(self) -> Dict[str, float]:
        """جلب رصيد الحساب الرئيسي"""
        result = await self._request('GET', '/sapi/v3/account')
        
        balances = {}
        for asset in result.get('balances', []):
            balance = float(asset.get('free', 0)) + float(asset.get('locked', 0))
            if balance > 0:
                balances[asset['asset']] = balance
                
        return balances
        
    async def get_total_portfolio_value(self, quote_asset: str = None) -> float:
        """
        حساب القيمة الإجمالية للمحفظة الرئيسية بعملة التقييم
        
        Args:
            quote_asset: عملة التقييم (USDC افتراضياً)
            
        Returns:
            float: القيمة الإجمالية
        """
        quote_asset = quote_asset or self.DEFAULT_COIN
        balances = await self.get_master_account_balance()
        
        total_value = 0.0
        
        for asset, amount in balances.items():
            if asset == quote_asset:
                total_value += amount
            else:
                try:
                    # محاولة الحصول على السعر
                    symbol = f"{asset}{quote_asset}"
                    ticker = await self._request(
                        'GET', 
                        '/api/v3/ticker/price', 
                        {'symbol': symbol}, 
                        signed=False
                    )
                    price = float(ticker.get('price', 0))
                    total_value += amount * price
                except:
                    # تجاهل الأصول التي لا يمكن تقييمها
                    pass
                    
        return total_value
        
    # ==================== التحويلات ====================
    
    async def transfer_to_master(
        self, 
        from_email: str, 
        asset: str, 
        amount: float
    ) -> TransferResult:
        """
        تحويل الأموال من حساب فرعي إلى الحساب الرئيسي (للتداول المشترك)
        
        Args:
            from_email: إيميل الحساب الفرعي
            asset: العملة
            amount: المبلغ
            
        Returns:
            TransferResult: نتيجة التحويل
        """
        params = {
            'fromEmail': from_email,
            'fromAccountType': 'SPOT',
            'toAccountType': 'SPOT',
            'asset': asset,
            'amount': str(amount)
        }
        
        result = await self._request('POST', '/sapi/v1/sub-account/universalTransfer', params)
        
        return TransferResult(
            txn_id=str(result.get('tranId', '')),
            from_email=from_email,
            to_email='master',
            asset=asset,
            amount=amount,
            status='success',
            timestamp=datetime.utcnow()
        )
        
    async def transfer_from_master(
        self, 
        to_email: str, 
        asset: str, 
        amount: float
    ) -> TransferResult:
        """
        تحويل الأموال من الحساب الرئيسي إلى حساب فرعي (للسحب)
        
        Args:
            to_email: إيميل الحساب الفرعي
            asset: العملة
            amount: المبلغ
            
        Returns:
            TransferResult: نتيجة التحويل
        """
        params = {
            'toEmail': to_email,
            'fromAccountType': 'SPOT',
            'toAccountType': 'SPOT',
            'asset': asset,
            'amount': str(amount)
        }
        
        result = await self._request('POST', '/sapi/v1/sub-account/universalTransfer', params)
        
        return TransferResult(
            txn_id=str(result.get('tranId', '')),
            from_email='master',
            to_email=to_email,
            asset=asset,
            amount=amount,
            status='success',
            timestamp=datetime.utcnow()
        )
        
    # ==================== سجل الإيداعات ====================
    
    async def get_deposit_history(
        self, 
        sub_account_email: str,
        coin: str = None,
        limit: int = 100
    ) -> List[Dict]:
        """جلب سجل الإيداعات للحساب الفرعي"""
        coin = coin or self.DEFAULT_COIN
        params = {
            'email': sub_account_email,
            'coin': coin,
            'limit': limit
        }
        
        result = await self._request('GET', '/sapi/v1/capital/deposit/subHisrec', params)
        return result if isinstance(result, list) else []
        
    # ==================== السحب ====================
    
    async def withdraw_from_sub_account(
        self,
        from_email: str,
        to_address: str,
        coin: str,
        amount: float,
        network: str = None
    ) -> Dict:
        """
        سحب الأموال من حساب فرعي إلى عنوان خارجي
        
        الخطوات:
        1. تحويل من الحساب الفرعي إلى الحساب الرئيسي
        2. سحب من الحساب الرئيسي إلى العنوان الخارجي
        """
        network = network or self.DEFAULT_NETWORK
        
        # الخطوة 1: تحويل إلى الحساب الرئيسي
        await self.transfer_to_master(from_email, coin, amount)
        
        # الخطوة 2: سحب إلى العنوان الخارجي
        params = {
            'coin': coin,
            'address': to_address,
            'amount': str(amount),
            'network': network
        }
        
        result = await self._request('POST', '/sapi/v1/capital/withdraw/apply', params)
        
        logger.info(f"Withdrawal initiated: {amount} {coin} to {to_address}")
        return result


# Singleton instance
_binance_service: Optional[BinanceSubAccountService] = None


def get_binance_service() -> BinanceSubAccountService:
    """الحصول على نسخة واحدة من الخدمة (Singleton)"""
    global _binance_service
    if _binance_service is None:
        _binance_service = BinanceSubAccountService()
    return _binance_service
