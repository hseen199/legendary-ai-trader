from binance.client import Client
from binance.exceptions import BinanceAPIException
from typing import Optional, Dict, List
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)


class BinanceService:
    """Service for interacting with Binance API"""
    
    def __init__(self):
        self.client = Client(
            settings.BINANCE_API_KEY,
            settings.BINANCE_API_SECRET,
            testnet=settings.BINANCE_TESTNET
        )
    
    # ============ Sub-Account Management ============
    
    async def create_sub_account(self, sub_account_string: str) -> Optional[str]:
        """
        Create a virtual sub-account for a new user
        Returns the sub-account email
        """
        try:
            result = self.client.create_sub_account(
                subAccountString=sub_account_string
            )
            return result.get("email")
        except BinanceAPIException as e:
            logger.error(f"Failed to create sub-account: {e}")
            return None
    
    async def get_sub_account_list(self) -> List[Dict]:
        """Get list of all sub-accounts"""
        try:
            result = self.client.get_sub_account_list()
            return result.get("subAccounts", [])
        except BinanceAPIException as e:
            logger.error(f"Failed to get sub-account list: {e}")
            return []
    
    # ============ Deposit Management ============
    
    async def get_deposit_address(
        self, 
        email: str, 
        coin: str = "USDT", 
        network: str = "TRC20"
    ) -> Optional[Dict]:
        """
        Get deposit address for a sub-account
        Returns: {"address": "...", "coin": "...", "network": "..."}
        """
        try:
            result = self.client.get_subaccount_deposit_address(
                email=email,
                coin=coin,
                network=network
            )
            return {
                "address": result.get("address"),
                "coin": result.get("coin"),
                "network": network
            }
        except BinanceAPIException as e:
            logger.error(f"Failed to get deposit address: {e}")
            return None
    
    async def get_deposit_history(
        self, 
        email: str, 
        coin: Optional[str] = None,
        status: Optional[int] = None,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None
    ) -> List[Dict]:
        """
        Get deposit history for a sub-account
        Status: 0-pending, 1-success, 6-credited but cannot withdraw
        """
        try:
            params = {"email": email}
            if coin:
                params["coin"] = coin
            if status is not None:
                params["status"] = status
            if start_time:
                params["startTime"] = start_time
            if end_time:
                params["endTime"] = end_time
            
            result = self.client.get_subaccount_deposit_history(**params)
            return result if isinstance(result, list) else []
        except BinanceAPIException as e:
            logger.error(f"Failed to get deposit history: {e}")
            return []
    
    # ============ Transfer Management ============
    
    async def transfer_to_master(
        self, 
        from_email: str, 
        asset: str, 
        amount: float
    ) -> Optional[int]:
        """
        Transfer assets from sub-account to master account
        Returns transaction ID
        """
        try:
            result = self.client.make_universal_transfer(
                fromEmail=from_email,
                fromAccountType="SPOT",
                toAccountType="SPOT",
                asset=asset,
                amount=amount
            )
            return result.get("tranId")
        except BinanceAPIException as e:
            logger.error(f"Failed to transfer to master: {e}")
            return None
    
    async def transfer_to_sub_account(
        self, 
        to_email: str, 
        asset: str, 
        amount: float
    ) -> Optional[int]:
        """
        Transfer assets from master account to sub-account
        Returns transaction ID
        """
        try:
            result = self.client.make_universal_transfer(
                toEmail=to_email,
                fromAccountType="SPOT",
                toAccountType="SPOT",
                asset=asset,
                amount=amount
            )
            return result.get("tranId")
        except BinanceAPIException as e:
            logger.error(f"Failed to transfer to sub-account: {e}")
            return None
    
    # ============ Withdrawal Management ============
    
    async def withdraw_from_sub_account(
        self, 
        email: str, 
        coin: str, 
        amount: float, 
        address: str, 
        network: str
    ) -> Optional[str]:
        """
        Withdraw from sub-account to external address
        Returns withdrawal ID
        """
        try:
            # First transfer to master, then withdraw
            # Note: Direct sub-account withdrawal may require additional setup
            result = self.client.withdraw(
                coin=coin,
                amount=amount,
                address=address,
                network=network
            )
            return result.get("id")
        except BinanceAPIException as e:
            logger.error(f"Failed to withdraw: {e}")
            return None
    
    # ============ Balance Management ============
    
    async def get_sub_account_balance(self, email: str) -> List[Dict]:
        """Get balance of a sub-account"""
        try:
            result = self.client.get_subaccount_assets(email=email)
            return result.get("balances", [])
        except BinanceAPIException as e:
            logger.error(f"Failed to get sub-account balance: {e}")
            return []
    
    async def get_master_account_balance(self) -> List[Dict]:
        """Get balance of master account"""
        try:
            result = self.client.get_account()
            return result.get("balances", [])
        except BinanceAPIException as e:
            logger.error(f"Failed to get master balance: {e}")
            return []
    
    async def get_total_assets_usd(self) -> float:
        """
        Get total assets in master account in USD
        This is used for NAV calculation
        """
        try:
            balances = await self.get_master_account_balance()
            total = 0.0
            
            for balance in balances:
                asset = balance.get("asset")
                free = float(balance.get("free", 0))
                locked = float(balance.get("locked", 0))
                total_amount = free + locked
                
                if total_amount > 0:
                    if asset == "USDT":
                        total += total_amount
                    else:
                        # Get price in USDT
                        try:
                            ticker = self.client.get_symbol_ticker(symbol=f"{asset}USDT")
                            price = float(ticker.get("price", 0))
                            total += total_amount * price
                        except:
                            pass
            
            return total
        except Exception as e:
            logger.error(f"Failed to calculate total assets: {e}")
            return 0.0


# Singleton instance
binance_service = BinanceService()
