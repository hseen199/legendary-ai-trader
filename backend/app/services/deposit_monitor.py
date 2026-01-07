"""
Deposit Monitor Service
This service runs in the background to monitor and process new deposits
"""
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.models import User, Balance, Transaction
from app.services.binance_service import binance_service
from app.services.nav_service import nav_service
from app.services.email_service import email_service
from app.core.database import AsyncSessionLocal
import logging
import asyncio

logger = logging.getLogger(__name__)


class DepositMonitor:
    """Monitor and process deposits from Binance sub-accounts"""
    
    def __init__(self):
        self.processed_tx_hashes = set()
    
    async def check_deposits(self):
        """Check for new deposits across all sub-accounts"""
        async with AsyncSessionLocal() as db:
            try:
                # Get all users with sub-accounts
                result = await db.execute(
                    select(User).where(User.sub_account_email.isnot(None))
                )
                users = result.scalars().all()
                
                for user in users:
                    await self._check_user_deposits(db, user)
                
            except Exception as e:
                logger.error(f"Error checking deposits: {e}")
    
    async def _check_user_deposits(self, db: AsyncSession, user: User):
        """Check deposits for a specific user"""
        try:
            # Get deposit history from Binance
            deposits = await binance_service.get_deposit_history(
                email=user.sub_account_email,
                status=1  # Only successful deposits
            )
            
            for deposit in deposits:
                tx_hash = deposit.get("txId")
                
                # Skip if already processed
                if tx_hash in self.processed_tx_hashes:
                    continue
                
                # Check if already in database
                result = await db.execute(
                    select(Transaction).where(Transaction.tx_hash == tx_hash)
                )
                if result.scalar_one_or_none():
                    self.processed_tx_hashes.add(tx_hash)
                    continue
                
                # Process new deposit
                await self._process_deposit(db, user, deposit)
                self.processed_tx_hashes.add(tx_hash)
                
        except Exception as e:
            logger.error(f"Error checking deposits for user {user.id}: {e}")
    
    async def _process_deposit(self, db: AsyncSession, user: User, deposit: dict):
        """Process a new deposit"""
        try:
            amount = float(deposit.get("amount", 0))
            coin = deposit.get("coin", "USDT")
            network = deposit.get("network", "TRC20")
            tx_hash = deposit.get("txId")
            
            # Calculate units
            units, nav = await nav_service.calculate_units_for_deposit(db, amount)
            
            # Create transaction record
            transaction = Transaction(
                user_id=user.id,
                type="deposit",
                amount_usd=amount,
                units_transacted=units,
                nav_at_transaction=nav,
                coin=coin,
                network=network,
                tx_hash=tx_hash,
                status="completed",
                completed_at=datetime.utcnow()
            )
            db.add(transaction)
            
            # Update user balance
            result = await db.execute(
                select(Balance).where(Balance.user_id == user.id)
            )
            balance = result.scalar_one_or_none()
            
            if balance:
                balance.units += units
                balance.last_deposit_at = datetime.utcnow()
            else:
                balance = Balance(
                    user_id=user.id,
                    units=units,
                    last_deposit_at=datetime.utcnow()
                )
                db.add(balance)
            
            # Transfer to master account
            await binance_service.transfer_to_master(
                from_email=user.sub_account_email,
                asset=coin,
                amount=amount
            )
            
            await db.commit()
            
            # Send notification
            await email_service.send_deposit_confirmed(
                user.email,
                amount,
                units
            )
            
            logger.info(f"Processed deposit: {amount} {coin} for user {user.id}")
            
        except Exception as e:
            logger.error(f"Error processing deposit: {e}")
            await db.rollback()


# Singleton instance
deposit_monitor = DepositMonitor()


async def run_deposit_monitor(interval_seconds: int = 60):
    """Run deposit monitor in a loop"""
    logger.info("Starting deposit monitor...")
    while True:
        await deposit_monitor.check_deposits()
        await asyncio.sleep(interval_seconds)
