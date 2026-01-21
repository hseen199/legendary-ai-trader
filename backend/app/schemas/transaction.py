from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime

# ============ Deposit Schemas ============
class DepositAddressResponse(BaseModel):
    address: str
    network: str
    coin: str
    qr_code_url: Optional[str] = None

class DepositHistoryItem(BaseModel):
    id: int
    amount: float
    coin: str
    network: str
    status: str
    is_active: Optional[bool] = True
    tx_hash: Optional[str] = None
    units_received: Optional[float] = None
    nav_at_deposit: Optional[float] = None
    created_at: datetime
    completed_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True

# ============ Withdrawal Schemas ============
class WithdrawalRequest(BaseModel):
    amount: float
    to_address: str
    network: str
    coin: str = "USDT"

class WithdrawalRequestResponse(BaseModel):
    id: int
    amount: float
    units_to_withdraw: float
    to_address: str
    network: str
    coin: str
    status: str
    is_active: Optional[bool] = True
    requested_at: datetime
    reviewed_at: Optional[datetime] = None
    rejection_reason: Optional[str] = None
    completed_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True

class WithdrawalConfirm(BaseModel):
    token: str

# ============ Transaction Schemas ============
class TransactionResponse(BaseModel):
    id: int
    type: str
    amount_usd: float
    units_transacted: Optional[float] = None
    nav_at_transaction: Optional[float] = None
    coin: str
    status: str
    is_active: Optional[bool] = True  # جعله اختيارياً مع قيمة افتراضية
    tx_hash: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True

# ============ Trading History Schemas ============
class TradeResponse(BaseModel):
    id: int
    symbol: str
    side: str
    order_type: str
    price: float
    quantity: float
    total_value: float
    pnl: Optional[float] = None
    pnl_percent: Optional[float] = None
    executed_at: datetime
    
    class Config:
        from_attributes = True

# ============ NAV Schemas ============
class NAVResponse(BaseModel):
    current_nav: float
    total_assets_usd: float
    total_units: float
    change_24h: Optional[float] = None
    change_7d: Optional[float] = None
    change_30d: Optional[float] = None

class NAVHistoryItem(BaseModel):
    nav_value: float
    total_assets_usd: float
    timestamp: datetime
    
    class Config:
        from_attributes = True

# ============ Dashboard Schemas ============
class UserDashboard(BaseModel):
    # Balance Info
    balance: float
    units: float
    current_nav: float
    
    # Profit/Loss
    total_deposited: float
    current_value: float
    profit_loss: float
    profit_loss_percent: float
    
    # Status
    can_withdraw: bool
    lock_period_ends: Optional[datetime] = None
    
    # Recent Activity
    recent_transactions: List[TransactionResponse]
    pending_withdrawals: List[WithdrawalRequestResponse]

# ============ Admin Schemas ============
class AdminWithdrawalReview(BaseModel):
    action: str  # approve or reject
    reason: Optional[str] = None

class AdminStats(BaseModel):
    total_users: int
    active_users: int
    total_assets_usd: float
    total_units: float
    current_nav: float
    pending_withdrawals: int
    total_deposits_today: float
    total_withdrawals_today: float
    high_water_mark: float
    total_fees_collected: float
    emergency_mode: str

class AdminUserResponse(BaseModel):
    id: int
    email: str
    full_name: Optional[str] = None
    status: str
    is_active: bool
    is_admin: bool
    units: float
    current_value_usd: float
    total_deposited: float
    total_withdrawn: float
    created_at: datetime
    last_login: Optional[datetime] = None
    
    class Config:
        from_attributes = True
