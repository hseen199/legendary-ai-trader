from app.schemas.user import (
    UserRegister,
    UserLogin,
    Token,
    TokenData,
    UserBase,
    UserResponse,
    UserUpdate,
    PasswordChange,
    BalanceResponse,
    TrustedAddressCreate,
    TrustedAddressResponse,
    OTPSendRequest,
    OTPVerifyRequest
)
from app.schemas.transaction import (
    DepositAddressResponse,
    DepositHistoryItem,
    WithdrawalRequest,
    WithdrawalRequestResponse,
    WithdrawalConfirm,
    TransactionResponse,
    TradeResponse,
    NAVResponse,
    NAVHistoryItem,
    UserDashboard,
    AdminWithdrawalReview,
    AdminStats,
    AdminUserResponse
)

# Aliases for backward compatibility
UserCreate = UserRegister

__all__ = [
    # User schemas
    "UserRegister",
    "UserCreate",
    "UserLogin",
    "Token",
    "TokenData",
    "UserBase",
    "UserResponse",
    "UserUpdate",
    "PasswordChange",
    "BalanceResponse",
    "TrustedAddressCreate",
    "TrustedAddressResponse",
    "OTPSendRequest",
    "OTPVerifyRequest",
    # Transaction schemas
    "DepositAddressResponse",
    "DepositHistoryItem",
    "WithdrawalRequest",
    "WithdrawalRequestResponse",
    "WithdrawalConfirm",
    "TransactionResponse",
    "TradeResponse",
    "NAVResponse",
    "NAVHistoryItem",
    "UserDashboard",
    "AdminWithdrawalReview",
    "AdminStats",
    "AdminUserResponse"
]
