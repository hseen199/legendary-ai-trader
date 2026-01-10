from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    # App Settings
    APP_NAME: str = "Legendary AI Trader"
    DEBUG: bool = True
    API_V1_PREFIX: str = "/api/v1"
    
    # Database
    DATABASE_URL: str = "postgresql+asyncpg://user:password@localhost:5432/crypto_platform"
    
    # JWT Settings
    SECRET_KEY: str = "your-super-secret-key-change-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24  # 24 hours
    
    # Binance API
    BINANCE_API_KEY: str = ""
    BINANCE_API_SECRET: str = ""
    BINANCE_TESTNET: bool = True  # Use testnet for development
    
    # NOWPayments API
    NOWPAYMENTS_API_KEY: str = ""
    NOWPAYMENTS_PUBLIC_KEY: str = ""
    NOWPAYMENTS_IPN_SECRET: str = ""
    NOWPAYMENTS_API_URL: str = "https://api.nowpayments.io/v1"
    
    # Email Settings
    SMTP_HOST: str = "smtp.gmail.com"
    SMTP_PORT: int = 587
    SMTP_USER: str = ""
    SMTP_PASSWORD: str = ""
    EMAIL_FROM: str = "noreply@legendaryai.com"
    
    # Platform Settings
    LOCK_PERIOD_DAYS: int = 7  # فترة القفل
    PERFORMANCE_FEE_PERCENT: float = 20.0  # رسوم الأداء
    RESERVE_PERCENT: float = 20.0  # نسبة الاحتياطي
    DRAWDOWN_LIMIT_PERCENT: float = 20.0  # حد وقف الخسارة
    INITIAL_NAV: float = 1.0  # قيمة الوحدة الأولية
    MIN_DEPOSIT: float = 100.0  # الحد الأدنى للإيداع
    MIN_WITHDRAWAL: float = 50.0  # الحد الأدنى للسحب
    
    # Admin
    ADMIN_EMAIL: str = "admin@legendaryai.com"
    
    # Frontend URL (for callbacks)
    FRONTEND_URL: str = "http://localhost:5173"
    BACKEND_URL: str = "http://localhost:8000"
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
