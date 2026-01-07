from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    # App Settings
    APP_NAME: str = "Crypto Investment Platform"
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
    
    # Email Settings
    SMTP_HOST: str = "smtp.gmail.com"
    SMTP_PORT: int = 587
    SMTP_USER: str = ""
    SMTP_PASSWORD: str = ""
    EMAIL_FROM: str = "noreply@cryptoplatform.com"
    
    # Platform Settings
    LOCK_PERIOD_DAYS: int = 7  # فترة القفل
    PERFORMANCE_FEE_PERCENT: float = 20.0  # رسوم الأداء
    RESERVE_PERCENT: float = 20.0  # نسبة الاحتياطي
    DRAWDOWN_LIMIT_PERCENT: float = 20.0  # حد وقف الخسارة
    INITIAL_NAV: float = 1.0  # قيمة الوحدة الأولية
    
    # Admin
    ADMIN_EMAIL: str = "admin@cryptoplatform.com"
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
