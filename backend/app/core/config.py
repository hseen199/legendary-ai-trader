"""
Asinax - Application Configuration
"""
from pydantic_settings import BaseSettings
from typing import Optional, List
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # ===== App Settings =====
    APP_NAME: str = "Asinax"
    DEBUG: bool = False
    ENVIRONMENT: str = "production"
    LOG_LEVEL: str = "INFO"
    API_V1_PREFIX: str = "/api/v1"
    
    # ===== Database =====
    DATABASE_URL: str = "postgresql+asyncpg://asinax:password@localhost:5432/asinax"
    DATABASE_ECHO: bool = False
    
    # ===== Redis =====
    REDIS_URL: str = "redis://localhost:6379/0"
    REDIS_CACHE_TTL: int = 3600
    
    # ===== JWT Settings =====
    SECRET_KEY: str = "your-super-secret-key-change-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60  # 1 hour
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    
    # ===== Binance API =====
    BINANCE_API_KEY: str = ""
    BINANCE_API_SECRET: str = ""
    BINANCE_MASTER_ACCOUNT: str = ""
    BINANCE_TESTNET: bool = False
    
    # ===== NOWPayments API =====
    NOWPAYMENTS_API_KEY: str = ""
    NOWPAYMENTS_PUBLIC_KEY: str = ""
    NOWPAYMENTS_IPN_SECRET: str = ""
    NOWPAYMENTS_API_URL: str = "https://api.nowpayments.io/v1"
    
    # ===== Wallet Addresses =====
    SOL_WALLET_ADDRESS: str = "GikxUq45vmnwCXVQVaGcU4EZrPmejZv5ikcDkuTANRFw"
    BNB_WALLET_ADDRESS: str = "0x35d755b7eae343c6a903ba178ceabfcf29aa4409"
    
    # ===== Email Settings =====
    SMTP_HOST: str = "smtp.gmail.com"
    SMTP_PORT: int = 587
    SMTP_USER: str = ""
    SMTP_PASSWORD: str = ""
    EMAIL_FROM: str = "noreply@asinax.com"
    EMAIL_FROM_NAME: str = "Asinax"
    
    # ===== Platform Settings =====
    LOCK_PERIOD_DAYS: int = 30  # فترة القفل بعد الإيداع
    PERFORMANCE_FEE_PERCENT: float = 20.0  # رسوم الأداء
    MANAGEMENT_FEE_PERCENT: float = 0.0  # رسوم الإدارة
    RESERVE_PERCENT: float = 20.0  # نسبة الاحتياطي
    DRAWDOWN_LIMIT_PERCENT: float = 20.0  # حد وقف الخسارة
    INITIAL_NAV: float = 1.0  # قيمة الوحدة الأولية
    
    # ===== Deposit/Withdrawal Limits =====
    MIN_DEPOSIT: float = 100.0  # الحد الأدنى للإيداع (دولار)
    MAX_DEPOSIT: float = 1000000.0  # الحد الأقصى للإيداع
    MIN_WITHDRAWAL: float = 10.0  # الحد الأدنى للسحب
    MAX_WITHDRAWAL: float = 100000.0  # الحد الأقصى للسحب
    
    # ===== VIP Levels =====
    VIP_BRONZE_MIN: float = 0.0
    VIP_SILVER_MIN: float = 1000.0
    VIP_GOLD_MIN: float = 10000.0
    VIP_PLATINUM_MIN: float = 50000.0
    
    # ===== Referral Settings =====
    REFERRAL_BONUS_PERCENT: float = 5.0  # نسبة مكافأة الإحالة
    
    # ===== Admin =====
    ADMIN_EMAIL: str = "admin@asinax.com"
    ADMIN_PASSWORD: str = "Admin@123456"
    ADMIN_NAME: str = "System Admin"
    
    # ===== URLs =====
    FRONTEND_URL: str = "https://asinax.com"
    BACKEND_URL: str = "https://asinax.com"
    
    # ===== CORS =====
    CORS_ORIGINS: str = "https://asinax.com,https://www.asinax.com"
    CORS_ALLOW_CREDENTIALS: bool = True
    
    # ===== OpenAI =====
    OPENAI_API_KEY: str = ""
    LLM_MODEL: str = "gpt-4.1-mini"
    LLM_TEMPERATURE: float = 0.7
    
    # ===== Monitoring =====
    SENTRY_DSN: str = ""
    MONITORING_ENABLED: bool = False
    
    @property
    def cors_origins_list(self) -> List[str]:
        """Get CORS origins as a list"""
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",")]
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()


# Global settings instance
settings = get_settings()
