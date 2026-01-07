from app.api.routes.auth import router as auth_router
from app.api.routes.wallet import router as wallet_router
from app.api.routes.admin import router as admin_router
from app.api.routes.dashboard import router as dashboard_router
from app.api.routes.investor import router as investor_router
from app.api.routes.bot import router as bot_router
from app.api.routes.analytics import router as analytics_router
from app.api.routes.marketing import router as marketing_router
from app.api.routes.support import router as support_router
from app.api.routes.security import router as security_router

__all__ = [
    "auth_router",
    "wallet_router",
    "admin_router",
    "dashboard_router",
    "investor_router",
    "bot_router",
    "analytics_router",
    "marketing_router",
    "support_router",
    "security_router"
]
