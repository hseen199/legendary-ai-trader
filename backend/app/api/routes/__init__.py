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
from app.api.routes.deposits import router as deposits_router
from app.api.routes.notifications import router as notifications_router
from app.api.routes.two_factor import router as two_factor_router

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
    "security_router",
    "deposits_router",
    "notifications_router",
    "two_factor_router"
]

from app.api.routes.agent import router as agent_router
from app.api.routes.webhook import router as webhook_router
from app.api.routes.agent_webhook import router as agent_webhook_router
from app.api.routes.vip import router as vip_router
from app.api.routes.reports import router as reports_router
from app.api.routes.communication import router as communication_router
from app.api.routes.admin_agent import router as admin_agent_router
from app.api.routes.user import router as user_router
from app.api.routes.admin_users import router as admin_users_router
from app.api.routes.platform_settings import router as platform_settings_router
