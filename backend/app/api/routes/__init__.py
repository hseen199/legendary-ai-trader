from app.api.routes.auth import router as auth_router
from app.api.routes.wallet import router as wallet_router
from app.api.routes.admin import router as admin_router
from app.api.routes.dashboard import router as dashboard_router

__all__ = [
    "auth_router",
    "wallet_router",
    "admin_router",
    "dashboard_router"
]
