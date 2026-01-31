"""
Asinax - Main Application Entry Point
"""
import logging
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import time
from app.core.config import settings
from app.core.database import init_db
from app.api.routes import (
    agent_webhook_router,
    agent_router,
    webhook_router,
    auth_router, 
    wallet_router, 
    admin_router, 
    dashboard_router,
    deposits_router,
    analytics_router,
    marketing_router,
    support_router,
    security_router,
    notifications_router,
    vip_router,
    reports_router,
    communication_router
)
# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    # Startup
    logger.info("🚀 Starting Asinax API...")
    await init_db()
    logger.info("✅ Database initialized")
    logger.info(f"📍 Environment: {settings.ENVIRONMENT}")
    logger.info(f"🔧 Debug mode: {settings.DEBUG}")
    yield
    # Shutdown
    logger.info("👋 Asinax API shutting down...")
# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    description="Asinax - منصة التداول الذكي بالذكاء الاصطناعي",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None,
)
# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list if settings.ENVIRONMENT == "production" else ["*"],
    allow_credentials=settings.CORS_ALLOW_CREDENTIALS,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Request timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response
# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )
# Include routers
app.include_router(auth_router, prefix=settings.API_V1_PREFIX, tags=["auth"])
app.include_router(wallet_router, prefix=settings.API_V1_PREFIX, tags=["wallet"])
app.include_router(dashboard_router, prefix=settings.API_V1_PREFIX, tags=["dashboard"])
app.include_router(admin_router, prefix=settings.API_V1_PREFIX, tags=["admin"])
app.include_router(deposits_router, prefix=settings.API_V1_PREFIX + "/deposits", tags=["deposits"])
app.include_router(analytics_router, prefix=settings.API_V1_PREFIX, tags=["analytics"])
app.include_router(marketing_router, prefix=settings.API_V1_PREFIX, tags=["marketing"])
app.include_router(support_router, prefix=settings.API_V1_PREFIX, tags=["support"])
app.include_router(security_router, prefix=settings.API_V1_PREFIX, tags=["security"])
app.include_router(agent_router, prefix=settings.API_V1_PREFIX, tags=["agent"])
app.include_router(webhook_router, prefix=settings.API_V1_PREFIX, tags=["webhook"])
app.include_router(notifications_router, prefix=settings.API_V1_PREFIX, tags=["notifications"])
app.include_router(agent_webhook_router, prefix=settings.API_V1_PREFIX, tags=["agent-webhook"])
app.include_router(vip_router, prefix=settings.API_V1_PREFIX, tags=["vip"])
app.include_router(reports_router, prefix=settings.API_V1_PREFIX, tags=["reports"])
app.include_router(communication_router, prefix=settings.API_V1_PREFIX, tags=["communication"])
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": settings.APP_NAME,
        "version": "1.0.0",
        "status": "running",
        "environment": settings.ENVIRONMENT,
        "docs": "/docs" if settings.DEBUG else "disabled"
    }
@app.get("/health")
async def health_check():
    """Health check endpoint for Docker/Kubernetes"""
    return {
        "status": "healthy",
        "app": settings.APP_NAME,
        "version": "1.0.0"
    }
@app.get("/api/v1/status")
async def api_status():
    """API status endpoint"""
    return {
        "status": "operational",
        "version": "1.0.0",
        "services": {
            "database": "connected",
            "nowpayments": "configured" if settings.NOWPAYMENTS_API_KEY else "not_configured",
            "binance": "configured" if settings.BINANCE_API_KEY else "not_configured",
            "email": "configured" if settings.SMTP_USER else "not_configured"
        }
    }
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )
