from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.core.config import settings
from app.core.database import init_db
from app.api.routes import (
    auth_router, 
    wallet_router, 
    admin_router, 
    dashboard_router,
    deposits_router,
    analytics_router,
    marketing_router,
    support_router,
    security_router
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    # Startup
    await init_db()
    print("Database initialized")
    yield
    # Shutdown
    print("Application shutting down")


# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    description="Legendary AI Trader - منصة التداول الذكي بالذكاء الاصطناعي",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth_router, prefix=settings.API_V1_PREFIX)
app.include_router(wallet_router, prefix=settings.API_V1_PREFIX)
app.include_router(dashboard_router, prefix=settings.API_V1_PREFIX)
app.include_router(admin_router, prefix=settings.API_V1_PREFIX)
app.include_router(deposits_router, prefix=settings.API_V1_PREFIX, tags=["deposits"])
app.include_router(analytics_router, prefix=settings.API_V1_PREFIX, tags=["analytics"])
app.include_router(marketing_router, prefix=settings.API_V1_PREFIX, tags=["marketing"])
app.include_router(support_router, prefix=settings.API_V1_PREFIX, tags=["support"])
app.include_router(security_router, prefix=settings.API_V1_PREFIX, tags=["security"])


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": settings.APP_NAME,
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG
    )
