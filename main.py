"""
FastAPI Main Application for LawChain Backend
"""

import time
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.middleware.trustedhost import TrustedHostMiddleware

from config.settings import settings
from app.core.api import router as api_router
from app.utils.helpers import setup_logging, ensure_directories
from app.services.lawchain_service import lawchain_service

# Setup logging
logger = setup_logging()

# Application start time
app_start_time = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    # Startup
    logger.info("🚀 Starting LawChain Backend API...")
    
    # Ensure directories exist
    ensure_directories()
    
    # Initialize LawChain services in background
    try:
        logger.info("🔄 Initializing LawChain services...")
        # You can choose to initialize both or just one by default
        # lawchain_service.initialize_both()
        logger.info("✅ LawChain services ready for on-demand initialization")
    except Exception as e:
        logger.warning(f"⚠️ Service initialization failed: {str(e)}")
        logger.info("🔄 Services will be initialized on first request")
    
    logger.info(f"🎉 LawChain Backend API started successfully!")
    logger.info(f"📊 Server running on {settings.HOST}:{settings.PORT}")
    logger.info(f"📖 API Documentation: http://{settings.HOST}:{settings.PORT}/docs")
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down LawChain Backend API...")


# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    description=settings.APP_DESCRIPTION,
    version=settings.APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url=f"{settings.API_V1_PREFIX}/openapi.json",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=settings.CORS_CREDENTIALS,
    allow_methods=settings.CORS_METHODS,
    allow_headers=settings.CORS_HEADERS,
)

# Add trusted host middleware for security
app.add_middleware(
    TrustedHostMiddleware, 
    allowed_hosts=["localhost", "127.0.0.1", settings.HOST]
)


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Global exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error",
            "message": str(exc) if settings.DEBUG else "An error occurred"
        }
    )


# Include API routes
app.include_router(
    api_router,
    prefix=settings.API_V1_PREFIX,
    tags=["LawChain API"]
)


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "🏛️ LawChain Backend API",
        "version": settings.APP_VERSION,
        "description": settings.APP_DESCRIPTION,
        "docs": f"{settings.API_V1_PREFIX}/docs",
        "health": f"{settings.API_V1_PREFIX}/health",
        "status": "running",
        "uptime_seconds": time.time() - app_start_time
    }


# Additional middleware for request timing
@app.middleware("http")
async def add_process_time_header(request, call_next):
    """Add processing time to response headers"""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(round(process_time, 4))
    return response


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=False,  # Disable auto-reload to prevent continuous restarts
        log_level=settings.LOG_LEVEL.lower(),
        timeout_keep_alive=300,  # 5 minutes for keep-alive
        timeout_graceful_shutdown=300,  # 5 minutes for graceful shutdown
        limit_max_requests=1000,  # Maximum number of requests
        limit_concurrency=100  # Maximum concurrent connections
    )
