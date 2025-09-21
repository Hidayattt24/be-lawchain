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
from app.core.api_structured import router as api_router
from app.utils.helpers import setup_logging, ensure_directories

# Setup logging
logger = setup_logging()

# Application start time
app_start_time = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    # Startup
    logger.info("Starting LawChain Structured RAG Backend API...")
    logger.info(f"AI Model: {settings.OLLAMA_LLM_MODEL} (Google Gemma2:2b)")
    logger.info("🔥 SISTEM RAG TERSTRUKTUR - DUAL SERVICE ARCHITECTURE")
    logger.info("📊 LangChain & Native implementations available")
    logger.info("🎯 Vector Store: vector_store_structured dengan metadata precision")
    logger.info("Available Services: LangChain framework & Native implementation")
    logger.info("Model Optimization: Gemma2:2b (1.6GB) untuk performance optimal")
    
    # Ensure directories exist
    ensure_directories()

    # Initialize services info (lazy loading)
    try:
        logger.info("Structured RAG services available for on-demand initialization:")
        logger.info("- LangChain: LangChain framework dengan custom retriever")
        logger.info("- Native: Pure Python implementation tanpa LangChain")
        logger.info("Vector Store: Menggunakan vector_store_structured dengan hard filtering")
        logger.info("Services ready for lazy initialization pada first requeollst")
    except Exception as e:
        logger.warning(f"Service info logging failed: {str(e)}")
        logger.info("Services will be initialized on first request")
    
    logger.info("LawChain Structured RAG Backend API started successfully!")
    logger.info(f"Server running on {settings.HOST}:{settings.PORT}")
    logger.info(f"API Documentation: http://{settings.HOST}:{settings.PORT}/docs")
    
    yield    # Shutdown
    logger.info("Shutting down LawChain Structured RAG Backend API...")


# Create FastAPI app
app = FastAPI(
    title="🏛️ LawChain API - Enhanced RAG System",
    description="""
    Sistem chatbot hukum Indonesia berbasis RAG (Retrieval-Augmented Generation) 
    dengan arsitektur dual service untuk fleksibilitas dan performa optimal.
    
    **🎯 Fitur Utama:**
    - Context Detection: Filter otomatis pertanyaan di luar konteks hukum
    - Dual Service: LangChain framework + Native implementation  
    - Enhanced Accuracy: Document ranking dan confidence scoring
    - Fallback System: Gemma2:2b untuk pertanyaan hukum umum
    
    **📚 Database:** UUD 1945 dari sumber resmi (MKRI, MPR, BPHN)
    
    **🤖 AI Model:** Gemma2:2b (1.6GB) optimized untuk domain hukum Indonesia
    
    **📖 Documentation:** Lihat `/docs` untuk interactive API documentation
    """,
    version="2.0.0",
    contact={
        "name": "LawChain Development Team",
        "url": "https://github.com/lawchain/api",
        "email": "dev@lawchain.id"
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT"
    },
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
    """
    🏛️ LawChain API - Enhanced RAG System
    
    Sistem chatbot hukum Indonesia dengan teknologi RAG terdepan.
    """
    return {
        "message": "🏛️ LawChain Enhanced RAG Backend API",
        "version": "2.0.0",
        "description": "Dual Service Architecture - LangChain & Native RAG implementations",
        "features": {
            "context_detection": "✅ Automatic legal context filtering", 
            "dual_service": "✅ LangChain + Native implementations",
            "enhanced_accuracy": "✅ Document ranking & confidence scoring",
            "fallback_system": "✅ Gemma2:2b for general legal questions"
        },
        "system_info": {
            "ai_model": "Gemma2:2b (1.6GB optimized)",
            "vector_store": "FAISS with structured metadata",
            "database": "UUD 1945 from verified sources (MKRI, MPR, BPHN)",
            "embeddings": "nomic-embed-text"
        },
        "endpoints": {
            "ask_question": f"{settings.API_V1_PREFIX}/ask",
            "health_check": f"{settings.API_V1_PREFIX}/health",
            "service_status": f"{settings.API_V1_PREFIX}/services/status",
            "documentation": "/docs",
            "redoc": "/redoc"
        },
        "available_services": ["langchain", "native"],
        "status": "running",
        "uptime_seconds": time.time() - app_start_time,
        "architecture": "enhanced_dual_service_rag"
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
        timeout_keep_alive=600,  # Increased to 10 minutes for keep-alive
        timeout_graceful_shutdown=600,  # Increased to 10 minutes for graceful shutdown
        limit_max_requests=1000,  # Maximum number of requests
        limit_concurrency=50    # Reduced concurrency to improve per-request performance
    )
