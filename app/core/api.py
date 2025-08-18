"""
API Core functionality
"""

import time
from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import Dict, Any

from app.models.schemas import (
    QuestionRequest, QuestionResponse, ErrorResponse, 
    HealthResponse, SystemInfoResponse, RebuildRequest, RebuildResponse
)
from app.services.lawchain_service import lawchain_service
from app.utils.helpers import (
    validate_ollama_connection, check_vector_store_exists,
    get_data_files, format_timestamp, calculate_uptime
)
from config.settings import settings

import logging

logger = logging.getLogger(__name__)

# Router instance
router = APIRouter()

# Application start time for uptime calculation
app_start_time = time.time()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        ollama_status = validate_ollama_connection()
        
        services = {
            "ollama": ollama_status['connection'],
            "langchain_vectorstore": check_vector_store_exists("langchain"),
            "native_vectorstore": check_vector_store_exists("native"),
            "data_files": len(get_data_files()) > 0
        }
        
        # Determine overall status
        if all(services.values()):
            status = "healthy"
        elif any(services.values()):
            status = "partial"
        else:
            status = "unhealthy"
        
        return HealthResponse(
            status=status,
            version=settings.APP_VERSION,
            timestamp=format_timestamp(),
            services=services,
            uptime=calculate_uptime(app_start_time)
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@router.get("/system/info", response_model=SystemInfoResponse)
async def get_system_info():
    """Get system information"""
    try:
        ollama_status = validate_ollama_connection()
        system_info = lawchain_service.get_system_info()
        
        vector_stores = {
            "langchain": check_vector_store_exists("langchain"),
            "native": check_vector_store_exists("native")
        }
        
        return SystemInfoResponse(
            app_name=settings.APP_NAME,
            version=settings.APP_VERSION,
            environment=settings.ENVIRONMENT,
            ollama_status=ollama_status['connection'],
            available_models=ollama_status.get('available_models', []),
            vector_stores=vector_stores,
            total_documents=system_info.get('total_documents', 0),
            total_chunks=system_info.get('total_chunks', 0)
        )
        
    except Exception as e:
        logger.error(f"Failed to get system info: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get system info: {str(e)}")


@router.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """Ask a question to the LawChain system"""
    try:
        logger.info(f"Received question: {request.question[:50]}... (method: {request.method})")
        
        # Validate method
        if request.method not in ["langchain", "native"]:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid method. Use 'langchain' or 'native'"
            )
        
        # Check if the requested method is initialized
        if request.method == "langchain" and not lawchain_service.initialization_status['langchain']:
            # Try to initialize if not done yet
            logger.info("LangChain not initialized, attempting initialization...")
            lawchain_service.initialize_langchain()
        
        if request.method == "native" and not lawchain_service.initialization_status['native']:
            # Try to initialize if not done yet
            logger.info("Native not initialized, attempting initialization...")
            lawchain_service.initialize_native()
        
        # Process the question
        response = lawchain_service.ask_question(
            question=request.question,
            method=request.method,
            max_docs=request.max_docs or 5
        )
        
        if not response.get('success', True):
            raise HTTPException(
                status_code=500,
                detail=response.get('error', 'Unknown error occurred')
            )
        
        # Convert to response model
        return QuestionResponse(**response)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/system/rebuild", response_model=RebuildResponse)
async def rebuild_vector_store(request: RebuildRequest, background_tasks: BackgroundTasks):
    """Rebuild vector store"""
    try:
        logger.info(f"Rebuilding vector store: {request.method}")
        
        if request.method not in ["langchain", "native", "both"]:
            raise HTTPException(
                status_code=400,
                detail="Invalid method. Use 'langchain', 'native', or 'both'"
            )
        
        # If force rebuild or vector store doesn't exist, proceed
        should_rebuild = request.force
        
        if not should_rebuild:
            if request.method == "langchain":
                should_rebuild = not check_vector_store_exists("langchain")
            elif request.method == "native":
                should_rebuild = not check_vector_store_exists("native")
            elif request.method == "both":
                should_rebuild = not (check_vector_store_exists("langchain") and check_vector_store_exists("native"))
        
        if not should_rebuild:
            return RebuildResponse(
                success=True,
                message=f"Vector store for {request.method} already exists. Use force=true to rebuild.",
                method=request.method,
                processing_time=0.0,
                timestamp=format_timestamp()
            )
        
        # Rebuild vector store
        result = lawchain_service.rebuild_vector_store(request.method)
        
        if not result['success']:
            raise HTTPException(status_code=500, detail=result['message'])
        
        return RebuildResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error rebuilding vector store: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/system/status")
async def get_status():
    """Get detailed system status"""
    try:
        ollama_status = validate_ollama_connection()
        system_info = lawchain_service.get_system_info()
        
        return {
            "timestamp": format_timestamp(),
            "uptime_seconds": calculate_uptime(app_start_time),
            "ollama": ollama_status,
            "lawchain": {
                "langchain_initialized": system_info.get('langchain_initialized', False),
                "native_initialized": system_info.get('native_initialized', False),
                "total_documents": system_info.get('total_documents', 0),
                "total_chunks": system_info.get('total_chunks', 0)
            },
            "vector_stores": {
                "langchain": check_vector_store_exists("langchain"),
                "native": check_vector_store_exists("native")
            },
            "data": {
                "files_count": len(get_data_files()),
                "files": get_data_files()
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
