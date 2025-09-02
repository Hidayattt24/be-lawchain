"""
Core utilities and helper functions
"""

import logging
import os
import time
import requests
from datetime import datetime
from typing import Dict, List, Any, Optional
from config.settings import settings


def setup_logging():
    """Setup logging configuration"""
    os.makedirs(settings.LOGS_DIR, exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(settings.LOG_FILE),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def validate_ollama_connection() -> Dict[str, Any]:
    """Validate Ollama service connection and available models"""
    try:
        # Check if Ollama is running
        response = requests.get(f"{settings.OLLAMA_BASE_URL}/api/tags", timeout=5)
        response.raise_for_status()
        
        models_data = response.json()
        available_models = [model['name'] for model in models_data.get('models', [])]
        
        # Check required models
        required_models = [settings.OLLAMA_LLM_MODEL, settings.OLLAMA_EMBEDDING_MODEL]
        missing_models = []
        
        for required_model in required_models:
            if not any(required_model in model for model in available_models):
                missing_models.append(required_model)
        
        return {
            'status': 'healthy' if not missing_models else 'partial',
            'available_models': available_models,
            'missing_models': missing_models,
            'connection': True
        }
        
    except requests.exceptions.ConnectionError:
        return {
            'status': 'unhealthy',
            'available_models': [],
            'missing_models': [],
            'connection': False,
            'error': 'Ollama service not reachable'
        }
    except Exception as e:
        return {
            'status': 'error',
            'available_models': [],
            'missing_models': [],
            'connection': False,
            'error': str(e)
        }


def check_vector_store_exists(method: str = "optimized") -> bool:
    """Check if optimized vector store exists"""
    if method == "optimized":
        return os.path.exists(os.path.join(settings.VECTOR_STORE_OPTIMIZED_PATH, "index.faiss"))
    # Legacy support for backward compatibility
    elif method == "langchain":
        return os.path.exists(os.path.join(settings.VECTOR_STORE_OPTIMIZED_PATH, "index.faiss"))
    elif method == "native":
        return os.path.exists(os.path.join(settings.VECTOR_STORE_OPTIMIZED_PATH, "index.faiss"))
    return False


def get_data_files() -> List[str]:
    """Get list of PDF files in data directory"""
    if not os.path.exists(settings.DATA_DIR):
        return []
    
    return [f for f in os.listdir(settings.DATA_DIR) if f.endswith('.pdf')]


def calculate_uptime(start_time: float) -> float:
    """Calculate uptime in seconds"""
    return time.time() - start_time


def format_timestamp() -> str:
    """Get current timestamp in ISO format"""
    return datetime.now().isoformat()


def format_processing_time(start_time: float) -> float:
    """Calculate processing time from start time"""
    return round(time.time() - start_time, 3)


def format_time_duration(duration: float) -> str:
    """Format duration in seconds to readable string"""
    return f"{duration:.3f}s"


def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe file operations"""
    import re
    return re.sub(r'[<>:"/\\|?*]', '_', filename)


def ensure_directories():
    """Ensure all required directories exist"""
    directories = [
        settings.DATA_DIR,
        settings.LOGS_DIR,
        settings.STORAGE_DIR,
        settings.VECTOR_STORE_OPTIMIZED_PATH
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
