"""
Configuration settings for LawChain Backend API
"""

import os
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings"""
    
    # Application Info
    APP_NAME: str = "LawChain Backend API"
    APP_VERSION: str = "1.0.0"
    APP_DESCRIPTION: str = "Backend API untuk Chatbot Hukum UUD 1945 berbasis RAG"
    
    # Environment
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    
    # API Settings
    API_V1_PREFIX: str = "/api/v1"
    HOST: str = "127.0.0.1"
    PORT: int = 8000
    
    # CORS Settings
    CORS_ORIGINS: list = ["http://localhost:3000", "http://127.0.0.1:3000"]
    CORS_CREDENTIALS: bool = True
    CORS_METHODS: list = ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
    CORS_HEADERS: list = ["*"]
    
    # Ollama Settings
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_LLM_MODEL: str = "llama3.1:8b"
    OLLAMA_EMBEDDING_MODEL: str = "nomic-embed-text"
    OLLAMA_TEMPERATURE: float = 0.1
    OLLAMA_TIMEOUT: int = 600  # Increased to 10 minutes for more reliable processing
    
    # Document Processing
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    SIMILARITY_THRESHOLD: float = 0.7
    MAX_RETRIEVED_DOCS: int = 5
    
    # File Paths
    DATA_DIR: str = "data"
    VECTOR_STORE_LANGCHAIN_PATH: str = "storage/vector_store_faiss"
    VECTOR_STORE_NATIVE_PATH: str = "storage/vector_store_native"
    LOGS_DIR: str = "logs"
    STORAGE_DIR: str = "storage"
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "logs/lawchain.log"
    LOG_MAX_SIZE: int = 10485760  # 10MB
    LOG_BACKUP_COUNT: int = 5
    
    # Rate Limiting
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_WINDOW: int = 3600  # 1 hour
    
    # Security
    SECRET_KEY: Optional[str] = None
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # Database (jika diperlukan nanti)
    DATABASE_URL: Optional[str] = None
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Create settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get application settings"""
    return settings
