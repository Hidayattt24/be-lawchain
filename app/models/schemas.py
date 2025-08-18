"""
Pydantic models for API requests and responses
"""

from datetime import datetime
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class QuestionRequest(BaseModel):
    """Request model for asking questions"""
    question: str = Field(..., min_length=1, max_length=1000, description="Pertanyaan tentang UUD 1945")
    method: str = Field(default="langchain", description="Method to use: 'langchain' or 'native'")
    max_docs: Optional[int] = Field(default=5, ge=1, le=10, description="Maximum number of documents to retrieve")


class MetricsModel(BaseModel):
    """Model for response metrics"""
    semantic_similarity: float = Field(..., description="Semantic similarity score")
    content_coverage: float = Field(..., description="Content coverage score")
    answer_relevance: float = Field(..., description="Answer relevance score")
    source_quality: float = Field(..., description="Source quality score")
    legal_context: float = Field(..., description="Legal context score")
    answer_completeness: float = Field(..., description="Answer completeness score")
    confidence_score: float = Field(..., description="Overall confidence score")
    estimated_accuracy: float = Field(..., description="Estimated accuracy percentage")


class SourceDocument(BaseModel):
    """Model for source document information"""
    dokumen: str = Field(..., description="Document filename")
    judul: str = Field(..., description="Document title")
    sumber_url: str = Field(..., description="Source URL")
    institusi: str = Field(..., description="Institution")
    priority_score: int = Field(..., description="Priority score")
    halaman: str = Field(..., description="Page number")
    chunk_id: int = Field(..., description="Chunk ID")
    similarity_score: float = Field(..., description="Similarity score")
    preview: str = Field(..., description="Content preview")


class QuestionResponse(BaseModel):
    """Response model for question answers"""
    success: bool = Field(..., description="Success status")
    pertanyaan: str = Field(..., description="Original question")
    jawaban: str = Field(..., description="Answer")
    method: str = Field(..., description="Method used")
    metrics: MetricsModel = Field(..., description="Response metrics")
    jumlah_sumber: int = Field(..., description="Number of source documents")
    sumber_dokumen: List[SourceDocument] = Field(..., description="Source documents")
    timestamp: str = Field(..., description="Response timestamp")
    processing_time: Optional[float] = Field(None, description="Processing time in seconds")


class ErrorResponse(BaseModel):
    """Error response model"""
    success: bool = Field(default=False, description="Success status")
    error: str = Field(..., description="Error message")
    error_code: Optional[str] = Field(None, description="Error code")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat(), description="Error timestamp")


class HealthResponse(BaseModel):
    """Health check response model"""
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    timestamp: str = Field(..., description="Check timestamp")
    services: Dict[str, bool] = Field(..., description="Service availability")
    uptime: Optional[float] = Field(None, description="Uptime in seconds")


class SystemInfoResponse(BaseModel):
    """System information response model"""
    app_name: str = Field(..., description="Application name")
    version: str = Field(..., description="Application version")
    environment: str = Field(..., description="Environment")
    ollama_status: bool = Field(..., description="Ollama service status")
    available_models: List[str] = Field(..., description="Available Ollama models")
    vector_stores: Dict[str, bool] = Field(..., description="Vector store availability")
    total_documents: int = Field(..., description="Total loaded documents")
    total_chunks: int = Field(..., description="Total chunks in vector store")


class RebuildRequest(BaseModel):
    """Request model for rebuilding vector store"""
    method: str = Field(..., description="Method to rebuild: 'langchain', 'native', or 'both'")
    force: bool = Field(default=False, description="Force rebuild even if exists")


class RebuildResponse(BaseModel):
    """Response model for rebuild operation"""
    success: bool = Field(..., description="Success status")
    message: str = Field(..., description="Result message")
    method: str = Field(..., description="Method used")
    processing_time: float = Field(..., description="Processing time in seconds")
    timestamp: str = Field(..., description="Completion timestamp")
