"""
Pydantic models for API requests and responses
Compatible with frontend chatbot interface
"""

from datetime import datetime
from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field


class QuestionRequest(BaseModel):
    """Request model for asking questions - Frontend compatible"""
    question: str = Field(..., min_length=1, max_length=1000, description="Pertanyaan tentang UUD 1945")
    method: Literal["langchain", "native"] = Field(default="langchain", description="Service method to use")
    max_docs: Optional[int] = Field(default=5, ge=1, le=10, description="Maximum number of documents to retrieve")


class SourceDocument(BaseModel):
    """Model for source document information - Frontend compatible"""
    dokumen: str = Field(..., description="Document filename")
    judul: str = Field(..., description="Document title")
    sumber_url: str = Field(..., description="Document source URL")
    institusi: str = Field(..., description="Institution name")
    priority_score: int = Field(..., description="Priority score")
    halaman: str = Field(..., description="Page number")
    chunk_id: int = Field(..., description="Chunk identifier")
    similarity_score: float = Field(..., description="Similarity score")
    preview: str = Field(..., description="Content preview")


class Metrics(BaseModel):
    """Model for detailed metrics - Frontend compatible"""
    semantic_similarity: float = Field(..., description="Semantic similarity score")
    content_coverage: float = Field(..., description="Content coverage score")
    answer_relevance: float = Field(..., description="Answer relevance score")
    source_quality: float = Field(..., description="Source quality score")
    legal_context: float = Field(..., description="Legal context score")
    answer_completeness: float = Field(..., description="Answer completeness score")
    confidence_score: float = Field(..., description="Overall confidence score")
    estimated_accuracy: float = Field(..., description="Estimated accuracy percentage")


class QuestionResponse(BaseModel):
    """Response model for question answers - Frontend compatible"""
    success: bool = Field(..., description="Success status")
    pertanyaan: str = Field(..., description="Original question")
    jawaban: str = Field(..., description="Answer to the question")
    method: str = Field(..., description="Method used (langchain/native)")
    metrics: Metrics = Field(..., description="Response metrics")
    jumlah_sumber: int = Field(..., description="Number of source documents")
    sumber_dokumen: List[SourceDocument] = Field(default_factory=list, description="Source documents")
    timestamp: str = Field(..., description="Response timestamp")
    processing_time: Optional[float] = Field(None, description="Processing time in seconds")


class HealthResponse(BaseModel):
    """Health check response model"""
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    timestamp: str = Field(..., description="Check timestamp")
    services: Dict[str, bool] = Field(..., description="Service availability")
    uptime: Optional[float] = Field(None, description="Uptime in seconds")


class MetricsModel(BaseModel):
    """Model for metrics information"""
    response_time: Optional[float] = Field(None, description="Response time in seconds")
    tokens_used: Optional[int] = Field(None, description="Number of tokens used")
    model_name: Optional[str] = Field(None, description="Model name used")
    confidence_score: Optional[float] = Field(None, description="Confidence score")
    source_count: Optional[int] = Field(None, description="Number of source documents")


class ErrorResponse(BaseModel):
    """Error response model"""
    success: bool = Field(default=False, description="Success status")
    error: str = Field(..., description="Error message")
    error_code: Optional[str] = Field(None, description="Error code")


# Additional models for legacy API compatibility
class RebuildRequest(BaseModel):
    """Request model for rebuilding vector store"""
    force: bool = Field(default=False, description="Force rebuild even if exists")
    

class RebuildResponse(BaseModel):
    """Response model for rebuild operation"""
    success: bool = Field(..., description="Rebuild success status")
    message: str = Field(..., description="Rebuild status message")
    details: Optional[Dict[str, Any]] = Field(None, description="Rebuild details")
    

class SystemInfoResponse(BaseModel):
    """System information response model"""
    status: str = Field(..., description="System status")
    vector_store: Dict[str, Any] = Field(..., description="Vector store information")
    models: Dict[str, Any] = Field(..., description="Model information")
    system: Dict[str, Any] = Field(..., description="System information")
    

class DocUploadRequest(BaseModel):
    """Document upload request model"""
    filename: str = Field(..., description="Document filename")
    

class UploadResponse(BaseModel):
    """Upload response model"""
    success: bool = Field(..., description="Upload success status")
    message: str = Field(..., description="Upload status message")
    filename: Optional[str] = Field(None, description="Uploaded filename")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat(), description="Error timestamp")


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
