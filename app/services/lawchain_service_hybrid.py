"""
Service wrapper for LawChain implementations - Hybrid compatibility layer
Maps legacy langchain/native calls to optimized backend
"""

import time
import logging
from typing import Dict, Any, Optional
from datetime import datetime

from app.utils.helpers import format_timestamp, format_time_duration


logger = logging.getLogger(__name__)


class LawChainService:
    """Service wrapper yang menyediakan kompatibilitas untuk langchain dan native via optimized backend"""
    
    def __init__(self):
        self.optimized_service = None
        self.initialization_status = {
            'langchain': False,
            'native': False,
            'optimized': False
        }
        
        # Auto-initialize optimized service
        try:
            from app.services.lawchain_optimized_service import optimized_lawchain_service
            self.optimized_service = optimized_lawchain_service
            
            # Set status based on optimized service
            if self.optimized_service.get_status()['ready']:
                self.initialization_status['langchain'] = True
                self.initialization_status['native'] = True
                self.initialization_status['optimized'] = True
                logger.info("LawChain service initialized using optimized backend")
        except Exception as e:
            logger.warning(f"Auto-initialization failed: {str(e)}")
    
    def check_vector_stores_exist(self):
        """Check if vector stores exist (using optimized backend)"""
        if self.optimized_service:
            optimized_exists = self.optimized_service.check_optimized_vector_store_exists()
            return {
                'langchain': optimized_exists,   # Alias untuk kompatibilitas
                'native': optimized_exists,      # Alias untuk kompatibilitas
                'optimized': optimized_exists
            }
        return {'langchain': False, 'native': False, 'optimized': False}
    
    def initialize_langchain(self):
        """Initialize langchain (menggunakan optimized backend)"""
        try:
            if self.optimized_service:
                success = self.optimized_service.initialize_optimized()
                self.initialization_status['langchain'] = success
                if success:
                    logger.info("LangChain initialized successfully via optimized backend")
                return success
            return False
        except Exception as e:
            logger.error(f"Failed to initialize langchain: {str(e)}")
            return False
    
    def initialize_native(self):
        """Initialize native (menggunakan optimized backend)"""
        try:
            if self.optimized_service:
                success = self.optimized_service.initialize_optimized()
                self.initialization_status['native'] = success
                if success:
                    logger.info("Native initialized successfully via optimized backend")
                return success
            return False
        except Exception as e:
            logger.error(f"Failed to initialize native: {str(e)}")
            return False
    
    async def ask_question(self, question: str, method: str = "langchain", max_docs: int = 5) -> Dict[str, Any]:
        """Ask question using specified method (semua method menggunakan optimized backend)"""
        start_time = time.time()
        
        try:
            logger.info(f"Processing question with {method} (via optimized): {question[:100]}...")
            
            # Semua method menggunakan optimized service
            if self.optimized_service:
                # Pastikan service terinisialisasi
                if not self.optimized_service.get_status()['ready']:
                    self.optimized_service.initialize_optimized()
                
                # Gunakan optimized service
                result = await self.optimized_service.answer_question(question, True)
                
                # Format response untuk kompatibilitas
                if result.get('success', True):
                    processing_time = time.time() - start_time
                    
                    return {
                        "success": True,
                        "question": question,
                        "answer": result.get("answer", ""),
                        "sources": result.get("sources", []),
                        "method": method,
                        "model": "gemma2:2b",
                        "processing_time": format_time_duration(processing_time),
                        "timestamp": format_timestamp(),
                        "via_optimized": True,
                        "metadata": result.get("metadata", {})
                    }
                else:
                    return {
                        "success": False,
                        "error": result.get("error", "Unknown error"),
                        "question": question,
                        "method": method,
                        "processing_time": format_time_duration(time.time() - start_time),
                        "timestamp": format_timestamp()
                    }
            else:
                raise Exception("Optimized service not available")
                
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Error in {method} question answering: {str(e)}")
            
            return {
                "success": False,
                "error": str(e),
                "question": question,
                "method": method,
                "processing_time": format_time_duration(processing_time),
                "timestamp": format_timestamp()
            }
    
    def rebuild_vector_store(self, method: str) -> Dict[str, Any]:
        """Rebuild vector store untuk method tertentu (menggunakan optimized backend)"""
        start_time = time.time()
        
        try:
            logger.info(f"Rebuilding vector store for {method} (via optimized)...")
            
            if self.optimized_service:
                result = self.optimized_service.rebuild_optimized_vector_store()
                
                # Update initialization status
                if result.get("success"):
                    if method in ["langchain", "both"]:
                        self.initialization_status['langchain'] = True
                    if method in ["native", "both"]:
                        self.initialization_status['native'] = True
                    self.initialization_status['optimized'] = True
                
                # Format response
                result["method"] = method
                result["via_optimized"] = True
                result["message"] = f"Vector store for {method} rebuilt successfully via optimized backend"
                
                return result
            else:
                raise Exception("Optimized service not available")
                
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Error rebuilding vector store for {method}: {str(e)}"
            logger.error(error_msg)
            
            return {
                "success": False,
                "error": error_msg,
                "method": method,
                "processing_time": format_time_duration(processing_time),
                "timestamp": format_timestamp()
            }
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information"""
        try:
            if self.optimized_service:
                # Get info from optimized service (synchronous version)
                stores = self.check_vector_stores_exist()
                
                return {
                    "total_documents": 5,  # Static info
                    "total_chunks": 741,   # Static info  
                    "langchain_ready": self.initialization_status['langchain'],
                    "native_ready": self.initialization_status['native'],
                    "optimized_ready": self.initialization_status['optimized'],
                    "vector_stores": stores,
                    "via_optimized": True,
                    "model": "gemma2:2b"
                }
            else:
                return {
                    "total_documents": 0,
                    "total_chunks": 0,
                    "error": "Optimized service not available"
                }
        except Exception as e:
            logger.error(f"Error getting system info: {str(e)}")
            return {
                "total_documents": 0,
                "total_chunks": 0,
                "error": str(e)
            }


# Create global instance
lawchain_service = LawChainService()
