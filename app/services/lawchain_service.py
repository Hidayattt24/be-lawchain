"""
Service wrapper for LawChain implementations
"""

import time
import logging
from typing import Dict, Any, Optional
from datetime import datetime

from app.utils.helpers import format_timestamp, format_processing_time


logger = logging.getLogger(__name__)


class LawChainService:
    """Service wrapper untuk kedua implementasi LawChain"""
    
    def __init__(self):
        self.langchain_instance: Optional[Any] = None
        self.native_instance: Optional[Any] = None
        self.initialization_status = {
            'langchain': False,
            'native': False
        }
    
    def initialize_langchain(self, force_rebuild: bool = False):
        """Initialize LangChain implementation"""
        try:
            logger.info("Initializing LangChain implementation...")
            
            # Import dan inisialisasi LangChain implementation
            from lawchain_indonesia import LawChainIndonesia
            
            self.langchain_instance = LawChainIndonesia()
            self.langchain_instance.initialize(force_rebuild_vectorstore=force_rebuild)
            
            self.initialization_status['langchain'] = True
            logger.info("LangChain implementation initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize LangChain implementation: {str(e)}")
            self.initialization_status['langchain'] = False
            raise
    
    def initialize_native(self, force_rebuild: bool = False):
        """Initialize Native implementation"""
        try:
            logger.info("Initializing Native implementation...")
            
            # Import dan inisialisasi Native implementation
            from lawchain_native import LawChainNative
            
            self.native_instance = LawChainNative()
            self.native_instance.initialize(force_rebuild_vectorstore=force_rebuild)
            
            self.initialization_status['native'] = True
            logger.info("Native implementation initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Native implementation: {str(e)}")
            self.initialization_status['native'] = False
            raise
    
    def initialize_both(self, force_rebuild: bool = False):
        """Initialize both implementations"""
        logger.info("Initializing both implementations...")
        
        try:
            self.initialize_langchain(force_rebuild)
        except Exception as e:
            logger.warning(f"LangChain initialization failed: {str(e)}")
        
        try:
            self.initialize_native(force_rebuild)
        except Exception as e:
            logger.warning(f"Native initialization failed: {str(e)}")
        
        if not any(self.initialization_status.values()):
            raise Exception("Failed to initialize any implementation")
    
    def ask_question(self, question: str, method: str = "langchain", max_docs: int = 5) -> Dict[str, Any]:
        """Ask question using specified method"""
        start_time = time.time()
        
        try:
            if method == "langchain":
                if not self.initialization_status['langchain'] or not self.langchain_instance:
                    raise ValueError("LangChain implementation not initialized")
                
                logger.info(f"Processing question with LangChain: {question[:50]}...")
                response = self.langchain_instance.ask_question(question)
                
            elif method == "native":
                if not self.initialization_status['native'] or not self.native_instance:
                    raise ValueError("Native implementation not initialized")
                
                logger.info(f"Processing question with Native: {question[:50]}...")
                response = self.native_instance.ask_question(question)
                
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            # Add processing time
            response['processing_time'] = format_processing_time(start_time)
            response['success'] = True
            
            logger.info(f"Question processed successfully in {response['processing_time']}s")
            return response
            
        except Exception as e:
            logger.error(f"Error processing question: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'pertanyaan': question,
                'method': method,
                'processing_time': format_processing_time(start_time),
                'timestamp': format_timestamp()
            }
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information"""
        try:
            info = {
                'langchain_initialized': self.initialization_status['langchain'],
                'native_initialized': self.initialization_status['native'],
                'total_documents': 0,
                'total_chunks': 0
            }
            
            # Get info from initialized instances
            if self.langchain_instance and self.initialization_status['langchain']:
                info['langchain_documents'] = getattr(self.langchain_instance, 'total_documents', 0)
                info['langchain_chunks'] = getattr(self.langchain_instance, 'total_chunks', 0)
                info['total_documents'] = max(info['total_documents'], info['langchain_documents'])
                info['total_chunks'] = max(info['total_chunks'], info['langchain_chunks'])
            
            if self.native_instance and self.initialization_status['native']:
                info['native_documents'] = getattr(self.native_instance, 'total_documents', 0)
                info['native_chunks'] = getattr(self.native_instance, 'total_chunks', 0)
                info['total_documents'] = max(info['total_documents'], info['native_documents'])
                info['total_chunks'] = max(info['total_chunks'], info['native_chunks'])
            
            return info
            
        except Exception as e:
            logger.error(f"Error getting system info: {str(e)}")
            return {
                'langchain_initialized': False,
                'native_initialized': False,
                'total_documents': 0,
                'total_chunks': 0,
                'error': str(e)
            }
    
    def rebuild_vector_store(self, method: str) -> Dict[str, Any]:
        """Rebuild vector store for specified method"""
        start_time = time.time()
        
        try:
            if method == "langchain":
                logger.info("Rebuilding LangChain vector store...")
                self.initialize_langchain(force_rebuild=True)
                
            elif method == "native":
                logger.info("Rebuilding Native vector store...")
                self.initialize_native(force_rebuild=True)
                
            elif method == "both":
                logger.info("Rebuilding both vector stores...")
                self.initialize_both(force_rebuild=True)
                
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            processing_time = format_processing_time(start_time)
            logger.info(f"Vector store rebuilt successfully in {processing_time}s")
            
            return {
                'success': True,
                'message': f"Vector store for {method} rebuilt successfully",
                'method': method,
                'processing_time': processing_time,
                'timestamp': format_timestamp()
            }
            
        except Exception as e:
            logger.error(f"Error rebuilding vector store: {str(e)}")
            return {
                'success': False,
                'message': f"Failed to rebuild vector store: {str(e)}",
                'method': method,
                'processing_time': format_processing_time(start_time),
                'timestamp': format_timestamp()
            }


# Global service instance
lawchain_service = LawChainService()
