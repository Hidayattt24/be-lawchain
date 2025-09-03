"""
LawChain Native - Native RAG Implementation untuk UUD 1945  
Implementasi manual RAG pipeline tanpa LangChain, menggunakan vector_store_structured
"""

import os
import warnings
import time
import requests
import json
import numpy as np
import pickle
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass
import math
import re

# Handle OpenMP conflicts
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

# Suppress warnings
warnings.filterwarnings("ignore")

# Core libraries
import faiss

# Setup logging
logger = logging.getLogger(__name__)

@dataclass
class QueryResult:
    """Result dari query dengan metadata lengkap"""
    answer: str
    success: bool
    processing_time: float
    sources_count: int
    source_details: List[Dict[str, Any]]
    confidence: float
    method: str = "native"
    error: Optional[str] = None

class NativeOllamaEmbedding:
    """Wrapper untuk Ollama embedding tanpa LangChain"""
    
    def __init__(self, model: str = "nomic-embed-text", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url
        self.embedding_url = f"{base_url}/api/embeddings"
    
    def embed_query(self, text: str) -> List[float]:
        """Generate embedding untuk satu query"""
        try:
            response = requests.post(
                self.embedding_url,
                json={
                    "model": self.model,
                    "prompt": text
                },
                timeout=120
            )
            response.raise_for_status()
            return response.json()["embedding"]
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return [0.0] * 768  # Default dimension

class NativeOllamaLLM:
    """Wrapper untuk Ollama LLM tanpa LangChain"""
    
    def __init__(self, model: str = "gemma2:2b", base_url: str = "http://localhost:11434", temperature: float = 0.1):
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.generate_url = f"{base_url}/api/generate"
    
    def generate(self, prompt: str) -> str:
        """Generate response dari Ollama"""
        try:
            response = requests.post(
                self.generate_url,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": self.temperature
                    }
                },
                timeout=300
            )
            response.raise_for_status()
            return response.json()["response"]
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Error: {str(e)}"

class LawChainNative:
    """LawChain implementation tanpa LangChain menggunakan vector_store_structured"""
    
    def __init__(self):
        self.vector_store = None
        self.embeddings_model = None
        self.llm = None
        self.documents_metadata = []
        self.vector_store_path = "storage/vector_store_structured"
        self.max_retrieval_docs = 5
        
        logger.info("🏛️ LawChain Native Service initialized")
    
    def initialize(self) -> bool:
        """Initialize semua komponen Native service"""
        try:
            logger.info("🔄 Initializing LawChain Native Service...")
            
            # Validate Ollama
            self._validate_ollama()
            
            # Setup embeddings
            self._setup_embeddings()
            
            # Load vector store
            self._load_vector_store()
            
            # Setup LLM
            self._setup_llm()
            
            logger.info("✅ LawChain Native Service berhasil diinisialisasi!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Initialization failed: {str(e)}")
            return False
    
    def _validate_ollama(self):
        """Validasi status Ollama"""
        logger.info("🔍 Validating Ollama status...")
        
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code != 200:
                raise Exception("Ollama API not responsive")
            
            models = response.json().get("models", [])
            required_models = ["nomic-embed-text", "gemma2:2b"]
            
            for model in required_models:
                found = any(model in m.get("name", "") for m in models)
                if not found:
                    logger.warning(f"⚠️ Model {model} not found in Ollama")
            
            logger.info("✅ Ollama is running and models are available")
            
        except Exception as e:
            raise Exception(f"Ollama validation failed: {str(e)}")
    
    def _setup_embeddings(self):
        """Setup embedding model"""
        logger.info("🔮 Setting up embeddings...")
        
        self.embeddings_model = NativeOllamaEmbedding()
        
        # Test embedding
        test_embed = self.embeddings_model.embed_query("test")
        logger.info(f"✅ Embeddings ready (dimension: {len(test_embed)})")
    
    def _load_vector_store(self):
        """Load existing FAISS vector store"""
        logger.info("📦 Loading vector store...")
        
        index_path = os.path.join(self.vector_store_path, "index.faiss")
        metadata_path = os.path.join(self.vector_store_path, "index.pkl")
        
        if not os.path.exists(index_path) or not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Vector store not found at {self.vector_store_path}")
        
        # Load FAISS index
        self.vector_store = faiss.read_index(index_path)
        
        # Load metadata from pickle file
        with open(metadata_path, 'rb') as f:
            data = pickle.load(f)
            
        # Handle different pickle structures
        if isinstance(data, dict):
            if "docstore" in data:
                # LangChain FAISS format
                docstore = data["docstore"]
                if hasattr(docstore, "_dict"):
                    self.documents_metadata = list(docstore._dict.values())
                else:
                    # Try to extract from docstore
                    self.documents_metadata = []
                    for i in range(self.vector_store.ntotal):
                        try:
                            doc = docstore.search(str(i))
                            self.documents_metadata.append(doc.metadata if hasattr(doc, 'metadata') else {})
                        except:
                            self.documents_metadata.append({})
            else:
                # Direct metadata format
                self.documents_metadata = data
        elif isinstance(data, list):
            # Direct list of metadata
            self.documents_metadata = data
        else:
            # Fallback - create empty metadata
            self.documents_metadata = [{} for _ in range(self.vector_store.ntotal)]
        
        logger.info(f"✅ Vector store loaded: {self.vector_store.ntotal} documents with {len(self.documents_metadata)} metadata entries")
    
    def _setup_llm(self):
        """Setup LLM"""
        logger.info("🤖 Setting up LLM...")
        
        self.llm = NativeOllamaLLM()
        
        # Test LLM
        test_response = self.llm.generate("Apa itu UUD 1945?")
        logger.info(f"✅ LLM ready: {test_response[:50]}...")
    
    def _search_similar_documents(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search similar documents dengan hard filtering untuk pasal/ayat"""
        
        # Generate query embedding
        query_embedding = np.array([self.embeddings_model.embed_query(query)])
        
        # Analisis query untuk hard filtering
        pasal_match = re.search(r'pasal\s+(\d+)', query.lower())
        ayat_match = re.search(r'ayat\s+(\d+)', query.lower())
        
        if pasal_match:
            pasal_number = int(pasal_match.group(1))
            
            # Filter metadata berdasarkan pasal
            filtered_indices = []
            for i, metadata in enumerate(self.documents_metadata):
                # Handle different metadata formats
                if isinstance(metadata, dict):
                    metadata_dict = metadata
                elif hasattr(metadata, 'metadata'):
                    metadata_dict = metadata.metadata
                else:
                    continue
                    
                if metadata_dict.get("pasal_number") == pasal_number:
                    # Jika ada ayat spesifik, filter juga berdasarkan ayat
                    if ayat_match:
                        ayat_number = int(ayat_match.group(1))
                        if metadata_dict.get("ayat_number") == ayat_number:
                            filtered_indices.append(i)
                    else:
                        filtered_indices.append(i)
            
            if filtered_indices:
                logger.info(f"🎯 Hard filter found {len(filtered_indices)} documents for Pasal {pasal_number}")
                
                # Search only dalam filtered documents
                filtered_vectors = np.array([self.vector_store.reconstruct(i) for i in filtered_indices])
                
                # Similarity search
                scores = np.dot(query_embedding, filtered_vectors.T).flatten()
                top_indices = np.argsort(scores)[::-1][:k]
                
                results = []
                for idx in top_indices:
                    original_idx = filtered_indices[idx]
                    metadata = self.documents_metadata[original_idx]
                    
                    # Extract metadata properly
                    if isinstance(metadata, dict):
                        metadata_dict = metadata
                    elif hasattr(metadata, 'metadata'):
                        metadata_dict = metadata.metadata
                        # Also get page content if available
                        if hasattr(metadata, 'page_content'):
                            metadata_dict["page_content"] = metadata.page_content
                    else:
                        metadata_dict = {}
                    
                    results.append({
                        "index": original_idx,
                        "score": float(scores[idx]),
                        "metadata": metadata_dict
                    })
                
                return results
        
        # Fallback ke regular similarity search
        logger.info("🔍 Using regular similarity search")
        scores, indices = self.vector_store.search(query_embedding, k)
        
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.documents_metadata):
                metadata = self.documents_metadata[idx]
                
                # Extract metadata properly
                if isinstance(metadata, dict):
                    metadata_dict = metadata
                elif hasattr(metadata, 'metadata'):
                    metadata_dict = metadata.metadata
                    if hasattr(metadata, 'page_content'):
                        metadata_dict["page_content"] = metadata.page_content
                else:
                    metadata_dict = {}
                
                results.append({
                    "index": int(idx),
                    "score": float(score),
                    "metadata": metadata_dict
                })
        
        return results
    
    def _create_prompt(self, question: str, context_docs: List[Dict]) -> str:
        """Create enhanced prompt untuk LLM dengan akurasi dan relevansi maksimal"""
        
        context_text = ""
        for i, doc in enumerate(context_docs[:3]):  # Limit ke 3 docs teratas
            metadata = doc["metadata"]
            content = metadata.get("page_content", "")
            
            pasal_info = ""
            if metadata.get("pasal_number"):
                pasal_info = f"Pasal {metadata['pasal_number']}"
                if metadata.get("ayat_number"):
                    pasal_info += f" ayat ({metadata['ayat_number']})"
            
            context_text += f"\n{pasal_info}: {content}\n"

        enhanced_prompt = f"""Anda adalah asisten hukum Indonesia yang sangat ahli dalam UUD 1945. 
Anda memiliki kemampuan luar biasa untuk menganalisis pertanyaan secara mendalam dan memberikan jawaban yang sangat relevan, akurat, dan komprehensif.

INSTRUKSI UTAMA:
1. ANALISIS PERTANYAAN: Pahami secara mendalam apa yang benar-benar ditanyakan pengguna
2. JAWABAN TERFOKUS: Jawab sesuai dengan inti pertanyaan, jangan menyimpang
3. AKURASI TINGGI: Gunakan HANYA informasi dari dokumen UUD 1945 yang tersedia
4. RELEVANSI MAKSIMAL: Pastikan setiap kalimat jawaban berkaitan langsung dengan pertanyaan

STRUKTUR JAWABAN WAJIB:
📋 **JAWABAN LANGSUNG**: Jawab pertanyaan secara langsung di paragraf pertama (1-2 kalimat)

📖 **DASAR HUKUM**: Sebutkan pasal/ayat UUD 1945 yang menjadi dasar jawaban:
- Kutip bunyi pasal/ayat yang tepat dan relevan
- Jelaskan mengapa pasal ini menjawab pertanyaan pengguna

📝 **PENJELASAN MENDALAM**: 
- Uraikan makna dan tujuan dari ketentuan tersebut
- Jelaskan bagaimana hal ini berkaitan dengan sistem ketatanegaraan Indonesia
- Berikan konteks sejarah atau filosofis jika diperlukan

💡 **IMPLIKASI PRAKTIS**:
- Bagaimana ketentuan ini diterapkan dalam praktik bernegara
- Contoh konkret dalam kehidupan sehari-hari atau pemerintahan
- Hubungan dengan lembaga negara atau warga negara

🎯 **KESIMPULAN**:
- Ringkas jawaban dalam 1-2 kalimat yang menjawab langsung pertanyaan awal
- Tekankan poin utama yang paling penting

PRINSIP KUALITAS JAWABAN:
✅ Jawaban harus LANGSUNG menjawab pertanyaan (tidak bertele-tele)
✅ Setiap paragraf harus RELEVAN dengan pertanyaan
✅ Gunakan bahasa yang mudah dipahami tapi tetap akurat
✅ Jika pertanyaan meminta contoh, berikan contoh yang spesifik
✅ Jika pertanyaan meminta penjelasan, berikan analisis yang mendalam

⚠️ PANTANGAN:
❌ Jangan memberikan informasi yang tidak ada dalam dokumen
❌ Jangan menjawab hal yang tidak ditanyakan
❌ Jangan terlalu umum, jawab sesuai spesifik pertanyaan
❌ Jangan mengulangi informasi yang sama berkali-kali

DOKUMEN UUD 1945:
{context_text}

PERTANYAAN PENGGUNA: {question}

JAWABAN KOMPREHENSIF DAN AKURAT:"""
        
        return enhanced_prompt
    
    def _is_legal_context(self, question: str) -> bool:
        """Check apakah pertanyaan masih dalam konteks hukum Indonesia"""
        legal_keywords = [
            # UUD dan Konstitusi
            'uud', 'undang-undang', 'konstitusi', 'pasal', 'ayat', 'bab',
            # Lembaga Negara
            'presiden', 'wakil presiden', 'dpr', 'dpd', 'mpr', 'mahkamah', 'mk', 'ma',
            'kpu', 'bawaslu', 'kementerian', 'menteri', 
            # Sistem Hukum
            'hukum', 'peraturan', 'perundangan', 'legal', 'juridis', 'yuridis',
            'sanksi', 'pidana', 'perdata', 'administrasi', 'tata negara',
            # Pemerintahan
            'pemerintahan', 'negara', 'republik', 'indonesia', 'nkri',
            'kedaulatan', 'rakyat', 'demokrasi', 'pancasila',
            # Peradilan
            'pengadilan', 'hakim', 'jaksa', 'kepolisian', 'kejaksaan',
            # Pemilu dan Politik
            'pemilu', 'pilpres', 'pileg', 'pilkada',
            # Pemerintah Daerah
            'gubernur', 'bupati', 'walikota', 'camat', 'lurah',
            # Terms English
            'constitutional', 'legislation', 'law', 'legal system',
            # Spesifik Tugas/Fungsi
            'tugas', 'fungsi', 'wewenang', 'kewajiban', 'hak', 'kewenangan'
        ]
        
        question_lower = question.lower()
        
        # Must contain at least one legal keyword
        has_legal_keyword = any(keyword in question_lower for keyword in legal_keywords)
        
        # Additional check: exclude common non-legal topics
        non_legal_indicators = [
            'makanan', 'makan', 'masak', 'resep', 'kucing', 'anjing', 'hewan',
            'olahraga', 'musik', 'film', 'game', 'teknologi', 'komputer',
            'smartphone', 'aplikasi', 'sosmed', 'facebook', 'instagram',
            'cuaca', 'alam', 'travel', 'wisata', 'hotel', 'restoran',
            'fashion', 'kecantikan', 'kesehatan', 'obat', 'dokter',
            'sekolah', 'universitas', 'pelajaran', 'matematika', 'fisika',
            'kimia', 'biologi', 'sejarah', 'geografi'
        ]
        
        has_non_legal = any(indicator in question_lower for indicator in non_legal_indicators)
        
        # Return True only if has legal keywords AND no strong non-legal indicators
        return has_legal_keyword and not has_non_legal
    
    def _generate_fallback_answer(self, question: str) -> str:
        """Generate jawaban dari Gemma2:2b untuk pertanyaan hukum umum"""
        
        if not self._is_legal_context(question):
            return 'Maaf, kami belum bisa menjawab pertanyaan di luar konteks hukum Indonesia. Saya adalah asisten khusus untuk UUD 1945 dan hukum Indonesia. Silakan tanyakan tentang pasal-pasal UUD 1945, lembaga negara, atau sistem hukum Indonesia yang bisa saya bantu jawab.'
        
        # Enhanced prompt untuk Gemma2:2b tentang hukum Indonesia secara umum
        enhanced_fallback_prompt = f"""Anda adalah asisten hukum Indonesia yang sangat ahli dan berpengalaman dalam sistem hukum Indonesia. Anda mampu memberikan jawaban yang akurat, relevan, dan komprehensif.

PERTANYAAN: {question}

INSTRUKSI KHUSUS:
1. Analisis pertanyaan dengan mendalam untuk memahami apa yang benar-benar ditanyakan
2. Berikan jawaban yang fokus dan langsung menjawab pertanyaan
3. Gunakan pengetahuan komprehensif tentang hukum Indonesia
4. Berikan informasi yang akurat dan up-to-date
5. Jika berkaitan dengan UUD 1945, berikan informasi umum yang tepat

STRUKTUR JAWABAN YANG DIHARAPKAN:
📋 **JAWABAN LANGSUNG**: Jawab inti pertanyaan secara langsung (1-2 kalimat)

📖 **DASAR HUKUM**: 
- Sebutkan peraturan perundang-undangan yang relevan
- Jika menyangkut UUD 1945, sebutkan pasal/bab yang terkait (secara umum)

📝 **PENJELASAN KOMPREHENSIF**:
- Uraikan secara detail dan mudah dipahami
- Jelaskan konteks dalam sistem hukum Indonesia
- Berikan latar belakang atau sejarah singkat jika relevan

💡 **IMPLEMENTASI PRAKTIS**:
- Bagaimana hal ini diterapkan dalam praktek
- Contoh konkret dalam kehidupan sehari-hari
- Hubungan dengan lembaga negara atau masyarakat

🎯 **KESIMPULAN & SARAN**:
- Ringkas poin utama jawaban
- Berikan saran untuk informasi lebih detail (konsultasi ahli hukum, rujuk dokumen resmi)

PRINSIP KUALITAS:
✅ Jawaban harus akurat dan dapat dipertanggungjawabkan
✅ Fokus pada pertanyaan, tidak menyimpang
✅ Berikan konteks yang memadai untuk pemahaman
✅ Gunakan bahasa yang profesional namun mudah dipahami

Jawaban Anda:"""

        try:
            return self.llm.generate(enhanced_fallback_prompt)
        except Exception as e:
            logger.error(f"Fallback generation failed: {e}")
            return f'Informasi tentang "{question}" tidak tersedia dalam database UUD 1945 kami. Untuk informasi hukum Indonesia yang lebih lengkap, silakan konsultasi dengan ahli hukum atau rujuk dokumen hukum resmi.'
    
    def query(self, question: str) -> QueryResult:
        """Process question dan return structured result"""
        start_time = time.time()
        
        try:
            # Auto-initialize if not initialized
            if self.embeddings_model is None or self.vector_store is None or self.llm is None:
                logger.info("🔄 Auto-initializing service...")
                if not self.initialize():
                    return QueryResult(
                        answer="Maaf, service tidak dapat diinisialisasi.",
                        success=False,
                        processing_time=round(time.time() - start_time, 2),
                        sources_count=0,
                        source_details=[],
                        confidence=0.0,
                        method="native",
                        error="Service initialization failed"
                    )
            
            logger.info(f"🔍 Processing question: {question}")
            
            # FIRST: Check if question is in legal context
            if not self._is_legal_context(question):
                logger.info("🚫 Question outside legal context, providing polite response")
                return QueryResult(
                    answer='Maaf, kami belum bisa menjawab pertanyaan di luar konteks hukum Indonesia. Saya adalah asisten khusus untuk UUD 1945 dan hukum Indonesia. Silakan tanyakan tentang pasal-pasal UUD 1945, lembaga negara, atau sistem hukum Indonesia yang bisa saya bantu jawab.',
                    success=True,  # Changed to True so it's not treated as an error
                    processing_time=round(time.time() - start_time, 2),
                    sources_count=0,
                    source_details=[],
                    confidence=0.0,
                    method="native",
                    error=None  # No error, just out of context
                )
            
            # Search similar documents
            similar_docs = self._search_similar_documents(question, self.max_retrieval_docs)
            
            if not similar_docs:
                logger.warning("⚠️ No relevant documents found, checking legal context...")
                # Check if question is in legal context
                if self._is_legal_context(question):
                    logger.info("📚 Legal context detected, using Gemma2:2b fallback")
                    answer = self._generate_fallback_answer(question)
                    
                    return QueryResult(
                        answer=answer,
                        success=True,
                        processing_time=round(time.time() - start_time, 2),
                        sources_count=0,
                        source_details=[],
                        confidence=75.0,  # Good confidence for fallback legal answers
                        method="native"
                    )
                else:
                    logger.info("🚫 Non-legal context, providing standard response")
                    answer = f'Maaf "{question}" kami tidak bisa menjawab nya karena di luar konteks Hukum di Indonesia. Tanyakan terkait UUD pasti kami bisa menjawab nya.'
                    
                    return QueryResult(
                        answer=answer,
                        success=False,
                        processing_time=round(time.time() - start_time, 2),
                        sources_count=0,
                        source_details=[],
                        confidence=0.0,
                        method="native",
                        error="Outside legal context"
                    )
            
            # Create prompt dan generate answer
            prompt = self._create_prompt(question, similar_docs)
            answer = self.llm.generate(prompt)
            
            # Process source details
            source_details = []
            for doc in similar_docs:
                metadata = doc["metadata"]
                source_details.append({
                    "content": metadata.get("page_content", "")[:200] + "..." if len(metadata.get("page_content", "")) > 200 else metadata.get("page_content", ""),
                    "pasal_number": metadata.get("pasal_number"),
                    "ayat_number": metadata.get("ayat_number"),
                    "bab_number": metadata.get("bab_number"),
                    "bab_title": metadata.get("bab_title"),
                    "page_number": metadata.get("page_number"),
                    "authority_level": metadata.get("authority_level", "high"),
                    "source_file": metadata.get("source_file"),
                    "confidence_score": metadata.get("chunk_quality_score", 85.0),
                    "similarity_score": doc["score"]
                })
            
            processing_time = time.time() - start_time
            confidence = self._calculate_confidence(answer, similar_docs, question)
            
            logger.info(f"✅ Query processed in {processing_time:.2f}s with confidence {confidence}%")
            
            return QueryResult(
                answer=answer,
                success=True,
                processing_time=round(processing_time, 2),
                sources_count=len(similar_docs),
                source_details=source_details,
                confidence=confidence,
                method="native"
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Error processing question: {str(e)}"
            logger.error(error_msg)
            
            return QueryResult(
                answer=f"Maaf, terjadi kesalahan: {str(e)}",
                success=False,
                processing_time=round(processing_time, 2),
                sources_count=0,
                source_details=[],
                confidence=0.0,
                method="native",
                error=str(e)
            )
    
    def _calculate_confidence(self, answer: str, sources: List[Dict], question: str) -> float:
        """Calculate confidence score"""
        
        if not sources:
            return 10.0
        
        # Base score dari jumlah sources
        source_score = min(len(sources) * 5, 25)
        
        # Similarity score dari top document
        similarity_score = min(sources[0]["score"] * 100, 25) if sources else 0
        
        # Authority score dari metadata
        authority_score = 0
        for doc in sources:
            metadata = doc["metadata"]
            if metadata.get("authority_level") == "highest":
                authority_score += 5
            elif metadata.get("authority_level") == "high":
                authority_score += 3
            else:
                authority_score += 1
        authority_score = min(authority_score, 20)
        
        # Exact match untuk pasal/ayat
        exact_match_score = 0
        question_lower = question.lower()
        if "pasal" in question_lower:
            for doc in sources:
                metadata = doc["metadata"]
                if metadata.get("pasal_number"):
                    exact_match_score += 15
                    if metadata.get("ayat_number") and "ayat" in question_lower:
                        exact_match_score += 10
                    break
        
        # Answer quality
        answer_quality = min(len(answer) / 100 * 5, 15)
        
        total_confidence = source_score + similarity_score + authority_score + exact_match_score + answer_quality
        
        return round(min(total_confidence, 100.0), 1)
    
    def health_check(self) -> Dict[str, Any]:
        """Health check untuk service"""
        try:
            # Test basic functionality
            test_result = self.query("Apa itu UUD 1945?")
            
            return {
                "status": "healthy",
                "service": "native",
                "vector_store_loaded": self.vector_store is not None,
                "llm_ready": self.llm is not None,
                "embeddings_ready": self.embeddings_model is not None,
                "test_query_success": test_result.success,
                "test_confidence": test_result.confidence,
                "total_documents": self.vector_store.ntotal if self.vector_store else 0
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "service": "native", 
                "error": str(e)
            }

# Global instance untuk reuse
_native_service = None

def get_native_service() -> LawChainNative:
    """Get atau create global Native service instance"""
    global _native_service
    
    if _native_service is None:
        _native_service = LawChainNative()
        if not _native_service.initialize():
            raise RuntimeError("Failed to initialize Native service")
    
    return _native_service

# Test function
if __name__ == "__main__":
    print("🧪 Testing LawChain Native Service...")
    print("=" * 60)
    
    service = LawChainNative()
    
    if service.initialize():
        print("\n✅ Service initialized successfully")
        
        # Test queries
        test_questions = [
            "Apa bunyi Pasal 1 ayat 1?",
            "Jelaskan tentang kekuasaan presiden", 
            "Sebutkan wewenang MPR"
        ]
        
        for question in test_questions:
            print(f"\n❓ Question: {question}")
            result = service.query(question)
            
            print(f"✅ Success: {result.success}")
            print(f"📝 Answer: {result.answer[:100]}...")
            print(f"⏱️ Time: {result.processing_time}s")
            print(f"📊 Sources: {result.sources_count}")
            print(f"🎯 Confidence: {result.confidence}%")
    else:
        print("❌ Failed to initialize service")
