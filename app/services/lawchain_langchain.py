"""
LawChain LangChain Service - Structured RAG Implementation
Menggunakan LangChain framework dengan vector_store_structured
"""

import os
import re
import time
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from langchain.schema import BaseRetriever

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
    method: str = "langchain"
    error: Optional[str] = None

class LawChainLangChain:
    """LawChain service menggunakan LangChain framework"""
    
    def __init__(self):
        self.vector_store = None
        self.embeddings = None
        self.llm = None
        self.qa_chain = None
        self.retriever = None
        self.vector_store_path = "storage/vector_store_structured"
        self.max_retrieval_docs = 5
        
        logger.info("🏛️ LawChain LangChain Service initialized")
    
    def initialize(self) -> bool:
        """Initialize semua komponen LangChain"""
        try:
            logger.info("🔄 Initializing LawChain LangChain Service...")
            
            # Setup embeddings
            self._setup_embeddings()
            
            # Load vector store
            self._load_vector_store()
            
            # Setup LLM
            self._setup_llm()
            
            # Create retriever dan QA chain
            self._create_retriever()
            self._create_qa_chain()
            
            logger.info("✅ LawChain LangChain Service berhasil diinisialisasi!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Initialization failed: {str(e)}")
            return False
    
    def _setup_embeddings(self):
        """Setup Ollama embeddings"""
        logger.info("🔮 Setting up embeddings...")
        
        self.embeddings = OllamaEmbeddings(
            model="nomic-embed-text",
            base_url="http://localhost:11434"
        )
        
        # Test embedding
        test_embed = self.embeddings.embed_query("test")
        logger.info(f"✅ Embeddings ready (dimension: {len(test_embed)})")
    
    def _load_vector_store(self):
        """Load existing vector store"""
        logger.info("📦 Loading vector store...")
        
        if not os.path.exists(self.vector_store_path):
            raise FileNotFoundError(f"Vector store not found at {self.vector_store_path}")
        
        self.vector_store = FAISS.load_local(
            self.vector_store_path,
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        
        logger.info("✅ Vector store loaded successfully")
    
    def _setup_llm(self):
        """Setup Ollama LLM"""
        logger.info("🤖 Setting up LLM...")
        
        self.llm = Ollama(
            model="gemma2:2b",
            base_url="http://localhost:11434",
            temperature=0.1,
            top_p=0.9,
            num_predict=512,
            stop=["Human:", "Assistant:"]
        )
        
        # Test LLM
        test_response = self.llm("Apa itu UUD 1945?")
        logger.info(f"✅ LLM ready: {test_response[:50]}...")
    
    def _create_retriever(self):
        """Create retriever dengan hard filtering capability"""
        logger.info("🔍 Creating retriever...")
        
        # Use standard retriever with MMR
        self.retriever = self.vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": self.max_retrieval_docs,
                "lambda_mult": 0.8,
                "fetch_k": 20
            }
        )
        
        logger.info("✅ Retriever created")
    
    def _create_qa_chain(self):
        """Create QA chain dengan prompt yang dioptimasi"""
        logger.info("🔗 Creating QA chain...")
        
        # Enhanced template prompt untuk response yang lebih akurat dan relevan
        template = """Anda adalah asisten hukum Indonesia yang sangat ahli dalam UUD 1945. 
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
{context}

PERTANYAAN PENGGUNA: {question}

JAWABAN KOMPREHENSIF DAN AKURAT:"""

        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
        
        # Create QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )
        
        logger.info("✅ QA chain created successfully")
        return True
    
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
    
    def _rank_documents_by_relevance(self, docs, question: str) -> List:
        """Enhanced document ranking untuk meningkatkan akurasi"""
        question_lower = question.lower()
        
        # Keywords yang meningkatkan prioritas
        priority_keywords = {
            'presiden': 3.0, 'wakil presiden': 3.0, 'dpr': 2.5, 'dpd': 2.5, 'mpr': 2.5,
            'mahkamah': 3.0, 'mk': 2.5, 'ma': 2.5, 'menteri': 2.0, 'pemerintahan': 2.0,
            'kekuasaan': 2.5, 'kehakiman': 3.0, 'eksekutif': 2.5, 'legislatif': 2.5,
            'yudikatif': 2.5, 'amandemen': 2.0, 'pasal': 2.0, 'ayat': 2.0
        }
        
        scored_docs = []
        for doc in docs:
            content_lower = doc.page_content.lower() if hasattr(doc, 'page_content') else str(doc).lower()
            
            # Base score dari similarity
            base_score = getattr(doc, 'similarity_score', 0.7)
            
            # Bonus untuk keyword matching
            keyword_bonus = 0.0
            for keyword, weight in priority_keywords.items():
                if keyword in question_lower and keyword in content_lower:
                    keyword_bonus += weight * 0.1  # Max 0.3 bonus
            
            # Bonus untuk exact pasal/ayat match
            pasal_match = re.search(r'pasal\s+(\d+)', question_lower)
            if pasal_match:
                pasal_num = pasal_match.group(1)
                if f'pasal {pasal_num}' in content_lower:
                    keyword_bonus += 0.4  # High bonus for exact pasal match
            
            # Penalty untuk dokumen yang terlalu pendek
            length_penalty = 0.0
            if len(content_lower) < 50:
                length_penalty = -0.2
            
            final_score = min(base_score + keyword_bonus + length_penalty, 1.0)
            scored_docs.append((doc, final_score))
        
        # Sort by score descending
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, score in scored_docs]
    
    def _calculate_enhanced_confidence(self, docs, answer: str, question: str) -> float:
        """Calculate enhanced confidence score berdasarkan multiple factors"""
        base_confidence = 0.6  # Base confidence
        
        # Factor 1: Document quality dan relevance
        doc_quality_score = 0.0
        if docs:
            avg_doc_length = sum(len(str(doc)) for doc in docs) / len(docs)
            if avg_doc_length > 200:  # Good document length
                doc_quality_score += 0.1
            if len(docs) >= 3:  # Multiple supporting documents
                doc_quality_score += 0.1
        
        # Factor 2: Answer completeness
        answer_completeness = 0.0
        if len(answer) > 100:  # Detailed answer
            answer_completeness += 0.1
        if any(marker in answer.lower() for marker in ['pasal', 'ayat', 'uud']):
            answer_completeness += 0.1
        
        # Factor 3: Question-answer alignment
        question_words = set(question.lower().split())
        answer_words = set(answer.lower().split())
        overlap_ratio = len(question_words & answer_words) / len(question_words) if question_words else 0
        alignment_score = min(overlap_ratio * 0.2, 0.2)
        
        final_confidence = min(base_confidence + doc_quality_score + answer_completeness + alignment_score, 0.95)
        return final_confidence * 100  # Convert to percentage
    
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
            return self.llm.invoke(enhanced_fallback_prompt)
        except Exception as e:
            logger.error(f"Fallback generation failed: {e}")
            return f'Informasi tentang "{question}" tidak tersedia dalam database UUD 1945 kami. Untuk informasi hukum Indonesia yang lebih lengkap, silakan konsultasi dengan ahli hukum atau rujuk dokumen hukum resmi.'
    
    def query(self, question: str) -> QueryResult:
        """Process question dan return structured result dengan hard filtering"""
        start_time = time.time()
        
        try:
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
                    method="langchain",
                    error=None  # No error, just out of context
                )
            
            # Analisis query untuk hard filtering
            pasal_match = re.search(r'pasal\s+(\d+)', question.lower())
            ayat_match = re.search(r'ayat\s+(\d+)', question.lower())
            
            # Hard filtering jika ada pasal/ayat spesifik
            if pasal_match:
                pasal_number = int(pasal_match.group(1))
                filter_dict = {"pasal_number": pasal_number}
                
                if ayat_match:
                    ayat_number = int(ayat_match.group(1))
                    filter_dict["ayat_number"] = ayat_number
                    logger.info(f"🎯 HARD FILTER: Pasal {pasal_number} ayat {ayat_number}")
                else:
                    logger.info(f"🎯 HARD FILTER: Pasal {pasal_number}")
                
                try:
                    # Search dengan filter
                    source_docs = self.vector_store.similarity_search(
                        question,
                        k=self.max_retrieval_docs,
                        filter=filter_dict
                    )
                    
                    if source_docs:
                        logger.info(f"✅ Hard filter found {len(source_docs)} documents")
                        # Create context dan generate answer
                        context = "\n\n".join([doc.page_content for doc in source_docs])
                        
                        enhanced_prompt = """Anda adalah asisten hukum Indonesia yang sangat ahli dalam UUD 1945. 
Anda memiliki kemampuan luar biasa untuk menganalisis pertanyaan secara mendalam dan memberikan jawaban yang sangat relevan, akurat, dan komprehensif.

INSTRUKSI KHUSUS UNTUK PENCARIAN PASAL/AYAT:
- Pengguna mencari informasi spesifik tentang pasal/ayat tertentu
- Berikan jawaban yang fokus dan tepat sasaran
- Jangan terlalu panjang, tapi pastikan informatif dan akurat

STRUKTUR JAWABAN:
📋 **JAWABAN LANGSUNG**: Langsung sampaikan isi pasal/ayat yang dicari

📖 **ISI LENGKAP**: Kutip bunyi lengkap pasal/ayat dari dokumen

📝 **PENJELASAN SINGKAT**: 
- Jelaskan makna dan tujuan ketentuan tersebut
- Bagaimana hal ini berkaitan dengan sistem ketatanegaraan

💡 **RELEVANSI**: Mengapa pasal/ayat ini penting dalam UUD 1945

DOKUMEN UUD 1945:
{context}

PERTANYAAN: {question}

JAWABAN AKURAT DAN TERFOKUS:"""
                        
                        prompt = enhanced_prompt.format(context=context, question=question)
                        answer = self.llm.invoke(prompt)
                        
                    else:
                        logger.warning("⚠️ Hard filter found no documents, checking if legal context...")
                        # Check if question is legal context
                        if self._is_legal_context(question):
                            logger.info("📚 Legal context detected, using Gemma2:2b fallback")
                            answer = self._generate_fallback_answer(question)
                            source_docs = []  # No source docs for fallback
                        else:
                            logger.info("🚫 Non-legal context, providing standard response")
                            answer = f'Maaf "{question}" kami tidak bisa menjawab nya karena di luar konteks Hukum di Indonesia. Tanyakan terkait UUD pasti kami bisa menjawab nya.'
                            source_docs = []
                        
                except Exception as e:
                    logger.warning(f"⚠️ Hard filter failed: {str(e)}, trying fallback")
                    # Check if question is legal context for fallback
                    if self._is_legal_context(question):
                        logger.info("📚 Legal context detected, using Gemma2:2b fallback")
                        answer = self._generate_fallback_answer(question)
                        source_docs = []
                    else:
                        logger.info("🚫 Non-legal context, providing standard response")
                        answer = f'Maaf "{question}" kami tidak bisa menjawab nya karena di luar konteks Hukum di Indonesia. Tanyakan terkait UUD pasti kami bisa menjawab nya.'
                        source_docs = []
            else:
                # Regular search untuk pertanyaan hukum umum (tanpa pasal spesifik)
                logger.info("🔍 Performing regular search for legal question")
                
                # Try regular QA first to see if there are relevant documents
                result = self.qa_chain.invoke({"query": question})
                source_docs = result.get("source_documents", [])
                
                # Check if documents found are actually relevant
                if source_docs and len(source_docs) > 0:
                    # Use documents found
                    answer = result["result"]
                    logger.info(f"✅ Found {len(source_docs)} relevant documents in database")
                else:
                    # No relevant documents, use Gemma2:2b fallback for legal questions
                    logger.info("� No relevant documents, using Gemma2:2b fallback for legal question")
                    answer = self._generate_fallback_answer(question)
                    source_docs = []
            
            # Process source details
            source_details = []
            for doc in source_docs:
                metadata = doc.metadata
                source_details.append({
                    "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                    "pasal_number": metadata.get("pasal_number"),
                    "ayat_number": metadata.get("ayat_number"),
                    "bab_number": metadata.get("bab_number"),
                    "bab_title": metadata.get("bab_title"),
                    "page_number": metadata.get("page_number"),
                    "authority_level": metadata.get("authority_level", "high"),
                    "source_file": metadata.get("source_file"),
                    "confidence_score": metadata.get("chunk_quality_score", 85.0)
                })
            
            processing_time = time.time() - start_time
            confidence = self._calculate_enhanced_confidence(source_docs, answer, question)
            
            logger.info(f"✅ Query processed in {processing_time:.2f}s with confidence {confidence}%")
            
            return QueryResult(
                answer=answer,
                success=True,
                processing_time=round(processing_time, 2),
                sources_count=len(source_docs),
                source_details=source_details,
                confidence=confidence,
                method="langchain"
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
                method="langchain",
                error=str(e)
            )
    
    def _calculate_confidence(self, answer: str, sources: List[Document], question: str) -> float:
        """Calculate confidence score berdasarkan berbagai faktor"""
        
        if not sources:
            return 10.0
        
        # Base confidence dari jumlah sources (max 20 points)
        source_score = min(len(sources) * 4, 20)
        
        # Authority score berdasarkan metadata (max 25 points)
        authority_score = 0
        for doc in sources:
            metadata = doc.metadata
            if metadata.get("authority_level") == "highest":
                authority_score += 5
            elif metadata.get("authority_level") == "high":
                authority_score += 3
            else:
                authority_score += 1
        authority_score = min(authority_score, 25)
        
        # Exact match score untuk pasal/ayat spesifik (max 25 points)
        exact_match_score = 0
        question_lower = question.lower()
        if "pasal" in question_lower:
            for doc in sources:
                metadata = doc.metadata
                if metadata.get("pasal_number"):
                    exact_match_score += 8
                    if metadata.get("ayat_number"):
                        exact_match_score += 5
                    break
        exact_match_score = min(exact_match_score, 25)
        
        # Answer quality score (max 20 points)
        answer_quality = min(len(answer) / 100 * 10, 20)
        
        # Content relevance (max 10 points)
        content_relevance = 10 if any("UUD" in doc.page_content or "pasal" in doc.page_content.lower() for doc in sources) else 5
        
        total_confidence = source_score + authority_score + exact_match_score + answer_quality + content_relevance
        
        return round(min(total_confidence, 100.0), 1)
    
    def health_check(self) -> Dict[str, Any]:
        """Health check untuk service"""
        try:
            # Test basic functionality
            test_result = self.query("Apa itu UUD 1945?")
            
            return {
                "status": "healthy",
                "service": "langchain",
                "vector_store_loaded": self.vector_store is not None,
                "llm_ready": self.llm is not None,
                "qa_chain_ready": self.qa_chain is not None,
                "test_query_success": test_result.success,
                "test_confidence": test_result.confidence
            }
        except Exception as e:
            return {
                "status": "unhealthy", 
                "service": "langchain",
                "error": str(e)
            }

# Global instance untuk reuse
_langchain_service = None

def get_langchain_service() -> LawChainLangChain:
    """Get atau create global LangChain service instance"""
    global _langchain_service
    
    if _langchain_service is None:
        _langchain_service = LawChainLangChain()
        if not _langchain_service.initialize():
            raise RuntimeError("Failed to initialize LangChain service")
    
    return _langchain_service

# Test function
if __name__ == "__main__":
    print("🧪 Testing LawChain LangChain Service...")
    print("=" * 60)
    
    service = LawChainLangChain()
    
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
