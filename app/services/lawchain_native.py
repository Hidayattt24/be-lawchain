"""
LawChain Native - Pengembangan Chatbot Hukum UUD 1945 Berbasis RAG Tanpa LangChain
Implementasi manual RAG pipeline dengan library dasar Python
"""

import os
import warnings
import time
import requests
import json
import numpy as np
import pickle
from typing import List, Dict, Any, Tuple
from datetime import datetime
import math
import re

# Handle OpenMP conflicts - set this BEFORE importing any libraries that use OpenMP
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'  # Limit threads to prevent conflicts

# Suppress warnings
warnings.filterwarnings("ignore")

# Core libraries (tanpa LangChain)
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import faiss


class SimpleTextSplitter:
    """Text splitter sederhana tanpa LangChain"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def split_text(self, text: str, metadata: Dict = None) -> List[Dict]:
        """Split text menjadi chunks dengan overlap"""
        chunks = []
        start = 0
        chunk_id = 0
        
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            
            # Cari boundary kata terdekat
            if end < len(text):
                while end > start and text[end] != ' ':
                    end -= 1
                if end == start:
                    end = start + self.chunk_size
            
            chunk_text = text[start:end].strip()
            if len(chunk_text) > 50:  # Skip chunks yang terlalu pendek
                chunk_data = {
                    'content': chunk_text,
                    'metadata': {
                        **(metadata or {}),
                        'chunk_id': chunk_id,
                        'start_pos': start,
                        'end_pos': end,
                        'chunk_size': len(chunk_text),
                        'created_at': datetime.now().isoformat()
                    }
                }
                chunks.append(chunk_data)
                chunk_id += 1
            
            start = max(start + self.chunk_size - self.chunk_overlap, start + 1)
        
        return chunks


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
                timeout=120  # Increased timeout for slower local processing
            )
            response.raise_for_status()
            return response.json()["embedding"]
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return [0.0] * 768  # Default dimension
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings untuk multiple documents"""
        embeddings = []
        for i, text in enumerate(texts):
            if i % 10 == 0:
                print(f"Processing embedding {i+1}/{len(texts)}")
            embeddings.append(self.embed_query(text))
            time.sleep(0.1)  # Rate limiting
        return embeddings


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
                timeout=300  # Increased timeout to 5 minutes for local LLM processing
            )
            response.raise_for_status()
            return response.json()["response"]
        except Exception as e:
            print(f"Error generating response: {e}")
            return "Maaf, terjadi error dalam menghasilkan jawaban."


class NativeFAISSVectorStore:
    """Vector store sederhana menggunakan FAISS"""
    
    def __init__(self, embedding_dimension: int = 768):
        self.embedding_dimension = embedding_dimension
        self.index = faiss.IndexFlatIP(embedding_dimension)  # Inner Product (cosine similarity)
        self.documents = []
        self.embeddings = []
    
    def add_documents(self, chunks: List[Dict], embeddings: List[List[float]]):
        """Tambahkan dokumen dan embeddings ke vector store"""
        self.documents.extend(chunks)
        self.embeddings.extend(embeddings)
        
        # Normalize embeddings untuk cosine similarity
        normalized_embeddings = []
        for emb in embeddings:
            emb_array = np.array(emb, dtype=np.float32)
            norm = np.linalg.norm(emb_array)
            if norm > 0:
                emb_array = emb_array / norm
            normalized_embeddings.append(emb_array)
        
        # Add to FAISS index
        embeddings_matrix = np.vstack(normalized_embeddings)
        self.index.add(embeddings_matrix)
    
    def similarity_search(self, query_embedding: List[float], k: int = 5) -> List[Dict]:
        """Cari dokumen yang paling mirip"""
        # Normalize query embedding
        query_array = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
        norm = np.linalg.norm(query_array)
        if norm > 0:
            query_array = query_array / norm
        
        # Search
        scores, indices = self.index.search(query_array, k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.documents):
                doc = self.documents[idx].copy()
                doc['similarity_score'] = float(score)
                results.append(doc)
        
        return results
    
    def get_all_documents(self) -> List[Dict]:
        """Get all stored documents untuk manual search"""
        return self.documents.copy()
    
    def save(self, filepath: str):
        """Simpan vector store"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, f"{filepath}.faiss")
        
        # Save documents and metadata
        with open(f"{filepath}.pkl", 'wb') as f:
            pickle.dump({
                'documents': self.documents,
                'embeddings': self.embeddings,
                'embedding_dimension': self.embedding_dimension
            }, f)
    
    def load(self, filepath: str):
        """Load vector store"""
        # Load FAISS index
        self.index = faiss.read_index(f"{filepath}.faiss")
        
        # Load documents and metadata
        with open(f"{filepath}.pkl", 'rb') as f:
            data = pickle.load(f)
            self.documents = data['documents']
            self.embeddings = data['embeddings']
            self.embedding_dimension = data['embedding_dimension']


class LawChainNative:
    """LawChain implementation tanpa LangChain"""
    
    def __init__(self):
        self.documents = []
        self.chunks = []
        self.vector_store = None
        self.embeddings_model = None
        self.llm = None
        
        # Konfigurasi
        self.chunk_size = 1000
        self.chunk_overlap = 200
        
        # Statistics
        self.total_documents = 0
        self.total_chunks = 0
        
        # Mapping PDF metadata dengan priority score
        self.pdf_metadata = {
            'UUD1945-BPHN.pdf': {
                'judul': 'UUD 1945 - Badan Pembinaan Hukum Nasional (BPHN)',
                'sumber': 'https://bphn.go.id/data/documents/uud_1945.pdf',
                'institusi': 'Badan Pembinaan Hukum Nasional',
                'priority_score': 95
            },
            'UUD1945-MKRI.pdf': {
                'judul': 'UUD 1945 Asli - Mahkamah Konstitusi RI (MKRI)',
                'sumber': 'https://www.mkri.id/public/content/infoumum/regulation/pdf/UUD45%20ASLI.pdf',
                'institusi': 'Mahkamah Konstitusi Republik Indonesia',
                'priority_score': 100
            },
            'UUD1945-MPR.pdf': {
                'judul': 'UUD 1945 - Majelis Permusyawaratan Rakyat (MPR)',
                'sumber': 'https://jdih.bapeten.go.id/unggah/dokumen/peraturan/4-full.pdf',
                'institusi': 'Majelis Permusyawaratan Rakyat',
                'priority_score': 90
            },
            'UUD1945.pdf': {
                'judul': 'UUD 1945 - Dewan Kehormatan Penyelenggara Pemilu (DKPP)',
                'sumber': 'https://dkpp.go.id/wp-content/uploads/2018/11/UUD-Nomor-Tahun-1945-UUD1945.pdf',
                'institusi': 'Dewan Kehormatan Penyelenggara Pemilu',
                'priority_score': 85
            },
            'UUD1945-BUKU.pdf': {
                'judul': 'UUD 1945 - Buku Panduan Lengkap MPR RI',
                'sumber': 'https://mpr.go.id/img/sosialisasi/file/1610334013_file_mpr.pdf',
                'institusi': 'Majelis Permusyawaratan Rakyat Republik Indonesia',
                'priority_score': 110
            }
        }
        
        print("🏛️ Selamat datang di LawChain Native - Chatbot UUD 1945 (Tanpa LangChain)")
        print("=" * 70)
    
    def validate_ollama_status(self):
        """Validasi status Ollama"""
        print("🔍 Memvalidasi status Ollama...")
        
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=3)
            if response.status_code != 200:
                raise Exception("API Ollama tidak responsif")
            
            models = response.json().get('models', [])
            required_models = ['gemma2:2b', 'nomic-embed-text']
            available_models = [model['name'] for model in models]
            
            missing_models = [model for model in required_models 
                             if not any(model in available for available in available_models)]
            
            if missing_models:
                raise Exception(f"Model tidak tersedia: {missing_models}")
            
            print("✅ Ollama dan model tersedia")
            return True
            
        except requests.exceptions.ConnectionError:
            raise Exception("❌ Ollama tidak berjalan atau tidak dapat diakses")
        except Exception as e:
            raise Exception(f"❌ Validasi Ollama gagal: {str(e)}")
    
    def load_documents(self, data_dir: str = "data"):
        """Load dokumen PDF menggunakan PyMuPDF"""
        print(f"📂 Memuat dokumen dari folder: {data_dir}")
        
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Folder {data_dir} tidak ditemukan!")
        
        pdf_files = [f for f in os.listdir(data_dir) if f.endswith('.pdf')]
        
        if not pdf_files:
            raise FileNotFoundError(f"Tidak ada file PDF ditemukan di folder {data_dir}")
        
        for pdf_file in pdf_files:
            file_path = os.path.join(data_dir, pdf_file)
            metadata = self.pdf_metadata.get(pdf_file, {'judul': pdf_file, 'priority_score': 70})
            
            # Priority indicator
            priority = metadata.get('priority_score', 70)
            if priority >= 110:
                priority_indicator = "🌟 PREMIUM"
            elif priority >= 100:
                priority_indicator = "⭐ EXCELLENT" 
            elif priority >= 95:
                priority_indicator = "✅ VERY GOOD"
            elif priority >= 90:
                priority_indicator = "👍 GOOD"
            else:
                priority_indicator = "📝 STANDARD"
            
            print(f"  📄 Memproses: {metadata['judul']} {priority_indicator}")
            print(f"      📁 File: {pdf_file}")
            print(f"      📊 Priority: {priority}/110")
            
            try:
                # Buka PDF dengan PyMuPDF
                doc = fitz.open(file_path)
                pages_text = []
                
                for page_num in range(len(doc)):
                    page = doc.load_page(page_num)
                    text = page.get_text()
                    
                    if text.strip():  # Skip empty pages
                        page_data = {
                            'content': text,
                            'metadata': {
                                'source_file': pdf_file,
                                'page': page_num,
                                'document_type': 'UUD 1945',
                                'judul': metadata['judul'],
                                'institusi': metadata.get('institusi', 'Unknown'),
                                'sumber_url': metadata.get('sumber', 'Unknown'),
                                'priority_score': priority,
                                'loaded_at': datetime.now().isoformat()
                            }
                        }
                        pages_text.append(page_data)
                
                doc.close()
                self.documents.extend(pages_text)
                print(f"      ✅ Berhasil memuat {len(pages_text)} halaman")
                
            except Exception as e:
                print(f"      ❌ Error memuat {pdf_file}: {str(e)}")
        
        self.total_documents = len(pdf_files)
        print(f"\n📊 Total dokumen dimuat: {self.total_documents}")
        print(f"📊 Total halaman: {len(self.documents)}")
    
    def split_documents(self):
        """Split dokumen menjadi chunks"""
        print("\n🔄 Membagi dokumen menjadi chunks...")
        
        text_splitter = SimpleTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        
        all_chunks = []
        for doc in self.documents:
            chunks = text_splitter.split_text(
                doc['content'], 
                doc['metadata']
            )
            all_chunks.extend(chunks)
        
        self.chunks = all_chunks
        self.total_chunks = len(self.chunks)
        
        print(f"✅ Berhasil membuat {self.total_chunks} chunks")
        print(f"📏 Ukuran chunk: {self.chunk_size} karakter")
        print(f"📏 Overlap: {self.chunk_overlap} karakter")
    
    def create_embeddings(self):
        """Buat embedding model"""
        print("\n🔮 Membuat embeddings dengan Ollama Native...")
        
        self.validate_ollama_status()
        
        try:
            self.embeddings_model = NativeOllamaEmbedding(
                model="nomic-embed-text",
                base_url="http://localhost:11434"
            )
            
            # Test embedding
            print("🧪 Testing embedding...")
            test_embedding = self.embeddings_model.embed_query("Test embedding")
            print(f"✅ Embedding berhasil dibuat (dimensi: {len(test_embedding)})")
            
        except Exception as e:
            print(f"❌ Error membuat embedding: {str(e)}")
            raise
    
    def create_vector_store(self):
        """Buat vector store dengan FAISS native"""
        from config.settings import settings
        vector_store_path = settings.VECTOR_STORE_NATIVE_PATH.replace("/", os.sep)
        
        # Pastikan direktori ada
        os.makedirs(vector_store_path, exist_ok=True)
        
        # Cek apakah vector store sudah ada (file dengan nama index.faiss dan index.pkl)
        faiss_file = os.path.join(vector_store_path, "index.faiss")
        pkl_file = os.path.join(vector_store_path, "index.pkl")
        
        if os.path.exists(faiss_file) and os.path.exists(pkl_file):
            print(f"\n📁 Vector store ditemukan di '{vector_store_path}'")
            print("🔄 Memuat vector store yang sudah ada...")
            
            try:
                self.vector_store = NativeFAISSVectorStore()
                # Load dengan path tanpa extension karena method load akan menambahkan .faiss dan .pkl
                base_path = os.path.join(vector_store_path, "index")
                self.vector_store.load(base_path)
                print("✅ Vector store berhasil dimuat dari cache")
                print("⚡ Proses lebih cepat karena menggunakan data yang sudah ada!")
                return
                
            except Exception as e:
                print(f"⚠️ Error memuat vector store: {str(e)}")
                print("🔄 Akan membuat vector store baru...")
        
        # Buat vector store baru
        print(f"\n🗄️ Membuat vector store baru dengan FAISS Native...")
        print(f"📊 Memproses {len(self.chunks)} chunks...")
        print("⏳ Estimasi waktu: 3-7 menit (hanya sekali)")
        
        if not self.chunks:
            raise ValueError("Tidak ada chunks untuk diproses!")
        
        try:
            start_time = time.time()
            
            # Generate embeddings untuk semua chunks
            chunk_texts = [chunk['content'] for chunk in self.chunks]
            print("🔮 Generating embeddings...")
            embeddings = self.embeddings_model.embed_documents(chunk_texts)
            
            # Buat vector store
            self.vector_store = NativeFAISSVectorStore()
            self.vector_store.add_documents(self.chunks, embeddings)
            
            elapsed_time = time.time() - start_time
            print(f"✅ Vector store berhasil dibuat dalam {elapsed_time:.1f} detik")
            
            # Simpan vector store
            base_path = os.path.join(vector_store_path, "index")
            self.vector_store.save(base_path)
            print(f"💾 Vector store disimpan ke '{vector_store_path}'")
            print("🎯 Selanjutnya akan menggunakan cache untuk startup yang lebih cepat!")
            
        except Exception as e:
            print(f"❌ Error membuat vector store: {str(e)}")
            raise
    
    def setup_llm(self):
        """Setup LLM dengan Ollama native"""
        print("\n🤖 Mengatur LLM dengan Ollama Native...")
        
        self.validate_ollama_status()
        
        try:
            self.llm = NativeOllamaLLM(
                model="gemma2:2b",
                base_url="http://localhost:11434",
                temperature=0.1
            )
            
            # Test LLM
            print("🧪 Testing LLM...")
            test_response = self.llm.generate("Halo")
            print(f"✅ LLM berhasil diatur dan diuji: {test_response[:50]}...")
            
        except Exception as e:
            print(f"❌ Error mengatur LLM: {str(e)}")
            raise
    
    def _validate_uud_context(self, question: str) -> Dict[str, Any]:
        """Validasi apakah pertanyaan berkaitan dengan UUD 1945"""
        
        # Kata kunci yang menunjukkan pertanyaan tentang UUD 1945
        uud_keywords = {
            'uud', 'undang-undang dasar', 'konstitusi', 'pasal', 'ayat', 'bab',
            'pancasila', 'bhinneka tunggal ika', 'nkri', 'republik indonesia',
            'presiden', 'wakil presiden', 'menteri', 'dpr', 'dpd', 'mpr',
            'mahkamah konstitusi', 'mahkamah agung', 'kekuasaan kehakiman',
            'kekuasaan eksekutif', 'kekuasaan legislatif', 'bpk', 'komisi yudisial',
            'hak asasi manusia', 'hak warga negara', 'kewajiban warga negara',
            'wilayah negara', 'lambang negara', 'bahasa negara', 'bendera negara',
            'lagu kebangsaan', 'ekonomi nasional', 'kesejahteraan sosial',
            'pendidikan nasional', 'pertahanan negara', 'keamanan negara'
        }
        
        # Kata kunci yang menunjukkan pertanyaan di luar konteks hukum
        non_legal_keywords = {
            'resep', 'masakan', 'memasak', 'kuliner', 'makanan', 'minuman',
            'olahraga', 'sepak bola', 'basket', 'game', 'permainan',
            'teknologi', 'programming', 'coding', 'komputer', 'software',
            'musik', 'lagu', 'artis', 'film', 'movie', 'entertainment',
            'fashion', 'gaya', 'trend', 'belanja', 'shopping',
            'wisata', 'travel', 'liburan', 'destinasi', 'pariwisata',
            'kesehatan', 'penyakit', 'obat', 'dokter', 'rumah sakit',
            'cuaca', 'ramalan cuaca', 'iklim', 'meteorologi',
            'matematika', 'fisika', 'kimia', 'biologi', 'science',
            'cerita', 'dongeng', 'novel', 'puisi', 'sastra'
        }
        
        question_lower = question.lower()
        
        # Cek apakah ada kata kunci non-legal yang jelas
        non_legal_found = any(keyword in question_lower for keyword in non_legal_keywords)
        
        if non_legal_found:
            return {
                'is_relevant': False,
                'response': f"Maaf, pertanyaan '{question}' yang Anda berikan berada di luar konteks sistem AI LawChain ini. AI ini secara khusus dirancang untuk membantu memberikan informasi dan analisis mengenai Undang-Undang Dasar 1945 (UUD 1945) Republik Indonesia. Silakan ajukan pertanyaan yang berkaitan dengan konstitusi, pasal-pasal UUD 1945, hak dan kewajiban warga negara, struktur pemerintahan, atau aspek hukum konstitusional lainnya."
            }
        
        # Cek apakah ada kata kunci UUD yang relevan
        uud_score = sum(1 for keyword in uud_keywords if keyword in question_lower)
        
        # Jika tidak ada kata kunci UUD sama sekali, lakukan pengecekan lebih lanjut
        if uud_score == 0:
            # Cek apakah pertanyaan memiliki pola yang menunjukkan pertanyaan hukum
            legal_patterns = [
                'hukum', 'aturan', 'peraturan', 'ketentuan', 'kewenangan', 'tugas',
                'fungsi', 'wewenang', 'tanggung jawab', 'kekuasaan', 'kedaulatan',
                'negara', 'pemerintah', 'pemerintahan', 'nasional', 'kebangsaan'
            ]
            
            pattern_score = sum(1 for pattern in legal_patterns if pattern in question_lower)
            
            # Jika tidak ada indikasi hukum sama sekali
            if pattern_score == 0:
                return {
                    'is_relevant': False,
                    'response': f"Maaf, pertanyaan '{question}' yang Anda berikan tampaknya berada di luar konteks sistem AI LawChain ini. AI ini secara khusus berfokus pada Undang-Undang Dasar 1945 (UUD 1945) dan aspek konstitusional Republik Indonesia. Mohon ajukan pertanyaan yang berkaitan dengan UUD 1945, seperti tentang pasal-pasal tertentu, struktur pemerintahan, hak dan kewajiban warga negara, atau prinsip-prinsip konstitusional lainnya."
                }
        
        # Jika lolos validasi, pertanyaan dianggap relevan
        return {
            'is_relevant': True,
            'response': None
        }
    
    def calculate_comprehensive_metrics(self, query: str, retrieved_docs: List, answer: str) -> Dict[str, float]:
        """Menghitung metrik akurasi komprehensif (sama seperti LangChain version)"""
        
        # 1. Semantic Similarity Score
        semantic_score = self._calculate_semantic_similarity(query, retrieved_docs)
        
        # 2. Content Coverage Score 
        coverage_score = self._calculate_content_coverage(query, retrieved_docs)
        
        # 3. Answer Relevance Score
        answer_relevance = self._calculate_answer_relevance(query, answer)
        
        # 4. Source Quality Score
        source_quality = self._calculate_source_quality(retrieved_docs)
        
        # 5. Legal Context Score
        legal_context = self._calculate_legal_context_score(query, answer)
        
        # 6. Answer Completeness Score
        completeness = self._calculate_answer_completeness(query, answer)
        
        # 7. Overall Confidence Score
        confidence_score = (
            semantic_score * 0.20 + 
            coverage_score * 0.15 + 
            answer_relevance * 0.25 + 
            source_quality * 0.15 +
            legal_context * 0.15 +
            completeness * 0.10
        )
        
        # 8. Estimated Accuracy
        estimated_accuracy = self._calculate_estimated_accuracy(
            semantic_score, coverage_score, answer_relevance, 
            source_quality, legal_context, completeness
        )
        
        return {
            'semantic_similarity': semantic_score,
            'content_coverage': coverage_score, 
            'answer_relevance': answer_relevance,
            'source_quality': source_quality,
            'legal_context': legal_context,
            'answer_completeness': completeness,
            'confidence_score': confidence_score,
            'estimated_accuracy': estimated_accuracy
        }
    
    def _calculate_semantic_similarity(self, query: str, retrieved_docs: List) -> float:
        """Hitung semantic similarity menggunakan embedding"""
        if not retrieved_docs:
            return 0.0
        
        try:
            # Embed query
            query_embedding = self.embeddings_model.embed_query(query)
            
            # Hitung similarity dengan retrieved docs
            similarities = []
            for doc in retrieved_docs:
                # Gunakan similarity score yang sudah dihitung FAISS
                if 'similarity_score' in doc:
                    # Convert dari inner product ke percentage
                    similarity = max(0, doc['similarity_score'] * 100)
                    similarities.append(similarity)
            
            return sum(similarities) / len(similarities) if similarities else 50.0
            
        except Exception:
            return 50.0
    
    def _calculate_content_coverage(self, query: str, retrieved_docs: List) -> float:
        """Hitung coverage konten"""
        if not retrieved_docs:
            return 0.0
        
        important_words = self._extract_key_terms(query)
        if not important_words:
            return 50.0
        
        total_coverage = 0
        for doc in retrieved_docs:
            doc_words = set(doc['content'].lower().split())
            coverage = len([word for word in important_words if word in doc_words])
            total_coverage += coverage / len(important_words)
        
        return min((total_coverage / len(retrieved_docs)) * 100, 100)
    
    def _calculate_answer_relevance(self, query: str, answer: str) -> float:
        """Hitung relevansi jawaban"""
        if not answer or len(answer.strip()) < 10:
            return 0.0
        
        query_terms = set(self._extract_key_terms(query))
        answer_terms = set(answer.lower().split())
        
        if query_terms:
            overlap = len(query_terms.intersection(answer_terms))
            relevance = (overlap / len(query_terms)) * 70
        else:
            relevance = 30
        
        # Bonus untuk konteks hukum
        context_indicators = ['pasal', 'bab', 'uud', 'ayat', 'hak', 'kewajiban', 'republik', 'indonesia']
        context_score = len([term for term in context_indicators if term in answer.lower()])
        context_bonus = min(context_score * 5, 30)
        
        return min(relevance + context_bonus, 100)
    
    def _calculate_source_quality(self, retrieved_docs: List) -> float:
        """Hitung kualitas sumber dengan priority score"""
        if not retrieved_docs:
            return 0.0
        
        quality_score = 0
        for doc in retrieved_docs:
            source_file = doc['metadata'].get('source_file', '')
            
            if source_file in self.pdf_metadata:
                priority = self.pdf_metadata[source_file]['priority_score']
                quality_score += priority
            else:
                quality_score += 70
        
        avg_quality = quality_score / len(retrieved_docs)
        return min(avg_quality, 100)
    
    def _calculate_legal_context_score(self, query: str, answer: str) -> float:
        """Hitung konteks hukum"""
        if not answer:
            return 0.0
        
        legal_keywords = {
            'pasal', 'ayat', 'bab', 'uud', 'undang-undang', 'dasar', 'konstitusi',
            'hak', 'kewajiban', 'warga', 'negara', 'republik', 'indonesia',
            'presiden', 'dpr', 'mpr', 'mahkamah', 'pancasila', 'bhinneka'
        }
        
        answer_words = set(answer.lower().split())
        legal_context_count = len([word for word in legal_keywords if word in answer_words])
        
        base_score = min(legal_context_count * 8, 80)
        
        # Bonus untuk referensi spesifik
        pasal_references = len(re.findall(r'pasal\s+\d+', answer.lower()))
        bab_references = len(re.findall(r'bab\s+[ivx]+', answer.lower()))
        
        reference_bonus = min((pasal_references * 10) + (bab_references * 5), 20)
        
        return min(base_score + reference_bonus, 100)
    
    def _calculate_answer_completeness(self, query: str, answer: str) -> float:
        """Hitung kelengkapan jawaban"""
        if not answer:
            return 0.0
        
        answer_length = len(answer)
        if answer_length < 50:
            length_score = 20
        elif answer_length < 150:
            length_score = 50
        elif answer_length < 300:
            length_score = 75
        else:
            length_score = 90
        
        structure_indicators = [
            'yaitu', 'adalah', 'antara lain', 'contoh', 'misalnya',
            'pertama', 'kedua', 'ketiga', 'selain itu', 'namun'
        ]
        structure_count = len([ind for ind in structure_indicators if ind in answer.lower()])
        structure_score = min(structure_count * 10, 30)
        
        uncertainty_phrases = ['tidak tahu', 'tidak dapat', 'tidak ada informasi', 'maaf']
        if any(phrase in answer.lower() for phrase in uncertainty_phrases):
            uncertainty_penalty = -20
        else:
            uncertainty_penalty = 0
        
        return max(min(length_score * 0.7 + structure_score + uncertainty_penalty, 100), 0)
    
    def _calculate_estimated_accuracy(self, semantic: float, coverage: float, 
                                     relevance: float, quality: float, 
                                     legal: float, completeness: float) -> float:
        """Hitung estimasi akurasi"""
        
        base_accuracy = (
            semantic * 0.25 +
            coverage * 0.20 +
            relevance * 0.25 +
            quality * 0.15 +
            legal * 0.10 +
            completeness * 0.05
        )
        
        # Bonus berdasarkan kualitas sumber
        if quality >= 110:
            source_bonus = 8.0
        elif quality >= 100:
            source_bonus = 5.0
        elif quality >= 95:
            source_bonus = 3.0
        elif quality >= 90:
            source_bonus = 2.0
        else:
            source_bonus = 0.0
        
        adjusted_accuracy = base_accuracy + source_bonus
        
        if adjusted_accuracy >= 85:
            final_accuracy = min(adjusted_accuracy * 1.05, 97)
        elif adjusted_accuracy >= 70:
            final_accuracy = adjusted_accuracy * 1.02
        elif adjusted_accuracy >= 50:
            final_accuracy = adjusted_accuracy * 0.98
        else:
            final_accuracy = adjusted_accuracy * 0.95
        
        return min(max(final_accuracy, 10), 97)
    
    def _extract_key_terms(self, text: str) -> List[str]:
        """Ekstrak kata kunci penting"""
        legal_keywords = {
            'hak', 'asasi', 'manusia', 'kewajiban', 'pasal', 'bab', 'ayat',
            'presiden', 'dpr', 'mpr', 'mahkamah', 'konstitusi', 'pancasila',
            'negara', 'rakyat', 'bangsa', 'indonesia', 'demokrasi', 'hukum'
        }
        
        words = [word.lower().strip('.,!?:;()[]') for word in text.split()]
        important_words = []
        
        for word in words:
            if word in legal_keywords or (len(word) > 4 and word.isalpha()):
                important_words.append(word)
        
        return list(set(important_words))
    
    def hybrid_search(self, question: str, k: int = 5) -> List[Dict]:
        """
        Hybrid search: Kombinasi keyword search untuk struktur hukum (Pasal, Bab) 
        dengan semantic search untuk konten umum
        """
        try:
            # Deteksi query struktural (Pasal, Bab, dll)
            structural_patterns = [
                r'\bpasal\s+(\d+[a-z]*)\b',
                r'\bbab\s+([IVXLCDM]+)\b', 
                r'\bpasal\s+(\w+)\b',
                r'\bbagian\s+(\w+)\b'
            ]
            
            is_structural = any(re.search(pattern, question.lower()) for pattern in structural_patterns)
            
            if is_structural:
                print("🔍 Detected structural query, using keyword search...")
                
                # Extract structural terms dari query
                structural_terms = []
                for pattern in structural_patterns:
                    matches = re.findall(pattern, question.lower())
                    structural_terms.extend(matches)
                
                # Keyword search di vector store
                keyword_results = []
                if hasattr(self.vector_store, 'search_by_keywords'):
                    keyword_results = self.vector_store.search_by_keywords(structural_terms, k=k)
                else:
                    # Fallback: manual search through stored documents
                    all_docs = self.vector_store.get_all_documents()
                    for doc in all_docs:
                        content_lower = doc['content'].lower()
                        if any(term in content_lower for term in structural_terms):
                            keyword_results.append(doc)
                
                if keyword_results:
                    print(f"✅ Found {len(keyword_results)} results via keyword search")
                    return keyword_results[:k]
            
            # Fallback ke semantic search
            print("🔍 Using semantic search...")
            query_embedding = self.embeddings_model.embed_query(question)
            semantic_results = self.vector_store.similarity_search(query_embedding, k=k)
            
            print(f"✅ Found {len(semantic_results)} results via semantic search")
            return semantic_results
            
        except Exception as e:
            print(f"❌ Hybrid search error: {str(e)}")
            # Fallback ke semantic search
            query_embedding = self.embeddings_model.embed_query(question)
            return self.vector_store.similarity_search(query_embedding, k=k)
    
    def ask_question_with_custom_qa(self, question: str) -> Dict[str, Any]:
        """
        Custom QA method menggunakan hybrid search dan direct LLM invocation
        (mirip dengan implementasi LangChain yang sudah berhasil)
        """
        if not self.vector_store or not self.llm:
            raise ValueError("Sistem belum diinisialisasi!")
        
        print(f"\n❓ Pertanyaan: {question}")
        
        try:
            # Validasi konteks UUD 1945 terlebih dahulu
            context_validation = self._validate_uud_context(question)
            if not context_validation['is_relevant']:
                print("❌ Pertanyaan di luar konteks UUD 1945")
                return {
                    'pertanyaan': question,
                    'jawaban': context_validation['response'],
                    'metrics': {
                        'semantic_similarity': 0.0,
                        'content_coverage': 0.0,
                        'answer_relevance': 0.0,
                        'source_quality': 0.0,
                        'legal_context': 0.0,
                        'answer_completeness': 0.0,
                        'confidence_score': 0.0,
                        'estimated_accuracy': 0.0
                    },
                    'jumlah_sumber': 0,
                    'sumber_dokumen': [],
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'out_of_context': True
                }
            
            # Validasi Ollama
            print("🔍 Validating Ollama...")
            try:
                self.validate_ollama_status()
                print("✅ Ollama status OK")
            except Exception as e:
                print(f"❌ Validasi gagal: {str(e)}")
                raise ValueError(f"Ollama validation failed: {str(e)}")
            
            print("🔍 Mencari jawaban dengan hybrid search...")
            
            # HYBRID RETRIEVAL: Gunakan hybrid search
            print("📋 Using hybrid search for document retrieval...")
            try:
                retrieved_docs = self.hybrid_search(question, k=5)
                print(f"✅ Found {len(retrieved_docs)} relevant documents")
            except Exception as e:
                print(f"❌ Hybrid search failed: {str(e)}")
                raise ValueError(f"Hybrid search failed: {str(e)}")
            
            # AUGMENTATION: Buat context dari retrieved docs
            print("📝 Building context...")
            try:
                context_parts = []
                for i, doc in enumerate(retrieved_docs, 1):
                    context_parts.append(f"[Dokumen {i}]:\n{doc['content']}")
                
                context = "\n\n".join(context_parts)
                print(f"✅ Context built with {len(context)} characters")
            except Exception as e:
                print(f"❌ Context building failed: {str(e)}")
                raise ValueError(f"Context building failed: {str(e)}")
            
            # GENERATION: Improved prompt dengan explicit context instructions
            print("🤖 Generating answer with custom QA...")
            try:
                prompt_template = """
Anda adalah asisten hukum profesional yang bertugas memberikan jawaban berbasis dokumen hukum Indonesia khususnya UUD 1945.

INSTRUKSI LENGKAP:
1. Analisis konteks dokumen dengan teliti
2. Berikan jawaban yang akurat berdasarkan informasi yang tersedia
3. WAJIB menyertakan referensi pasal/ayat yang spesifik jika tersedia
4. Gunakan bahasa Indonesia formal dan profesional
5. Struktur jawaban dengan jelas dan sistematis
6. Jika informasi tidak lengkap, sampaikan keterbatasan tersebut

FORMAT JAWABAN:
- Mulai dengan penjelasan umum konsep yang ditanyakan
- Cantumkan dasar hukum (pasal/ayat) yang relevan dengan format: "Pasal [nomor] ayat ([nomor])"
- Berikan analisis dan interpretasi
- Tutup dengan kesimpulan yang jelas

KONTEKS LENGKAP DARI DOKUMEN UUD 1945:
{context}

PERTANYAAN YANG HARUS DIJAWAB SECARA DETAIL: {question}

JAWABAN TERSTRUKTUR (berdasarkan konteks UUD 1945):
"""
                
                prompt = prompt_template.format(context=context, question=question)
                answer = self.llm.generate(prompt)
                print(f"✅ Answer generated with {len(answer)} characters")
            except Exception as e:
                print(f"❌ Answer generation failed: {str(e)}")
                raise ValueError(f"Answer generation failed: {str(e)}")
            
            # EVALUATION: Hitung metrik
            print("📊 Calculating metrics...")
            try:
                metrics = self.calculate_comprehensive_metrics(question, retrieved_docs, answer)
                print("✅ Metrics calculated successfully")
            except Exception as e:
                print(f"⚠️ Metrics calculation failed: {str(e)}")
                # Use default metrics if calculation fails
                metrics = {
                    'semantic_similarity': 50.0,
                    'content_coverage': 50.0,
                    'answer_relevance': 50.0,
                    'source_quality': 70.0,
                    'legal_context': 50.0,
                    'answer_completeness': 50.0,
                    'confidence_score': 50.0,
                    'estimated_accuracy': 50.0
                }
            
            # Format sumber dokumen
            sources = []
            for i, doc in enumerate(retrieved_docs):
                metadata = doc['metadata']
                source_file = metadata.get('source_file', 'Unknown')
                
                pdf_meta = self.pdf_metadata.get(source_file, {
                    'judul': source_file,
                    'sumber': 'Tidak diketahui',
                    'institusi': 'Tidak diketahui',
                    'priority_score': 70
                })
                
                sources.append({
                    'dokumen': source_file,
                    'judul': pdf_meta['judul'],
                    'sumber_url': pdf_meta['sumber'],
                    'institusi': pdf_meta['institusi'],
                    'priority_score': pdf_meta['priority_score'],
                    'halaman': str(metadata.get('page', 'Unknown')),
                    'chunk_id': metadata.get('chunk_id', i),
                    'similarity_score': doc.get('similarity_score', 0.0),
                    'preview': doc['content'][:150] + "..." if len(doc['content']) > 150 else doc['content']
                })
            
            response = {
                'pertanyaan': question,
                'jawaban': answer,
                'metrics': metrics,
                'jumlah_sumber': len(sources),
                'sumber_dokumen': sources,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'method': 'Native RAG dengan Hybrid Search'
            }
            
            return response
            
        except Exception as e:
            print(f"❌ Error saat memproses pertanyaan: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
    
    def ask_question(self, question: str) -> Dict[str, Any]:
        """Proses pertanyaan dengan RAG pipeline native dengan error handling"""
        if not self.vector_store or not self.llm:
            raise ValueError("Sistem belum diinisialisasi!")
        
        print(f"\n❓ Pertanyaan: {question}")
        
        try:
            # Validasi Ollama dengan timeout
            print("🔍 Validating Ollama...")
            try:
                self.validate_ollama_status()
                print("✅ Ollama status OK")
            except Exception as e:
                print(f"❌ Validasi gagal: {str(e)}")
                raise ValueError(f"Ollama validation failed: {str(e)}")
            
            print("🔍 Mencari jawaban...")
            
            # RETRIEVAL: Cari dokumen relevan dengan timeout protection
            print("📋 Retrieving relevant documents...")
            try:
                query_embedding = self.embeddings_model.embed_query(question)
                retrieved_docs = self.vector_store.similarity_search(query_embedding, k=5)
                print(f"✅ Found {len(retrieved_docs)} relevant documents")
            except Exception as e:
                print(f"❌ Document retrieval failed: {str(e)}")
                raise ValueError(f"Document retrieval failed: {str(e)}")
            
            # AUGMENTATION: Buat context dari retrieved docs
            print("📝 Building context...")
            try:
                context_parts = []
                for i, doc in enumerate(retrieved_docs, 1):
                    context_parts.append(f"[Dokumen {i}]:\n{doc['content']}")
                
                context = "\n\n".join(context_parts)
                print(f"✅ Context built with {len(context)} characters")
            except Exception as e:
                print(f"❌ Context building failed: {str(e)}")
                raise ValueError(f"Context building failed: {str(e)}")
            
            # GENERATION: Buat prompt dan generate jawaban
            print("🤖 Generating answer...")
            try:
                prompt_template = """
Anda adalah asisten hukum profesional yang bertugas memberikan jawaban berbasis dokumen hukum Indonesia khususnya UUD 1945.

INSTRUKSI LENGKAP:
1. Analisis konteks dokumen dengan teliti
2. Berikan jawaban yang akurat berdasarkan informasi yang tersedia
3. WAJIB menyertakan referensi pasal/ayat yang spesifik jika tersedia
4. Gunakan bahasa Indonesia formal dan profesional
5. Struktur jawaban dengan jelas dan sistematis
6. Jika informasi tidak lengkap, sampaikan keterbatasan tersebut

FORMAT JAWABAN:
- Mulai dengan penjelasan umum konsep yang ditanyakan
- Cantumkan dasar hukum (pasal/ayat) yang relevan dengan format: "Pasal [nomor] ayat ([nomor])"
- Berikan analisis dan interpretasi
- Tutup dengan kesimpulan yang jelas

KONTEKS DOKUMEN:
{context}

PERTANYAAN: {question}

JAWABAN TERSTRUKTUR (dalam bahasa Indonesia):
"""
                
                prompt = prompt_template.format(context=context, question=question)
                answer = self.llm.generate(prompt)
                print(f"✅ Answer generated with {len(answer)} characters")
            except Exception as e:
                print(f"❌ Answer generation failed: {str(e)}")
                raise ValueError(f"Answer generation failed: {str(e)}")
            
            # EVALUATION: Hitung metrik
            print("📊 Calculating metrics...")
            try:
                metrics = self.calculate_comprehensive_metrics(question, retrieved_docs, answer)
                print("✅ Metrics calculated successfully")
            except Exception as e:
                print(f"⚠️ Metrics calculation failed: {str(e)}")
                # Use default metrics if calculation fails
                metrics = {
                    'semantic_similarity': 50.0,
                    'content_coverage': 50.0,
                    'answer_relevance': 50.0,
                    'source_quality': 70.0,
                    'legal_context': 50.0,
                    'answer_completeness': 50.0,
                    'confidence_score': 50.0,
                    'estimated_accuracy': 50.0
                }
            
            # Format sumber dokumen
            sources = []
            for i, doc in enumerate(retrieved_docs):
                metadata = doc['metadata']
                source_file = metadata.get('source_file', 'Unknown')
                
                pdf_meta = self.pdf_metadata.get(source_file, {
                    'judul': source_file,
                    'sumber': 'Tidak diketahui',
                    'institusi': 'Tidak diketahui',
                    'priority_score': 70
                })
                
                sources.append({
                    'dokumen': source_file,
                    'judul': pdf_meta['judul'],
                    'sumber_url': pdf_meta['sumber'],
                    'institusi': pdf_meta['institusi'],
                    'priority_score': pdf_meta['priority_score'],
                    'halaman': str(metadata.get('page', 'Unknown')),  # Ensure string type
                    'chunk_id': metadata.get('chunk_id', i),
                    'similarity_score': doc.get('similarity_score', 0.0),
                    'preview': doc['content'][:150] + "..." if len(doc['content']) > 150 else doc['content']
                })
            
            response = {
                'pertanyaan': question,
                'jawaban': answer,
                'metrics': metrics,
                'jumlah_sumber': len(sources),
                'sumber_dokumen': sources,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'method': 'Native RAG (Tanpa LangChain)'
            }
            
            return response
            
        except Exception as e:
            print(f"❌ Error saat memproses pertanyaan: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
    
    def display_response(self, response: Dict[str, Any]):
        """Display response dengan format yang sama seperti LangChain version"""
        metrics = response['metrics']
        
        print(f"\n{'=' * 70}")
        print(f"🤖 JAWABAN LAWCHAIN NATIVE")
        print(f"{'=' * 70}")
        print(f"📝 Pertanyaan: {response['pertanyaan']}")
        print(f"🔧 Method: {response['method']}")
        print(f"\n💬 Jawaban:")
        print(f"{response['jawaban']}")
        
        # Kategorisasi akurasi
        accuracy = metrics['estimated_accuracy']
        if accuracy >= 90:
            accuracy_level = "🟢 SANGAT TINGGI"
            accuracy_emoji = "🎯"
        elif accuracy >= 80:
            accuracy_level = "🟡 TINGGI"
            accuracy_emoji = "👍"
        elif accuracy >= 70:
            accuracy_level = "🟠 SEDANG"
            accuracy_emoji = "⚠️"
        elif accuracy >= 60:
            accuracy_level = "🔴 RENDAH"
            accuracy_emoji = "⚠️"
        else:
            accuracy_level = "🔴 SANGAT RENDAH"
            accuracy_emoji = "❌"
        
        print(f"\n{'=' * 70}")
        print(f"📊 ANALISIS AKURASI & KUALITAS JAWABAN (NATIVE RAG)")
        print(f"{'=' * 70}")
        print(f"🎯 TINGKAT AKURASI ESTIMASI: {accuracy:.1f}% {accuracy_level}")
        print(f"🎓 CONFIDENCE SCORE: {metrics['confidence_score']:.1f}%")
        
        print(f"\n📈 BREAKDOWN METRIK DETAIL:")
        print(f"   🔍 Semantic Similarity: {metrics['semantic_similarity']:.1f}% - Kemiripan makna dengan dokumen")
        print(f"   📋 Content Coverage: {metrics['content_coverage']:.1f}% - Cakupan konten relevan")
        print(f"   💡 Answer Relevance: {metrics['answer_relevance']:.1f}% - Relevansi jawaban dengan pertanyaan") 
        print(f"   📚 Source Quality: {metrics['source_quality']:.1f}% - Kualitas sumber dokumen")
        print(f"   ⚖️ Legal Context: {metrics['legal_context']:.1f}% - Penggunaan konteks hukum")
        print(f"   ✅ Answer Completeness: {metrics['answer_completeness']:.1f}% - Kelengkapan jawaban")
        
        # Interpretasi hasil
        print(f"\n🔬 INTERPRETASI HASIL:")
        if accuracy >= 85:
            print(f"   {accuracy_emoji} Jawaban sangat dapat diandalkan dengan referensi yang kuat")
        elif accuracy >= 75:
            print(f"   {accuracy_emoji} Jawaban dapat diandalkan dengan sedikit verifikasi")
        elif accuracy >= 65:
            print(f"   {accuracy_emoji} Jawaban cukup baik namun disarankan verifikasi lebih lanjut")
        else:
            print(f"   {accuracy_emoji} Jawaban perlu verifikasi menyeluruh dengan sumber lain")
        
        print(f"\n📚 SUMBER REFERENSI ({response['jumlah_sumber']} dokumen):")
        
        if response['sumber_dokumen']:
            for i, source in enumerate(response['sumber_dokumen'], 1):
                priority = source.get('priority_score', 70)
                if priority >= 110:
                    quality_indicator = "🌟 PREMIUM"
                    quality_color = "🟨"
                elif priority >= 100:
                    quality_indicator = "⭐ EXCELLENT"
                    quality_color = "🟩"
                elif priority >= 95:
                    quality_indicator = "✅ VERY GOOD"
                    quality_color = "🟢"
                elif priority >= 90:
                    quality_indicator = "👍 GOOD"
                    quality_color = "🔵"
                else:
                    quality_indicator = "📝 STANDARD"
                    quality_color = "⚪"
                
                print(f"\n  📄 SUMBER {i}: {quality_color} {quality_indicator}")
                print(f"     🏛️ Judul: {source['judul']}")
                print(f"     🏢 Institusi: {source['institusi']}")
                print(f"     🔗 Link Sumber: {source['sumber_url']}")
                print(f"     📊 Priority Score: {priority}/110")
                print(f"     🎯 Similarity Score: {source.get('similarity_score', 0):.3f}")
                print(f"     📁 File: {source['dokumen']} (Halaman {source['halaman']})")
                print(f"     📝 Preview: {source['preview']}")
        
        print(f"\n⏰ Waktu Respons: {response['timestamp']}")
        print(f"{'=' * 70}")
        
        # Rekomendasi penggunaan
        print(f"💡 REKOMENDASI PENGGUNAAN:")
        if accuracy >= 90:
            print(f"   ✅ Dapat digunakan sebagai referensi utama")
        elif accuracy >= 80:
            print(f"   ✅ Dapat digunakan dengan sedikit cross-check")
        elif accuracy >= 70:
            print(f"   ⚠️ Gunakan sebagai starting point, perlu verifikasi")
        else:
            print(f"   ❌ Perlu verifikasi menyeluruh sebelum digunakan")
        
        print(f"{'=' * 70}")
    
    def initialize(self, force_rebuild_vectorstore=False):
        """Inisialisasi lengkap sistem RAG Native dengan error handling"""
        try:
            print("🔧 Starting Native RAG initialization...")
            
            # 1. Load documents
            print("📂 Step 1: Loading documents...")
            self.load_documents()
            
            # 2. Split documents
            print("🔄 Step 2: Splitting documents...")
            self.split_documents()
            
            # 3. Create embeddings
            print("🔮 Step 3: Creating embeddings...")
            self.create_embeddings()
            
            # 4. Create vector store
            print("🗄️ Step 4: Creating vector store...")
            if force_rebuild_vectorstore and os.path.exists("vector_store_native"):
                print("🔄 Force rebuild: Menghapus vector store lama...")
                import shutil
                shutil.rmtree("vector_store_native")
            
            self.create_vector_store()
            
            # 5. Setup LLM
            print("🤖 Step 5: Setting up LLM...")
            self.setup_llm()
            
            print(f"\n{'=' * 70}")
            print("🎉 LAWCHAIN NATIVE SIAP DIGUNAKAN!")
            print(f"{'=' * 70}")
            print(f"📊 Statistik:")
            print(f"   • Total dokumen: {self.total_documents}")
            print(f"   • Total halaman: {len(self.documents)}")
            print(f"   • Total chunks: {self.total_chunks}")
            print(f"   • Model LLM: gemma2:2b (Native)")
            print(f"   • Model Embedding: nomic-embed-text (Native)")
            print(f"   • Vector Store: FAISS (Native)")
            print(f"   • Framework: Custom RAG (Tanpa LangChain)")
            print(f"{'=' * 70}")
            
        except Exception as e:
            print(f"❌ Error inisialisasi Native: {str(e)}")
            print(f"❌ Error type: {type(e).__name__}")
            import traceback
            print(f"❌ Traceback: {traceback.format_exc()}")
            # Clean up on failure
            self.cleanup_on_error()
            raise
    
    def cleanup_on_error(self):
        """Clean up resources on initialization error"""
        try:
            if hasattr(self, 'vector_store') and self.vector_store:
                del self.vector_store
            if hasattr(self, 'embeddings_model') and self.embeddings_model:
                del self.embeddings_model
            if hasattr(self, 'llm') and self.llm:
                del self.llm
            # Force garbage collection
            import gc
            gc.collect()
        except:
            pass
    
    def run_interactive_chat(self):
        """Chat interaktif"""
        print("\n💬 Mode Chat Interaktif (Native RAG)")
        print("Ketik 'quit', 'exit', atau 'keluar' untuk mengakhiri")
        print("Ketik 'rebuild' untuk membangun ulang vector store")
        print("-" * 70)
        
        while True:
            try:
                question = input("\n🙋 Tanya UUD 1945 (Native): ").strip()
                
                if question.lower() in ['quit', 'exit', 'keluar', 'q']:
                    print("\n👋 Terima kasih telah menggunakan LawChain Native!")
                    break
                
                if question.lower() in ['rebuild', 'rebuild-vector']:
                    print("\n🔄 Membangun ulang vector store...")
                    self.initialize(force_rebuild_vectorstore=True)
                    print("✅ Vector store berhasil dibangun ulang!")
                    continue
                
                if not question:
                    print("❗ Silakan masukkan pertanyaan yang valid")
                    continue
                
                # Proses pertanyaan
                response = self.ask_question(question)
                self.display_response(response)
                
            except KeyboardInterrupt:
                print("\n\n👋 Chat dihentikan oleh user. Terima kasih!")
                break
            except Exception as e:
                print(f"\n❌ Error: {str(e)}")
                print("💡 Silakan coba lagi atau restart aplikasi")


def main():
    """Fungsi utama"""
    try:
        # Buat instance LawChain Native
        lawchain = LawChainNative()
        
        # Inisialisasi sistem
        lawchain.initialize()
        
        # Jalankan chat interaktif
        lawchain.run_interactive_chat()
        
    except Exception as e:
        print(f"\n❌ Error fatal: {str(e)}")
        print("💡 Pastikan:")
        print("   1. Ollama sudah berjalan")
        print("   2. Model gemma2:2b dan nomic-embed-text sudah di-pull")
        print("   3. Folder 'data' berisi file PDF UUD 1945")


if __name__ == "__main__":
    main()