"""
LawChain - Pengembangan Chatbot Hukum UUD 1945 Berbasis RAG dengan Model LLaMA
Versi Indonesia dengan Output Bahasa Indonesia
"""

import os
import warnings
import time
import requests
import math
from typing import List, Dict, Any
import json
from datetime import datetime

# Suppress warnings
warnings.filterwarnings("ignore")

# LangChain imports
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


class LawChainIndonesia:
    """Chatbot RAG untuk UUD 1945 dengan output berbahasa Indonesia"""
    
    def __init__(self):
        self.documents = []
        self.text_chunks = []
        self.vector_store = None
        self.llm = None
        self.embeddings = None
        self.qa_chain = None
        self.chunk_metadata = []
        
        # Konfigurasi yang lebih baik untuk dokumen hukum
        self.chunk_size = 800  # Lebih kecil untuk granularitas yang lebih baik
        self.chunk_overlap = 150  # Overlap yang lebih kecil tapi proporsional
        
        # Statistics
        self.total_documents = 0
        self.total_chunks = 0
        
        # Mapping nama file ke judul dan sumber
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
                'priority_score': 110  # Skor tertinggi karena paling komprehensif
            }
        }
        
        print("🏛️ Selamat datang di LawChain - Chatbot Hukum UUD 1945")
        print("=" * 60)
    
    def validate_ollama_status(self):
        """Validasi status Ollama sebelum operasi"""
        print("🔍 Memvalidasi status Ollama...")
        
        try:
            # Cek API Ollama
            response = requests.get("http://localhost:11434/api/tags", timeout=3)
            if response.status_code != 200:
                raise Exception("API Ollama tidak responsif")
            
            models = response.json().get('models', [])
            required_models = ['llama3.1:8b', 'nomic-embed-text']
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
        """Memuat dokumen PDF UUD 1945"""
        print(f"📂 Memuat dokumen dari folder: {data_dir}")
        
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Folder {data_dir} tidak ditemukan!")
        
        pdf_files = [f for f in os.listdir(data_dir) if f.endswith('.pdf')]
        
        if not pdf_files:
            raise FileNotFoundError(f"Tidak ada file PDF ditemukan di folder {data_dir}")
        
        for pdf_file in pdf_files:
            file_path = os.path.join(data_dir, pdf_file)
            metadata = self.pdf_metadata.get(pdf_file, {'judul': pdf_file, 'priority_score': 70})
            
            # Indikator priority
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
                loader = PyMuPDFLoader(file_path)
                docs = loader.load()
                
                # Tambahkan metadata untuk setiap dokumen
                for doc in docs:
                    doc.metadata.update({
                        'source_file': pdf_file,
                        'document_type': 'UUD 1945',
                        'loaded_at': datetime.now().isoformat(),
                        'judul': metadata['judul'],
                        'institusi': metadata.get('institusi', 'Unknown'),
                        'sumber_url': metadata.get('sumber', 'Unknown'),
                        'priority_score': metadata.get('priority_score', 70)
                    })
                
                self.documents.extend(docs)
                print(f"      ✅ Berhasil memuat {len(docs)} halaman")
                
            except Exception as e:
                print(f"      ❌ Error memuat {pdf_file}: {str(e)}")
        
        self.total_documents = len(pdf_files)
        print(f"\n📊 Total dokumen dimuat: {self.total_documents}")
        print(f"📊 Total halaman: {len(self.documents)}")
    
    def split_documents(self):
        """Membagi dokumen menjadi chunks yang lebih kecil"""
        print("\n🔄 Membagi dokumen menjadi chunks...")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            # Separators yang lebih cocok untuk dokumen hukum
            separators=[
                "\n\nPasal ",      # Separator antar pasal
                "\n\nBab ",        # Separator antar bab  
                "\n\nBagian ",     # Separator antar bagian
                "\n\nParagraf ",   # Separator antar paragraf
                "\n\n",            # Double newline
                "\n",              # Single newline
                ". ",              # Sentence ending
                " ",               # Space
                ""                 # Character level
            ]
        )
        
        self.text_chunks = text_splitter.split_documents(self.documents)
        
        # Simpan metadata untuk setiap chunk
        for i, chunk in enumerate(self.text_chunks):
            chunk.metadata.update({
                'chunk_id': i,
                'chunk_size': len(chunk.page_content),
                'created_at': datetime.now().isoformat()
            })
        
        self.total_chunks = len(self.text_chunks)
        print(f"✅ Berhasil membuat {self.total_chunks} chunks")
        print(f"📏 Ukuran chunk: {self.chunk_size} karakter")
        print(f"📏 Overlap: {self.chunk_overlap} karakter")
    
    def create_embeddings(self):
        """Membuat embeddings menggunakan Ollama dengan validasi"""
        print("\n🔮 Membuat embeddings dengan Ollama...")
        
        # Validasi Ollama dulu
        self.validate_ollama_status()
        
        try:
            self.embeddings = OllamaEmbeddings(
                model="nomic-embed-text",
                base_url="http://localhost:11434"
                # Note: timeout parameter removed as it's not supported in newer versions
            )
            
            # Test embedding dengan validasi ulang
            print("🧪 Testing embedding...")
            test_text = "Test embedding"
            test_embedding = self.embeddings.embed_query(test_text)
            print(f"✅ Embedding berhasil dibuat (dimensi: {len(test_embedding)})")
            
        except Exception as e:
            print(f"❌ Error membuat embedding: {str(e)}")
            print("💡 Pastikan Ollama berjalan dengan: ollama serve")
            print("💡 Dan model tersedia dengan: ollama pull nomic-embed-text")
            raise
    
    def create_vector_store(self):
        """Membuat atau memuat vector store dengan FAISS"""
        from config.settings import settings
        vector_store_path = settings.VECTOR_STORE_LANGCHAIN_PATH
        
        # Pastikan direktori ada
        os.makedirs(os.path.dirname(vector_store_path), exist_ok=True)
        
        # Cek apakah vector store sudah ada
        if os.path.exists(vector_store_path):
            print(f"\n📦 Vector store ditemukan di '{vector_store_path}'")
            print("🔄 Memuat vector store yang sudah ada...")
            
            try:
                self.vector_store = FAISS.load_local(
                    vector_store_path, 
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                print("✅ Vector store berhasil dimuat dari cache")
                print("⚡ Proses lebih cepat karena menggunakan data yang sudah ada!")
                return
                
            except Exception as e:
                print(f"⚠️ Error memuat vector store: {str(e)}")
                print("🔄 Akan membuat vector store baru...")
        
        # Jika tidak ada atau gagal dimuat, buat yang baru
        print(f"\n🗄️ Membuat vector store baru dengan FAISS...")
        print(f"📊 Memproses {len(self.text_chunks)} chunks...")
        print("⏳ Estimasi waktu: 2-5 menit (hanya sekali)")
        
        if not self.text_chunks:
            raise ValueError("Tidak ada chunks untuk diproses!")
        
        try:
            start_time = time.time()
            
            self.vector_store = FAISS.from_documents(
                documents=self.text_chunks,
                embedding=self.embeddings
            )
            
            elapsed_time = time.time() - start_time
            print(f"✅ Vector store berhasil dibuat dalam {elapsed_time:.1f} detik")
            
            # Simpan vector store
            self.vector_store.save_local(vector_store_path)
            print(f"💾 Vector store disimpan ke '{vector_store_path}'")
            print("🎯 Selanjutnya akan menggunakan cache untuk startup yang lebih cepat!")
            
        except Exception as e:
            print(f"❌ Error membuat vector store: {str(e)}")
            raise
    
    def setup_llm(self):
        """Setup LLM dengan Ollama dengan validasi"""
        print("\n🤖 Mengatur LLM dengan Ollama...")
        
        # Validasi Ollama dulu  
        self.validate_ollama_status()
        
        try:
            self.llm = Ollama(
                model="llama3.1:8b",
                base_url="http://localhost:11434",
                temperature=0.1
                # Note: timeout parameter removed as it's not supported in newer versions
            )
            
            # Test LLM dengan validasi ulang
            print("🧪 Testing LLM...")
            test_response = self.llm.invoke("Halo")
            print(f"✅ LLM berhasil diatur dan diuji: {test_response[:50]}...")
            
        except Exception as e:
            print(f"❌ Error mengatur LLM: {str(e)}")
            print("💡 Pastikan Ollama berjalan dengan: ollama serve")
            print("💡 Dan model tersedia dengan: ollama pull llama3.1:8b")
            raise
    
    def create_qa_chain(self):
        """Membuat QA chain dengan prompt bahasa Indonesia"""
        print("\n🔗 Membuat QA chain...")
        
        # Prompt template dalam bahasa Indonesia yang lebih spesifik
        prompt_template = """
Kamu adalah asisten hukum ahli yang menguasai Undang-Undang Dasar 1945 (UUD 1945). 
Tugasmu adalah menjawab pertanyaan tentang UUD 1945 dengan akurat berdasarkan konteks dokumen yang diberikan.

INSTRUKSI PENTING:
1. Jawab HANYA berdasarkan informasi yang ada dalam KONTEKS di bawah ini
2. Jika pertanyaan menanyakan tentang pasal tertentu, cari informasi pasal tersebut dalam konteks
3. Berikan jawaban yang lengkap dan akurat dalam bahasa Indonesia
4. Sertakan nomor pasal, ayat, dan bunyi lengkap pasal jika tersedia dalam konteks
5. Jika informasi tidak ada dalam konteks, jawab: "Maaf, informasi tentang [topik] tidak ditemukan dalam dokumen UUD 1945 yang tersedia."

KONTEKS DARI UUD 1945:
{context}

PERTANYAAN: {question}

JAWABAN (berdasarkan konteks UUD 1945 di atas):
"""
        
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # Buat retriever dengan konfigurasi yang lebih baik
        retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": 10,  # Ambil lebih banyak dokumen untuk analisis yang lebih baik
                "fetch_k": 20,  # Fetch lebih banyak kandidat
            }
        )
        
        # Buat QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )
        
        print("✅ QA chain berhasil dibuat")
    
    def ask_question_with_custom_qa(self, question: str) -> Dict[str, Any]:
        """Custom QA yang memaksa menggunakan hybrid search results"""
        if not self.llm:
            raise ValueError("LLM belum diinisialisasi!")
        
        print(f"\n❓ Pertanyaan: {question}")
        
        # Validasi Ollama sebelum proses
        try:
            self.validate_ollama_status()
            print("✅ Ollama status OK")
        except Exception as e:
            print(f"❌ Validasi gagal: {str(e)}")
            print("💡 Silakan restart Ollama dan coba lagi")
            raise
        
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
        
        print("🔍 Mencari jawaban dengan hybrid search...")
        
        try:
            # Gunakan hybrid search untuk mendapatkan dokumen terbaik
            search_results = self.hybrid_search(question, k=10)
            
            # Buat context dari hasil hybrid search
            context_docs = [doc for doc, score in search_results]
            context = "\n\n".join([doc.page_content for doc in context_docs])
            
            # Buat prompt template yang lebih comprehensive
            prompt_template = """
Kamu adalah ahli hukum konstitusi Indonesia yang sangat menguasai Undang-Undang Dasar 1945 (UUD 1945). 
Tugasmu adalah memberikan analisis yang mendalam dan penjelasan yang sangat detail berdasarkan konteks dokumen UUD 1945.

INSTRUKSI KHUSUS:
1. WAJIB gunakan HANYA informasi yang tersedia dalam KONTEKS di bawah ini
2. Untuk pertanyaan tentang pasal, bab, atau ketentuan tertentu: berikan bunyi lengkap dan jelaskan maknanya
3. Untuk pertanyaan tentang tugas, wewenang, atau fungsi: analisis berdasarkan seluruh dokumen yang relevan dalam konteks
4. Berikan penjelasan yang SANGAT DETAIL dan KOMPREHENSIF dalam bahasa Indonesia yang formal
5. Sertakan referensi pasal, ayat, bab, atau sumber yang spesifik
6. Jika informasi tersebar di beberapa bagian dokumen, gabungkan untuk memberikan gambaran lengkap
7. Untuk pertanyaan umum, berikan analisis mendalam berdasarkan prinsip-prinsip yang terkandung dalam konteks
8. Hanya jika benar-benar tidak ada informasi relevan, katakan: "Maaf, informasi tentang [topik] tidak ditemukan dalam dokumen UUD 1945 yang tersedia."

KONTEKS LENGKAP DARI DOKUMEN UUD 1945:
{context}

PERTANYAAN YANG HARUS DIJAWAB SECARA DETAIL: {question}

ANALISIS MENDALAM DAN PENJELASAN DETAIL (berdasarkan konteks UUD 1945):
"""
            
            # Buat prompt dengan context dari hybrid search
            final_prompt = prompt_template.format(context=context, question=question)
            
            # Direct call ke LLM
            answer = self.llm.invoke(final_prompt)
            
            # Simulate result format seperti RetrievalQA
            result = {
                'result': answer,
                'source_documents': context_docs
            }
            
            # Hitung metrik komprehensif
            metrics = self.calculate_comprehensive_metrics(
                question, 
                result.get('source_documents', []),
                result['result']
            )
            
            # Format sumber dokumen dengan similarity scores yang benar dari hybrid search
            sources = []
            source_docs = result.get('source_documents', [])
            
            # Match source documents dengan search results untuk mendapatkan scores yang benar
            for i, doc in enumerate(source_docs):
                # Cari matching document di search results
                similarity_score = 0.0
                for search_doc, score in search_results:
                    if (search_doc.page_content == doc.page_content and 
                        search_doc.metadata.get('page') == doc.metadata.get('page')):
                        similarity_score = float(score)
                        break
                
                source_file = doc.metadata.get('source_file', 'Unknown')
                
                # Dapatkan metadata dari mapping
                metadata = self.pdf_metadata.get(source_file, {
                    'judul': source_file,
                    'sumber': 'Tidak diketahui',
                    'institusi': 'Tidak diketahui', 
                    'priority_score': 70
                })
                
                sources.append({
                    'dokumen': source_file,
                    'judul': metadata['judul'],
                    'sumber_url': metadata['sumber'],
                    'institusi': metadata['institusi'],
                    'priority_score': metadata['priority_score'],
                    'halaman': str(doc.metadata.get('page', 'Unknown')),
                    'chunk_id': doc.metadata.get('chunk_id', i),
                    'similarity_score': similarity_score,  # Now with correct score from hybrid search
                    'preview': doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content
                })
            
            response = {
                'pertanyaan': question,
                'jawaban': result['result'],
                'metrics': metrics,
                'jumlah_sumber': len(sources),
                'sumber_dokumen': sources,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            return response
            
        except Exception as e:
            print(f"❌ Error saat memproses pertanyaan: {str(e)}")
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
        """Menghitung berbagai metrik untuk evaluasi jawaban dan akurasi"""
        
        # 1. Semantic Similarity Score (menggunakan embedding)
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
        
        # 7. Overall Confidence Score (weighted average)
        confidence_score = (
            semantic_score * 0.20 + 
            coverage_score * 0.15 + 
            answer_relevance * 0.25 + 
            source_quality * 0.15 +
            legal_context * 0.15 +
            completeness * 0.10
        )
        
        # 8. Estimated Accuracy (berdasarkan multiple factors)
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
        """Menghitung similarity semantik menggunakan embedding"""
        if not retrieved_docs:
            return 0.0
        
        try:
            # Embed query
            query_embedding = self.embeddings.embed_query(query)
            
            # Embed retrieved documents
            similarities = []
            for doc in retrieved_docs:
                doc_text = doc.page_content[:500]  # Ambil 500 karakter pertama
                doc_embedding = self.embeddings.embed_query(doc_text)
                similarity = self._cosine_similarity(query_embedding, doc_embedding)
                similarities.append(similarity)
            
            return (sum(similarities) / len(similarities)) * 100
            
        except Exception:
            return 50.0  # Default score jika error
    
    def _calculate_content_coverage(self, query: str, retrieved_docs: List) -> float:
        """Menghitung seberapa lengkap dokumen yang diambil menjawab query"""
        if not retrieved_docs:
            return 0.0
        
        # Ekstrak kata kunci penting dari query
        important_words = self._extract_key_terms(query)
        
        if not important_words:
            return 50.0
        
        # Cek coverage di retrieved documents
        total_coverage = 0
        for doc in retrieved_docs:
            doc_words = set(doc.page_content.lower().split())
            coverage = len([word for word in important_words if word in doc_words])
            total_coverage += coverage / len(important_words)
        
        return min((total_coverage / len(retrieved_docs)) * 100, 100)
    
    def _calculate_answer_relevance(self, query: str, answer: str) -> float:
        """Menghitung relevansi jawaban terhadap pertanyaan"""
        if not answer or len(answer.strip()) < 10:
            return 0.0
        
        # Cek apakah jawaban mengandung kata kunci dari pertanyaan
        query_terms = set(self._extract_key_terms(query))
        answer_terms = set(answer.lower().split())
        
        # Hitung overlap dan konteks
        if query_terms:
            overlap = len(query_terms.intersection(answer_terms))
            relevance = (overlap / len(query_terms)) * 70
        else:
            relevance = 30
        
        # Bonus untuk indikator konteks hukum
        context_indicators = ['pasal', 'bab', 'uud', 'ayat', 'hak', 'kewajiban', 'republik', 'indonesia']
        context_score = len([term for term in context_indicators if term in answer.lower()])
        context_bonus = min(context_score * 5, 30)
        
        return min(relevance + context_bonus, 100)
    
    def _calculate_source_quality(self, retrieved_docs: List) -> float:
        """Menghitung kualitas sumber dokumen dengan priority score"""
        if not retrieved_docs:
            return 0.0
        
        quality_score = 0
        for doc in retrieved_docs:
            source_file = doc.metadata.get('source_file', '')
            
            # Dapatkan priority score dari metadata
            if source_file in self.pdf_metadata:
                priority = self.pdf_metadata[source_file]['priority_score']
                quality_score += priority
            else:
                # Default score untuk file tidak dikenal
                quality_score += 70
        
        # Normalisasi ke skala 0-100
        avg_quality = quality_score / len(retrieved_docs)
        return min(avg_quality, 100)
    
    def _calculate_legal_context_score(self, query: str, answer: str) -> float:
        """Menghitung seberapa baik jawaban menggunakan konteks hukum"""
        if not answer:
            return 0.0
        
        # Kata kunci konteks hukum UUD 1945
        legal_keywords = {
            'pasal', 'ayat', 'bab', 'uud', 'undang-undang', 'dasar', 'konstitusi',
            'hak', 'kewajiban', 'warga', 'negara', 'republik', 'indonesia',
            'presiden', 'dpr', 'mpr', 'mahkamah', 'pancasila', 'bhinneka',
            'kedaulatan', 'rakyat', 'demokrasi', 'hukum', 'keadilan'
        }
        
        answer_words = set(answer.lower().split())
        legal_context_count = len([word for word in legal_keywords if word in answer_words])
        
        # Skor berdasarkan jumlah konteks hukum yang digunakan
        base_score = min(legal_context_count * 8, 80)
        
        # Bonus untuk referensi spesifik (Pasal X, Bab Y, dll)
        import re
        pasal_references = len(re.findall(r'pasal\s+\d+', answer.lower()))
        bab_references = len(re.findall(r'bab\s+[ivx]+', answer.lower()))
        
        reference_bonus = min((pasal_references * 10) + (bab_references * 5), 20)
        
        return min(base_score + reference_bonus, 100)
    
    def _calculate_answer_completeness(self, query: str, answer: str) -> float:
        """Menghitung kelengkapan jawaban"""
        if not answer:
            return 0.0
        
        # Panjang jawaban (indikator kelengkapan)
        answer_length = len(answer)
        if answer_length < 50:
            length_score = 20
        elif answer_length < 150:
            length_score = 50
        elif answer_length < 300:
            length_score = 75
        else:
            length_score = 90
        
        # Struktur jawaban (ada penjelasan, contoh, dll)
        structure_indicators = [
            'yaitu', 'adalah', 'antara lain', 'contoh', 'misalnya',
            'pertama', 'kedua', 'ketiga', 'selain itu', 'namun'
        ]
        structure_count = len([ind for ind in structure_indicators if ind in answer.lower()])
        structure_score = min(structure_count * 10, 30)
        
        # Skor negatif untuk jawaban yang mengaku tidak tahu
        uncertainty_phrases = ['tidak tahu', 'tidak dapat', 'tidak ada informasi', 'maaf']
        if any(phrase in answer.lower() for phrase in uncertainty_phrases):
            uncertainty_penalty = -20
        else:
            uncertainty_penalty = 0
        
        return max(min(length_score * 0.7 + structure_score + uncertainty_penalty, 100), 0)
    
    def _calculate_estimated_accuracy(self, semantic: float, coverage: float, 
                                     relevance: float, quality: float, 
                                     legal: float, completeness: float) -> float:
        """Menghitung estimasi akurasi berdasarkan semua metrik dengan bobot khusus"""
        
        # Weighted calculation dengan emphasis pada faktor kunci
        base_accuracy = (
            semantic * 0.25 +      # Similarity semantik sangat penting
            coverage * 0.20 +      # Coverage konten penting
            relevance * 0.25 +     # Relevansi jawaban krusial
            quality * 0.15 +       # Kualitas sumber penting
            legal * 0.10 +         # Konteks hukum bonus
            completeness * 0.05    # Kelengkapan sebagai faktor minor
        )
        
        # Bonus akurasi berdasarkan kualitas sumber tertinggi
        if quality >= 110:  # File BUKU (paling komprehensif)
            source_bonus = 8.0
        elif quality >= 100:  # File MKRI (otoritatif)
            source_bonus = 5.0
        elif quality >= 95:   # File BPHN (resmi)
            source_bonus = 3.0
        elif quality >= 90:   # File MPR (resmi)
            source_bonus = 2.0
        else:
            source_bonus = 0.0
        
        # Adjustment berdasarkan threshold dengan bonus
        adjusted_accuracy = base_accuracy + source_bonus
        
        if adjusted_accuracy >= 85:
            final_accuracy = min(adjusted_accuracy * 1.05, 97)  # Boost untuk skor tinggi
        elif adjusted_accuracy >= 70:
            final_accuracy = adjusted_accuracy * 1.02
        elif adjusted_accuracy >= 50:
            final_accuracy = adjusted_accuracy * 0.98
        else:
            final_accuracy = adjusted_accuracy * 0.95  # Penalty untuk skor rendah
        
        return min(max(final_accuracy, 10), 97)  # Cap antara 10-97%
    
    def _extract_key_terms(self, text: str) -> List[str]:
        """Ekstrak kata kunci penting dari teks"""
        # Kata kunci hukum yang penting untuk UUD 1945
        legal_keywords = {
            'hak', 'asasi', 'manusia', 'kewajiban', 'pasal', 'bab', 'ayat',
            'presiden', 'dpr', 'mpr', 'mahkamah', 'konstitusi', 'pancasila',
            'negara', 'rakyat', 'bangsa', 'indonesia', 'demokrasi', 'hukum',
            'kedaulatan', 'pemerintah', 'kekuasaan', 'legislatif', 'eksekutif',
            'yudikatif', 'pemilu', 'partai', 'otonomi', 'daerah'
        }
        
        words = [word.lower().strip('.,!?:;()[]') for word in text.split()]
        important_words = []
        
        for word in words:
            # Ambil kata yang ada di legal keywords atau kata panjang (>4 karakter)
            if word in legal_keywords or (len(word) > 4 and word.isalpha()):
                important_words.append(word)
        
        return list(set(important_words))
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Menghitung cosine similarity antara dua vektor"""
        if len(vec1) != len(vec2):
            return 0.0
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(a * a for a in vec2))
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)
    
    def debug_search(self, question: str, k: int = 10):
        """Method untuk debugging search results"""
        print(f"\n🔍 DEBUG: Searching for: {question}")
        
        if not self.vector_store:
            print("❌ Vector store not initialized")
            return
        
        # Search dengan similarity
        docs = self.vector_store.similarity_search_with_score(question, k=k)
        
        print(f"📊 Found {len(docs)} documents")
        for i, (doc, score) in enumerate(docs):
            print(f"\n--- Document {i+1} (Score: {score:.4f}) ---")
            print(f"Source: {doc.metadata.get('source_file', 'Unknown')}")
            print(f"Page: {doc.metadata.get('page', 'Unknown')}")
            print(f"Content preview: {doc.page_content[:200]}...")
            
        return docs
    
    def hybrid_search(self, query: str, k: int = 10):
        """Hybrid search combining keyword and semantic search"""
        import re
        
        # Check if query contains specific article reference
        pasal_pattern = r'[Pp]asal\s+(\d+)'
        pasal_match = re.search(pasal_pattern, query)
        
        if pasal_match:
            # If query mentions specific article, prioritize keyword search
            pasal_num = pasal_match.group(1)
            print(f"🔍 Detected Pasal {pasal_num} query, using hybrid search...")
            
            # Find chunks containing the specific article
            keyword_results = []
            for i, chunk in enumerate(self.text_chunks):
                if f'Pasal {pasal_num}' in chunk.page_content:
                    # Calculate a "keyword score" based on relevance
                    score = 100.0  # High priority for exact matches
                    
                    # Boost score if chunk has actual content (not just header/footer)
                    if len(chunk.page_content.strip()) > 50:
                        score -= 10
                    
                    # Boost score for higher priority documents
                    source_file = chunk.metadata.get('source_file', '')
                    priority = self.pdf_metadata.get(source_file, {}).get('priority_score', 70)
                    score -= (110 - priority)  # Higher priority = lower distance score
                    
                    keyword_results.append((chunk, score))
            
            # Sort by score (lower is better, like distance)
            keyword_results.sort(key=lambda x: x[1])
            
            if keyword_results:
                print(f"  ✅ Found {len(keyword_results)} keyword matches for Pasal {pasal_num}")
                
                # Return top keyword results, but also add some semantic results
                semantic_results = self.vector_store.similarity_search_with_score(query, k=k//2)
                
                # Combine results, prioritizing keyword matches
                combined_results = keyword_results[:k//2] + semantic_results[:k//2]
                return combined_results[:k]
            else:
                print(f"  ❌ No keyword matches found for Pasal {pasal_num}, falling back to semantic search")
        
        # Default to semantic search
        return self.vector_store.similarity_search_with_score(query, k=k)
    
    def ask_question(self, question: str) -> Dict[str, Any]:
        """Mengajukan pertanyaan ke chatbot dengan validasi real-time"""
        if not self.qa_chain:
            raise ValueError("QA chain belum diinisialisasi!")
        
        print(f"\n❓ Pertanyaan: {question}")
        
        # Validasi Ollama sebelum proses
        try:
            self.validate_ollama_status()
            print("✅ Ollama status OK")
        except Exception as e:
            print(f"❌ Validasi gagal: {str(e)}")
            print("� Silakan restart Ollama dan coba lagi")
            raise
        
        print("�🔍 Mencari jawaban...")
        
        try:
            # Debug search first menggunakan hybrid search
            print(f"🔍 Debugging hybrid search untuk: {question}")
            search_results = self.hybrid_search(question, k=10)
            
            print(f"📊 Found {len(search_results)} relevant documents:")
            for i, (doc, score) in enumerate(search_results[:3]):
                print(f"  {i+1}. Score: {score:.4f} | Page: {doc.metadata.get('page', '?')} | Preview: {doc.page_content[:100]}...")
            
            # Untuk QA chain, kita perlu menggunakan retriever yang sudah ada
            # Tapi kita bisa override source documents dengan hasil hybrid search
            result = self.qa_chain.invoke({"query": question})
            
            # Override source documents dengan hasil hybrid search
            if search_results:
                # Convert search results to documents (remove scores)
                hybrid_docs = [doc for doc, score in search_results]
                result['source_documents'] = hybrid_docs[:10]  # Take top 10
            
            # Hitung metrik komprehensif  
            metrics = self.calculate_comprehensive_metrics(
                question, 
                result.get('source_documents', []),
                result['result']
            )
            
            # Format sumber dokumen dengan similarity scores yang benar dari hybrid search
            sources = []
            source_docs = result.get('source_documents', [])
            
            # Match source documents dengan search results untuk mendapatkan scores yang benar
            for i, doc in enumerate(source_docs):
                # Cari matching document di search results
                similarity_score = 0.0
                for search_doc, score in search_results:
                    if (search_doc.page_content == doc.page_content and 
                        search_doc.metadata.get('page') == doc.metadata.get('page')):
                        similarity_score = float(score)
                        break
                
                source_file = doc.metadata.get('source_file', 'Unknown')
                
                # Dapatkan metadata dari mapping
                metadata = self.pdf_metadata.get(source_file, {
                    'judul': source_file,
                    'sumber': 'Tidak diketahui',
                    'institusi': 'Tidak diketahui', 
                    'priority_score': 70
                })
                
                sources.append({
                    'dokumen': source_file,
                    'judul': metadata['judul'],
                    'sumber_url': metadata['sumber'],
                    'institusi': metadata['institusi'],
                    'priority_score': metadata['priority_score'],
                    'halaman': str(doc.metadata.get('page', 'Unknown')),
                    'chunk_id': doc.metadata.get('chunk_id', i),
                    'similarity_score': similarity_score,  # Now with correct score from hybrid search
                    'preview': doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content
                })
            
            response = {
                'pertanyaan': question,
                'jawaban': result['result'],
                'metrics': metrics,
                'jumlah_sumber': len(sources),
                'sumber_dokumen': sources,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            return response
            
        except Exception as e:
            print(f"❌ Error saat memproses pertanyaan: {str(e)}")
            raise
    
    def display_response(self, response: Dict[str, Any]):
        """Menampilkan respons dengan metrik akurasi yang komprehensif"""
        metrics = response['metrics']
        
        print(f"\n{'=' * 70}")
        print(f"🤖 JAWABAN LAWCHAIN")
        print(f"{'=' * 70}")
        print(f"📝 Pertanyaan: {response['pertanyaan']}")
        print(f"\n💬 Jawaban:")
        print(f"{response['jawaban']}")
        
        # Kategorisasi tingkat akurasi
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
        print(f"📊 ANALISIS AKURASI & KUALITAS JAWABAN")
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
        
        print(f"\n� SUMBER REFERENSI ({response['jumlah_sumber']} dokumen):")
        
        if response['sumber_dokumen']:
            for i, source in enumerate(response['sumber_dokumen'], 1):
                # Indikator kualitas berdasarkan priority score
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
        """Inisialisasi lengkap sistem RAG"""
        try:
            # 1. Load documents
            self.load_documents()
            
            # 2. Split documents
            self.split_documents()
            
            # 3. Create embeddings
            self.create_embeddings()
            
            # 4. Create vector store (dengan opsi rebuild)
            if force_rebuild_vectorstore and os.path.exists("vector_store_faiss"):
                print("🔄 Force rebuild: Menghapus vector store lama...")
                import shutil
                shutil.rmtree("vector_store_faiss")
            
            self.create_vector_store()
            
            # 5. Setup LLM
            self.setup_llm()
            
            # 6. Create QA chain
            self.create_qa_chain()
            
            print(f"\n{'=' * 60}")
            print("🎉 LAWCHAIN SIAP DIGUNAKAN!")
            print(f"{'=' * 60}")
            print(f"📊 Statistik:")
            print(f"   • Total dokumen: {self.total_documents}")
            print(f"   • Total halaman: {len(self.documents)}")
            print(f"   • Total chunks: {self.total_chunks}")
            print(f"   • Model LLM: llama3.1:8b")
            print(f"   • Model Embedding: nomic-embed-text")
            print(f"   • Vector Store: FAISS")
            print(f"{'=' * 60}")
            
        except Exception as e:
            print(f"❌ Error inisialisasi: {str(e)}")
            raise
    
    def run_interactive_chat(self):
        """Menjalankan chat interaktif"""
        print("\n💬 Mode Chat Interaktif")
        print("Ketik 'quit', 'exit', atau 'keluar' untuk mengakhiri")
        print("Ketik 'rebuild' untuk membangun ulang vector store")
        print("Ketik 'hapus' untuk menghapus riwayat percakapan")
        print("-" * 60)
        
        while True:
            try:
                question = input("\n🙋 Tanya UUD 1945: ").strip()
                
                if question.lower() in ['quit', 'exit', 'keluar', 'q']:
                    print("\n👋 Terima kasih telah menggunakan LawChain!")
                    break
                
                if question.lower() in ['rebuild', 'rebuild-vector']:
                    print("\n🔄 Membangun ulang vector store...")
                    self.initialize(force_rebuild_vectorstore=True)
                    print("✅ Vector store berhasil dibangun ulang!")
                    continue
                
                if question.lower() in ['hapus', 'clear', 'cls']:
                    # Reset chat history jika ada implementasi
                    print("🗑️ Riwayat percakapan dihapus!")
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
        # Buat instance LawChain
        lawchain = LawChainIndonesia()
        
        # Inisialisasi sistem
        lawchain.initialize()
        
        # Jalankan chat interaktif
        lawchain.run_interactive_chat()
        
    except Exception as e:
        print(f"\n❌ Error fatal: {str(e)}")
        print("💡 Pastikan:")
        print("   1. Ollama sudah berjalan")
        print("   2. Model llama3.1:8b dan nomic-embed-text sudah di-pull")
        print("   3. Folder 'data' berisi file PDF UUD 1945")


if __name__ == "__main__":
    main()
