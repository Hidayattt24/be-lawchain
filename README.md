# LawChain - Chatbot Hukum UUD 1945

**Pengembangan Chatbot Hukum UUD 1945 Berbasis Retrieval Augmented Generation dengan Model LLaMA**

## 📝 Deskripsi

LawChain adalah chatbot pintar yang dirancang khusus untuk menjawab pertanyaan tentang Undang-Undang Dasar 1945 (UUD 1945). Sistem ini menggunakan teknologi **Retrieval Augmented Generation (RAG)** dengan model **LLaMA3** yang berjalan secara lokal melalui **Ollama**, mengikuti panduan resmi LangChain RAG Tutorial.

### 🎯 Fitur Utama

- ✅ **RAG Pipeline LangChain**: Mengikuti alur resmi Load → Split → Embed → Store → Retrieve → Generate
- ✅ **LLaMA3 Lokal**: Menggunakan Ollama untuk menjalankan LLaMA3 secara lokal dan private
- ✅ **Memori Kontekstual**: Chatbot mengingat percakapan sebelumnya untuk jawaban yang lebih pintar
- ✅ **Multi-dokumen UUD 1945**: Mendukung 4-5 PDF UUD 1945 dari berbagai sumber resmi
- ✅ **Output Bahasa Indonesia**: Respon lengkap dengan referensi pasal dan metadata
- ✅ **Vector Search FAISS**: Pencarian semantik cepat dengan similarity scoring

## 🛠️ Teknologi yang Digunakan

- **🦜 LangChain**: Framework utama untuk aplikasi RAG dan LLM
- **🏗️ LangChain Community**: Integrasi dengan Ollama dan vector stores
- **🔍 FAISS**: Facebook AI Similarity Search untuk vector database
- **🦙 Ollama**: Platform untuk menjalankan LLaMA3 secara lokal
- **🤖 LLaMA3**: Large Language Model dari Meta (8B parameters)
- **📄 PyMuPDF**: Library untuk memproses dokumen PDF UUD 1945
- **🔤 Sentence Transformers**: Model embedding multilingual untuk Bahasa Indonesia

## 🏗️ Arsitektur RAG Pipeline

Mengikuti tutorial resmi LangChain: https://python.langchain.com/docs/tutorials/rag/

```
📋 PHASE 1: INDEXING (Offline Processing)
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   PDF Docs      │───▶│   Text Chunks   │───▶│  Vector Store   │
│  (UUD 1945)     │    │  (1000 chars)   │    │    (FAISS)      │
│   4-5 files     │    │   240 chunks    │    │   + Metadata    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
        │                       │                       │
        ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ PyMuPDFLoader   │    │RecursiveChar    │    │ OllamaEmbeddings│
│   Load PDFs     │    │ TextSplitter    │    │ nomic-embed-text│
│  Extract Text   │    │ chunk_size=1000 │    │   768 dim       │
└─────────────────┘    └─────────────────┘    └─────────────────┘

📋 PHASE 2: RETRIEVAL & GENERATION (Runtime)
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   User Query    │───▶│   Similarity    │───▶│   Top-4 Chunks  │
│ "Hak asasi di   │    │   Search FAISS  │    │  + Source Info  │
│  UUD 1945?"     │    │  (cosine sim)   │    │  + Page Number  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
                                                        ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Final Answer   │◀───│   LLaMA3 via    │◀───│   Prompt +      │
│  + References   │    │    Ollama       │    │   Context +     │
│  + Metadata     │    │  temp=0.7       │    │   History       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🔄 Alur Kerja RAG Berdasarkan LangChain Tutorial

### 1️⃣ **Load** - Document Loading

```python
# Load multiple PDF UUD 1945
from langchain_community.document_loaders import PyMuPDFLoader

for pdf_file in pdf_files:
    loader = PyMuPDFLoader(pdf_path)
    documents.extend(loader.load())
```

### 2️⃣ **Split** - Text Chunking

```python
# Split documents into chunks
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = text_splitter.split_documents(documents)
```

### 3️⃣ **Embed** - Vector Embeddings

```python
# Create embeddings using Ollama
from langchain_community.embeddings import OllamaEmbeddings

embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url="http://localhost:11434"
)
```

### 4️⃣ **Store** - Vector Database

```python
# Store in FAISS vector database
from langchain_community.vectorstores import FAISS

vector_store = FAISS.from_documents(
    documents=chunks,
    embedding=embeddings
)
```

### 5️⃣ **Retrieve** - Similarity Search

```python
# Retrieve relevant documents
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)
```

### 6️⃣ **Generate** - LLM Response

```python
# Generate answer using LLaMA3
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA

llm = Ollama(model="llama3", temperature=0.7)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)
```

## 📋 Prasyarat

1. **Python 3.10+**
2. **Ollama** terinstall di sistem
3. **Model LLaMA3** dan **nomic-embed-text** sudah di-download
4. **File PDF UUD 1945** di folder `data/`
5. **Virtual Environment** sudah diaktifkan

## 🚀 Instalasi & Setup

### 1. Aktifkan Virtual Environment

```bash
# Pastikan berada di direktori project
cd D:\.KKP\LawChain\LLM-LawChain

# Aktifkan virtual environment
.\.venv\Scripts\activate
```

### 2. Install Dependencies

```bash
# Install semua package yang diperlukan
pip install -r requirements.txt
```

### 3. Setup Ollama dan Models

```bash
# Jalankan Ollama server (terminal terpisah)
ollama serve

# Download model LLaMA3 dan embedding
ollama pull llama3
ollama pull nomic-embed-text
```

### 4. Verifikasi Data PDF

Pastikan file PDF UUD 1945 ada di folder `data/`:

- `UUD1945-BPHN.pdf`
- `UUD1945-MKRI.pdf`
- `UUD1945-MPR.pdf`
- `UUD1945.pdf`

### 5. Jalankan LawChain

```bash
python lawchain_indonesia.py
```

## 📖 Cara Penggunaan

### Contoh Interaksi Lengkap

```bash
🏛️ Selamat datang di LawChain - Chatbot Hukum UUD 1945
============================================================

📂 Memuat dokumen dari folder: data
✅ Berhasil memuat: UUD1945-BPHN.pdf (26 halaman)
✅ Berhasil memuat: UUD1945-MKRI.pdf (25 halaman)
✅ Berhasil memuat: UUD1945-MPR.pdf (27 halaman)
✅ Berhasil memuat: UUD1945.pdf (25 halaman)

📊 Total dokumen dimuat: 4
📊 Total halaman: 103

🔄 Membagi dokumen menjadi chunks...
✅ Berhasil membuat 240 chunks

🔮 Membuat embeddings dengan Ollama...
✅ Embedding berhasil dibuat (dimensi: 768)

🗄️ Membuat vector store dengan FAISS...
✅ Vector store berhasil dibuat

🤖 Setup LLM dengan Ollama...
✅ LLM berhasil disetup (Model: llama3)

⚙️ Membuat QA chain...
✅ QA chain berhasil dibuat

============================================================
🎉 LAWCHAIN SIAP DIGUNAKAN!
============================================================

💡 Ketik 'keluar' untuk exit, 'hapus' untuk clear history
❓ Pertanyaan Anda: Apa itu hak asasi manusia menurut UUD 1945?

🤔 Memproses pertanyaan...

🏛️ Jawaban LawChain:
Menurut UUD 1945, hak asasi manusia (HAM) adalah hak dasar yang melekat pada setiap manusia sebagai anugerah Tuhan Yang Maha Esa. Hak asasi manusia diatur dalam Bab XA UUD 1945, khususnya dalam Pasal 28A hingga 28J.

Beberapa hak asasi manusia yang dijamin dalam UUD 1945 antara lain:

1. **Hak untuk hidup** (Pasal 28A)
2. **Hak berkeluarga dan melanjutkan keturunan** (Pasal 28B)
3. **Hak mengembangkan diri dan memperoleh pendidikan** (Pasal 28C)
4. **Hak atas pengakuan, jaminan, perlindungan, dan kepastian hukum** (Pasal 28D)
5. **Hak beragama dan berkeyakinan** (Pasal 28E)

📊 Skor Relevansi: 95%

📚 Referensi:
   • Sumber: UUD1945-MKRI.pdf, Halaman 15, Chunk ID: 3
   • Sumber: UUD1945-BPHN.pdf, Halaman 12, Chunk ID: 7
   • Metadata: {"source": "UUD1945-MKRI.pdf", "page": 15, "type": "hak_asasi"}

❓ Pertanyaan Anda:
```

### Perintah Khusus

- `keluar` atau `exit` - Keluar dari aplikasi
- `hapus` atau `clear` - Menghapus riwayat percakapan

## 🧠 Bagaimana RAG Bekerja di LawChain

Mengikuti prinsip RAG dari LangChain Tutorial:

### **Indexing Phase** (Sekali saja):

1. **Load**: PDF UUD 1945 dimuat menggunakan PyMuPDFLoader
2. **Split**: Dokumen dibagi menjadi chunks (1000 karakter, overlap 200)
3. **Embed**: Setiap chunk diubah menjadi vector embeddings (nomic-embed-text)
4. **Store**: Embeddings disimpan di FAISS dengan metadata lengkap

### **Retrieval Phase** (Setiap query):

1. **Query Embedding**: Pertanyaan user diubah menjadi vector
2. **Similarity Search**: FAISS mencari 4 chunks paling relevan
3. **Context Extraction**: Chunks relevan + metadata diekstrak

### **Generation Phase** (Setiap query):

1. **Prompt Engineering**: Context + query + history digabung
2. **LLM Generation**: LLaMA3 menghasilkan jawaban berdasarkan context
3. **Response Formatting**: Jawaban + referensi + skor relevansi

## 🔧 Konfigurasi

### Parameter RAG Optimal:

```python
# Text Splitting
chunk_size = 1000        # Ukuran chunk optimal untuk konteks hukum
chunk_overlap = 200      # Overlap untuk kontinuitas informasi

# Retrieval
k = 4                    # Top-4 dokumen relevan
search_type = "similarity"

# Generation
temperature = 0.7        # Balance antara kreativitas dan konsistensi
model = "llama3"         # LLaMA3 8B parameters
```

### Environment Variables (.env):

```env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3
EMBEDDING_MODEL=nomic-embed-text
```

## 📊 Contoh Pertanyaan yang Didukung

### Pertanyaan Dasar:

- "Apa itu Pancasila menurut UUD 1945?"
- "Jelaskan tentang hak asasi manusia di UUD 1945"
- "Bagaimana sistem pemerintahan Indonesia?"

### Pertanyaan Spesifik:

- "Apa bunyi Pasal 28A UUD 1945?"
- "Jelaskan tentang amandemen UUD 1945"
- "Apa saja kewajiban warga negara menurut UUD 1945?"

### Pertanyaan Kontekstual:

- "Bagaimana hubungan Pancasila dengan hak asasi manusia?"
- "Apa perbedaan UUD 1945 sebelum dan sesudah amandemen?"

## 🔍 Troubleshooting

### Error: Ollama not running

```bash
# Solusi: Jalankan Ollama server
ollama serve
```

### Error: Model not found

```bash
# Download models yang diperlukan
ollama pull llama3
ollama pull nomic-embed-text
```

### Error: No PDF files found

```
Pastikan file PDF UUD 1945 ada di folder 'data/'
```

### Memory/Performance Issues

```python
# Kurangi parameter jika ada masalah memory:
chunk_size = 800        # dari 1000
k = 3                   # dari 4
```

## 🚀 Fitur Pengembangan Lanjutan

### Yang Bisa Ditambahkan:

1. **Web Interface**: Streamlit/FastAPI untuk GUI
2. **API Endpoint**: REST API untuk integrasi sistem lain
3. **Advanced RAG**: Hybrid search + reranking
4. **Fine-tuning**: Custom model untuk hukum Indonesia
5. **Multi-format**: Support DOCX, TXT, HTML
6. **Database**: PostgreSQL untuk persistent chat history

### Optimisasi RAG:

1. **Semantic Chunking**: Chunking berdasarkan struktur pasal
2. **Custom Embeddings**: Embedding model khusus hukum Indonesia
3. **Query Expansion**: Ekspansi query dengan sinonim hukum
4. **Answer Validation**: Validasi jawaban dengan confidence scoring

## 📄 Struktur Project

```
LLM-LawChain/
├── 📄 lawchain_indonesia.py    # File utama RAG chatbot
├── 📁 data/                    # Folder PDF UUD 1945
│   ├── UUD1945-BPHN.pdf
│   ├── UUD1945-MKRI.pdf
│   ├── UUD1945-MPR.pdf
│   └── UUD1945.pdf
├── 📁 vector_store_faiss/      # FAISS database (auto-generated)
├── 📁 .venv/                   # Virtual environment
├── 📄 requirements.txt         # Dependencies
├── 📄 .env                     # Environment variables
└── 📄 README.md               # Dokumentasi
```

## 👥 Kontributor

- **Developer**: [Your Name]
- **Project**: LawChain - Chatbot Hukum UUD 1945
- **Framework**: LangChain RAG Tutorial Implementation
- **Institution**: [Your Institution]

---

**⚖️ Disclaimer**: Project ini menggunakan teknologi AI untuk membantu pemahaman UUD 1945. Selalu verifikasi informasi dengan sumber resmi untuk keperluan legal yang serius.

**🔗 Referensi**:

- [LangChain RAG Tutorial](https://python.langchain.com/docs/tutorials/rag/)
- [Ollama Documentation](https://ollama.ai/)
- [UUD 1945 Resmi](https://www.dpr.go.id/)
  #
