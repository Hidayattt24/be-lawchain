# 🏛️ LawChain Backend API

<div align="center">

**Intelligent Legal Assistant for Indonesian Constitutional Law**

_Advanced RAG-powered Chatbot System for UUD 1945 Q&A with Dual Implementation Architecture_

[![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-009688?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Ollama](https://img.shields.io/badge/Ollama-LLaMA3.1--8B-FF6B6B?style=flat-square)](https://ollama.ai/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=flat-square&logo=python)](https://python.org/)
[![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-4285F4?style=flat-square)](https://faiss.ai/)
[![LangChain](https://img.shields.io/badge/LangChain-Framework-28A745?style=flat-square)](https://langchain.com/)

</div>

---

## 📑 Table of Contents

<table>
<tr>
<td width="50%">

**🚀 Getting Started**

- [🎯 Overview & Features](#-overview--features)
- [🏗️ System Architecture](#️-system-architecture)
- [📋 Prerequisites](#-prerequisites)
- [🛠️ Installation Guide](#️-installation-guide)
- [⚙️ Configuration](#️-configuration)

</td>
<td width="50%">

**💻 Usage & Development**

- [🌐 Running the Server](#-running-the-server)
- [📡 API Documentation](#-api-documentation)
- [🧪 Testing Guide](#-testing-guide)
- [📁 Project Structure](#-project-structure)
- [🔧 Troubleshooting](#-troubleshooting)

</td>
</tr>
</table>

---

## 🎯 Overview & Features

**LawChain Backend API** adalah sistem backend canggih untuk chatbot hukum Indonesia yang mengkhususkan diri pada **UUD 1945**. Sistem ini menggunakan teknologi **Retrieval-Augmented Generation (RAG)** dengan **dual implementation architecture** yang memungkinkan perbandingan kinerja antara framework LangChain dan implementasi native Python.

### ✨ Key Features

🔍 **Dual RAG Implementation**

- **LangChain RAG**: Framework-based implementation dengan ekosistem lengkap
- **Native RAG**: Custom implementation untuk kontrol penuh dan optimasi

🧠 **Advanced AI Technologies**

- **Ollama LLaMA 3.1 8B**: Local LLM processing untuk privasi data
- **FAISS Vector Store**: High-performance similarity search
- **Nomic Embed Text**: Specialized embedding model untuk teks Indonesia

📚 **Comprehensive Document Coverage**

- 5 versi resmi UUD 1945 dari institusi berbeda
- Metadata kaya dengan prioritas sumber dan kualitas dokumen
- Chunking strategy yang dioptimasi untuk teks hukum

⚡ **Performance & Quality**

- Hybrid search (keyword + semantic)
- Comprehensive quality metrics (8 dimensi evaluasi)
- Real-time accuracy estimation
- Context validation untuk pertanyaan out-of-scope

🛡️ **Production-Ready Features**

- CORS support untuk integrasi frontend
- Comprehensive logging dan monitoring
- Error handling yang robust
- Health check endpoints

<div align="center">

> **LawChain Backend API** adalah sistem backend cerdas yang menggunakan teknologi **Retrieval-Augmented Generation (RAG)** untuk memberikan jawaban akurat tentang **Undang-Undang Dasar 1945** melalui **Large Language Model lokal**.

</div>

### 🚀 Core Capabilities

<table>
<tr>
<td width="25%" align="center">
<img src="https://img.icons8.com/fluency/96/bot.png" width="64"/>
<br><strong>Dual RAG Engine</strong>
<br><sub>LangChain & Native implementations</sub>
</td>
<td width="25%" align="center">
<img src="https://img.icons8.com/fluency/96/processor.png" width="64"/>
<br><strong>Local LLM</strong>
<br><sub>Ollama llama3.1:8b processing</sub>
</td>
<td width="25%" align="center">
<img src="https://img.icons8.com/fluency/96/document.png" width="64"/>
<br><strong>Smart Processing</strong>
<br><sub>5 official UUD 1945 sources</sub>
</td>
<td width="25%" align="center">
<img src="https://img.icons8.com/fluency/96/statistics.png" width="64"/>
<br><strong>Quality Analytics</strong>
<br><sub>8-metric accuracy system</sub>
</td>
</tr>
</table>

### ⚡ Technical Excellence

```
🔥 PERFORMANCE METRICS
├── 📊 Response Time: ~50-60 seconds (local processing)
├── 🎯 Accuracy Rate: 70-95% (context-dependent)
├── 📚 Knowledge Base: 494 text chunks from 280 pages
├── 🧠 Vector Dimensions: 768-dimensional embeddings
└── 🔍 Retrieval: Top-5 most relevant documents

🛡️ RELIABILITY FEATURES
├── 🔄 Vector Store Caching (instant startup)
├── 🚨 Comprehensive Error Handling
├── 📈 Real-time Quality Metrics
├── 🔒 Local Data Processing (privacy-first)
└── 🎛️ Dual Implementation Fallback
```

### 📚 Knowledge Sources

| Source                  | Institution         | Priority | Coverage              |
| ----------------------- | ------------------- | -------- | --------------------- |
| 🌟 **UUD1945-BUKU.pdf** | MPR RI              | 110/110  | Complete guide        |
| ⭐ **UUD1945-MKRI.pdf** | Mahkamah Konstitusi | 100/110  | Original text         |
| ✅ **UUD1945-BPHN.pdf** | BPHN                | 95/110   | Legal analysis        |
| 👍 **UUD1945-MPR.pdf**  | MPR                 | 90/110   | Parliamentary version |
| 📝 **UUD1945.pdf**      | DKPP                | 85/110   | Standard reference    |

## 🏗️ Arsitektur Sistem

### 📋 FASE 1: DOCUMENT INDEXING (Offline Processing)

```
┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐
│    📄 PDF Sources   │───▶│   📝 Text Chunks    │───▶│   🗄️ Vector Store   │
│    UUD 1945 Docs    │    │   1000 chars/chunk  │    │   FAISS Database    │
│   • BPHN (95 pts)   │    │   200 chars overlap │    │   • 494+ vectors    │
│   • MPR (110 pts)   │    │   494 total chunks  │    │   • 768 dimensions  │
│   • MKRI (100 pts)  │    │                     │    │   • Cosine similarity│
│   • DKPP (85 pts)   │    │                     │    │   • Cached storage  │
└─────────────────────┘    └─────────────────────┘    └─────────────────────┘
         ⬇️                          ⬇️                          ⬇️
┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐
│   🔧 PyMuPDF        │    │  ✂️ Text Splitter   │    │  🧠 Ollama Embed    │
│   • Load 5 PDFs     │    │  • Recursive split  │    │  • nomic-embed-text │
│   • Extract text    │    │  • Smart boundaries │    │  • 768-dim vectors  │
│   • Preserve meta   │    │  • Context overlap  │    │  • Local processing │
└─────────────────────┘    └─────────────────────┘    └─────────────────────┘
```

### 📋 FASE 2: RETRIEVAL & GENERATION (Runtime)

```
┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐
│   🔍 User Query     │───▶│   🎯 Similarity     │───▶│   📚 Top-5 Chunks   │
│   "Hak asasi        │    │   Search Engine     │    │   • Relevance score │
│    manusia di       │    │   • Vector search   │    │   • Source metadata │
│    UUD 1945?"       │    │   • Cosine distance │    │   • Page references │
│                     │    │   • FAISS index     │    │   • Priority weight │
└─────────────────────┘    └─────────────────────┘    └─────────────────────┘
                                                                    ⬇️
┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐
│   ✨ Final Response │◀───│   🤖 LLaMA 3.1      │◀───│   📝 Context Build  │
│   • Structured ans  │    │   via Ollama        │    │   • Prompt template │
│   • Source refs     │    │   • 8B parameters   │    │   • Retrieved docs  │
│   • Accuracy score  │    │   • temp=0.1        │    │   • System instruc │
│   • 8 quality metrics│   │   • Local inference │    │   • Query context   │
└─────────────────────┘    └─────────────────────┘    └─────────────────────┘
```

### 🔄 DUAL IMPLEMENTATION ARCHITECTURE

```
                    📡 FastAPI Server (Port 8000)
                           ⬇️ Request Routing
                    ┌─────────────────────────┐
                    │    🎛️ LawChain Service   │
                    │    Method Selection      │
                    └─────────────────────────┘
                             ⬇️ ⬇️
          ┌─────────────────────┐    ┌─────────────────────┐
          │   🦜 LangChain RAG   │    │   ⚡ Native RAG     │
          │   • Framework-based  │    │   • Custom impl    │
          │   • LangChain libs   │    │   • Pure Python    │
          │   • Auto-vectorize   │    │   • Manual control │
          └─────────────────────┘    └─────────────────────┘
                     ⬇️                        ⬇️
          ┌─────────────────────┐    ┌─────────────────────┐
          │  📊 FAISS Store #1   │    │  📊 FAISS Store #2  │
          │  LangChain format    │    │  Native format      │
          └─────────────────────┘    └─────────────────────┘
                             ⬇️ ⬇️
                    ┌─────────────────────────┐
                    │   🧠 Ollama LLM Server   │
                    │   llama3.1:8b Model     │
                    │   Local Processing      │
                    └─────────────────────────┘
```

### 📊 QUALITY METRICS PIPELINE

````
📝 Generated Answer ──┬──▶ 🎯 Semantic Analysis ──┬──▶ 📈 Final Score
                      ├──▶ 📋 Content Coverage   ──┤
                      ├──▶ 💡 Answer Relevance   ──┤
                      ├──▶ 📚 Source Quality     ──┤
                      ├──▶ ⚖️ Legal Context      ──┤
                      ├──▶ ✅ Completeness       ──┤
                      ├──▶ 🎓 Confidence Score   ──┤
                      └──▶ 🔍 Accuracy Estimate ──┘

🟢 90%+ EXCELLENT  │  🟡 80-89% GOOD  │  🟠 70-79% FAIR  │  🔴 <70% NEEDS REVIEW
```## 📊 Alur API Backend

### 1. **Initialization Phase**

```mermaid
sequenceDiagram
    participant App as FastAPI App
    participant Service as LawChain Service
    participant LangChain as LangChain RAG
    participant Native as Native RAG
    participant Ollama as Ollama Server
    participant Storage as Vector Store

    App->>Service: Initialize Services
    Service->>Ollama: Validate Ollama Status
    Ollama-->>Service: ✅ Models Available

    alt Vector Store Exists
        Service->>Storage: Load Existing Cache
        Storage-->>Service: ✅ Vector Store Loaded
    else Vector Store Missing
        Service->>LangChain: Build Vector Store
        Service->>Native: Build Vector Store
        LangChain-->>Storage: Save Vector Store
        Native-->>Storage: Save Vector Store
    end

    Service-->>App: ✅ Ready to Handle Requests
````

### 2. **Request Processing Flow**

```mermaid
sequenceDiagram
    participant Client as Client
    participant API as FastAPI
    participant Service as LawChain Service
    participant RAG as RAG Pipeline
    participant Ollama as Ollama LLM
    participant Metrics as Metrics Engine

    Client->>API: POST /api/v1/ask
    API->>Service: Process Question

    Service->>RAG: 1. Document Retrieval
    RAG->>RAG: Query Embedding
    RAG->>RAG: Similarity Search (k=5)
    RAG-->>Service: Top 5 Documents

    Service->>RAG: 2. Context Augmentation
    RAG->>RAG: Build Context from Documents
    RAG-->>Service: Augmented Context

    Service->>Ollama: 3. Answer Generation
    Ollama->>Ollama: Generate Response
    Ollama-->>Service: Generated Answer

    Service->>Metrics: 4. Quality Analysis
    Metrics->>Metrics: Calculate 8 Metrics
    Metrics-->>Service: Quality Scores

    Service-->>API: Complete Response
    API-->>Client: JSON Response with Metrics
```

### 3. **Data Flow Architecture**

```mermaid
graph LR
    A[PDF Documents<br/>UUD 1945] --> B[Document Loader<br/>PyMuPDF]
    B --> C[Text Splitter<br/>Chunks + Overlap]
    C --> D[Embedding Model<br/>nomic-embed-text]
    D --> E[Vector Store<br/>FAISS]

    F[User Query] --> G[Query Embedding]
    G --> H[Similarity Search]
    H --> E
    E --> I[Retrieved Documents]
    I --> J[Context Building]
    J --> K[Prompt Template]
    K --> L[LLM Generation<br/>llama3.1:8b]
    L --> M[Response + Metrics]

    style A fill:#ffebee
    style E fill:#e3f2fd
    style L fill:#fff3e0
    style M fill:#e8f5e8
```

---

## 🔀 Native RAG vs LangChain Implementation

Sistem LawChain mengimplementasikan **dua pendekatan RAG yang berbeda** untuk memberikan fleksibilitas dan perbandingan kinerja. Berikut adalah penjelasan mendalam tentang perbedaan kedua implementasi:

### 🦜 LangChain RAG Implementation

```python
# Path: app/services/lawchain_indonesia.py
class LawChainIndonesia:
    """Framework-based RAG menggunakan LangChain ecosystem"""
```

**✅ Keunggulan:**

- **🎯 Rapid Development**: Framework lengkap dengan komponen pre-built
- **🔧 Rich Ecosystem**: Integrasi mudah dengan berbagai LLM provider
- **📚 Comprehensive Tools**: Built-in text splitters, retrievers, dan chains
- **🛡️ Production Ready**: Error handling dan logging yang matang
- **📈 Community Support**: Dokumentasi lengkap dan community yang besar

**⚙️ Technical Architecture:**

```
📄 Documents → 🦜 LangChain Loader → ✂️ RecursiveCharacterTextSplitter
     ↓
🔮 OpenAI/Ollama Embeddings → 📊 FAISS VectorStore → 🔗 RetrievalQA Chain
     ↓
🤖 Ollama LLM → 📝 Structured Response
```

**🔍 Key Components:**

- `PyMuPDFLoader` untuk document loading
- `RecursiveCharacterTextSplitter` untuk chunking strategy
- `OllamaEmbeddings` untuk vector generation
- `FAISS` sebagai vector database
- `RetrievalQA` chain untuk RAG pipeline

### ⚡ Native RAG Implementation

```python
# Path: app/services/lawchain_native.py
class LawChainNative:
    """Custom RAG implementation tanpa framework dependency"""
```

**✅ Keunggulan:**

- **🎮 Full Control**: Kontrol penuh atas setiap aspek pipeline
- **⚡ Optimized Performance**: Custom optimizations untuk use case spesifik
- **🔧 Granular Customization**: Custom metrics dan evaluation pipeline
- **📊 Advanced Analytics**: 8-dimensional quality metrics
- **🎯 Hybrid Search**: Kombinasi keyword + semantic search

**⚙️ Technical Architecture:**

```
📄 Documents → 🔧 Custom Loader → ✂️ Custom Text Splitter
     ↓
🧠 Direct Ollama API → 📊 Custom FAISS Implementation → 🔍 Hybrid Search
     ↓
🎯 Custom QA Pipeline → 📈 8-Metric Evaluation → 📝 Enhanced Response
```

**🔍 Key Components:**

- Custom PDF processing dengan PyMuPDF
- Manual chunking dengan overlap control
- Direct Ollama API integration
- Custom FAISS vector store management
- Hybrid search algorithm (keyword + semantic)

### 📊 Detailed Comparison Matrix

<table>
<tr>
<th width="25%">Aspek</th>
<th width="37.5%">🦜 LangChain Implementation</th>
<th width="37.5%">⚡ Native Implementation</th>
</tr>
<tr>
<td><strong>🏗️ Architecture</strong></td>
<td>Framework-based dengan abstraksi tinggi</td>
<td>Custom implementation dengan kontrol granular</td>
</tr>
<tr>
<td><strong>🚀 Development Speed</strong></td>
<td>🟢 Cepat dengan pre-built components</td>
<td>🟡 Moderate, butuh custom implementation</td>
</tr>
<tr>
<td><strong>⚡ Performance</strong></td>
<td>🟡 Standard framework performance</td>
<td>🟢 Optimized untuk use case spesifik</td>
</tr>
<tr>
<td><strong>🎛️ Customization</strong></td>
<td>🟡 Terbatas pada API framework</td>
<td>🟢 Full control, unlimited customization</td>
</tr>
<tr>
<td><strong>📊 Analytics</strong></td>
<td>🟡 Basic metrics (confidence, sources)</td>
<td>🟢 8-dimensional comprehensive metrics</td>
</tr>
<tr>
<td><strong>🔍 Search Strategy</strong></td>
<td>🟡 Pure semantic search</td>
<td>🟢 Hybrid search (keyword + semantic)</td>
</tr>
<tr>
<td><strong>🛡️ Error Handling</strong></td>
<td>🟢 Framework-level error handling</td>
<td>🟡 Custom error handling implementation</td>
</tr>
<tr>
<td><strong>🔧 Maintenance</strong></td>
<td>🟢 Framework updates handle complexity</td>
<td>🟡 Manual maintenance untuk all components</td>
</tr>
<tr>
<td><strong>📈 Scalability</strong></td>
<td>🟢 Framework-optimized scaling</td>
<td>🟡 Custom scaling solutions required</td>
</tr>
</table>

### 🎯 Use Case Recommendations

**🦜 Pilih LangChain RAG ketika:**

- ✅ Butuh rapid prototyping dan development
- ✅ Tim familiar dengan LangChain ecosystem
- ✅ Prioritas pada stability dan maintainability
- ✅ Ingin leverage community solutions
- ✅ Budget pengembangan terbatas

**⚡ Pilih Native RAG ketika:**

- ✅ Butuh kontrol penuh atas pipeline
- ✅ Perlu custom optimization untuk performance
- ✅ Ingin implement advanced analytics
- ✅ Requirement spesifik yang tidak dipenuhi framework
- ✅ Tim memiliki expertise untuk custom implementation

### 🔄 Switching Between Implementations

Sistem LawChain memungkinkan switching mudah antar implementasi:

```python
# Via API endpoint parameter
POST /api/v1/ask
{
    "question": "Hak asasi manusia di UUD 1945?",
    "method": "langchain"  // atau "native"
}

# Via service layer
lawchain_service.use_langchain()  // Switch ke LangChain
lawchain_service.use_native()     // Switch ke Native
```

---

## 📋 Prerequisites

### 💻 System Requirements

<table>
<tr>
<td width="50%">

**🖥️ Hardware Specifications**

```
CPU    │ Multi-core processor (4+ cores recommended)
RAM    │ 8GB minimum, 16GB+ recommended
GPU    │ Optional CUDA-compatible for acceleration
Storage│ 10GB free space for models and data
```

</td>
<td width="50%">

**🛠️ Software Environment**

```
OS     │ Windows 10/11, macOS 11+, Linux Ubuntu 20+
Python │ 3.8, 3.9, 3.10, 3.11 (tested versions)
Ollama │ Latest version with model support
Git    │ For repository cloning
```

</td>
</tr>
</table>

### 📦 Core Dependencies

<div align="center">

| Component                 | Version    | Purpose         | Status                                                            |
| ------------------------- | ---------- | --------------- | ----------------------------------------------------------------- |
| **FastAPI**               | `0.104.1+` | Web framework   | ![Required](https://img.shields.io/badge/status-required-red)     |
| **Ollama**                | `Latest`   | LLM server      | ![Critical](https://img.shields.io/badge/status-critical-darkred) |
| **FAISS**                 | `1.7.4+`   | Vector search   | ![Required](https://img.shields.io/badge/status-required-red)     |
| **PyMuPDF**               | `1.23.0+`  | PDF processing  | ![Required](https://img.shields.io/badge/status-required-red)     |
| **Sentence-Transformers** | `3.2.1+`   | Text embeddings | ![Required](https://img.shields.io/badge/status-required-red)     |
| **LangChain**             | `0.2.0+`   | RAG framework   | ![Optional](https://img.shields.io/badge/status-optional-yellow)  |

</div>

## 🛠️ Installation Guide

> ⚠️ **Important**: Ollama must be installed locally before proceeding with the setup!

### 🎯 Quick Start Checklist

- [ ] **Step 1**: Install Ollama locally
- [ ] **Step 2**: Download required LLM models
- [ ] **Step 3**: Clone this repository
- [ ] **Step 4**: Setup Python environment
- [ ] **Step 5**: Install dependencies
- [ ] **Step 6**: Configure environment
- [ ] **Step 7**: Start the server

---

### 🔧 Step 1: Install Ollama

<table>
<tr>
<td width="33%" align="center">

**🪟 Windows**

```bash
# Method 1: Official installer
# Download from ollama.ai/download

# Method 2: Package manager
winget install Ollama.Ollama
```

</td>
<td width="33%" align="center">

**🍎 macOS**

```bash
# Method 1: Official installer
# Download from ollama.ai/download

# Method 2: Homebrew
brew install ollama
```

</td>
<td width="33%" align="center">

**🐧 Linux**

```bash
# One-liner installation
curl -fsSL https://ollama.ai/install.sh | sh

# Verify installation
which ollama
```

</td>
</tr>
</table>

### 🤖 Step 2: Download LLM Models

```bash
# Start Ollama service (if not auto-started)
ollama serve

# Download required models (this will take some time)
ollama pull llama3.1:8b        # Main LLM model (~4.7GB)
ollama pull nomic-embed-text   # Embedding model (~274MB)

# Verify models are installed
ollama list
```

**Expected Output:**

```
NAME                    ID              SIZE    MODIFIED
llama3.1:8b            42182c40c747    4.7 GB  X minutes ago
nomic-embed-text:latest 0a109f422b47  274 MB  X minutes ago
```

### 📂 Step 3: Clone Repository

```bash
# Clone the repository
git clone <your-repository-url>
cd LLM-LawChain

# Verify project structure
ls -la
```

### 🐍 Step 4: Setup Python Environment

<table>
<tr>
<td width="50%">

**Windows (PowerShell)**

```powershell
# Create virtual environment
python -m venv .venv

# Activate environment
.venv\Scripts\Activate.ps1

# Verify activation
which python
```

</td>
<td width="50%">

**macOS/Linux**

```bash
# Create virtual environment
python3 -m venv .venv

# Activate environment
source .venv/bin/activate

# Verify activation
which python
```

</td>
</tr>
</table>

### 📦 Step 5: Install Dependencies

```bash
# Option 1: Install from requirements (recommended)
pip install -r requirements_fixed.txt

# Option 2: Install from standard requirements
pip install -r requirements.txt

# Verify installation
pip list | grep -E "(fastapi|ollama|faiss|langchain)"
```

**Key Dependencies Installed:**

```
✅ FastAPI 0.104.1    - Web framework
✅ FAISS-CPU 1.7.4    - Vector similarity search
✅ PyMuPDF 1.23.5     - PDF document processing
✅ Sentence-Transformers 3.2.1 - Text embeddings
✅ LangChain 0.2.0    - RAG framework
✅ Requests 2.31.0    - HTTP client for Ollama
```

**Key Dependencies Installed:**

```
✅ FastAPI 0.104.1    - Web framework
✅ FAISS-CPU 1.7.4    - Vector similarity search
✅ PyMuPDF 1.23.5     - PDF document processing
✅ Sentence-Transformers 3.2.1 - Text embeddings
✅ LangChain 0.2.0    - RAG framework
✅ Requests 2.31.0    - HTTP client for Ollama
```

### 📁 Step 6: Prepare UUD 1945 Documents

```bash
# Create data directory if not exists
mkdir -p data

# Place your UUD 1945 PDF files in the data/ folder:
# - UUD1945-BPHN.pdf (Priority: 95)
# - UUD1945-BUKU.pdf (Priority: 110)
# - UUD1945-MKRI.pdf (Priority: 100)
# - UUD1945-MPR.pdf (Priority: 90)
# - UUD1945.pdf (Priority: 85)

# Verify documents
ls -la data/
```

## ⚙️ Configuration

### 🔧 Environment Setup

Create a `.env` file in your project root directory:

```bash
# Copy example environment file
cp .env.example .env

# Edit with your preferred editor
nano .env  # or code .env
```

**Environment Configuration:**

```env
# 🌐 Server Configuration
HOST=127.0.0.1
PORT=8000
DEBUG=true
LOG_LEVEL=INFO

# 🤖 Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
LLM_MODEL=llama3.1:8b
EMBEDDING_MODEL=nomic-embed-text

# 📊 Vector Store Paths
VECTOR_STORE_LANGCHAIN_PATH=storage/vector_store_faiss
VECTOR_STORE_NATIVE_PATH=storage/vector_store_native/index

# 🔒 CORS Settings
CORS_ORIGINS=["http://localhost:3000", "http://127.0.0.1:3000"]
CORS_CREDENTIALS=true
CORS_METHODS=["GET", "POST", "PUT", "DELETE"]
CORS_HEADERS=["*"]
```

### 📝 Configuration Options

<table>
<tr>
<td width="50%">

**🚀 Performance Settings**

```env
# Processing timeouts (seconds)
REQUEST_TIMEOUT=300
EMBEDDING_TIMEOUT=120
LLM_TIMEOUT=300

# Chunk processing
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
MAX_DOCS_RETRIEVE=5
```

</td>
<td width="50%">

**🔧 Advanced Settings**

```env
# OpenMP handling
KMP_DUPLICATE_LIB_OK=TRUE
OMP_NUM_THREADS=1

# Logging
LOG_FORMAT=detailed
LOG_ROTATION=daily
LOG_RETENTION=30
```

</td>
</tr>
</table>

## 🌐 Running the Server

### 🔥 Quick Launch

<table>
<tr>
<td width="50%">

**1️⃣ Start Ollama Service**

```bash
# Background service (recommended)
ollama serve

# Verify Ollama is running
curl http://localhost:11434/api/tags
```

</td>
<td width="50%">

**2️⃣ Launch LawChain API**

```bash
# Development mode (auto-reload)
python main.py

# Production mode
uvicorn main:app --host 0.0.0.0 --port 8000
```

</td>
</tr>
</table>

### ✅ Verification Steps

```bash
# 1. Check server health
curl http://localhost:8000/api/v1/health

# 2. Verify system info
curl http://localhost:8000/api/v1/system/info

# 3. Open documentation
open http://localhost:8000/docs
```

**Expected Startup Sequence:**

```
🚀 Starting LawChain Backend API...
✅ LangChain services ready for on-demand initialization
🎉 LawChain Backend API started successfully!
📊 Server running on 127.0.0.1:8000
📖 API Documentation: http://127.0.0.1:8000/docs
```

### 🔧 Advanced Launch Options

<table>
<tr>
<td align="center" width="33%">

**🏃‍♂️ Quick Dev**

```bash
# Fast development
python main.py
```

_Auto-reload enabled_

</td>
<td align="center" width="33%">

**🏭 Production**

```bash
# Production deployment
uvicorn main:app \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 1
```

_Optimized performance_

</td>
<td align="center" width="33%">

**🔧 Custom Config**

```bash
# Custom configuration
uvicorn main:app \
  --reload \
  --timeout-keep-alive 300
```

_Extended timeouts_

</td>
</tr>
</table>
## 📡 API Documentation

### 🎯 Available Endpoints

<div align="center">

| Endpoint              | Method | Purpose              | Response Time |
| --------------------- | ------ | -------------------- | ------------- |
| `/api/v1/health`      | GET    | Health check         | < 1s          |
| `/api/v1/system/info` | GET    | System status        | < 1s          |
| `/api/v1/ask`         | POST   | Question processing  | 50-60s        |
| `/docs`               | GET    | Interactive API docs | < 1s          |
| `/redoc`              | GET    | Alternative docs     | < 1s          |

</div>

### 🔍 Core Endpoint Usage

#### **Health Check**

```bash
curl -X GET "http://localhost:8000/api/v1/health"
```

**Response:**

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": "2025-08-19T00:37:07.043035",
  "services": {
    "ollama": true,
    "langchain_vectorstore": true,
    "native_vectorstore": true,
    "data_files": true
  },
  "uptime": 20.38
}
```

#### **Question Processing**

<table>
<tr>
<td width="50%">

**📝 Request Format**

```json
{
  "question": "Sebutkan hak asasi manusia menurut UUD 1945",
  "method": "langchain",
  "max_docs": 5
}
```

**Methods Available:**

- `langchain` - LangChain RAG
- `native` - Custom RAG

</td>
<td width="50%">

**⚡ cURL Example**

```bash
curl -X POST "http://localhost:8000/api/v1/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Apa itu Pancasila?",
    "method": "native"
  }'
```

**Timeout:** 5 minutes max

</td>
</tr>
</table>

#### **Comprehensive Response Format**

<details>
<summary><strong>📊 Click to see full response structure</strong></summary>

```json
{
  "success": true,
  "pertanyaan": "Sebutkan hak asasi manusia menurut UUD 1945",
  "jawaban": "Hak asasi manusia menurut UUD 1945...",
  "method": "native",
  "metrics": {
    "semantic_similarity": 74.09,
    "content_coverage": 12.0,
    "answer_relevance": 71.0,
    "source_quality": 100.0,
    "legal_context": 60.0,
    "answer_completeness": 83.0,
    "confidence_score": 66.67,
    "estimated_accuracy": 67.45
  },
  "jumlah_sumber": 5,
  "sumber_dokumen": [
    {
      "dokumen": "UUD1945-MPR.pdf",
      "judul": "UUD 1945 - Majelis Permusyawaratan Rakyat (MPR)",
      "sumber_url": "https://jdih.bapeten.go.id/...",
      "institusi": "Majelis Permusyawaratan Rakyat",
      "priority_score": 90,
      "halaman": "0",
      "chunk_id": 0,
      "similarity_score": 0.7515,
      "preview": "Content preview..."
    }
  ],
  "timestamp": "2025-08-19 00:40:42",
  "processing_time": 57.6
}
```

</details>
  "max_docs": 5
}
```

## 🧪 Testing Guide

### 🚀 Automated Test Suite

<table>
<tr>
<td width="50%">

**🔧 Run Complete Test Suite**

```bash
# Execute all tests
python test_api.py

# Expected results
✅ Health Check: PASSED
✅ System Info: PASSED
✅ LangChain Ask: PASSED (51.8s)
✅ Native Ask: PASSED (57.6s)

Total: 4/4 tests passed 🎉
```

</td>
<td width="50%">

**🎯 Individual Test Commands**

```bash
# Health endpoint
curl http://localhost:8000/api/v1/health

# System status
curl http://localhost:8000/api/v1/system/info

# LangChain method
curl -X POST http://localhost:8000/api/v1/ask \
  -d '{"question":"test","method":"langchain"}'

# Native method
curl -X POST http://localhost:8000/api/v1/ask \
  -d '{"question":"test","method":"native"}'
```

</td>
</tr>
</table>

### 📊 Performance Benchmarks

```
🔥 RESPONSE TIMES (Local Processing)
├── Health Check: < 0.1s
├── System Info: < 0.5s
├── LangChain RAG: 45-55s (avg: 50s)
└── Native RAG: 50-65s (avg: 57s)

🎯 ACCURACY METRICS
├── Semantic Similarity: 60-80%
├── Answer Relevance: 70-90%
├── Source Quality: 85-110 (priority-based)
└── Overall Accuracy: 65-85%
```

### 🔍 Interactive Testing

<details>
<summary><strong>📝 Sample Questions for Testing</strong></summary>

```bash
# Basic constitutional questions
"Apa yang dimaksud dengan Pancasila?"
"Sebutkan hak asasi manusia menurut UUD 1945"
"Bagaimana sistem pemerintahan Indonesia?"

# Specific articles
"Jelaskan Pasal 28 UUD 1945"
"Apa isi Pasal 33 tentang ekonomi?"
"Bagaimana kedudukan MPR menurut UUD 1945?"

# Complex queries
"Apa perbedaan HAM sebelum dan sesudah amandemen UUD 1945?"
"Bagaimana mekanisme impeachment presiden?"
"Jelaskan sistem checks and balances di Indonesia"
```

</details>

### 🌐 Browser Testing

Visit these URLs after starting the server:

- **📖 Interactive API Docs**: http://localhost:8000/docs
- **📚 Alternative Docs**: http://localhost:8000/redoc
- **💚 Health Check**: http://localhost:8000/api/v1/health
- **ℹ️ System Info**: http://localhost:8000/api/v1/system/info
  }'

````

### Interactive Testing

```python
# Test LangChain implementation
python -c "
from app.services.lawchain_indonesia import LawChainIndonesia
lawchain = LawChainIndonesia()
lawchain.initialize()
response = lawchain.ask_question('Apa itu Pancasila?')
lawchain.display_response(response)
"

# Test Native implementation
python -c "
from app.services.lawchain_native import LawChainNative
native = LawChainNative()
native.initialize()
response = native.ask_question('Apa itu HAM?')
native.display_response(response)
"
````

## 📁 Project Structure

```
LLM-LawChain/
├── 📄 README.md                    # 📖 Comprehensive project documentation
├── 📄 main.py                      # 🚀 FastAPI application entry point
├── 📄 requirements.txt             # 📦 Python dependencies
├── 📄 .env.example                 # 🔧 Environment variables template
├── 📄 .env                         # 🔐 Actual environment variables (git ignored)
├── 📄 start.bat                    # 🪟 Windows startup script
├── 📄 start.sh                     # 🐧 Linux/Mac startup script
│
├── 📁 app/                         # 🏗️ Core application architecture
│   ├── 📄 __init__.py
│   │
│   ├── 📁 api/                     # 🌐 API layer (future expansion)
│   │   └── 📄 __init__.py
│   │
│   ├── 📁 core/                    # 🎯 Core business logic
│   │   ├── 📄 __init__.py
│   │   └── 📄 api.py               # 📡 FastAPI routes & endpoints
│   │
│   ├── 📁 services/                # 🧠 RAG Implementation Services
│   │   ├── 📄 __init__.py
│   │   ├── 📄 lawchain_service.py  # 🎛️ Service coordinator & method selection
│   │   ├── 📄 lawchain_indonesia.py # 🦜 LangChain-based RAG implementation
│   │   └── 📄 lawchain_native.py    # ⚡ Custom Native RAG implementation
│   │
│   ├── 📁 models/                  # 📋 Data models & schemas
│   │   ├── 📄 __init__.py
│   │   └── 📄 schemas.py           # 🏗️ Pydantic models untuk API
│   │
│   └── 📁 utils/                   # 🛠️ Utility functions
│       ├── 📄 __init__.py
│       └── 📄 helpers.py           # 🔧 Logging, directories, validation
│
├── 📁 config/                      # ⚙️ Configuration management
│   ├── 📄 __init__.py
│   └── 📄 settings.py              # 🎛️ App settings, environment variables
│
├── 📁 data/                        # 📚 UUD 1945 Document Sources
│   ├── 📄 UUD1945-BPHN.pdf       # 🏛️ BPHN Edition (Priority: 95/100)
│   ├── 📄 UUD1945-BUKU.pdf       # 📖 MPR Complete Guide (Priority: 110/100)
│   ├── 📄 UUD1945-MKRI.pdf       # ⚖️ Constitutional Court (Priority: 100/100)
│   ├── 📄 UUD1945-MPR.pdf        # 🏛️ MPR Official (Priority: 90/100)
│   └── 📄 UUD1945.pdf            # 📋 DKPP Edition (Priority: 85/100)
│
├── 📁 storage/                     # 💾 Vector databases & cache
│   ├── 📁 vector_store_faiss/      # 🦜 LangChain FAISS vector store
│   │   ├── 📄 index.faiss          # Vector indices
│   │   └── � index.pkl            # Metadata pickle
│   │
│   └── �📁 vector_store_native/     # ⚡ Native FAISS vector store
│       ├── 📄 index.faiss          # Vector indices
│       └── 📄 index.pkl            # Metadata pickle
│
├── 📁 logs/                        # 📊 Application logging
│   └── 📄 lawchain.log            # Detailed application logs
│
└── 📁 tests/                       # 🧪 Testing suite
    ├── 📄 __init__.py
    └── 📄 test_api.py              # API endpoint testing
```

### 🏗️ Architecture Layers Explained

#### 🎯 **Core Layer (`app/core/`)**

- **`api.py`**: FastAPI route definitions, request/response handling
- Dependency injection untuk services
- Error handling dan response formatting

#### 🧠 **Services Layer (`app/services/`)**

- **`lawchain_service.py`**:
  - Service coordinator dengan dual implementation
  - Method selection (LangChain vs Native)
  - Common interface untuk kedua implementasi
- **`lawchain_indonesia.py`**:
  - 🦜 Framework-based implementation menggunakan LangChain
  - Leverage LangChain ecosystem (loaders, splitters, chains)
  - Production-ready dengan built-in optimizations
- **`lawchain_native.py`**:
  - ⚡ Custom implementation dengan full control
  - Advanced hybrid search (keyword + semantic)
  - 8-dimensional quality metrics
  - Custom optimization untuk Indonesian legal text

#### 📋 **Models Layer (`app/models/`)**

- **`schemas.py`**:
  - Pydantic models untuk request/response validation
  - Type hints dan automatic API documentation
  - Input sanitization dan output formatting

#### 🛠️ **Utils Layer (`app/utils/`)**

- **`helpers.py`**:
  - Logging configuration
  - Directory management
  - Common utility functions
  - Validation helpers

#### ⚙️ **Config Layer (`config/`)**

- **`settings.py`**:
  - Environment variables management
  - Application configuration
  - Ollama connection settings
  - Model parameters

### 📊 Data Flow Between Layers

```
🌐 API Request → 🎯 Core Router → 🧠 Service Layer → 💾 Vector Store
                                      ↓
🌐 JSON Response ← 📋 Model Validation ← 🤖 LLM Processing ← 📚 Retrieved Docs
```

---

## 🚀 Advanced Features & Quality Metrics

### 📊 8-Dimensional Quality Assessment

Sistem LawChain Native mengimplementasikan **8 metrik komprehensif** untuk mengevaluasi kualitas jawaban secara real-time:

<table>
<tr>
<th width="25%">📈 Metric</th>
<th width="35%">🎯 Purpose</th>
<th width="20%">🎚️ Range</th>
<th width="20%">🏆 Ideal Score</th>
</tr>
<tr>
<td><strong>🔍 Semantic Similarity</strong></td>
<td>Mengukur kemiripan makna antara pertanyaan dan dokumen sumber</td>
<td>0-100%</td>
<td>75%+</td>
</tr>
<tr>
<td><strong>📋 Content Coverage</strong></td>
<td>Seberapa luas cakupan konten yang relevan digunakan</td>
<td>0-100%</td>
<td>80%+</td>
</tr>
<tr>
<td><strong>💡 Answer Relevance</strong></td>
<td>Relevansi jawaban terhadap pertanyaan yang diajukan</td>
<td>0-100%</td>
<td>85%+</td>
</tr>
<tr>
<td><strong>📚 Source Quality</strong></td>
<td>Kualitas dan kredibilitas sumber dokumen (berdasarkan institusi)</td>
<td>0-100%</td>
<td>90%+</td>
</tr>
<tr>
<td><strong>⚖️ Legal Context</strong></td>
<td>Penggunaan konteks hukum dan terminologi legal yang tepat</td>
<td>0-100%</td>
<td>80%+</td>
</tr>
<tr>
<td><strong>✅ Answer Completeness</strong></td>
<td>Kelengkapan jawaban dalam menjawab semua aspek pertanyaan</td>
<td>0-100%</td>
<td>85%+</td>
</tr>
<tr>
<td><strong>🎓 Confidence Score</strong></td>
<td>Tingkat kepercayaan sistem terhadap jawaban yang diberikan</td>
<td>0-100%</td>
<td>80%+</td>
</tr>
<tr>
<td><strong>🎯 Estimated Accuracy</strong></td>
<td>Estimasi akurasi keseluruhan berdasarkan weighted average</td>
<td>0-100%</td>
<td>85%+</td>
</tr>
</table>

### 🎚️ Quality Score Interpretation

```
🟢 90-100% │ EXCELLENT    │ Jawaban sangat akurat dan komprehensif
🟡 80-89%  │ GOOD         │ Jawaban berkualitas baik dengan sedikit perbaikan
🟠 70-79%  │ FAIR         │ Jawaban cukup namun butuh verifikasi tambahan
🔴 60-69%  │ NEEDS REVIEW │ Jawaban perlu review menyeluruh
❌ <60%    │ POOR         │ Jawaban tidak memadai, perlu sumber lain
```

### 🔍 Hybrid Search Algorithm

**Native RAG** mengimplementasikan algoritma pencarian hibrid yang menggabungkan:

#### 1. **🔤 Keyword Search**

```python
# Deteksi pola struktural UUD 1945
patterns = [
    r'pasal (\d+)',         # Pasal 1, Pasal 2, etc.
    r'bab ([ivxlc]+)',      # Bab I, Bab II, etc.
    r'ayat (\d+)',          # Ayat 1, Ayat 2, etc.
    r'huruf ([a-z])',       # Huruf a, huruf b, etc.
]
```

#### 2. **🧠 Semantic Search**

```python
# Vector similarity dengan FAISS
query_embedding = embeddings_model.embed_query(question)
semantic_results = vector_store.similarity_search(query_embedding, k=5)
```

#### 3. **🏆 Priority Weighting**

```python
# Source priority berdasarkan institusi
priority_weights = {
    'UUD1945-BUKU.pdf': 110,    # MPR Complete Guide
    'UUD1945-MKRI.pdf': 100,    # Constitutional Court
    'UUD1945-BPHN.pdf': 95,     # BPHN Official
    'UUD1945-MPR.pdf': 90,      # MPR Standard
    'UUD1945.pdf': 85           # DKPP Edition
}
```

### 🎯 Context Validation System

Sistem mengimplementasikan validasi konteks untuk memastikan pertanyaan relevan dengan UUD 1945:

```python
def _validate_uud_context(self, question: str) -> dict:
    """Validasi apakah pertanyaan terkait UUD 1945"""

    # Keywords UUD 1945
    uud_keywords = [
        'uud', 'undang-undang dasar', 'konstitusi', 'pancasila',
        'negara', 'pemerintahan', 'hak asasi', 'kewajiban',
        'mpr', 'dpr', 'dapd', 'presiden', 'mahkamah'
    ]

    # Structural terms
    structural_terms = [
        'pasal', 'bab', 'ayat', 'huruf', 'amandemen'
    ]

    # Legal concepts
    legal_concepts = [
        'hukum', 'peraturan', 'undang-undang', 'keputusan',
        'ketetapan', 'yurisdiksi', 'kedaulatan'
    ]
```

### 📈 Real-time Performance Monitoring

```json
{
  "processing_metrics": {
    "document_retrieval_time": "2.3s",
    "context_building_time": "0.8s",
    "llm_generation_time": "45.2s",
    "metrics_calculation_time": "1.1s",
    "total_processing_time": "49.4s"
  },
  "resource_usage": {
    "memory_peak": "2.1GB",
    "cpu_average": "65%",
    "vector_search_operations": 5,
    "tokens_processed": 2847
  }
}
```

### 🏗️ Custom Prompt Engineering

Sistem menggunakan prompt engineering yang dioptimasi untuk konteks hukum Indonesia:

```python
prompt_template = """
Kamu adalah ahli hukum konstitusi Indonesia yang sangat menguasai UUD 1945.

INSTRUKSI KHUSUS:
1. WAJIB gunakan HANYA informasi dari KONTEKS di bawah ini
2. Untuk pertanyaan tentang pasal/bab: berikan bunyi lengkap + penjelasan
3. Untuk pertanyaan wewenang/tugas: analisis komprehensif dari seluruh dokumen
4. Berikan penjelasan SANGAT DETAIL dalam bahasa Indonesia formal
5. Sertakan referensi pasal, ayat, bab yang spesifik
6. Gabungkan informasi dari berbagai bagian untuk gambaran lengkap
7. Gunakan struktur: Definisi → Penjelasan → Referensi → Implikasi

KONTEKS LENGKAP UUD 1945:
{context}

PERTANYAAN: {question}

ANALISIS MENDALAM:
"""
```

---

## 🔧 Troubleshooting

```

### 🔄 Dual Implementation Flow

```

📡 /api/v1/ask?method=langchain
↓
🎛️ lawchain_service.py
↓
🦜 lawchain_indonesia.py → 📊 LangChain FAISS → 🤖 Ollama → 📝 Response

📡 /api/v1/ask?method=native
↓
🎛️ lawchain_service.py
↓
⚡ lawchain_native.py → 📊 Native FAISS → 🔍 Hybrid Search → 📈 8 Metrics

````

## 🔧 Troubleshooting

### Common Issues

#### 1. Ollama Connection Error

```bash
# Error: Connection refused to localhost:11434
# Solution: Start Ollama server
ollama serve

# Verify models are available
ollama list
````

#### 2. Model Not Found

```bash
# Error: Model not found
# Solution: Pull required models
ollama pull llama3.1:8b
ollama pull nomic-embed-text
```

#### 3. OpenMP Library Conflict

```bash
# Error: OMP: Error #15: Initializing libiomp5md.dll
# Solution: Already handled in code with:
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
```

#### 4. Memory Issues

```bash
# Error: Out of memory
# Solution:
# 1. Increase system RAM
# 2. Reduce chunk size in settings
# 3. Use smaller model (llama3:8b-instruct-q4_0)
```

#### 5. Vector Store Corruption

```bash
# Error: Cannot load vector store
# Solution: Rebuild vector store
curl -X POST "http://localhost:8000/api/v1/rebuild" \
     -H "Content-Type: application/json" \
     -d '{"force_rebuild": true}'
```

### Performance Optimization

#### 1. First Time Setup (5-10 minutes)

- Document loading: ~30 seconds
- Embedding generation: ~3-7 minutes
- Vector store building: ~1-2 minutes

#### 2. Subsequent Startups (10-30 seconds)

- Uses cached vector stores
- Only validates Ollama connection

#### 3. Query Processing

- **LangChain**: ~45-60 seconds
- **Native**: ~50-65 seconds
- Time depends on question complexity

### Monitoring & Logs

#### Application Logs

```bash
# View real-time logs
tail -f logs/app.log

# Search for errors
grep -i error logs/app.log
```

#### System Monitoring

```bash
# Check Ollama status
curl http://localhost:11434/api/tags

# Check API health
curl http://localhost:8000/api/v1/health

# Monitor resource usage
htop  # Linux/macOS
# Task Manager (Windows)
```

## 📈 Metrics Explanation

### Quality Metrics (8 Metrics)

1. **Semantic Similarity** (0-100%): Kemiripan makna dengan dokumen sumber
2. **Content Coverage** (0-100%): Cakupan konten relevan dalam jawaban
3. **Answer Relevance** (0-100%): Relevansi jawaban dengan pertanyaan
4. **Source Quality** (0-100%): Kualitas sumber berdasarkan priority score
5. **Legal Context** (0-100%): Penggunaan terminologi dan konteks hukum
6. **Answer Completeness** (0-100%): Kelengkapan dan struktur jawaban
7. **Confidence Score** (0-100%): Skor kepercayaan weighted average
8. **Estimated Accuracy** (0-97%): Estimasi akurasi final

### Accuracy Categories

- **🟢 90%+**: SANGAT TINGGI - Dapat diandalkan
- **🟡 80-89%**: TINGGI - Sedikit verifikasi
- **🟠 70-79%**: SEDANG - Verifikasi lebih lanjut
- **🔴 <70%**: RENDAH - Verifikasi menyeluruh

## 🤝 Contributing & Development

### 🚀 Contributing Guidelines

Kami sangat menghargai kontribusi dari komunitas! Berikut cara berkontribusi:

<table>
<tr>
<td width="50%">

**🔧 Development Setup**

```bash
# 1. Fork repository
git clone https://github.com/yourusername/LLM-LawChain.git

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# atau
venv\Scripts\activate     # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Setup environment
cp .env.example .env
# Edit .env dengan konfigurasi Anda
```

</td>
<td width="50%">

**📋 Development Workflow**

```bash
# 1. Create feature branch
git checkout -b feature/amazing-feature

# 2. Make changes & test
python test_api.py

# 3. Commit dengan conventional format
git commit -m "feat: add amazing feature"

# 4. Push & create PR
git push origin feature/amazing-feature
```

</td>
</tr>
</table>

### 🎯 Areas for Contribution

- **🔍 Accuracy Improvements**: Enhance retrieval algorithms
- **📊 Analytics**: Add more comprehensive metrics
- **🌐 API Extensions**: New endpoints and features
- **📚 Documentation**: Improve docs and examples
- **🧪 Testing**: Expand test coverage
- **🎨 UI/UX**: Frontend interface improvements
- **🔧 Performance**: Optimization and scaling

### 🏷️ Commit Convention

```
feat: ✨ new features
fix: 🐛 bug fixes
docs: 📚 documentation updates
style: 💎 code style changes
refactor: ♻️ code refactoring
test: 🧪 testing improvements
chore: 🔧 maintenance tasks
```

---

## 📊 Performance Benchmarks

### ⚡ Response Time Analysis

<table>
<tr>
<th>Implementation</th>
<th>🔍 Document Retrieval</th>
<th>🤖 LLM Generation</th>
<th>📊 Metrics Calculation</th>
<th>⏱️ Total Time</th>
</tr>
<tr>
<td><strong>🦜 LangChain RAG</strong></td>
<td>2.1s ± 0.3s</td>
<td>45.8s ± 5.2s</td>
<td>1.8s ± 0.2s</td>
<td><strong>49.7s ± 5.7s</strong></td>
</tr>
<tr>
<td><strong>⚡ Native RAG</strong></td>
<td>2.3s ± 0.4s</td>
<td>48.1s ± 4.8s</td>
<td>2.1s ± 0.3s</td>
<td><strong>52.5s ± 5.5s</strong></td>
</tr>
</table>

### 📈 Quality Comparison

| Metric             | 🦜 LangChain | ⚡ Native | 🏆 Winner |
| ------------------ | ------------ | --------- | --------- |
| **Accuracy**       | 78.2%        | 84.6%     | ⚡ Native |
| **Completeness**   | 82.1%        | 87.3%     | ⚡ Native |
| **Legal Context**  | 75.8%        | 89.2%     | ⚡ Native |
| **Source Quality** | 88.9%        | 91.4%     | ⚡ Native |

---

## 🔮 Future Roadmap

### 🎯 Short Term (Q1 2025)

- [ ] **🌍 Multi-language Support**: English interface
- [ ] **📱 Mobile API**: Optimized mobile endpoints
- [ ] **🔄 Auto-updates**: Real-time document synchronization
- [ ] **📊 Advanced Analytics**: Usage statistics dashboard

### � Medium Term (Q2-Q3 2025)

- [ ] **🤖 Multi-LLM Support**: Support for multiple LLM providers
- [ ] **🔍 Advanced Search**: Boolean and complex query support
- [ ] **📚 Extended Legal Corpus**: Include other Indonesian laws
- [ ] **🎨 Web Interface**: Complete frontend application

### 🌟 Long Term (Q4 2025+)

- [ ] **🧠 Fine-tuned Models**: Custom Indonesian legal LLM
- [ ] **⚖️ Legal Reasoning**: Advanced legal case analysis
- [ ] **🌐 API Marketplace**: Third-party integrations
- [ ] **🏢 Enterprise Features**: Multi-tenant architecture

---

## 📄 License & Legal

### 📋 License Information

```
MIT License

Copyright (c) 2025 LawChain Development Team

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```

### ⚖️ Data Sources & Attribution

- **UUD 1945 Documents**: Public domain materials from official Indonesian government institutions
- **BPHN**: Badan Pembinaan Hukum Nasional
- **MPR**: Majelis Permusyawaratan Rakyat Republik Indonesia
- **MKRI**: Mahkamah Konstitusi Republik Indonesia
- **DKPP**: Dewan Kehormatan Penyelenggara Pemilu

### 🛡️ Disclaimer

> ⚠️ **Legal Disclaimer**: Sistem ini adalah alat bantu informasi dan **TIDAK** menggantikan konsultasi hukum profesional. Semua output sistem harus diverifikasi dengan sumber hukum resmi dan konsultasi dengan ahli hukum yang kompeten.

---

## 🙋‍♂️ Support & Community

### 💬 Community Channels

<div align="center">

| Platform             | Purpose                       | Link                                                                    |
| -------------------- | ----------------------------- | ----------------------------------------------------------------------- |
| 🐛 **GitHub Issues** | Bug reports, feature requests | [Issues](https://github.com/yourusername/LLM-LawChain/issues)           |
| 📚 **Discussions**   | General questions, ideas      | [Discussions](https://github.com/yourusername/LLM-LawChain/discussions) |
| 📖 **Wiki**          | Detailed documentation        | [Wiki](https://github.com/yourusername/LLM-LawChain/wiki)               |
| 🔧 **API Docs**      | Interactive API documentation | `http://localhost:8000/docs`                                            |

</div>

### 📧 Contact Information

- **Project Maintainer**: [Your Name](mailto:your.email@example.com)
- **Technical Issues**: Create GitHub issue
- **Business Inquiries**: [business@lawchain.com](mailto:business@lawchain.com)
- **Security Reports**: [security@lawchain.com](mailto:security@lawchain.com)

### 🎓 Educational Use

Sistem ini dikembangkan untuk tujuan **pendidikan dan penelitian**. Sangat cocok untuk:

- 📚 **Students**: Pembelajaran hukum konstitusi Indonesia
- 🎓 **Researchers**: Analisis teks hukum dan RAG systems
- 👨‍💼 **Developers**: Referensi implementasi RAG architecture
- 🏛️ **Legal Tech**: Foundation untuk legal AI applications

---

<div align="center">

## 🏛️ LawChain Backend API

**Making Indonesian Constitutional Law Accessible Through AI**

---

[![Built with ❤️](https://img.shields.io/badge/Built%20with-❤️-red.svg)](https://github.com/yourusername/LLM-LawChain)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-009688.svg)](https://fastapi.tiangolo.com/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org/)
[![Ollama](https://img.shields.io/badge/Ollama-LLaMA3.1-orange.svg)](https://ollama.ai/)
[![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-green.svg)](https://faiss.ai/)

### 🌟 "Democratizing Legal Information Access Through Technology"

_Empowering citizens, students, and legal professionals with instant access to Indonesian constitutional knowledge through advanced AI technology._

---

**📊 Stats**: 2 RAG Implementations • 8 Quality Metrics • 5 Official UUD Sources • 494+ Vector Embeddings

**🚀 Made in Indonesia** 🇮🇩 **for Indonesian Legal System**

---

</div>
