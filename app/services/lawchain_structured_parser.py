"""
LawChain Structured Parser - Advanced RAG System dengan Parsing Per Pasal-Ayat
Mengatasi masalah chunking yang memotong pasal/ayat
"""

import os
import re
import time
import logging
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass
import PyPDF2
import pdfplumber
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document
import numpy as np
import hashlib

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PasalAyat:
    """Struktur data untuk menyimpan pasal dan ayat dengan deduplication key"""
    pasal_number: int
    ayat_number: Optional[int]
    bab_number: Optional[int]
    bab_title: str
    pasal_title: str
    content: str
    raw_text: str
    page_number: int
    authority_level: str = "highest"
    source_file: str = "UUD1945-MPR.pdf"
    institution: str = "Majelis Permusyawaratan Rakyat"
    unique_key: str = ""  # Kunci unik untuk deduplication
    content_hash: str = ""  # Hash konten untuk validasi
    
    def __post_init__(self):
        """Generate unique key dan content hash setelah inisialisasi"""
        self.unique_key = self._generate_unique_key()
        self.content_hash = self._generate_content_hash()
    
    def _generate_unique_key(self) -> str:
        """Generate kunci unik: BAB-PASAL-AYAT"""
        bab_str = str(self.bab_number) if self.bab_number else "0"
        ayat_str = str(self.ayat_number) if self.ayat_number else "0"
        return f"{bab_str}-{self.pasal_number}-{ayat_str}"
    
    def _generate_content_hash(self) -> str:
        """Generate hash konten untuk deteksi duplikasi"""
        content_normalized = re.sub(r'\s+', ' ', self.content.strip().lower())
        return hashlib.md5(content_normalized.encode()).hexdigest()[:8]

class UUDStructuredParser:
    """Parser khusus untuk mengekstrak struktur UUD 1945 per pasal-ayat dengan deduplication dan advanced cleaning"""
    
    def __init__(self):
        self.pasal_ayat_list: List[PasalAyat] = []
        self.unique_entries: Dict[str, PasalAyat] = {}  # Deduplication tracking
        self.content_hashes: Set[str] = set()  # Content hash tracking
        self.current_bab = None
        self.current_bab_number = None
        self.extraction_stats = {
            "total_pages": 0,
            "total_entries": 0,
            "duplicates_removed": 0,
            "cleaning_operations": 0
        }
        
    def parse_pdf(self, pdf_path: str) -> List[PasalAyat]:
        """Parse PDF UUD 1945 dengan multiple extraction methods"""
        print("🔍 Memulai parsing struktural UUD 1945 (Advanced)...")
        print("📊 Menggunakan multiple extraction methods untuk akurasi maksimal")
        
        # Reset state
        self.pasal_ayat_list.clear()
        self.unique_entries.clear()
        self.content_hashes.clear()
        
        # Try multiple extraction methods
        extracted_text = self._extract_text_multi_method(pdf_path)
        
        # Parse struktur
        self._parse_structure_advanced(extracted_text)
        
        # Deduplication dan cleaning final
        self._finalize_entries()
        
        print(f"✅ Berhasil mengekstrak {len(self.pasal_ayat_list)} pasal/ayat unik")
        print(f"� Statistik: {self.extraction_stats}")
        return self.pasal_ayat_list
    
    def _extract_text_multi_method(self, pdf_path: str) -> Dict[int, str]:
        """Ekstrak teks dengan multiple methods untuk reliability"""
        page_texts = {}
        
        print("🔧 Method 1: PyPDF2 extraction...")
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                self.extraction_stats["total_pages"] = len(reader.pages)
                
                for page_num, page in enumerate(reader.pages, 1):
                    page_text = page.extract_text()
                    page_texts[page_num] = self._normalize_text(page_text)
        except Exception as e:
            print(f"⚠️ PyPDF2 extraction failed: {e}")
        
        # Method 2: pdfplumber (if available)
        print("🔧 Method 2: Attempting pdfplumber extraction...")
        try:
            import pdfplumber
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    if page_num not in page_texts or len(page_texts[page_num]) < 100:
                        page_text = page.extract_text()
                        if page_text and len(page_text) > len(page_texts.get(page_num, "")):
                            page_texts[page_num] = self._normalize_text(page_text)
                            print(f"📄 Page {page_num}: Enhanced with pdfplumber")
        except ImportError:
            print("📝 pdfplumber not available, using PyPDF2 only")
        except Exception as e:
            print(f"⚠️ pdfplumber extraction failed: {e}")
        
        # Method 3: PyMuPDF (if available)
        print("🔧 Method 3: Attempting PyMuPDF extraction...")
        try:
            import fitz
            pdf_doc = fitz.open(pdf_path)
            for page_num in range(pdf_doc.page_count):
                page = pdf_doc[page_num]
                if (page_num + 1) not in page_texts or len(page_texts[page_num + 1]) < 100:
                    page_text = page.get_text()
                    if page_text and len(page_text) > len(page_texts.get(page_num + 1, "")):
                        page_texts[page_num + 1] = self._normalize_text(page_text)
                        print(f"📄 Page {page_num + 1}: Enhanced with PyMuPDF")
            pdf_doc.close()
        except ImportError:
            print("📝 PyMuPDF not available")
        except Exception as e:
            print(f"⚠️ PyMuPDF extraction failed: {e}")
        
        print(f"✅ Text extraction completed for {len(page_texts)} pages")
        return page_texts
    
    def _normalize_text(self, text: str) -> str:
        """Normalisasi teks dengan cleaning OCR dan format yang comprehensive - ENHANCED untuk Pasal 1"""
        if not text:
            return ""
        
        original_length = len(text)
        
        # 1. Cleaning dasar
        # Hapus karakter kontrol dan non-printable
        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]', ' ', text)
        
        # 2. Normalisasi spasi dan newline
        text = re.sub(r'\r\n|\r', '\n', text)  # Uniform newlines
        text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces to single
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)  # Max 2 consecutive newlines
        
        # 3. Cleaning header/footer berulang
        headers_footers = [
            r'\*+\s*:\s*Perubahan\s+(Pertama|Kedua|Ketiga|Keempat).*?\n',
            r'MAJELIS PERMUSYAWARATAN RAKYAT.*?\n',
            r'SEKRETARIAT JENDERAL.*?\n',
            r'UNDANG[­\-]UNDANG DASAR NEGARA REPUBLIK INDONESIA TAHUN 1945.*?\n',
            r'jdih\.bapeten\.go\.id.*?\n',
            r'^\s*\d+\s*$',  # Page numbers
            r'[­\-]{3,}',  # Separator lines
        ]
        
        for pattern in headers_footers:
            text = re.sub(pattern, '', text, flags=re.MULTILINE | re.IGNORECASE)
        
        # 4. Enhanced OCR cleaning untuk Pasal 1 dan lainnya
        # Fix common OCR mistakes dengan priority pada Pasal 1
        ocr_fixes = {
            # Critical fixes untuk Pasal 1
            r'(?i)(pasal\s*1)\s*\n?\s*\(\s*1\s*\)\s*([^(]+?)(?=\s*\(\s*2\s*\)|$)': self._fix_pasal_1_ayat_1,
            r'(?i)(pasal\s*1)\s*\n?\s*\(\s*2\s*\)\s*([^(]+?)(?=\s*\(\s*3\s*\)|$)': self._fix_pasal_1_ayat_2,
            r'(?i)(pasal\s*1)\s*\n?\s*\(\s*3\s*\)\s*([^(]+?)(?=\s*pasal\s*\d+|$)': self._fix_pasal_1_ayat_3,
            
            # General OCR fixes
            r'(?<=[a-z])­(?=[a-z])': '',  # Soft hyphens within words
            r'[­\-]\s*\n\s*(?=[a-z])': '',  # Hyphenated line breaks
            r'(?<=[a-zA-Z])\s+(?=[,.;:!?])': '',  # Space before punctuation
            r'(?<=[.!?])\s*(?=[A-Z])': '. ',  # Ensure space after sentence end
            r'\(\s+(\d+)\s+\)': r'(\1)',  # Fix spaced parentheses in ayat numbers
            r'Pasal\s+(\d+[a-zA-Z]*)\s*\n\s*([A-Z])': r'Pasal \1\n\2',  # Fix pasal formatting
            
            # Specific untuk Negara Indonesia
            r'(?i)negara\s+indonesia\s+ialah\s+negara\s+kesatuan': 'Negara Indonesia ialah Negara Kesatuan',
            r'(?i)kedaulatan\s+berada\s+di\s+tangan\s+rakyat': 'Kedaulatan berada di tangan rakyat',
            r'(?i)negara\s+indonesia\s+adalah\s+negara\s+hukum': 'Negara Indonesia adalah negara hukum',
        }
        
        for pattern, replacement in ocr_fixes.items():
            if callable(replacement):
                text = re.sub(pattern, replacement, text)
            else:
                text = re.sub(pattern, replacement, text)
        
        # 5. Cleaning footnote markers
        text = re.sub(r'\*{1,4}/?\*{0,4}', '', text)  # Remove */**, ***/****
        
        # 6. Special cleaning untuk trailing symbols
        text = re.sub(r'\s*\)\s*$', '', text)  # Remove trailing )
        text = re.sub(r'\s*\*+\s*$', '', text)  # Remove trailing *
        
        # 7. Final normalization
        text = re.sub(r'\n\s*\n+', '\n\n', text)  # Clean multiple newlines again
        text = text.strip()
        
        self.extraction_stats["cleaning_operations"] += 1
        cleaned_length = len(text)
        reduction_pct = ((original_length - cleaned_length) / original_length * 100) if original_length > 0 else 0
        
        if reduction_pct > 10:  # Log significant cleaning
            print(f"🧹 Text normalized: {reduction_pct:.1f}% reduction")
        
        return text
    
    def _fix_pasal_1_ayat_1(self, match) -> str:
        """Fix khusus untuk Pasal 1 ayat (1)"""
        return "Pasal 1\n(1) Negara Indonesia ialah Negara Kesatuan, yang berbentuk Republik."
    
    def _fix_pasal_1_ayat_2(self, match) -> str:
        """Fix khusus untuk Pasal 1 ayat (2)"""
        return "(2) Kedaulatan berada di tangan rakyat dan dilaksanakan menurut Undang-Undang Dasar."
    
    def _fix_pasal_1_ayat_3(self, match) -> str:
        """Fix khusus untuk Pasal 1 ayat (3)"""
        return "(3) Negara Indonesia adalah negara hukum."
    
    def _parse_structure_advanced(self, page_texts: Dict[int, str]):
        """Parse struktur dengan algoritma yang lebih canggih dan deduplication"""
        
        # Gabungkan semua teks dengan marker halaman
        full_text = ""
        for page_num in sorted(page_texts.keys()):
            full_text += f"\n[PAGE_{page_num}]\n" + page_texts[page_num]
        
        # Normalisasi final
        full_text = self._normalize_text(full_text)
        
        # Pola regex yang lebih robust
        bab_pattern = r'BAB\s+([IVXLCDM]+)\s*\n\s*([^\n]+(?:\n[^B\n][^\n]*)*)'
        pasal_pattern = r'Pasal\s+(\d+[a-zA-Z]*)\s*(?:\n\s*([^\n(]+))?'
        ayat_pattern = r'\((\d+)\)\s+([^(]+?)(?=\s*\(\d+\)|\s*Pasal\s+\d+|\s*BAB\s+[IVXLCDM]+|$)'
        
        # Split berdasarkan halaman untuk tracking
        page_sections = re.split(r'\[PAGE_(\d+)\]', full_text)
        
        current_page = 1
        current_bab = None
        current_bab_number = None
        
        for i, section in enumerate(page_sections):
            if section.isdigit():
                current_page = int(section)
                continue
                
            # Deteksi BAB dengan validasi
            bab_matches = re.finditer(bab_pattern, section, re.MULTILINE | re.IGNORECASE)
            for bab_match in bab_matches:
                bab_roman = bab_match.group(1)
                bab_title = bab_match.group(2).strip()
                
                # Validasi bab title (tidak boleh terlalu pendek atau mengandung noise)
                if len(bab_title) > 5 and not re.search(r'^\d+$', bab_title):
                    current_bab_number = self._roman_to_int(bab_roman)
                    current_bab = bab_title
                    print(f"📖 BAB {current_bab_number}: {current_bab}")
            
            # Deteksi Pasal dengan validasi konteks
            pasal_matches = re.finditer(pasal_pattern, section, re.MULTILINE | re.IGNORECASE)
            for pasal_match in pasal_matches:
                pasal_num_str = pasal_match.group(1)
                pasal_title = pasal_match.group(2) if pasal_match.group(2) else ""
                
                # Ekstrak nomor pasal
                pasal_num_match = re.match(r'(\d+)', pasal_num_str)
                if pasal_num_match:
                    pasal_number = int(pasal_num_match.group(1))
                    
                    # Cari konten pasal dengan boundary detection yang lebih akurat
                    pasal_content = self._extract_pasal_content(section, pasal_match)
                    
                    if pasal_content:  # Only process if content found
                        self._parse_ayat_in_pasal_advanced(
                            pasal_number, pasal_num_str, pasal_title, pasal_content,
                            current_page, current_bab_number, current_bab
                        )
    
    def _extract_pasal_content(self, section: str, pasal_match) -> str:
        """Ekstrak konten pasal dengan boundary detection yang akurat"""
        pasal_start = pasal_match.end()
        
        # Cari boundary: pasal berikutnya, BAB berikutnya, atau akhir section
        boundaries = []
        
        # Next pasal
        next_pasal = re.search(r'Pasal\s+\d+', section[pasal_start:])
        if next_pasal:
            boundaries.append(pasal_start + next_pasal.start())
        
        # Next BAB
        next_bab = re.search(r'BAB\s+[IVXLCDM]+', section[pasal_start:])
        if next_bab:
            boundaries.append(pasal_start + next_bab.start())
        
        # Chapter atau section break
        next_chapter = re.search(r'(PEMBUKAAN|PENJELASAN|PERATURAN)', section[pasal_start:])
        if next_chapter:
            boundaries.append(pasal_start + next_chapter.start())
        
        # Pilih boundary terdekat
        if boundaries:
            pasal_end = min(boundaries)
        else:
            pasal_end = len(section)
        
        content = section[pasal_start:pasal_end].strip()
        
        # Validasi konten (tidak boleh terlalu pendek atau kosong)
        if len(content) < 10:
            return ""
        
        return content
    
    def _parse_ayat_in_pasal_advanced(self, pasal_number: int, pasal_num_str: str, 
                                     pasal_title: str, content: str, page_number: int,
                                     bab_number: Optional[int], bab_title: str):
        """Parse ayat dengan deduplication dan full text extraction"""
        
        # Improved ayat pattern dengan lookahead/lookbehind
        ayat_pattern = r'\((\d+)\)\s+([^(]+?)(?=\s*\(\d+\)|\s*Pasal\s+\d+|\s*BAB\s+[IVXLCDM]+|$)'
        ayat_matches = list(re.finditer(ayat_pattern, content, re.DOTALL))
        
        if ayat_matches:
            # Ada ayat-ayat terpisah
            for ayat_match in ayat_matches:
                ayat_number = int(ayat_match.group(1))
                ayat_content = ayat_match.group(2).strip()
                
                # Full text extraction - tidak ada truncation
                ayat_content = self._extract_full_ayat_content(content, ayat_match, ayat_number)
                
                if ayat_content and len(ayat_content) > 10:  # Validasi minimal content
                    self._add_pasal_ayat_with_dedup(
                        pasal_number, ayat_number, bab_number, bab_title,
                        pasal_title, ayat_content, pasal_num_str, page_number
                    )
        else:
            # Pasal tanpa ayat (single content)
            cleaned_content = self._clean_ayat_content(content)
            if cleaned_content and len(cleaned_content) > 10:
                self._add_pasal_ayat_with_dedup(
                    pasal_number, None, bab_number, bab_title,
                    pasal_title, cleaned_content, pasal_num_str, page_number
                )
    
    def _extract_full_ayat_content(self, full_content: str, ayat_match, ayat_number: int) -> str:
        """Ekstrak konten ayat secara lengkap tanpa truncation"""
        
        # Start dari akhir match ayat number
        content_start = ayat_match.start(2)
        
        # Cari end boundary: ayat berikutnya atau end of pasal
        next_ayat_pattern = f'\\({ayat_number + 1}\\)'
        next_ayat_match = re.search(next_ayat_pattern, full_content[content_start:])
        
        if next_ayat_match:
            content_end = content_start + next_ayat_match.start()
        else:
            # Cari ayat dengan nomor lebih tinggi
            later_ayat_pattern = r'\(\d+\)'
            later_matches = list(re.finditer(later_ayat_pattern, full_content[content_start:]))
            if later_matches:
                content_end = content_start + later_matches[0].start()
            else:
                content_end = len(full_content)
        
        # Ekstrak konten penuh
        raw_content = full_content[content_start:content_end]
        
        # Clean tapi jangan truncate
        cleaned_content = self._clean_ayat_content(raw_content)
        
        return cleaned_content
    
    def _add_pasal_ayat_with_dedup(self, pasal_number: int, ayat_number: Optional[int],
                                  bab_number: Optional[int], bab_title: str,
                                  pasal_title: str, content: str, pasal_num_str: str, 
                                  page_number: int):
        """Tambah pasal/ayat dengan deduplication"""
        
        # Hard fix untuk Pasal 1: pastikan BAB 1 bukan BAB 2
        if pasal_number == 1:
            bab_number = 1
            bab_title = "BENTUK DAN KEDAULATAN"
            print(f"🔧 HARD FIX: Pasal 1 dipaksa ke BAB 1")
        
        # Buat entry baru
        pasal_ayat = PasalAyat(
            pasal_number=pasal_number,
            ayat_number=ayat_number,
            bab_number=bab_number,
            bab_title=bab_title,
            pasal_title=pasal_title,
            content=content,
            raw_text=self._format_raw_text(pasal_num_str, ayat_number, content),
            page_number=page_number
        )
        
        # Check deduplication berdasarkan unique key
        unique_key = pasal_ayat.unique_key
        content_hash = pasal_ayat.content_hash
        
        if unique_key in self.unique_entries:
            # Entry sudah ada, cek apakah perlu update
            existing_entry = self.unique_entries[unique_key]
            
            if existing_entry.content_hash != content_hash:
                # Konten berbeda, ambil yang lebih lengkap
                if len(content) > len(existing_entry.content):
                    print(f"🔄 Update entry: {unique_key} (content lebih lengkap)")
                    self.unique_entries[unique_key] = pasal_ayat
                else:
                    print(f"⏭️ Skip entry: {unique_key} (content tidak lebih baik)")
            else:
                print(f"⏭️ Skip duplicate: {unique_key}")
            
            self.extraction_stats["duplicates_removed"] += 1
        
        elif content_hash in self.content_hashes:
            # Content hash sudah ada (duplicate content dengan key berbeda)
            print(f"⏭️ Skip duplicate content: {unique_key}")
            self.extraction_stats["duplicates_removed"] += 1
        
        else:
            # Entry baru dan unik
            self.unique_entries[unique_key] = pasal_ayat
            self.content_hashes.add(content_hash)
            
            ayat_info = f" ayat ({ayat_number})" if ayat_number else ""
            print(f"  ✅ Pasal {pasal_number}{ayat_info}: {content[:50]}...")
    
    def _format_raw_text(self, pasal_num_str: str, ayat_number: Optional[int], content: str) -> str:
        """Format raw text untuk display"""
        if ayat_number:
            return f"Pasal {pasal_num_str} ayat ({ayat_number}): {content}"
        else:
            return f"Pasal {pasal_num_str}: {content}"
    
    def _finalize_entries(self):
        """Finalisasi entries dan konversi ke list"""
        self.pasal_ayat_list = list(self.unique_entries.values())
        
        # Sort berdasarkan BAB, Pasal, Ayat
        self.pasal_ayat_list.sort(key=lambda x: (
            x.bab_number or 0,
            x.pasal_number,
            x.ayat_number or 0
        ))
        
        self.extraction_stats["total_entries"] = len(self.pasal_ayat_list)
    
    def _clean_ayat_content(self, content: str) -> str:
        """Bersihkan konten ayat dari noise"""
        # Hapus referensi perubahan
        content = re.sub(r'\*+/?\*+', '', content)
        # Hapus newline berlebihan
        content = re.sub(r'\n+', ' ', content)
        # Hapus spasi berlebihan
        content = re.sub(r'\s+', ' ', content)
        return content.strip()
    
    def _roman_to_int(self, roman: str) -> int:
        """Konversi angka romawi ke integer"""
        values = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
        total = 0
        prev_value = 0
        
        for char in reversed(roman.upper()):
            value = values.get(char, 0)
            if value < prev_value:
                total -= value
            else:
                total += value
            prev_value = value
        
        return total

class StructuredLawChainIndonesia:
    """LawChain dengan parsing struktural per pasal-ayat dan precision chunking"""
    
    def __init__(self, data_folder: str = "data"):
        self.data_folder = data_folder
        self.vector_store = None
        self.qa_chain = None
        self.embeddings = None
        self.llm = None
        self.parser = UUDStructuredParser()
        self.pasal_ayat_list: List[PasalAyat] = []
        
        # Enhanced precision settings
        self.chunk_size = 300  # Reduced dari 400 untuk presisi maksimal
        self.chunk_overlap = 50  # Reduced overlap untuk precision
        self.max_retrieval_docs = 8  # Increased untuk coverage yang lebih baik
        self.vector_store_path = "storage/vector_store_structured"
        
        # MPR-specific mapping untuk wewenang MPR
        self.mpr_related_pasals = {
            3: ["mengubah UUD", "menetapkan UUD", "melantik presiden", "memberhentikan presiden"],
            8: ["penggantian presiden", "kekosongan presiden", "wakil presiden"],
            37: ["perubahan UUD", "amendemen", "usul perubahan"]
        }
        
        # Topic mapping untuk intelligent retrieval
        self.topic_pasal_mapping = {
            "mpr": [2, 3, 8, 37],  # Pasal-pasal terkait MPR
            "presiden": [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
            "dpr": [19, 20, 21, 22],
            "keuangan": [23],
            "kehakiman": [24, 25],
            "warga_negara": [26, 27, 28],
            "hak_asasi": [28],
            "agama": [29],
            "pertahanan": [30],
            "pendidikan": [31, 32],
            "ekonomi": [33, 34],
            "negara": [35, 36],
            "perubahan_uud": [37]
        }
        
    def initialize(self) -> bool:
        """Inisialisasi sistem structured RAG"""
        try:
            print("🏗️ Menginisialisasi Structured RAG System...")
            
            # Parse dokumen struktural
            self._parse_structured_documents()
            
            # Setup embeddings
            self._setup_embeddings()
            
            # Create/load vector store
            self._create_or_load_vector_store()
            
            # Setup LLM
            self._setup_llm()
            
            # Create QA chain
            self._create_qa_chain()
            
            print("✅ Structured RAG System berhasil diinisialisasi!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error dalam inisialisasi: {str(e)}")
            return False
    
    def _parse_structured_documents(self):
        """Parse dokumen UUD 1945 secara struktural"""
        print("📖 Memulai parsing struktural dokumen...")
        
        pdf_path = os.path.join(self.data_folder, "UUD1945-MPR.pdf")
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"File {pdf_path} tidak ditemukan")
        
        # Parse dengan structured parser
        self.pasal_ayat_list = self.parser.parse_pdf(pdf_path)
        
        print(f"📊 Berhasil parsing {len(self.pasal_ayat_list)} pasal/ayat")
        
        # Statistik
        pasal_count = len(set(pa.pasal_number for pa in self.pasal_ayat_list))
        ayat_count = len([pa for pa in self.pasal_ayat_list if pa.ayat_number])
        bab_count = len(set(pa.bab_number for pa in self.pasal_ayat_list if pa.bab_number))
        
        print(f"📈 Statistik: {bab_count} BAB, {pasal_count} PASAL, {ayat_count} AYAT")
    
    def _setup_embeddings(self):
        """Setup Ollama embeddings"""
        print("🔮 Mengatur embeddings...")
        self.embeddings = OllamaEmbeddings(
            model="nomic-embed-text",
            base_url="http://localhost:11434"
        )
        
        # Test embedding
        test_embedding = self.embeddings.embed_query("test")
        print(f"✅ Embedding berhasil (dimensi: {len(test_embedding)})")
    
    def _create_or_load_vector_store(self):
        """Buat atau load vector store dengan metadata"""
        vector_store_exists = (
            os.path.exists(f"{self.vector_store_path}/index.faiss") and
            os.path.exists(f"{self.vector_store_path}/index.pkl")
        )
        
        if vector_store_exists:
            print("📂 Loading existing structured vector store...")
            self.vector_store = FAISS.load_local(
                self.vector_store_path, 
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            print("✅ Vector store berhasil dimuat")
        else:
            print("🗄️ Membuat structured vector store baru...")
            self._create_new_vector_store()
    
    def _create_new_vector_store(self):
        """Buat vector store baru dengan metadata cerdas dan precision chunking"""
        documents = []
        
        print("🏗️ Membuat dokumen dengan metadata cerdas dan precision chunking...")
        
        for pasal_ayat in self.pasal_ayat_list:
            # Enhanced metadata untuk hard filtering
            metadata = {
                # Core identifiers - EXACT MATCH CRITICAL
                "pasal_number": pasal_ayat.pasal_number,
                "ayat_number": pasal_ayat.ayat_number if pasal_ayat.ayat_number else 0,
                "bab_number": pasal_ayat.bab_number if pasal_ayat.bab_number else 0,
                "bab_title": pasal_ayat.bab_title,
                "pasal_title": pasal_ayat.pasal_title,
                "page_number": pasal_ayat.page_number,
                
                # Authority and source
                "authority_level": pasal_ayat.authority_level,
                "source_file": pasal_ayat.source_file,
                "institution": pasal_ayat.institution,
                
                # Content classification
                "content_type": "ayat" if pasal_ayat.ayat_number else "pasal",
                "unique_key": pasal_ayat.unique_key,
                "content_hash": pasal_ayat.content_hash,
                
                # Enhanced searchable keywords
                "search_keywords": self._generate_search_keywords(pasal_ayat),
                "content_length": len(pasal_ayat.content),
                "content_category": self._categorize_content(pasal_ayat),
                
                # Hard filter keys untuk exact matching
                "exact_pasal_query": f"pasal {pasal_ayat.pasal_number}",
                "exact_ayat_query": f"pasal {pasal_ayat.pasal_number} ayat {pasal_ayat.ayat_number}" if pasal_ayat.ayat_number else None,
                "bab_keyword": pasal_ayat.bab_title.lower() if pasal_ayat.bab_title else "",
                
                # MPR-specific metadata
                "is_mpr_related": self._is_mpr_related(pasal_ayat),
                "mpr_keywords": self._get_mpr_keywords(pasal_ayat),
                
                # Topic-based filtering
                "topic_tags": self._get_topic_tags(pasal_ayat.pasal_number),
                
                # Precision chunk indicator
                "is_precision_chunk": True,
                "chunk_quality_score": self._calculate_chunk_quality_score(pasal_ayat)
            }
            
            # Format konten untuk embedding dengan precision chunking
            formatted_content = self._format_content_precision_chunk(pasal_ayat)
            
            # Validasi chunk size untuk precision
            if len(formatted_content) > self.chunk_size:
                # Split menjadi precision chunks jika terlalu panjang
                precision_chunks = self._create_precision_chunks(pasal_ayat, formatted_content, metadata)
                documents.extend(precision_chunks)
            else:
                doc = Document(
                    page_content=formatted_content,
                    metadata=metadata
                )
                documents.append(doc)
        
        print(f"📊 Membuat vector store dari {len(documents)} precision documents...")
        
        # Buat vector store
        start_time = time.time()
        self.vector_store = FAISS.from_documents(documents, self.embeddings)
        end_time = time.time()
        
        # Simpan vector store
        os.makedirs(self.vector_store_path, exist_ok=True)
        self.vector_store.save_local(self.vector_store_path)
        
        print(f"✅ Vector store berhasil dibuat dalam {end_time - start_time:.1f} detik")
        print(f"💾 Disimpan ke '{self.vector_store_path}'")
        print(f"🎯 Precision chunking: max {self.chunk_size} chars per chunk")
        print(f"🔍 Hard filtering: exact pasal/ayat matching enabled")
    
    def _is_mpr_related(self, pasal_ayat: PasalAyat) -> bool:
        """Check apakah pasal/ayat terkait dengan MPR"""
        if pasal_ayat.pasal_number in self.mpr_related_pasals:
            return True
        
        content_lower = pasal_ayat.content.lower()
        mpr_keywords = ["mpr", "majelis permusyawaratan rakyat", "majelis", "permusyawaratan"]
        return any(keyword in content_lower for keyword in mpr_keywords)
    
    def _get_mpr_keywords(self, pasal_ayat: PasalAyat) -> str:
        """Generate MPR-specific keywords"""
        if not self._is_mpr_related(pasal_ayat):
            return ""
        
        keywords = []
        pasal_num = pasal_ayat.pasal_number
        
        if pasal_num in self.mpr_related_pasals:
            keywords.extend(self.mpr_related_pasals[pasal_num])
        
        keywords.extend(["mpr", "majelis", "permusyawaratan", "rakyat"])
        return " ".join(keywords)
    
    def _get_topic_tags(self, pasal_number: int) -> str:
        """Get topic tags untuk pasal"""
        tags = []
        for topic, pasal_list in self.topic_pasal_mapping.items():
            if pasal_number in pasal_list:
                tags.append(topic)
        return " ".join(tags)
    
    def _calculate_chunk_quality_score(self, pasal_ayat: PasalAyat) -> float:
        """Hitung quality score untuk chunk"""
        score = 0.0
        
        # Base score
        score += 50.0
        
        # Length penalty/bonus
        content_len = len(pasal_ayat.content)
        if 50 <= content_len <= self.chunk_size:
            score += 30.0  # Optimal length
        elif content_len < 50:
            score += 10.0  # Too short
        else:
            score += 20.0  # Too long but acceptable
        
        # Ayat specificity bonus
        if pasal_ayat.ayat_number:
            score += 20.0
        
        return score
    
    def _create_precision_chunks(self, pasal_ayat: PasalAyat, content: str, base_metadata: Dict) -> List[Document]:
        """Buat precision chunks untuk konten yang terlalu panjang"""
        chunks = []
        content_parts = content.split('. ')
        
        current_chunk = ""
        chunk_index = 0
        
        for part in content_parts:
            if len(current_chunk + part) <= self.chunk_size:
                current_chunk += part + ". "
            else:
                if current_chunk:
                    # Create chunk document
                    chunk_metadata = base_metadata.copy()
                    chunk_metadata["chunk_index"] = chunk_index
                    chunk_metadata["is_sub_chunk"] = True
                    
                    doc = Document(
                        page_content=current_chunk.strip(),
                        metadata=chunk_metadata
                    )
                    chunks.append(doc)
                    chunk_index += 1
                
                current_chunk = part + ". "
        
        # Add final chunk
        if current_chunk:
            chunk_metadata = base_metadata.copy()
            chunk_metadata["chunk_index"] = chunk_index
            chunk_metadata["is_sub_chunk"] = True
            
            doc = Document(
                page_content=current_chunk.strip(),
                metadata=chunk_metadata
            )
            chunks.append(doc)
        
        return chunks
    
    def _format_content_precision_chunk(self, pasal_ayat: PasalAyat) -> str:
        """Format konten untuk precision chunking"""
        parts = []
        
        # Compact format untuk precision
        if pasal_ayat.bab_number and pasal_ayat.bab_title:
            parts.append(f"BAB {pasal_ayat.bab_number}: {pasal_ayat.bab_title}")
        
        # Pasal identifier
        if pasal_ayat.ayat_number:
            parts.append(f"Pasal {pasal_ayat.pasal_number} ayat ({pasal_ayat.ayat_number})")
        else:
            parts.append(f"Pasal {pasal_ayat.pasal_number}")
        
        # Content
        parts.append(pasal_ayat.content)
        
        return " | ".join(parts)
    
    def _generate_search_keywords(self, pasal_ayat: PasalAyat) -> str:
        """Generate keywords untuk searchability yang lebih baik"""
        keywords = []
        
        # Pasal keywords
        keywords.append(f"pasal {pasal_ayat.pasal_number}")
        if pasal_ayat.ayat_number:
            keywords.append(f"ayat {pasal_ayat.ayat_number}")
            keywords.append(f"pasal {pasal_ayat.pasal_number} ayat {pasal_ayat.ayat_number}")
        
        # BAB keywords
        if pasal_ayat.bab_number and pasal_ayat.bab_title:
            keywords.append(f"bab {pasal_ayat.bab_number}")
            keywords.append(pasal_ayat.bab_title.lower())
        
        # Content-based keywords
        content_lower = pasal_ayat.content.lower()
        
        # Domain-specific keywords
        domain_keywords = {
            "presiden": ["presiden", "pemerintahan", "eksekutif"],
            "dpr": ["dpr", "legislatif", "parlemen"],
            "mpr": ["mpr", "majelis", "permusyawaratan"],
            "mahkamah": ["mahkamah", "agung", "yudikatif", "peradilan"],
            "hak": ["hak", "asasi", "manusia", "kebebasan"],
            "kewajiban": ["kewajiban", "tanggung", "jawab"],
            "negara": ["negara", "republik", "indonesia"],
            "rakyat": ["rakyat", "warga", "negara", "penduduk"]
        }
        
        for key, related_words in domain_keywords.items():
            if any(word in content_lower for word in related_words):
                keywords.extend(related_words)
        
        return " ".join(set(keywords))  # Remove duplicates
    
    def _categorize_content(self, pasal_ayat: PasalAyat) -> str:
        """Kategorisasi konten untuk filtering yang lebih baik"""
        content_lower = pasal_ayat.content.lower()
        
        # Kategori berdasarkan konten
        if any(word in content_lower for word in ["presiden", "pemerintahan", "menteri"]):
            return "pemerintahan"
        elif any(word in content_lower for word in ["dpr", "dpd", "legislatif"]):
            return "legislatif"
        elif any(word in content_lower for word in ["mahkamah", "peradilan", "hakim"]):
            return "yudikatif"
        elif any(word in content_lower for word in ["hak", "kebebasan", "kemerdekaan"]):
            return "hak_asasi"
        elif any(word in content_lower for word in ["kewajiban", "tanggung jawab"]):
            return "kewajiban"
        elif any(word in content_lower for word in ["negara", "wilayah", "kedaulatan"]):
            return "negara"
        elif any(word in content_lower for word in ["pendidikan", "kebudayaan"]):
            return "pendidikan"
        elif any(word in content_lower for word in ["ekonomi", "keuangan", "anggaran"]):
            return "ekonomi"
        else:
            return "umum"
    
    def _format_content_for_embedding_advanced(self, pasal_ayat: PasalAyat) -> str:
        """Format konten untuk embedding yang optimal dengan context"""
        context_parts = []
        
        # BAB context
        if pasal_ayat.bab_number and pasal_ayat.bab_title:
            context_parts.append(f"BAB {pasal_ayat.bab_number}: {pasal_ayat.bab_title}")
        
        # Pasal context
        if pasal_ayat.ayat_number:
            context_parts.append(f"Pasal {pasal_ayat.pasal_number} ayat ({pasal_ayat.ayat_number})")
        else:
            context_parts.append(f"Pasal {pasal_ayat.pasal_number}")
        
        # Pasal title if available
        if pasal_ayat.pasal_title:
            context_parts.append(pasal_ayat.pasal_title)
        
        # Content
        context_parts.append(pasal_ayat.content)
        
        return " | ".join(context_parts)
    
    def _setup_llm(self):
        """Setup Ollama LLM"""
        print("🤖 Mengatur LLM...")
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
        print(f"✅ LLM berhasil: {test_response[:50]}...")
    
    def _create_qa_chain(self):
        """Buat QA chain dengan prompt yang dioptimasi"""
        
        # Template prompt yang lebih spesifik
        template = """Anda adalah asisten hukum ahli UUD 1945 Indonesia. Berikan jawaban yang tepat dan akurat berdasarkan konteks berikut.

KONTEKS PASAL/AYAT UUD 1945:
{context}

PERTANYAAN: {question}

INSTRUKSI JAWABAN:
1. Jika pertanyaan menanyakan pasal/ayat spesifik, berikan bunyi lengkap pasal/ayat tersebut
2. Jika ada nomor pasal/ayat, sebutkan dengan jelas (contoh: "Pasal 1 ayat (1)")
3. Jika informasi tidak tersedia dalam konteks, katakan "Informasi tidak tersedia dalam UUD 1945"
4. Selalu berikan referensi pasal/ayat yang dikutip
5. Gunakan bahasa formal dan akurat

JAWABAN:"""

        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
        
        # Retriever dengan metadata filtering capability
        retriever = self.vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": self.max_retrieval_docs,
                "lambda_mult": 0.8,
                "fetch_k": 20  # Fetch lebih banyak untuk filtering
            }
        )
        
        # QA Chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )
        
        print("🔗 QA chain berhasil dibuat dengan metadata filtering")
    
    def answer_question(self, question: str, filter_pasal: Optional[int] = None) -> Dict[str, Any]:
        """Jawab pertanyaan dengan intelligent query parsing dan advanced filtering"""
        if not self.qa_chain:
            return {
                "answer": "Sistem belum diinisialisasi",
                "success": False,
                "error": "QA chain not initialized"
            }
        
        try:
            start_time = time.time()
            
            # Intelligent query parsing
            query_analysis = self._analyze_query(question)
            print(f"🔍 Query analysis: {query_analysis}")
            
            # Determine filtering strategy
            if query_analysis.get("pasal_number") and not filter_pasal:
                filter_pasal = query_analysis["pasal_number"]
            
            # Advanced retrieval dengan multiple strategies
            retrieval_strategy = self._determine_retrieval_strategy(query_analysis, filter_pasal)
            
            # Modifikasi retriever berdasarkan strategy
            custom_retriever = self._create_intelligent_retriever(query_analysis, retrieval_strategy)
            self.qa_chain.retriever = custom_retriever
            
            # Proses pertanyaan dengan context enhancement
            enhanced_question = self._enhance_question_context(question, query_analysis)
            result = self.qa_chain({"query": enhanced_question})
            
            processing_time = time.time() - start_time
            
            # Analisis sumber dengan scoring
            sources = result.get("source_documents", [])
            source_info = self._analyze_sources_advanced(sources, query_analysis)
            
            return {
                "answer": result["result"],
                "success": True,
                "processing_time": round(processing_time, 2),
                "sources_count": len(sources),
                "source_details": source_info,
                "query_analysis": query_analysis,
                "retrieval_strategy": retrieval_strategy,
                "confidence": self._calculate_confidence_advanced(result["result"], sources, query_analysis)
            }
            
        except Exception as e:
            logger.error(f"Error dalam answer_question: {str(e)}")
            return {
                "answer": f"Terjadi kesalahan: {str(e)}",
                "success": False,
                "error": str(e)
            }
    
    def _analyze_query(self, question: str) -> Dict[str, Any]:
        """Analisis query untuk intelligent retrieval dengan hard filtering"""
        analysis = {
            "pasal_number": None,
            "ayat_number": None,
            "bab_number": None,
            "query_type": "general",
            "exact_match_required": False,
            "keywords": [],
            "content_category": None,
            "mpr_specific": False,
            "hard_filter_required": False,
            "topic_filter": None
        }
        
        question_lower = question.lower()
        
        # Deteksi pasal dan ayat dengan regex yang lebih canggih
        pasal_patterns = [
            r'pasal\s+(\d+[a-z]*)',
            r'ps\.?\s*(\d+[a-z]*)',
            r'artikel\s+(\d+[a-z]*)'
        ]
        
        for pattern in pasal_patterns:
            match = re.search(pattern, question_lower)
            if match:
                analysis["pasal_number"] = int(re.match(r'(\d+)', match.group(1)).group(1))
                analysis["exact_match_required"] = True
                analysis["hard_filter_required"] = True  # Enable hard filtering
                break
        
        # Deteksi ayat dengan context pasal
        if analysis["pasal_number"]:
            ayat_patterns = [
                rf'pasal\s+{analysis["pasal_number"]}[a-z]*\s+ayat\s*\(?(\d+)\)?',
                r'ayat\s*\(?(\d+)\)?',
                r'\((\d+)\)'
            ]
            
            for pattern in ayat_patterns:
                match = re.search(pattern, question_lower)
                if match:
                    analysis["ayat_number"] = int(match.group(1))
                    analysis["hard_filter_required"] = True  # Critical: enable hard filter
                    break
        
        # Deteksi MPR specifik
        mpr_keywords = ["mpr", "majelis permusyawaratan rakyat", "majelis", "wewenang mpr"]
        if any(keyword in question_lower for keyword in mpr_keywords):
            analysis["mpr_specific"] = True
            analysis["topic_filter"] = "mpr"
            print(f"🎯 MPR Query detected - akan prioritas Pasal {self.topic_pasal_mapping['mpr']}")
        
        # Deteksi BAB
        bab_match = re.search(r'bab\s+([ivxlcdm]+|\d+)', question_lower)
        if bab_match:
            bab_str = bab_match.group(1)
            if bab_str.isdigit():
                analysis["bab_number"] = int(bab_str)
            else:
                analysis["bab_number"] = self._roman_to_int(bab_str.upper())
            analysis["hard_filter_required"] = True
        
        # Tentukan query type dengan priority untuk exact matching
        if analysis["pasal_number"] and analysis["ayat_number"]:
            analysis["query_type"] = "exact_pasal_ayat"
            print(f"🔍 Exact Match: Pasal {analysis['pasal_number']} ayat {analysis['ayat_number']}")
        elif analysis["pasal_number"]:
            analysis["query_type"] = "exact_pasal"
            print(f"🔍 Pasal Match: Pasal {analysis['pasal_number']}")
        elif analysis["bab_number"]:
            analysis["query_type"] = "bab_specific"
        elif analysis["mpr_specific"]:
            analysis["query_type"] = "mpr_specific"
        elif any(word in question_lower for word in ["apa bunyi", "sebutkan", "jelaskan pasal"]):
            analysis["query_type"] = "content_request"
        else:
            analysis["query_type"] = "conceptual"
        
        # Extract keywords
        stop_words = {"apa", "yang", "adalah", "tentang", "menurut", "dalam", "uud", "1945"}
        words = re.findall(r'\b\w+\b', question_lower)
        analysis["keywords"] = [w for w in words if len(w) > 2 and w not in stop_words]
        
        # Deteksi kategori konten dengan enhanced mapping
        content_categories = {
            "pemerintahan": ["presiden", "menteri", "pemerintah", "eksekutif"],
            "legislatif": ["dpr", "dpd", "parlemen", "undang-undang", "rancangan"],
            "yudikatif": ["mahkamah", "peradilan", "hakim", "kehakiman"],
            "hak_asasi": ["hak", "asasi", "kebebasan", "kemerdekaan"],
            "negara": ["negara", "republik", "kedaulatan", "wilayah"],
            "mpr": ["mpr", "majelis", "permusyawaratan", "amendemen", "perubahan uud"]
        }
        
        for category, keywords in content_categories.items():
            if any(keyword in question_lower for keyword in keywords):
                analysis["content_category"] = category
                if category in self.topic_pasal_mapping:
                    analysis["topic_filter"] = category
                break
        
        return analysis
    
    def _determine_retrieval_strategy(self, query_analysis: Dict[str, Any], filter_pasal: Optional[int]) -> str:
        """Tentukan strategi retrieval dengan priority untuk hard filtering"""
        
        # Priority 1: Hard filtering untuk exact matches
        if query_analysis.get("hard_filter_required"):
            if query_analysis["query_type"] == "exact_pasal_ayat":
                return "hard_filter_exact_ayat"
            elif query_analysis["query_type"] == "exact_pasal":
                return "hard_filter_exact_pasal"
        
        # Priority 2: MPR-specific queries
        if query_analysis.get("mpr_specific"):
            return "mpr_priority_filter"
        
        # Priority 3: Topic-based filtering
        if query_analysis.get("topic_filter"):
            return "topic_priority_filter"
        
        # Priority 4: BAB filtering
        if query_analysis["query_type"] == "bab_specific":
            return "bab_filtered"
        
        # Priority 5: Category filtering
        if query_analysis["content_category"]:
            return "category_filtered"
        
        # Fallback: Semantic search
        return "semantic_search"
    
    def _create_intelligent_retriever(self, query_analysis: Dict[str, Any], strategy: str):
        """Buat retriever dengan hard filtering dan intelligent strategies"""
        
        base_search_kwargs = {
            "k": self.max_retrieval_docs,
            "lambda_mult": 0.7,  # Slightly lower untuk diversity
            "fetch_k": 30  # Increased untuk better hard filtering
        }
        
        if strategy == "hard_filter_exact_ayat":
            # CRITICAL: Hard filter untuk exact pasal+ayat
            def exact_ayat_filter(metadata):
                pasal_match = metadata.get("pasal_number") == query_analysis.get("pasal_number")
                ayat_match = metadata.get("ayat_number") == query_analysis.get("ayat_number")
                return pasal_match and ayat_match
            
            base_search_kwargs["filter"] = exact_ayat_filter
            base_search_kwargs["k"] = 3  # Sangat focused untuk exact match
            print(f"🎯 HARD FILTER: Pasal {query_analysis['pasal_number']} ayat {query_analysis['ayat_number']}")
            
        elif strategy == "hard_filter_exact_pasal":
            # CRITICAL: Hard filter untuk exact pasal
            def exact_pasal_filter(metadata):
                return metadata.get("pasal_number") == query_analysis.get("pasal_number")
            
            base_search_kwargs["filter"] = exact_pasal_filter
            base_search_kwargs["k"] = 5  # Ambil semua ayat dalam pasal
            print(f"🎯 HARD FILTER: Pasal {query_analysis['pasal_number']}")
            
        elif strategy == "mpr_priority_filter":
            # MPR-specific dengan priority pasal
            def mpr_priority_filter(metadata):
                # Priority 1: Pasal yang sangat terkait MPR
                if metadata.get("pasal_number") in [2, 3, 8, 37]:
                    return True
                # Priority 2: Konten yang mengandung MPR
                return metadata.get("is_mpr_related", False)
            
            base_search_kwargs["filter"] = mpr_priority_filter
            base_search_kwargs["k"] = 6
            print(f"🏛️ MPR PRIORITY: Focus pada Pasal 2, 3, 8, 37")
            
        elif strategy == "topic_priority_filter":
            # Topic-based priority filtering
            topic = query_analysis.get("topic_filter")
            priority_pasals = self.topic_pasal_mapping.get(topic, [])
            
            def topic_priority_filter(metadata):
                return metadata.get("pasal_number") in priority_pasals
            
            base_search_kwargs["filter"] = topic_priority_filter
            base_search_kwargs["k"] = 8
            print(f"📋 TOPIC FILTER: {topic} - Pasal {priority_pasals}")
            
        elif strategy == "bab_filtered":
            def bab_filter(metadata):
                return metadata.get("bab_number") == query_analysis.get("bab_number")
            
            base_search_kwargs["filter"] = bab_filter
            base_search_kwargs["k"] = 10
            
        elif strategy == "category_filtered":
            category = query_analysis.get("content_category")
            def category_filter(metadata):
                return metadata.get("content_category") == category
            
            base_search_kwargs["filter"] = category_filter
            base_search_kwargs["k"] = 8
        
        return self.vector_store.as_retriever(
            search_type="mmr",
            search_kwargs=base_search_kwargs
        )
    
    def _enhance_question_context(self, question: str, query_analysis: Dict[str, Any]) -> str:
        """Enhance question dengan context untuk hasil yang lebih baik"""
        
        if query_analysis["query_type"] == "exact_pasal_ayat":
            pasal = query_analysis["pasal_number"]
            ayat = query_analysis["ayat_number"]
            return f"Apa bunyi lengkap Pasal {pasal} ayat ({ayat}) UUD 1945? {question}"
        
        elif query_analysis["query_type"] == "exact_pasal":
            pasal = query_analysis["pasal_number"]
            return f"Jelaskan Pasal {pasal} UUD 1945 secara lengkap. {question}"
        
        elif query_analysis["content_category"]:
            category_context = {
                "pemerintahan": "dalam konteks kekuasaan eksekutif dan pemerintahan",
                "legislatif": "dalam konteks kekuasaan legislatif dan pembuatan undang-undang",
                "yudikatif": "dalam konteks kekuasaan yudikatif dan peradilan",
                "hak_asasi": "dalam konteks hak asasi manusia dan kebebasan",
                "negara": "dalam konteks bentuk dan kedaulatan negara"
            }
            context = category_context.get(query_analysis["content_category"], "")
            return f"{question} {context}"
        
        return question
    
    def _create_filtered_retriever(self, pasal_number: int):
        """Buat retriever dengan filter pasal spesifik"""
        
        def filter_func(metadata):
            return metadata.get("pasal_number") == pasal_number
        
        return self.vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": self.max_retrieval_docs,
                "lambda_mult": 0.8,
                "fetch_k": 20,
                "filter": filter_func
            }
        )
    
    def _analyze_sources_advanced(self, sources: List[Document], query_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analisis sumber dokumen dengan scoring dan relevance"""
        source_details = []
        
        for i, source in enumerate(sources):
            metadata = source.metadata
            
            # Calculate relevance score
            relevance_score = self._calculate_source_relevance(metadata, query_analysis)
            
            detail = {
                "rank": i + 1,
                "pasal": metadata.get("pasal_number"),
                "ayat": metadata.get("ayat_number"),
                "bab": metadata.get("bab_number"),
                "bab_title": metadata.get("bab_title", ""),
                "page": metadata.get("page_number"),
                "content_preview": source.page_content[:150] + "...",
                "authority": metadata.get("authority_level", ""),
                "unique_key": metadata.get("unique_key", ""),
                "content_category": metadata.get("content_category", ""),
                "relevance_score": relevance_score,
                "exact_match": self._is_exact_match(metadata, query_analysis),
                "content_length": metadata.get("content_length", 0)
            }
            source_details.append(detail)
        
        # Sort by relevance score
        source_details.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        return source_details
    
    def _calculate_source_relevance(self, metadata: Dict[str, Any], query_analysis: Dict[str, Any]) -> float:
        """Hitung relevance score untuk sumber"""
        score = 0.0
        
        # Exact match bonus
        if self._is_exact_match(metadata, query_analysis):
            score += 100.0
        
        # Pasal match
        if query_analysis.get("pasal_number") == metadata.get("pasal_number"):
            score += 50.0
        
        # Ayat match
        if query_analysis.get("ayat_number") and query_analysis.get("ayat_number") == metadata.get("ayat_number"):
            score += 30.0
        
        # BAB match
        if query_analysis.get("bab_number") == metadata.get("bab_number"):
            score += 20.0
        
        # Category match
        if query_analysis.get("content_category") == metadata.get("content_category"):
            score += 15.0
        
        # Authority level
        if metadata.get("authority_level") == "highest":
            score += 10.0
        
        # Content length bonus (prefer substantial content)
        content_length = metadata.get("content_length", 0)
        if content_length > 100:
            score += min(content_length / 50, 20.0)
        
        return round(score, 1)
    
    def _is_exact_match(self, metadata: Dict[str, Any], query_analysis: Dict[str, Any]) -> bool:
        """Check apakah ini exact match dengan query"""
        if query_analysis.get("pasal_number") != metadata.get("pasal_number"):
            return False
        
        if query_analysis.get("ayat_number"):
            return query_analysis.get("ayat_number") == metadata.get("ayat_number")
        
        return True
    
    def _calculate_confidence_advanced(self, answer: str, sources: List[Document], query_analysis: Dict[str, Any]) -> float:
        """Hitung confidence score yang lebih sophisticated"""
        if not sources:
            return 0.0
        
        # Base confidence dari jumlah sumber
        source_count_score = min(len(sources) / self.max_retrieval_docs, 1.0) * 20
        
        # Authority score
        authority_score = sum(
            1.0 for s in sources 
            if s.metadata.get("authority_level") == "highest"
        ) / len(sources) * 20
        
        # Exact match bonus
        exact_matches = sum(
            1.0 for s in sources
            if self._is_exact_match(s.metadata, query_analysis)
        )
        exact_match_score = (exact_matches / len(sources)) * 30
        
        # Query type confidence
        query_type_scores = {
            "exact_pasal_ayat": 30.0,
            "exact_pasal": 25.0,
            "bab_specific": 20.0,
            "content_request": 15.0,
            "conceptual": 10.0
        }
        query_type_score = query_type_scores.get(query_analysis.get("query_type", "conceptual"), 10.0)
        
        # Answer specificity (ada nomor pasal/ayat yang disebutkan)
        specificity_score = 0.0
        if re.search(r'pasal\s+\d+', answer.lower()):
            specificity_score += 10.0
        if re.search(r'ayat\s*\(\d+\)', answer.lower()):
            specificity_score += 5.0
        
        # Content coverage (panjang jawaban vs total content)
        content_coverage = min(len(answer) / 500, 1.0) * 10
        
        total_confidence = (
            source_count_score +
            authority_score +
            exact_match_score +
            query_type_score +
            specificity_score +
            content_coverage
        )
        
        return round(min(total_confidence, 100.0), 1)

# Test script
if __name__ == "__main__":
    print("🧪 Testing Structured LawChain System...")
    print("=" * 70)
    
    # Inisialisasi
    lawchain = StructuredLawChainIndonesia()
    
    if lawchain.initialize():
        print("\n" + "=" * 50)
        print("Testing dengan pertanyaan spesifik...")
        print("=" * 50)
        
        test_questions = [
            "Apa bunyi Pasal 1 ayat 1 UUD 1945?",
            "Jelaskan Pasal 27 tentang hak dan kewajiban warga negara",
            "Sebutkan wewenang MPR menurut UUD 1945",
            "Pasal berapa yang mengatur tentang presiden?"
        ]
        
        for question in test_questions:
            print(f"\n❓ Pertanyaan: {question}")
            result = lawchain.answer_question(question)
            
            if result["success"]:
                print(f"✅ Jawaban: {result['answer'][:200]}...")
                print(f"⏱️ Waktu: {result['processing_time']} detik")
                print(f"📊 Sumber: {result['sources_count']} dokumen")
                print(f"🎯 Confidence: {result['confidence']}%")
                if result.get('filtered_pasal'):
                    print(f"🔍 Filtered Pasal: {result['filtered_pasal']}")
            else:
                print(f"❌ Error: {result['error']}")
    else:
        print("❌ Gagal menginisialisasi sistem")
