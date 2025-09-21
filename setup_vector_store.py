#!/usr/bin/env python3
"""
Script untuk setup vector store awal
Menggunakan LawChainStructuredParser untuk membuat vector store dari UUD1945-MPR.pdf
"""

import os
import sys
import logging
from app.services.lawchain_structured_parser import StructuredLawChainIndonesia

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main function untuk setup vector store"""
    print("🚀 LawChain Vector Store Setup")
    print("=" * 50)
    
    try:
        # Cek apakah data file exists
        pdf_path = "data/UUD1945-MPR.pdf"
        if not os.path.exists(pdf_path):
            print(f"❌ File {pdf_path} tidak ditemukan!")
            print("💡 Pastikan file UUD1945-MPR.pdf ada di folder data/")
            return False
        
        print(f"✅ Data file ditemukan: {pdf_path}")
        
        # Initialize structured parser
        print("\n🏗️ Menginisialisasi Structured LawChain Indonesia...")
        lawchain_parser = StructuredLawChainIndonesia()
        
        # Initialize system (akan membuat vector store)
        print("\n🔄 Memulai proses setup vector store...")
        success = lawchain_parser.initialize()
        
        if success:
            print("\n🎉 Vector store berhasil dibuat!")
            print("✅ Aplikasi siap digunakan")
            
            # Tampilkan info vector store
            if lawchain_parser.vector_store:
                print(f"📊 Vector store path: {lawchain_parser.vector_store_path}")
                print(f"📈 Total dokumen dalam vector store: {lawchain_parser.vector_store.index.ntotal}")
            
            return True
        else:
            print("\n❌ Gagal membuat vector store")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error dalam setup: {str(e)}")
        print(f"\n💥 Error: {str(e)}")
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
    
    print("\n🎯 Setup selesai! Anda sekarang bisa menjalankan:")
    print("   python main.py")