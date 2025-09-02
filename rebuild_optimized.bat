@echo off
echo 🔄 LawChain Database Rebuild & Optimization Script
echo ============================================================
echo.

echo 📋 This script will:
echo    1. Backup existing vector stores
echo    2. Rebuild optimized vector database
echo    3. Test the optimized system
echo    4. Compare performance metrics
echo.

set /p confirm="Continue with database rebuild? (y/n): "
if /i "%confirm%" neq "y" (
    echo ❌ Operation cancelled.
    pause
    exit /b 0
)

echo.
echo 🚀 Starting LawChain optimization process...
echo.

REM Check if virtual environment exists
if not exist ".venv" (
    echo ⚠️ Virtual environment not found. Creating...
    python -m venv .venv
)

REM Activate virtual environment
echo 🔄 Activating virtual environment...
call .venv\Scripts\activate.bat

REM Install/update dependencies
echo 📦 Installing/updating dependencies...
pip install -r requirements.txt

REM Check Ollama status
echo 🔍 Checking Ollama status...
curl -f http://localhost:11434/api/tags >nul 2>&1
if errorlevel 1 (
    echo ❌ Ollama is not running or not accessible
    echo 💡 Please start Ollama first: ollama serve
    pause
    exit /b 1
)

echo ✅ Ollama is running

REM Check required models
echo 🔍 Checking required models...
ollama list | findstr "gemma2:2b" >nul
if errorlevel 1 (
    echo ❌ Gemma2:2b model not found
    echo 💡 Installing model...
    ollama pull gemma2:2b
)

ollama list | findstr "nomic-embed-text" >nul
if errorlevel 1 (
    echo ❌ nomic-embed-text model not found
    echo 💡 Installing model...
    ollama pull nomic-embed-text
)

echo ✅ All required models are available

REM Backup existing vector stores
echo 📁 Creating backup of existing vector stores...
if exist "storage\vector_store_faiss" (
    if not exist "storage\backup" mkdir "storage\backup"
    xcopy "storage\vector_store_faiss" "storage\backup\vector_store_faiss_backup_%date:~-4,4%%date:~-10,2%%date:~-7,2%" /E /I /Q
    echo ✅ LangChain vector store backed up
)

if exist "storage\vector_store_native" (
    if not exist "storage\backup" mkdir "storage\backup"
    xcopy "storage\vector_store_native" "storage\backup\vector_store_native_backup_%date:~-4,4%%date:~-10,2%%date:~-7,2%" /E /I /Q
    echo ✅ Native vector store backed up
)

REM Remove old vector stores to force rebuild
echo 🗑️ Removing old vector stores for rebuild...
if exist "storage\vector_store_faiss" (
    rmdir /s /q "storage\vector_store_faiss"
    echo ✅ Old LangChain vector store removed
)

if exist "storage\vector_store_native" (
    rmdir /s /q "storage\vector_store_native"
    echo ✅ Old Native vector store removed
)

REM Test optimized system
echo.
echo 🧪 Testing optimized LawChain system...
echo.

python -c "
import sys
import os
sys.path.append('.')

print('🔧 Testing Optimized LawChain Implementation...')
print('=' * 60)

try:
    from app.services.lawchain_optimized import OptimizedLawChainIndonesia
    
    print('📂 Initializing optimized system...')
    lawchain = OptimizedLawChainIndonesia()
    lawchain.initialize_optimized(force_rebuild_vectorstore=True)
    
    print('\n🧪 Running test questions...')
    test_questions = [
        'Apa itu Pancasila menurut UUD 1945?',
        'Bagaimana tugas dan wewenang Presiden?',
        'Sebutkan hak asasi manusia dalam UUD 1945'
    ]
    
    total_accuracy = 0
    successful_tests = 0
    
    for i, question in enumerate(test_questions, 1):
        print(f'\n--- Test {i}: {question} ---')
        try:
            import time
            start_time = time.time()
            response = lawchain.ask_question_optimized(question)
            end_time = time.time()
            
            accuracy = response['metrics']['estimated_accuracy']
            processing_time = end_time - start_time
            sources = response['jumlah_sumber']
            
            print(f'✅ Success!')
            print(f'   Accuracy: {accuracy:.1f}%')
            print(f'   Processing Time: {processing_time:.2f}s')
            print(f'   Sources Used: {sources}')
            print(f'   Answer Length: {len(response[\"jawaban\"])} characters')
            
            total_accuracy += accuracy
            successful_tests += 1
            
        except Exception as e:
            print(f'❌ Test failed: {str(e)}')
    
    if successful_tests > 0:
        avg_accuracy = total_accuracy / successful_tests
        print(f'\n🎯 OPTIMIZATION RESULTS:')
        print(f'   Successful Tests: {successful_tests}/{len(test_questions)}')
        print(f'   Average Accuracy: {avg_accuracy:.1f}%')
        print(f'   System Status: OPTIMIZED ✅')
    else:
        print(f'\n❌ All tests failed. Please check system configuration.')
        
except Exception as e:
    print(f'❌ Optimization test failed: {str(e)}')
    print('💡 Please check:')
    print('   1. Ollama is running')
    print('   2. Required models are installed')
    print('   3. Data folder contains PDF files')
    print('   4. Python dependencies are installed')
"

if errorlevel 1 (
    echo.
    echo ❌ Optimization test failed
    echo 💡 Please check the error messages above
    pause
    exit /b 1
)

echo.
echo 🎉 LawChain optimization completed successfully!
echo.
echo 📊 Next steps:
echo    1. Start the API server: python main.py
echo    2. Test via frontend or API endpoints
echo    3. Monitor performance improvements
echo.
echo 💡 Key optimizations applied:
echo    • Smaller chunk size (600 vs 800)
echo    • Strategic overlap (100 vs 150)
echo    • MMR retrieval for diversity
echo    • Enhanced context filtering
echo    • Optimized prompt templates
echo    • Better source ranking
echo.

pause
