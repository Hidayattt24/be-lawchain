@echo off
REM LawChain Backend API Startup Script untuk Windows

echo 🏛️ Starting LawChain Backend API...

REM Check if virtual environment exists
if not exist ".venv" (
    echo ⚠️ Virtual environment not found. Creating...
    python -m venv .venv
)

REM Activate virtual environment
echo 🔄 Activating virtual environment...
call .venv\Scripts\activate.bat

REM Install dependencies
echo 📦 Installing dependencies...
pip install -r requirements.txt

REM Create necessary directories
echo 📁 Creating directories...
if not exist "logs" mkdir logs
if not exist "storage" mkdir storage
if not exist "data" mkdir data

REM Check if Ollama is running
echo 🔍 Checking Ollama status...
curl -f http://localhost:11434/api/tags >nul 2>&1
if errorlevel 1 (
    echo ❌ Ollama is not running or not accessible
    echo 💡 Please start Ollama first: ollama serve
    pause
    exit /b 1
)

REM Start the API server
echo 🚀 Starting FastAPI server...
python main.py

pause
