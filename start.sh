#!/bin/bash
# LawChain Backend API Startup Script

echo "🏛️ Starting LawChain Backend API..."

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "⚠️ Virtual environment not found. Creating..."
    python -m venv .venv
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source .venv/bin/activate

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p logs storage data config

# Check if Ollama is running
echo "🔍 Checking Ollama status..."
if ! curl -f http://localhost:11434/api/tags >/dev/null 2>&1; then
    echo "❌ Ollama is not running or not accessible"
    echo "💡 Please start Ollama first: ollama serve"
    exit 1
fi

# Start the API server
echo "🚀 Starting FastAPI server..."
python main.py
