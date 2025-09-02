@echo off
echo 🚀 Installing required Ollama models for LawChain...
echo.

echo 📥 Pulling Gemma2:2b model...
ollama pull gemma2:2b
if %errorlevel% neq 0 (
    echo ❌ Failed to pull gemma2:2b
    pause
    exit /b 1
)

echo 📥 Pulling nomic-embed-text model...
ollama pull nomic-embed-text
if %errorlevel% neq 0 (
    echo ❌ Failed to pull nomic-embed-text
    pause
    exit /b 1
)

echo.
echo ✅ All models installed successfully!
echo 📊 Verifying models...
ollama list

echo.
echo 🎉 Setup complete! You can now run LawChain with Gemma2:2b
echo 💡 Model changed from LLaMA 3.1:8B to Gemma2:2b for better performance
echo 💡 Gemma2:2b uses ~1.6GB RAM compared to ~4.9GB for LLaMA 3.1:8B
pause
