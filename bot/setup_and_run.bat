@echo off
TITLE Legendary AI Trader - One-Click Setup (Windows)

echo 🚀 Starting Legendary AI Trader Setup...

:: 1. Check Python installation
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Error: Python is not installed or not in PATH. Please install it first.
    pause
    exit /b
)

:: 2. Create Virtual Environment
if not exist "venv" (
    echo 📦 Creating virtual environment...
    python -m venv venv
) else (
    echo ✅ Virtual environment already exists.
)

:: 3. Activate Virtual Environment
call venv\Scripts\activate

:: 4. Upgrade pip
echo 🔄 Upgrading pip...
python -m pip install --upgrade pip

:: 5. Install PyTorch (CPU version for compatibility)
echo 🔥 Installing PyTorch (CPU)...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

:: 6. Install other requirements
if exist "requirements.txt" (
    echo 📚 Installing dependencies from requirements.txt...
    pip install -r requirements.txt
) else (
    echo ⚠️ Warning: requirements.txt not found.
)

:: 7. Setup Environment Variables
if not exist ".env" (
    if exist ".env.example" (
        echo 📝 Creating .env from .env.example...
        copy .env.example .env
        echo ⚠️ Please edit .env and add your API keys.
    ) else (
        echo ⚠️ Warning: .env.example not found.
    )
)

echo ✅ Setup complete!
echo 🚀 Running Legendary AI Trader...
python main_integrated.py
pause
