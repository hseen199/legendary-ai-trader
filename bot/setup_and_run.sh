#!/bin/bash

# ==========================================
# Legendary AI Trader - One-Click Setup (Linux/Mac)
# ==========================================

echo "🚀 Starting Legendary AI Trader Setup..."

# 1. Check Python installation
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 is not installed. Please install it first."
    exit 1
fi

# 2. Create Virtual Environment
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
else
    echo "✅ Virtual environment already exists."
fi

# 3. Activate Virtual Environment
source venv/bin/activate

# 4. Upgrade pip
echo "🔄 Upgrading pip..."
pip install --upgrade pip

# 5. Install PyTorch (CPU version for compatibility)
echo "🔥 Installing PyTorch (CPU)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 6. Install other requirements
if [ -f "requirements.txt" ]; then
    echo "📚 Installing dependencies from requirements.txt..."
    pip install -r requirements.txt
else
    echo "⚠️ Warning: requirements.txt not found."
fi

# 7. Setup Environment Variables
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        echo "📝 Creating .env from .env.example..."
        cp .env.example .env
        echo "⚠️ Please edit .env and add your API keys."
    else
        echo "⚠️ Warning: .env.example not found."
    fi
fi

echo "✅ Setup complete!"
echo "🚀 Running Legendary AI Trader..."
python main_integrated.py
