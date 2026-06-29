#!/usr/bin/env bash
# FinRAG Agent - Development Setup Script
set -euo pipefail

echo "=================================================="
echo "  FinRAG Agent - Setup Script"
echo "=================================================="

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
REQUIRED_MAJOR=3
REQUIRED_MINOR=10

echo "→ Checking Python version: $PYTHON_VERSION"
if ! python3 -c "import sys; assert sys.version_info >= (3, 10)" 2>/dev/null; then
    echo "✗ Python 3.10+ is required. Found: $PYTHON_VERSION"
    exit 1
fi
echo "✓ Python version OK"

# Create virtual environment
echo ""
echo "→ Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Activate venv
source venv/bin/activate
echo "✓ Virtual environment activated"

# Install dependencies
echo ""
echo "→ Installing Python dependencies..."
pip install --upgrade pip -q
pip install -r requirements.txt -q
echo "✓ Dependencies installed"

# Create .env if not exists
echo ""
echo "→ Setting up environment configuration..."
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "✓ Created .env from .env.example"
    echo "  ⚠️  Please edit .env with your configuration"
else
    echo "✓ .env already exists"
fi

# Create data directories
echo ""
echo "→ Creating data directories..."
mkdir -p data/pdfs data/vectors
echo "✓ Data directories ready"

# Check Ollama
echo ""
echo "→ Checking Ollama installation..."
if command -v ollama &>/dev/null; then
    echo "✓ Ollama is installed: $(ollama --version 2>/dev/null || echo 'version unknown')"
else
    echo "⚠️  Ollama not found. Install from: https://ollama.ai"
    echo "   After installing, run: ollama pull mistral"
fi

echo ""
echo "=================================================="
echo "  Setup complete!"
echo "=================================================="
echo ""
echo "Next steps:"
echo "  1. Edit .env to configure your Ollama model"
echo "  2. Start Ollama:  ollama serve"
echo "  3. Pull a model:  ollama pull mistral"
echo "  4. Start API:     uvicorn app.main:app --reload"
echo "  5. Open docs:     http://localhost:8000/docs"
echo ""
