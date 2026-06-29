# FinRAG Agent - Dockerfile
FROM python:3.11-slim

LABEL maintainer="FinRAG Agent"
LABEL description="Financial Document RAG System"
LABEL version="1.0.0"

# System dependencies for PDF processing
RUN apt-get update && apt-get install -y \
    libpoppler-cpp-dev \
    poppler-utils \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create data directories
RUN mkdir -p data/pdfs data/vectors

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Start the API server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
