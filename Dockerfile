# ==============================================================================
# Ask_Med: Backend API & Ingestion Engine Dockerfile
# Stage: Production Runtime (FastAPI + ChromaDB + BM25 + Cross-Encoder)
# Base Image: Python 3.11 Slim (Debian Bookworm)
# ==============================================================================
FROM python:3.11-slim

# Prevent python from buffering stdout/stderr and writing .pyc files
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    PORT=8000

# Set working directory
WORKDIR /app

# Install curl for Docker HEALTHCHECK and system utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker layer caching
COPY requirements.txt /app/requirements.txt

# Upgrade pip and install all python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code, configuration and evaluation assets
COPY src/ /app/src/
COPY configs/ /app/configs/
COPY data/ /app/data/

# Expose FastAPI port
EXPOSE 8000

# Healthcheck to probe operational readiness
HEALTHCHECK --interval=30s --timeout=10s --start-period=45s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start production Uvicorn server
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
