# syntax=docker/dockerfile:1
FROM python:3.12-slim

# System dependencies for scientific computing, image processing & healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgl1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
ARG TORCH_INDEX_URL="https://download.pytorch.org/whl/cpu"
RUN pip install --no-cache-dir torch --index-url "${TORCH_INDEX_URL}" && \
    pip install --no-cache-dir -r requirements.txt

# Copy source tree and essential assets
COPY configs/ ./configs/
COPY scripts/ ./scripts/
COPY src/ ./src/
COPY neuroscope/ ./neuroscope/
COPY frontend/ ./frontend/
COPY models/ ./models/
COPY data/ ./data/

# Ensure workspace and runtime scratch directories exist
RUN mkdir -p work/runs

# Environment configuration
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

EXPOSE 8000

# Create non-root user for security compliance
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD curl -f http://localhost:8000/api/status || exit 1

CMD ["python", "-m", "uvicorn", "app:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8000"]
