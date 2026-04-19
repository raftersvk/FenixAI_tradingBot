# ==============================================================================
# FenixAI Trading Bot - Multi-stage Dockerfile
# ==============================================================================
# Build: docker build -t fenix-trading-bot .
# Run: docker run -p 8000:8000 --env-file .env fenix-trading-bot
# ==============================================================================

# Stage 1: Builder - Install dependencies
FROM python:3.13-slim AS builder

WORKDIR /app

# Install build dependencies (rarely changes - cached)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libffi-dev \
    libssl-dev \
    curl \
    wget \
    && wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz \
    && tar -xzf ta-lib-0.4.0-src.tar.gz \
    && cd ta-lib/ \
    && ./configure --prefix=/usr \
    && make \
    && make install \
    && cd .. \
    && rm -rf ta-lib-0.4.0-src.tar.gz ta-lib/ \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# ========== Copy dependency files first (for Docker layer caching) ==========
# These files change rarely - Docker will cache this layer
COPY pyproject.toml ./
COPY requirements.txt ./
COPY README.md ./
COPY LICENSE ./

# Install Python dependencies from requirements.txt
# This layer is cached until requirements.txt changes
RUN pip install --upgrade pip wheel setuptools && \
    pip install -r requirements.txt

# ========== NOW copy source code (after dependencies are installed) ==========
# This layer will rebuild quickly when src/ changes (no pip install needed)
COPY src/ ./src/

# ==============================================================================
# Stage 2: Runtime - Minimal production image
# ==============================================================================
FROM python:3.13-slim AS runtime

WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libssl3 \
    ca-certificates \
    curl \
    wget \
    gnupg \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m -u 1000 fenix

# Install Chromium for Kaleido (Plotly chart export)
RUN apt-get update && apt-get install -y --no-install-recommends \
    chromium \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set Python environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    CHROMIUM_PATH=/usr/bin/chromium

# Copy application code
COPY --chown=fenix:fenix src/ ./src/
COPY --chown=fenix:fenix config/ ./config/

# Create necessary directories
RUN mkdir -p /app/logs /app/data /app/htmlcov && \
    chown -R fenix:fenix /app

# Switch to non-root user
USER fenix

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command - Start API server
CMD ["uvicorn", "src.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
