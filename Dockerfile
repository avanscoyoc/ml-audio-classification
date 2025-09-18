# Multi-stage build for ML Audio Classification
FROM python:3.10-slim AS builder

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV MPLCONFIGDIR=/tmp/matplotlib
ENV XDG_CACHE_HOME=/tmp/cache

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    wget \
    libsndfile1 \
    libsndfile1-dev \
    ffmpeg \
    pkg-config \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user early with home directory
RUN groupadd -r appuser && useradd -r -g appuser -m -d /home/appuser appuser

# Copy pyproject.toml first for better caching
COPY pyproject.toml ./

# Install Python dependencies (without source code)
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy source code
COPY src/ ./src/
COPY scripts/ ./scripts/

# Install the package in development mode with all dependencies
RUN pip install --no-cache-dir -e .

# Create directories for data, results, credentials, and caches
RUN mkdir -p /app/data /app/results /app/credentials /app/logs /tmp/matplotlib /tmp/cache && \
    chown -R appuser:appuser /app /tmp/matplotlib /tmp/cache /home/appuser

# Create a simple test to verify installation (without config validation)
RUN python -c "import sys; sys.path.insert(0, '/app/src'); import ml_audio_classification.models; print('✅ Package installed successfully')"

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD python -c "import ml_audio_classification; print('OK')" || exit 1

# Expose port for potential web interface
EXPOSE 8080

# Create entry point script
USER root
COPY scripts/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh && chown appuser:appuser /app/entrypoint.sh

USER appuser

# Default entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]

# Default command - show help
CMD ["python", "-m", "ml_audio_classification", "--help"]