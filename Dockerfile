# ========================================
# CTrackAI — Production Dockerfile
# Multi-stage build: builder → runtime
# Uses CPU-only PyTorch to avoid ~3GB CUDA libs
# ========================================

# ── Stage 1: Builder ──────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Create venv and install dependencies
# KEY: Install CPU-only torch FIRST to prevent sentence-transformers
# from pulling the full CUDA version (~3GB → ~200MB)
RUN python -m venv /build/venv && \
    /build/venv/bin/pip install --no-cache-dir --upgrade pip && \
    /build/venv/bin/pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu && \
    /build/venv/bin/pip install --no-cache-dir -r requirements.txt && \
    rm -rf /root/.cache/pip


# ── Stage 2: Runtime ──────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL maintainer="shreyasghate"
LABEL project="CTrackAI"

# Create non-root user
RUN groupadd -r ctrackai && useradd -r -g ctrackai -d /app -s /sbin/nologin ctrackai

WORKDIR /app

# Copy venv from builder
COPY --from=builder /build/venv /app/venv

ENV PATH="/app/venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Copy app code
COPY . .

# Create data dirs and set ownership
RUN mkdir -p /app/data /app/logs && \
    chown -R ctrackai:ctrackai /app

USER ctrackai

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["python", "-m", "gunicorn", "main:app", \
    "--worker-class", "uvicorn.workers.UvicornWorker", \
    "--workers", "2", \
    "--bind", "0.0.0.0:8000", \
    "--timeout", "120", \
    "--graceful-timeout", "30", \
    "--access-logfile", "-"]
