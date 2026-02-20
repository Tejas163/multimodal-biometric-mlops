# ── Stage 1: Builder ──────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt setup.py ./
COPY src/ src/

# Install into a prefix so we can copy cleanly into the runtime image
RUN pip install --upgrade pip \
    && pip install --prefix=/install -e ".[dev]"

# ── Stage 2: Runtime ──────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL maintainer="Tejas163"
LABEL description="Multimodal biometric ML inference service"

WORKDIR /app

# Only copy installed packages from builder — keeps image lean
COPY --from=builder /install /usr/local
COPY src/ src/
COPY conf/ conf/
COPY scripts/ scripts/

# Create unprivileged user for security
RUN useradd -m -u 1000 mluser && chown -R mluser /app
USER mluser

# MLflow tracking URI can be overridden at runtime
ENV MLFLOW_TRACKING_URI=mlruns/
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src

EXPOSE 8080

# Default: run inference pipeline (override with training cmd if needed)
CMD ["python", "-c", "from biometric_ml.utils.logging import setup_logging; setup_logging(); print('Biometric ML container ready.')"]
