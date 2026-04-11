# ── Stage 1: builder ────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Stage 2: runtime ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

COPY --from=builder /install /usr/local
COPY src/ ./src/
COPY docs/ ./docs/
COPY pyproject.toml .

RUN mkdir -p vector_store
RUN pip install --no-cache-dir -e . --no-deps

EXPOSE 8000

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Pre-build the index on container start if it doesn't exist, then serve
CMD ["sh", "-c", \
     "[ -f vector_store/index.faiss ] || safetyrag-ingest && \
      uvicorn safetyrag.api.main:app --host 0.0.0.0 --port 8000"]
