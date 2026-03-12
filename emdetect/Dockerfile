# ── Stage 1: builder ────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

COPY src/requirements.txt .
RUN pip install --prefix=/install --no-cache-dir -r requirements.txt

# Download spaCy models for English and German
RUN /install/bin/python -m spacy download en_core_web_sm && \
    /install/bin/python -m spacy download de_core_news_sm

# ── Stage 2: runtime ────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

RUN adduser --disabled-password --gecos "" appuser

WORKDIR /app

COPY --from=builder /install /usr/local

COPY src/ .

# HuggingFace model cache — mount as volume so it survives restarts
ENV HF_HOME=/app/.cache/huggingface
RUN mkdir -p /app/.cache && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
