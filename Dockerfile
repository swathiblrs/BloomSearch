FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

WORKDIR /app

COPY pyproject.toml README.md ./
COPY bloom_filter ./bloom_filter
COPY bloom_search ./bloom_search
COPY examples ./examples
COPY web ./web

RUN pip install --no-cache-dir ".[web]"

RUN useradd --create-home --uid 10001 appuser && chown -R appuser:appuser /app
USER appuser

CMD ["sh", "-c", "uvicorn bloom_search.web:app --host 0.0.0.0 --port ${PORT}"]

