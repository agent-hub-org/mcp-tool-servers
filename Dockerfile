# ── Stage 1: install deps into an in-project venv ──────────────────────────
FROM python:3.13-slim AS builder
WORKDIR /app

RUN pip install --no-cache-dir poetry

COPY pyproject.toml poetry.lock ./
RUN poetry config virtualenvs.in-project true && \
    poetry install --only main --no-interaction --no-ansi

# ── Stage 2: lean runtime image ─────────────────────────────────────────────
FROM python:3.13-slim
WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV="/app/.venv" \
    PATH="/app/.venv/bin:$PATH"

COPY --from=builder /app/.venv ./.venv
COPY . .

CMD ["python", "server.py"]
