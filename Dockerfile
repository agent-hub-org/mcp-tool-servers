FROM python:3.13-slim
WORKDIR /app
RUN pip install poetry --quiet
COPY pyproject.toml poetry.lock ./
RUN poetry config virtualenvs.create false && \
    poetry install --no-interaction --no-ansi --only main
COPY . .
CMD ["python", "server.py"]
