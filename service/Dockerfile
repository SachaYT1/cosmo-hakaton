FROM python:3.12-slim

RUN pip install --no-cache-dir uv

WORKDIR /srv/service
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-dev

COPY app ./app
COPY scripts ./scripts
# Каталог из предсказаний собирается заранее:
# uv run python scripts/ingest.py --data-dir ../train
COPY data ./data

EXPOSE 8000
CMD ["uv", "run", "--no-sync", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
