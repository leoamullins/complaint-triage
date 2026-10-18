FROM python:3.13-slim

COPY --from=ghcr.io/astral-sh/uv:0.11.32 /uv /usr/local/bin/uv

WORKDIR /app


COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --extra serve --no-install-project


COPY src ./src
RUN uv sync --frozen --no-dev --extra serve

ENV PATH="/app/.venv/bin:$PATH" \
    MODEL_DIR=leoamullins/complaint-roberta

EXPOSE 8000
CMD ["uvicorn", "complaint.api:app", "--host", "0.0.0.0", "--port", "8000"]
