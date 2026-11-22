# Hosted worker (Railway): the team bot + the periodic sync in ONE process (they share one SQLite database
# on a volume mounted at /data). The MCP server for Claude stays local. See README → «Розгортання на сервері».
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.13.9 /uv /uvx /usr/local/bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONUNBUFFERED=1 \
    DATA_DIR=/data

WORKDIR /app

# Dependencies first (cached layer), then the project itself. Exact versions come from uv.lock.
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev

# Secrets arrive only as (sealed) Railway variables: .env, data/ and sessions are excluded by .dockerignore.
CMD ["uv", "run", "--no-sync", "tg-connector", "bot", "--sync"]
