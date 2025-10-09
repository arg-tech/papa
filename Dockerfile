# With inspiration from uv's example multistage dockerfile.

# Lock with digest to prevent unecessary rebuilds (and improve reproducibility).
# Update as needed
FROM python:3.13-slim-trixie@sha256:087a9f3b880e8b2c7688debb9df2a5106e060225ebd18c264d5f1d7a73399db0 AS builder
WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:0.7@sha256:629240833dd25d03949509fc01ceff56ae74f5e5f0fd264da634dd2f70e9cc70 /uv /uvx /bin/

# ENV UV_COMPILE_BYTECODE=1 was temporarily disabled due to server issues
ENV UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=0

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-dev \
    && uv run -m spacy download en_core_web_sm \
    && uv run -m nltk.downloader vader_lexicon -d /app/.venv/nltk_data \
    && uv run -m nltk.downloader punkt_tab -d /app/.venv/nltk_data \
    && uv run -m nltk.downloader averaged_perceptron_tagger_eng -d /app/.venv/nltk_data

COPY src/ /app/src/

FROM python:3.13-slim-trixie@sha256:087a9f3b880e8b2c7688debb9df2a5106e060225ebd18c264d5f1d7a73399db0 AS runtime

COPY --from=builder /app /app

ENV PATH="/app/.venv/bin:$PATH"

# The python code attempts to use absolute imports for files relative to it.
ENV PYTHONPATH="/app/src"

ENTRYPOINT ["fastapi", "run", "/app/src"]
