# With inspiration from uv's example multistage dockerfile.

FROM python:3.13-slim-bookworm AS builder
WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:0.7 /uv /uvx /bin/

# Compile bytecode, copy from the cache,
# and disable uv downloading a managed python version.
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=0

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project --no-dev
COPY src/ /app/src/
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-dev \
      # Add spacy's data after uv's last sync so that it doesn't get removed.
      && uv run -m spacy download en_core_web_sm \
      && uv run -m nltk.downloader vader_lexicon -d /app/.venv/nltk_data \
      && uv run -m nltk.downloader punkt_tab -d /app/.venv/nltk_data \
      && uv run -m nltk.downloader averaged_perceptron_tagger_eng -d /app/.venv/nltk_data
      #&& uv run -m nltk.downloader all -d /app/.venv/nltk_data


FROM python:3.13-slim-bookworm AS runtime

COPY --from=builder /app /app

ENV PATH="/app/.venv/bin:$PATH"

# The python code attempts to use absolute imports for files relative to it.
ENV PYTHONPATH="/app/src"

ENTRYPOINT ["fastapi", "run", "/app/src"]
