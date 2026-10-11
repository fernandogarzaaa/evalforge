# Built and smoke-tested in CI (.github/workflows/ci.yml, job "docker").
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY evalforge ./evalforge
RUN pip install --no-cache-dir . \
    && useradd --create-home --uid 10001 evalforge \
    && mkdir -p /data \
    && chown evalforge:evalforge /data

USER evalforge
VOLUME ["/data"]
ENV EVALFORGE_DB=/data/evalforge.db
ENTRYPOINT ["evalforge"]
CMD ["--help"]
