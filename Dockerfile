# Untested locally: see "Known gaps" in README.md.
FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY evalforge ./evalforge
RUN pip install --no-cache-dir .
VOLUME ["/data"]
ENV EVALFORGE_DB=/data/evalforge.db
ENTRYPOINT ["evalforge"]
CMD ["--help"]
