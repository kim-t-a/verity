FROM python:3.12-slim

# PYTHONUNBUFFERED for live logs; PIP_NO_CACHE_DIR keeps the image small.
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY pyproject.toml ./
COPY verity ./verity
COPY migrations ./migrations
COPY alembic.ini ./

# The build backend is installed as a first step rather than letting pip's
# isolated build fetch it: this network is slow enough that a separate
# resolution pass sometimes returns no versions at all, and --no-build-isolation
# with setuptools already present needs no network for the backend at all.
RUN pip install "setuptools>=68" wheel && pip install --no-build-isolation .

EXPOSE 8000

# Migrations run before serving: a schema missing a column is the kind of
# partial state the product must never look ok in.
CMD ["sh", "-c", "alembic upgrade head && uvicorn verity.api.main:app --host 0.0.0.0 --port 8000"]
