# syntax=docker/dockerfile:1
# Tag and immutable digest verified against Docker Hub on 2026-10-25.
ARG PYTHON_IMAGE=python:3.12.14-slim-bookworm@sha256:392307d22300de8b5986851a12d9176dfc0fc073e65bf6523ebd7dcbeb23564e
FROM ${PYTHON_IMAGE} AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    DO_NOT_TRACK=true
WORKDIR /app
RUN groupadd --gid 10001 app \
    && useradd --uid 10001 --gid app --create-home app \
    && mkdir -p /models /mlflow \
    && chown app:app /models /mlflow
COPY requirements/ /tmp/requirements/

FROM base AS runtime
RUN pip install --no-cache-dir --require-hashes -r /tmp/requirements/runtime.txt
COPY --chown=app:app src/ /app/src/
# Windows build contexts may carry read-only directory modes. MLflow copies
# these directories into temporary artifacts and must be able to remove them.
RUN find /app/src -type d -exec chmod 755 {} + \
    && find /app/src -type f -exec chmod 644 {} +
USER app

FROM runtime AS train
USER root
RUN pip install --no-cache-dir --require-hashes -r /tmp/requirements/train.txt
USER app
# The host supplies GIT_COMMIT; Git is intentionally absent from this image.
ENV GIT_PYTHON_REFRESH=quiet
ENTRYPOINT ["python", "-m", "src.models.train"]
CMD ["--help"]

FROM train AS test
USER root
RUN pip install --no-cache-dir --require-hashes -r /tmp/requirements/test.txt
COPY --chown=app:app tests/ /app/tests/
COPY --chown=app:app pyproject.toml /app/pyproject.toml
COPY --chown=app:app scripts/verify_stack.py /app/scripts/verify_stack.py
USER app
ENTRYPOINT ["python", "-m", "pytest"]
CMD ["-q", "-p", "no:cacheprovider"]

FROM base AS mlflow
RUN pip install --no-cache-dir --require-hashes -r /tmp/requirements/mlflow.txt
USER app
ENTRYPOINT ["mlflow", "server"]
CMD ["--host", "0.0.0.0", "--port", "5000", "--backend-store-uri", "sqlite:////mlflow/mlflow.db", "--artifacts-destination", "/mlflow/artifacts", "--serve-artifacts", "--allowed-hosts", "localhost,127.0.0.1,mlflow,localhost:*,127.0.0.1:*,mlflow:*", "--workers", "1"]

FROM runtime AS api
ENTRYPOINT ["python", "-m", "uvicorn"]
CMD ["src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
