# Same Python base as the project's dev container
FROM python:3.12.14-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install the MLflow server with the pinned version
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Shell form so ${VAR} values come from the .env loaded by docker compose
CMD mlflow server \
    --host 0.0.0.0 \
    --port ${MLFLOW_PORT} \
    --backend-store-uri ${MLFLOW_BACKEND_STORE_URI} \
    --artifacts-destination ${MLFLOW_ARTIFACTS_DESTINATION} \
    --cors-allowed-origins ${MLFLOW_CORS_ALLOWED_ORIGINS} \
    --allowed-hosts "localhost:*,127.0.0.1:*,host.docker.internal:*,10.17.0.238:*" \
    --serve-artifacts