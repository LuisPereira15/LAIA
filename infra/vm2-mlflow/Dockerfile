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
    --serve-artifacts