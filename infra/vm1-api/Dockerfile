# Same Python base as the dev container used for training
FROM python:3.12.14-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Same pinned versions as training, so the model loads with identical libraries
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Application code: API + preprocessing pipeline used by the model
COPY src/ /app/src/

# Shell form reads ${API_PORT} from the .env; exec makes uvicorn receive stop signals directly
CMD exec uvicorn src.api.main:app --host 0.0.0.0 --port ${API_PORT}