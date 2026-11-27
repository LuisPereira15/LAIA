"""Runtime settings for the inference API, read from environment variables."""
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    """API configuration. Every value can be overridden through the environment."""

    mlflow_tracking_uri: str = os.getenv('MLFLOW_TRACKING_URI')
    model_name: str = os.getenv('MLFLOW_MODEL_NAME')
    model_alias: str = os.getenv('MODEL_ALIAS')

    # Requests above this size are rejected before any prediction runs.
    max_instances: int = int(os.getenv('API_MAX_INSTANCES'))

    # Large requests are predicted in chunks of this size to keep memory stable.
    batch_size: int = int(os.getenv('API_BATCH_SIZE'))

    # Debug-only: when true, the first `capture_raw_max` payloads received at
    # /classify and /labels are appended to /app/logs/raw_requests/<date>.jsonl.
    # Defaults off so production deployments never write capture files.
    capture_raw_requests: bool = os.getenv('CAPTURE_RAW_REQUESTS', '0') == '1'
    capture_raw_max: int = int(os.getenv('CAPTURE_RAW_MAX', '5'))


settings = Settings()