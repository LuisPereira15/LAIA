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

settings = Settings()