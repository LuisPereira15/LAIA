import os

from mlflow.tracking import MlflowClient


def promote_model():
    tracking_uri = os.environ["MLFLOW_TRACKING_URI"]
    model_name = os.environ["MLFLOW_MODEL_NAME"]
    from_alias = os.environ["FROM_ALIAS"]
    to_alias = os.environ["TO_ALIAS"]

    client = MlflowClient(tracking_uri=tracking_uri)

    version = client.get_model_version_by_alias(
        model_name,
        from_alias,
    )

    client.set_registered_model_alias(
        name=model_name,
        alias=to_alias,
        version=version.version,
    )

    print(
        f"Promoted {model_name} version {version.version} "
        f"from @{from_alias} to @{to_alias}"
    )


if __name__ == "__main__":
    promote_model()