"""Train an integration baseline from prepared splits, or explicit demo data."""
import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import tempfile
import warnings

import mlflow
from mlflow.models import infer_signature
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from src.contracts import CLASSES, FlightRecord
from src.models.artifacts import save_bundle
from src.models.demo import synthetic_dataset
from src.models.pipeline import make_pipeline

TARGET = "delay_severity"


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_hash():
    digest = hashlib.sha256()
    for path in sorted(Path("src").rglob("*.py")):
        digest.update(path.as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def read_prepared(path, max_rows):
    # Bound memory and never silently truncate or sample the scientific dataset.
    frame = pd.read_csv(path, dtype={"OP_CARRIER_AIRLINE_ID": "string",
                                    "CRS_DEP_TIME": "string", "CRS_ARR_TIME": "string"},
                        nrows=max_rows + 1)
    if len(frame) > max_rows:
        raise ValueError(f"{path} exceeds --max-rows={max_rows}. Supply a documented sample or raise the limit.")
    if frame.empty or TARGET not in frame:
        raise ValueError(f"{path} must contain flights and a {TARGET} column")
    labels = frame[TARGET]
    if labels.isna().any() or not set(labels).issubset(CLASSES):
        raise ValueError(f"{TARGET} must contain only {CLASSES}")
    # Normalize through exactly the same contract used by the API.
    records = frame.drop(columns=[TARGET]).astype(object).where(frame.drop(columns=[TARGET]).notna(), None).to_dict("records")
    normalized = [FlightRecord.model_validate(record).model_dump(mode="json") for record in records]
    return pd.DataFrame(normalized), labels.reset_index(drop=True)


def validate_splits(train_x, train_y, validation_x):
    if set(train_y) != set(CLASSES):
        raise ValueError("The training split must contain all three classes")
    # ISO dates have chronological lexicographic order. Same-day separation is
    # deliberately disallowed until a finer chronological policy is documented.
    if train_x["FL_DATE"].max() >= validation_x["FL_DATE"].min():
        raise ValueError("Training dates must be strictly before validation dates")


def model_export_options(pipeline, example, dependencies):
    # Explicit format: the serving bundle also uses trusted Python serialization.
    # Do not depend on MLflow changing its default format between releases.
    return {
        "sk_model": pipeline,
        "serialization_format": mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
        "signature": infer_signature(example, pipeline.predict(example)),
        "input_example": example,
        "code_paths": ["src"],
        "pip_requirements": [f"{name}=={value}" for name, value in dependencies.items()],
    }


def run_training(args):
    if args.demo:
        if args.train_csv or args.validation_csv:
            raise ValueError("--demo cannot be combined with CSV input")
        train_x, train_y = synthetic_dataset(seed=args.seed)
        validation_x, validation_y = synthetic_dataset(size=90, seed=args.seed + 1, start="2023-01-01")
        kind = "synthetic"
        provenance = {"generator": "src.models.demo.synthetic_dataset", "seed": args.seed}
    else:
        if not args.train_csv or not args.validation_csv:
            raise ValueError("Provide --train-csv and --validation-csv, or explicitly choose --demo")
        if not args.data_version or args.data_version == "not-set":
            raise ValueError("Prepared datasets require --data-version")
        train_x, train_y = read_prepared(args.train_csv, args.max_rows)
        validation_x, validation_y = read_prepared(args.validation_csv, args.max_rows)
        kind = "prepared"
        provenance = {"data_version": args.data_version,
                      "train_sha256": file_hash(args.train_csv),
                      "validation_sha256": file_hash(args.validation_csv)}

    validate_splits(train_x, train_y, validation_x)
    mlflow.set_tracking_uri(args.tracking_uri)
    experiment = "laia-infrastructure-demo" if args.demo else "laia-flight-delay"
    registered_name = "laia-demo-flight-delay" if args.demo else "laia-flight-delay"
    mlflow.set_experiment(experiment)
    pipeline = make_pipeline(args.seed)
    with mlflow.start_run(run_name=f"{kind}-logistic-baseline") as run:
        run_id = run.info.run_id
        mlflow.set_tags({"data_kind": kind, "git_commit": args.git_commit,
                         "purpose": "infrastructure_validation" if args.demo else "baseline_candidate"})
        mlflow.log_params({"seed": args.seed, "model": "logistic_regression",
                           "class_weight": "balanced", "max_iter": 1000,
                           "train_rows": len(train_x), "validation_rows": len(validation_x),
                           "train_start": train_x.FL_DATE.min(), "train_end": train_x.FL_DATE.max(),
                           "validation_start": validation_x.FL_DATE.min(),
                           "validation_end": validation_x.FL_DATE.max()})
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            pipeline.fit(train_x, train_y)
        predictions = pipeline.predict(validation_x)
        metrics = {"validation_accuracy": float(accuracy_score(validation_y, predictions)),
                   "validation_macro_f1": float(f1_score(validation_y, predictions, labels=CLASSES, average="macro", zero_division=0))}
        mlflow.log_metrics(metrics)
        report = classification_report(validation_y, predictions, labels=CLASSES, output_dict=True, zero_division=0)
        mlflow.log_dict(report, "evaluation/classification_report.json")
        mlflow.log_dict({"labels": list(CLASSES), "matrix": confusion_matrix(validation_y, predictions, labels=CLASSES).tolist()},
                        "evaluation/confusion_matrix.json")
        mlflow.log_dict(provenance, "data/provenance.json")
        dependencies = {name: version(name) for name in
                        ("scikit-learn", "pandas", "numpy", "pydantic", "joblib", "cloudpickle", "mlflow-skinny")}
        example = train_x.head(3)
        model_info = mlflow.sklearn.log_model(
            name="model", **model_export_options(pipeline, example, dependencies),
        )
        registered = mlflow.register_model(model_info.model_uri, registered_name)
        # Registration is evidence of versioning; it is NOT production promotion.
        MlflowClient().set_model_version_tag(registered_name, registered.version, "data_kind", kind)
        metadata = {
            "run_id": run_id, "model_version": str(registered.version),
            "registered_model": registered_name, "model_uri": model_info.model_uri,
            "data_kind": kind, "git_commit": args.git_commit,
            "source_sha256": source_hash(),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "classes": list(CLASSES), "provenance": provenance,
            "dependencies": dependencies, "metrics": metrics,
        }
        mlflow.log_dict(metadata, "deployment/metadata.json")
        # Keep the exact local serving bundle in the run as well, for recovery.
        with tempfile.TemporaryDirectory() as temp_dir:
            bundle_path = save_bundle(pipeline, metadata, temp_dir)
            mlflow.log_artifact(str(bundle_path), artifact_path="deployment")
        output = save_bundle(pipeline, metadata, args.output_dir)
        print(json.dumps({"model_path": str(output), **metadata}, indent=2))
        return metadata


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--demo", action="store_true", help="Synthetic infrastructure test, not a scientific result")
    result.add_argument("--train-csv")
    result.add_argument("--validation-csv")
    result.add_argument("--data-version", default=os.getenv("DATA_VERSION", "not-set"))
    result.add_argument("--output-dir", default="/models/candidate")
    result.add_argument("--max-rows", type=int, default=200_000)
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
    result.add_argument("--git-commit", default=os.getenv("GIT_COMMIT", "unknown"))
    return result


if __name__ == "__main__":
    arguments = parser().parse_args()
    if arguments.max_rows < 1:
        raise SystemExit("--max-rows must be positive")
    run_training(arguments)
