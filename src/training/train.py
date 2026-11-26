"""Train and compare models, test the best one on 2023 data, and save it in MLflow."""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import time
import uuid
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
import yaml
from sklearn.base import BaseEstimator, clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline

from src.preprocessing.pipeline import build_pipeline, feature_columns, to_bts_frame

PROJECT_ROOT = Path(__file__).resolve().parents[2]

EvaluationResult = tuple[dict, dict, np.ndarray]


def load_config(path: Path | None = None) -> dict:
    """Load the YAML config, apply environment variable overrides and validate it."""
    with open(path or PROJECT_ROOT / 'src/config/config.yaml', encoding='utf-8') as stream:
        config = yaml.safe_load(stream)
    for key in ['tracking_uri', 'experiment_name', 'model_name']:
        config['mlflow'][key] = os.getenv(f'MLFLOW_{key.upper()}', config['mlflow'][key])
    if os.getenv('DATA_PATH'):
        config['data']['processed_path'] = os.environ['DATA_PATH']
    for env, key in [('TRAIN_SAMPLE_SIZE', 'sample_size'),
                     ('VALIDATION_SAMPLE_SIZE', 'validation_sample_size'),
                     ('TEST_SAMPLE_SIZE', 'test_sample_size')]:
        if env in os.environ:
            value = os.environ[env]
            config['data'][key] = None if value.lower() in ['none', 'null', 'all'] else int(value)
    if os.getenv('ENABLED_MODELS'):
        config['training']['enabled_models'] = [v.strip() for v in os.environ['ENABLED_MODELS'].split(',')]
    if os.getenv('TRAINING_OUTPUT_DIR'):
        config['training']['output_dir'] = os.environ['TRAINING_OUTPUT_DIR']
    validate_config(config)
    return config


def validate_config(config: dict) -> None:
    """Raise ValueError if the config is inconsistent or unsupported."""
    data = config['data']
    if data['train_year'] >= data['test_year']:
        raise ValueError('train_year must precede test_year')
    if not 2 <= data['validation_start_month'] <= 12:
        raise ValueError('validation_start_month must be between 2 and 12')
    for key in ['sample_size', 'validation_sample_size', 'test_sample_size']:
        if data[key] is not None and data[key] < len(config['features']['target_classes']):
            raise ValueError(f'{key} must allow at least one row per target class, or be null')
    enabled = config['training']['enabled_models']
    if not enabled or set(enabled) - set(config['models']):
        raise ValueError('enabled_models must name at least one configured model')
    if config['training']['primary_metric'] not in ['f1_macro', 'f1_weighted', 'accuracy', 'balanced_accuracy']:
        raise ValueError('Unsupported primary_metric')
    if config['training']['minimum_improvement_over_dummy'] < 0:
        raise ValueError('minimum_improvement_over_dummy must be nonnegative')


def data_path(config: dict) -> Path:
    """Return the absolute path of the processed CSV."""
    return PROJECT_ROOT / config['data']['processed_path']


def partition_mask(df: pd.DataFrame, config: dict, partition: str) -> pd.Series:
    """Return a boolean mask selecting the rows of a temporal partition.

    Partitions: train (train year, before validation month), validation
    (train year, from validation month), refit (whole train year), test (test year).
    """
    d = config['data']
    year = df['YEAR']
    if partition == 'test':
        return year == d['test_year']
    in_train_year = year == d['train_year']
    if partition == 'train':
        return in_train_year & (df['MONTH'] < d['validation_start_month'])
    if partition == 'validation':
        return in_train_year & (df['MONTH'] >= d['validation_start_month'])
    if partition == 'refit':
        return in_train_year
    raise ValueError(f'Unknown partition: {partition}')


def column_dtypes(config: dict) -> dict:
    """Compact dtype per column so the whole CSV fits comfortably in RAM.

    Text columns become 'category' (stored once, referenced by small codes)
    and numbers use the smallest int/float that holds them. This keeps the
    full 12M-row table well under a gigabyte.
    """
    f = config['features']
    dtypes: dict[str, str] = {'YEAR': 'int16', f['target']: 'category', 'AIRLINE_ID': 'int32'}
    for col in f['temporal']:
        dtypes[col] = 'int16'
    for col in f['time_hhmm']:
        dtypes[col] = 'float32'
    for col in f['numeric']:
        dtypes[col] = 'float32'
    for col in f['categorical']:
        if col != 'AIRLINE_ID':
            dtypes[col] = 'category'
    return dtypes


def load_dataset(config: dict) -> pd.DataFrame:
    """Load the whole processed CSV into memory, with compact dtypes.

    Reads only the columns the model needs, each with the smallest dtype that
    fits (see column_dtypes), so the full dataset fits in RAM in one go.
    """
    columns = list(dict.fromkeys(['YEAR'] + feature_columns(config) + [config['features']['target']]))
    header = pd.read_csv(data_path(config), nrows=0)
    missing_columns = set(columns) - set(header.columns)
    if missing_columns:
        raise ValueError(f'Missing CSV columns: {sorted(missing_columns)}')
    return pd.read_csv(data_path(config), usecols=columns, dtype=column_dtypes(config))


def validate_dataset(df: pd.DataFrame, config: dict) -> None:
    """Check labels, years and months are valid and every partition has every class."""
    data, features = config['data'], config['features']
    classes = features['target_classes']
    target = features['target']
    if df[target].isna().any() or not df[target].isin(classes).all():
        raise ValueError('CSV target has missing or unexpected labels')
    if not df['YEAR'].isin([data['train_year'], data['test_year']]).all():
        raise ValueError('CSV contains missing or unexpected years')
    if not df['MONTH'].between(1, 12).all():
        raise ValueError('CSV contains invalid months')
    for partition in ['train', 'validation', 'refit', 'test']:
        present = set(df.loc[partition_mask(df, config, partition), target].unique())
        missing = set(classes) - present
        if missing:
            raise ValueError(f'{partition} is missing classes: {sorted(missing)}')


def sample_partition(df: pd.DataFrame, config: dict, partition: str, size: int | None) -> pd.DataFrame:
    """Return a stratified random sample of one partition (size=None -> all rows).

    Keeps the same fraction of every class, so the class proportions of the
    partition are preserved. The result is shuffled and reproducible.
    """
    rows = df[partition_mask(df, config, partition)]
    seed = config['data']['random_seed']
    if size is None or size >= len(rows):
        sample = rows
    else:
        fraction = size / len(rows)
        sample = rows.groupby(config['features']['target'], observed=True, group_keys=False).sample(
            frac=fraction, random_state=seed)
    return sample.sample(frac=1, random_state=seed).reset_index(drop=True)


def metrics_from_confusion(cm: np.ndarray, classes: list[str]) -> tuple[dict[str, float], dict[str, dict]]:
    """Compute global and per-class metrics from a confusion matrix.

    Returns a flat metrics dict (for MLflow) and a classification report dict.
    """
    support = cm.sum(axis=1)       # true rows per class
    predicted = cm.sum(axis=0)     # predicted columns per class
    tp = cm.diagonal()
    # Divide safely: a class never predicted / never present scores 0 instead of NaN.
    precision = np.divide(tp, predicted, out=np.zeros(len(classes)), where=predicted != 0)
    recall = np.divide(tp, support, out=np.zeros(len(classes)), where=support != 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros(len(classes)), where=precision + recall != 0)

    total = int(support.sum())
    if not total:
        raise ValueError('Cannot evaluate an empty partition')
    metrics = {
        'accuracy': float(tp.sum() / total),
        'balanced_accuracy': float(recall[support > 0].mean()),
        'f1_macro': float(f1.mean()),
        'f1_weighted': float(np.dot(f1, support) / total),
        'precision_weighted': float(np.dot(precision, support) / total),
        'recall_weighted': float(np.dot(recall, support) / total),
    }
    report = {}
    for i, cls in enumerate(classes):
        report[cls] = {'precision': float(precision[i]), 'recall': float(recall[i]),
                       'f1-score': float(f1[i]), 'support': int(support[i])}
        for metric, values in [('precision', precision), ('recall', recall), ('f1', f1)]:
            metrics[f'{metric}_{cls}'] = float(values[i])
    return metrics, report


def evaluate(model: Pipeline, frame: pd.DataFrame, config: dict) -> EvaluationResult:
    """Predict on a frame and return (metrics, report, confusion matrix)."""
    classes = config['features']['target_classes']
    target = config['features']['target']
    started = time.perf_counter()
    predictions = model.predict(frame[feature_columns(config)])
    if not np.isin(predictions, classes).all():
        raise ValueError('Model returned unexpected labels')
    cm = confusion_matrix(frame[target], predictions, labels=classes)
    metrics, report = metrics_from_confusion(cm, classes)
    metrics['evaluation_seconds'] = time.perf_counter() - started
    return metrics, report, cm


def log_evaluation(result: EvaluationResult, prefix: str, classes: list[str]) -> None:
    """Log metrics, confusion matrix (JSON and PNG) and report to the active MLflow run."""
    metrics, report, cm = result
    mlflow.log_metrics({f'{prefix}_{k}': v for k, v in metrics.items()})
    mlflow.log_dict({'labels': classes, 'matrix': cm.tolist()}, f'{prefix}/confusion_matrix.json')
    mlflow.log_dict(report, f'{prefix}/classification_report.json')
    mlflow.log_text(pd.DataFrame(report).T.to_string(), f'{prefix}/classification_report.txt')
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, cmap='Blues')
    fig.colorbar(im, ax=ax)
    ax.set(xticks=range(len(classes)), yticks=range(len(classes)), xticklabels=classes,
           yticklabels=classes, xlabel='Predicted label', ylabel='True label', title=prefix)
    plt.setp(ax.get_xticklabels(), rotation=35, ha='right')
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f'{cm[i, j]:,}', ha='center', va='center',
                    color='white' if cm[i, j] > cm.max() / 2 else 'black')
    fig.tight_layout()
    mlflow.log_figure(fig, f'{prefix}/confusion_matrix.png')
    plt.close(fig)


def log_feature_importance(model: Pipeline, top_n: int = 20) -> None:
    """Log the model's top feature importances to MLflow, when it exposes them.

    Tree models (RandomForest) have feature_importances_; the names come from
    the fitted preprocessor. Models without importances (LogisticRegression)
    are skipped.
    """
    classifier = model.named_steps['classifier']
    if not hasattr(classifier, 'feature_importances_'):
        return
    names = model.named_steps['preprocessor'].get_feature_names_out()
    ranked = sorted(zip(names, classifier.feature_importances_), key=lambda nv: nv[1], reverse=True)
    mlflow.log_dict({str(name): float(importance) for name, importance in ranked[:top_n]},
                    'feature_importance.json')


def dataset_sha256(path: Path) -> str:
    """Return the SHA256 hash of a file, used to fingerprint the dataset."""
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def make_classifier(model_type: str, params: dict) -> BaseEstimator:
    """Create a classifier from its config type name and parameters."""
    classifiers = {'RandomForestClassifier': RandomForestClassifier, 'LogisticRegression': LogisticRegression}
    return classifiers[model_type](**params)


def fit_with_warnings(pipeline: Pipeline, frame: pd.DataFrame, config: dict) -> bool:
    """Fit the pipeline, log warnings and fit time to MLflow, and return True if it converged."""
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', ConvergenceWarning)
        pipeline.fit(frame[feature_columns(config)], frame[config['features']['target']])
    messages = [str(w.message) for w in caught]
    converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
    mlflow.log_param('converged', converged)
    mlflow.log_text('\n'.join(messages) or 'No fit warnings', 'fit_warnings.txt')
    mlflow.log_metric('fit_seconds', time.perf_counter() - started)
    for message in messages:
        print(f'Fit warning: {message}', flush=True)
    return converged


def _resolve_commit(commit_sha: str | None) -> str:
    """Return the commit SHA from the argument or the COMMIT_SHA env var, validating its form."""
    commit = commit_sha or os.getenv('COMMIT_SHA')
    if not commit or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', commit):
        raise ValueError('COMMIT_SHA is required and must use letters, digits, dot, underscore or hyphen')
    return commit


def run_dummy_baseline(config: dict, train: pd.DataFrame, validation: pd.DataFrame, common_tags: dict,
                       commit: str) -> tuple[Pipeline, str, EvaluationResult]:
    """Fit a most-frequent dummy and log its validation metrics as the reference run.

    Returns the fitted dummy pipeline, its MLflow run id, and its evaluation
    result (whose primary metric is the score every candidate must beat).
    """
    classes = config['features']['target_classes']
    dummy = build_pipeline(config, DummyClassifier(strategy='most_frequent'))
    with mlflow.start_run(run_name=f'dummy_{commit[:7]}', tags={**common_tags, 'role': 'reference'}) as run:
        fit_with_warnings(dummy, train, config)
        dummy_result = evaluate(dummy, validation, config)
        log_evaluation(dummy_result, 'validation', classes)
        dummy_run_id = run.info.run_id
    return dummy, dummy_run_id, dummy_result


def search_best_candidate(config: dict, train: pd.DataFrame, validation: pd.DataFrame,
                          common_tags: dict, commit: str) -> tuple[dict | None, list[dict]]:
    """Grid-search every enabled model on the validation split, logging each run.

    Returns the best converged candidate (by the primary metric) and the list
    of every candidate's result. ``best`` is None when nothing converged.
    """
    classes = config['features']['target_classes']
    primary = config['training']['primary_metric']
    seed = config['data']['random_seed']

    candidate_results: list[dict] = []
    best = None
    for key in config['training']['enabled_models']:
        spec = config['models'][key]
        for grid_params in ParameterGrid(spec['param_grid']):
            # One seed in config controls both the sampling and the estimators.
            params = {**grid_params, 'random_state': seed}
            print(f'Training {key}: {params}', flush=True)
            with mlflow.start_run(run_name=f'{key}_{commit[:7]}', tags={**common_tags, 'role': 'candidate'}) as run:
                mlflow.log_params({'model_type': spec['type'], **params,
                                   'train_rows': len(train), 'validation_rows': len(validation)})
                mlflow.log_dict(config, 'config.json')
                pipeline = build_pipeline(config, make_classifier(spec['type'], params))
                converged = fit_with_warnings(pipeline, train, config)
                result = evaluate(pipeline, validation, config)
                log_evaluation(result, 'validation', classes)
                candidate = {'model': key, 'params': params, 'run_id': run.info.run_id,
                             'converged': converged, 'validation_metrics': result[0]}
                candidate_results.append(candidate)
                print(f'Validation {primary}: {result[0][primary]:.4f}; converged={converged}', flush=True)
                if converged and (best is None or result[0][primary] > best['validation_metrics'][primary]):
                    best = candidate
    return best, candidate_results


def log_and_verify_model(final_model: Pipeline, final_train: pd.DataFrame,
                         config: dict) -> tuple[object, pd.DataFrame, np.ndarray]:
    """Log the fitted model to MLflow and prove it reloads and handles unknowns.

    Ships only the importable preprocessing code (never the dataset), reloads
    the model through both the sklearn and pyfunc flavors, and checks it still
    returns valid labels for a previously unseen airport and airline.

    Returns the logged-model info, the BTS example frame and its predictions.
    """
    classes = config['features']['target_classes']
    example = to_bts_frame(final_train.head(5), config)
    expected = final_model.predict(example)
    signature = mlflow.models.infer_signature(example, expected)

    # Ship only importable runtime code, never src/data or the training CSV.
    with tempfile.TemporaryDirectory() as directory:
        code_root = Path(directory) / 'src'
        (code_root / 'preprocessing').mkdir(parents=True)
        for relative in ['__init__.py', 'preprocessing/__init__.py', 'preprocessing/pipeline.py']:
            shutil.copy2(PROJECT_ROOT / 'src' / relative, code_root / relative)
        info = mlflow.sklearn.log_model(
            final_model, name='model', signature=signature, input_example=example,
            code_paths=[str(code_root)],
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
            pip_requirements=[f'{name}=={importlib.metadata.version(name)}'
                              for name in ['scikit-learn', 'pandas', 'numpy', 'cloudpickle']])

    # Verify both consumers before the caller publishes any registry alias.
    loaded = mlflow.sklearn.load_model(info.model_uri)
    np.testing.assert_array_equal(loaded.predict(example), expected)
    np.testing.assert_array_equal(mlflow.pyfunc.load_model(info.model_uri).predict(example), expected)

    # Prove the model survives a previously unseen airport and airline.
    unseen = example.copy()
    unseen['ORIGIN'] = 'ZZZ'
    unseen['OP_CARRIER_AIRLINE_ID'] = 999999.0
    if not np.isin(loaded.predict(unseen), classes).all():
        raise RuntimeError('Reloaded model returned invalid unknown-category predictions')
    return info, example, expected


def train_all_models(config: dict, commit_sha: str | None = None) -> dict:
    """Run the full training workflow and register the best model in MLflow.

    Steps: load and validate the dataset, draw the train/validation/test
    samples, train a dummy baseline, grid-search every enabled model on
    validation, pick the best converged one, refit it on the full train year,
    evaluate once on the test year, verify the saved model reloads, then
    register it under the ``commit-<sha>`` alias. Promotion to staging and
    production is a separate step (see model_promotion).

    Returns:
        A summary dict, also saved to MLflow and to the local output dir.
    """
    commit = _resolve_commit(commit_sha)
    validate_config(config)
    mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
    mlflow.set_experiment(config['mlflow']['experiment_name'])

    cfg = config['data']
    classes = config['features']['target_classes']
    primary = config['training']['primary_metric']

    # Load the whole dataset once, then draw every sample from it in memory.
    fingerprint = dataset_sha256(data_path(config))
    df = load_dataset(config)
    validate_dataset(df, config)
    print(f'Dataset SHA256: {fingerprint}\nRows loaded: {len(df):,}', flush=True)

    train = sample_partition(df, config, 'train', cfg['sample_size'])
    validation = sample_partition(df, config, 'validation', cfg['validation_sample_size'])
    test = sample_partition(df, config, 'test', cfg['test_sample_size'])
    refit_train = (sample_partition(df, config, 'refit', cfg['sample_size'])
                   if config['training']['refit_on_full_train_year'] else train)
    del df  # free the full dataset; only the samples are needed from here on

    common_tags = {'commit_sha': commit, 'training_group': uuid.uuid4().hex, 'dataset_sha256': fingerprint}

    # Reference baseline, then grid-search for the best candidate.
    dummy, dummy_run_id, dummy_result = run_dummy_baseline(config, train, validation, common_tags, commit)
    reference_score = dummy_result[0][primary]
    best, candidate_results = search_best_candidate(config, train, validation, common_tags, commit)

    # The winner must exist and beat the dummy, or nothing is registered.
    if best is None:
        raise RuntimeError('No converged candidate. Inspect fit_warnings.txt in MLflow')
    improvement = best['validation_metrics'][primary] - reference_score
    if not np.isfinite(improvement) or improvement <= config['training']['minimum_improvement_over_dummy']:
        raise RuntimeError(f'No model registered: validation {primary} must improve on dummy ({reference_score:.4f})')

    final_model = build_pipeline(config, make_classifier(config['models'][best['model']]['type'], best['params']))

    with mlflow.start_run(run_name=f'final_{commit[:7]}',
                          tags={**common_tags, 'role': 'final', 'selected_run_id': best['run_id']}) as run:
        final_run_id = run.info.run_id
        mlflow.log_params({'selected_model': best['model'], **best['params'], 'train_rows': len(refit_train),
                           'test_rows': len(test),
                           'refit_on_full_train_year': config['training']['refit_on_full_train_year']})
        if not fit_with_warnings(final_model, refit_train, config):
            raise RuntimeError('Final refit did not converge; model will not be registered')
        final_result = evaluate(final_model, test, config)
        log_evaluation(final_result, 'test', classes)
        final_dummy = clone(dummy).fit(refit_train[feature_columns(config)], refit_train[config['features']['target']])
        log_evaluation(evaluate(final_dummy, test, config), 'dummy_test', classes)
        log_feature_importance(final_model)

        # Log the model artifact and prove it reloads and handles unknown categories.
        info, example, expected = log_and_verify_model(final_model, refit_train, config)
        if dataset_sha256(data_path(config)) != fingerprint:
            raise RuntimeError('Dataset changed during training; refusing to register')

        # Assemble the run summary and log the API request/response examples.
        summary = {
            'commit_sha': commit, 'dataset_sha256': fingerprint,
            'sample_counts': {'train': len(train), 'refit': len(refit_train),
                              'validation': len(validation), 'test': len(test)},
            'selected': best, 'candidates': candidate_results,
            'dummy_validation_run_id': dummy_run_id, 'dummy_validation_score': reference_score,
            'final_run_id': final_run_id, 'test_metrics': final_result[0],
            'model_uri': info.model_uri, 'reload_verified': True,
            'versions': {name: importlib.metadata.version(name) for name in ['scikit-learn', 'pandas', 'numpy', 'mlflow']},
        }
        mlflow.log_dict(config, 'config.json')
        mlflow.log_dict(summary, 'training_summary.json')
        mlflow.log_text(json.dumps({'instances': example.to_dict(orient='records')}, indent=2), 'api_request_example.json')
        mlflow.log_text(json.dumps({'classifications': expected.tolist()}, indent=2), 'api_response_example.json')

        # Register the version and tag it with the commit alias.
        version = mlflow.register_model(info.model_uri, config['mlflow']['model_name'])
        commit_alias = f'commit-{commit}'
        MlflowClient().set_registered_model_alias(config['mlflow']['model_name'], commit_alias, version.version)
        summary['registered_version'] = version.version
        summary['aliases'] = [commit_alias]
        mlflow.log_dict(summary, 'training_summary.json')

        # Persist the summary and the exact config next to the run output.
        out = PROJECT_ROOT / config['training']['output_dir'] / final_run_id
        out.mkdir(parents=True, exist_ok=True)
        (out / 'training_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        (out / 'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
        print(f'Model registered: {config["mlflow"]["model_name"]} v{version.version}\n'
              f'Summary: {out / "training_summary.json"}', flush=True)
    return summary


def main() -> None:
    """CLI entry point: parse --config and run training."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, help='Alternative YAML config')
    args = parser.parse_args()
    train_all_models(load_config(args.config))


if __name__ == '__main__':
    main()