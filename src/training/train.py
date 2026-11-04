"""Train and compare models, test the best one on 2023 data, and save it in MLflow."""

import argparse
from collections.abc import Iterable, Iterator
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
    if data['chunksize'] <= 0:
        raise ValueError('chunksize must be positive')
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
    mask = year == d['train_year']
    if partition == 'train':
        return mask & (df['MONTH'] < d['validation_start_month'])
    if partition == 'validation':
        return mask & (df['MONTH'] >= d['validation_start_month'])
    if partition == 'refit':
        return mask
    raise ValueError(f'Unknown partition: {partition}')


def inventory(config: dict) -> dict[str, dict[str, int]]:
    """Count rows per class in each partition, reading the CSV in chunks.

    Also checks that labels, years and months are valid and that every
    partition contains every class.
    """
    target = config['features']['target']
    classes = config['features']['target_classes']
    counts = {p: {c: 0 for c in classes} for p in ['train', 'validation', 'refit', 'test']}
    header = pd.read_csv(data_path(config), nrows=0)
    required = feature_columns(config) + ['YEAR', target]
    if set(required) - set(header.columns):
        raise ValueError(f'Missing CSV columns: {sorted(set(required) - set(header.columns))}')
    for chunk in pd.read_csv(data_path(config), usecols=['YEAR', 'MONTH', target], chunksize=config['data']['chunksize']):
        if chunk[target].isna().any() or not chunk[target].isin(classes).all():
            raise ValueError('CSV target has missing or unexpected labels')
        if not chunk['YEAR'].isin([config['data']['train_year'], config['data']['test_year']]).all():
            raise ValueError('CSV contains missing or unexpected years')
        if not (chunk['MONTH'].between(1, 12) & (chunk['MONTH'] % 1 == 0)).all():
            raise ValueError('CSV contains invalid months')
        for partition in counts:
            values = chunk.loc[partition_mask(chunk, config, partition), target].value_counts()
            for cls in classes:
                counts[partition][cls] += int(values.get(cls, 0))
    for partition, value in counts.items():
        if any(n == 0 for n in value.values()):
            raise ValueError(f'{partition} must contain every class; found {value}')
    return counts


def iter_partition(config: dict, partition: str) -> Iterator[pd.DataFrame]:
    """Yield the rows of a partition chunk by chunk."""
    cols = list(dict.fromkeys(['YEAR'] + feature_columns(config) + [config['features']['target']]))
    for chunk in pd.read_csv(data_path(config), usecols=cols, chunksize=config['data']['chunksize']):
        subset = chunk.loc[partition_mask(chunk, config, partition)]
        if len(subset):
            yield subset


def stratified_quotas(counts: dict[str, int], size: int) -> dict[str, int]:
    """Split a sample size across classes in proportion to their counts.

    Uses largest-remainder allocation so the total is exact and every
    class gets at least one row.
    """
    total = sum(counts.values())
    size = min(size, total)
    if size < len(counts):
        raise ValueError('Sample must contain at least one example per class')
    expected = {c: size * n / total for c, n in counts.items()}
    quotas = {c: max(1, int(v)) for c, v in expected.items()}
    while sum(quotas.values()) < size:
        cls = max((c for c in counts if quotas[c] < counts[c]), key=lambda c: expected[c] - quotas[c])
        quotas[cls] += 1
    while sum(quotas.values()) > size:
        cls = max((c for c in counts if quotas[c] > 1), key=lambda c: quotas[c] - expected[c])
        quotas[cls] -= 1
    return quotas


def sample_partition(config: dict, partition: str, counts: dict[str, int], size: int | None) -> pd.DataFrame:
    """Draw a stratified random sample of a partition without loading the whole CSV.

    Each chunk contributes a hypergeometric share of each class quota, so
    rows near the start of the file are not favoured. Use size=None for
    the full partition.
    """
    remaining_population = dict(counts)
    remaining_quota = stratified_quotas(counts, sum(counts.values()) if size is None else size)
    target = config['features']['target']
    # Different seed offset per partition so samples are independent but reproducible.
    offsets = {'train': 0, 'validation': 1, 'refit': 2, 'test': 3}
    rng = np.random.default_rng(config['data']['random_seed'] + offsets[partition])
    parts = []
    for chunk in iter_partition(config, partition):
        for cls, group in chunk.groupby(target, sort=False):
            n = len(group)
            take = int(rng.hypergeometric(n, remaining_population[cls] - n, remaining_quota[cls]))
            if take:
                parts.append(group.iloc[rng.choice(n, size=take, replace=False)].copy())
            remaining_population[cls] -= n
            remaining_quota[cls] -= take
    if any(remaining_quota.values()) or any(remaining_population.values()):
        raise ValueError('CSV changed during sampling; restart with an immutable dataset')
    result = pd.concat(parts, ignore_index=True)
    return result.iloc[rng.permutation(len(result))].reset_index(drop=True)


def metrics_from_confusion(cm: np.ndarray, classes: list[str]) -> tuple[dict[str, float], dict[str, dict]]:
    """Compute global and per-class metrics from a confusion matrix.

    Returns a flat metrics dict (for MLflow) and a classification report dict.
    """
    support = cm.sum(axis=1)
    predicted = cm.sum(axis=0)
    tp = cm.diagonal()
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


def evaluate(model: Pipeline, batches: Iterable[pd.DataFrame], config: dict) -> EvaluationResult:
    """Predict batch by batch and return (metrics, report, confusion matrix)."""
    classes = config['features']['target_classes']
    target = config['features']['target']
    cm = np.zeros((len(classes), len(classes)), dtype=np.int64)
    started = time.perf_counter()
    for batch in batches:
        predictions = model.predict(batch[feature_columns(config)])
        if not np.isin(predictions, classes).all():
            raise ValueError('Model returned unexpected labels')
        cm += confusion_matrix(batch[target], predictions, labels=classes)
    metrics, report = metrics_from_confusion(cm, classes)
    metrics['evaluation_seconds'] = time.perf_counter() - started
    return metrics, report, cm


def frame_batches(frame: pd.DataFrame, config: dict) -> Iterator[pd.DataFrame]:
    """Split an in-memory DataFrame into chunksize batches."""
    size = config['data']['chunksize']
    return (frame.iloc[i:i + size] for i in range(0, len(frame), size))


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


def dataset_sha256(path: Path) -> str:
    """Return the SHA256 hash of a file, used to fingerprint the dataset."""
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def get_model_instance(model_type: str, params: dict) -> BaseEstimator:
    """Create a classifier from its config type name and parameters."""
    models = {'RandomForestClassifier': RandomForestClassifier, 'LogisticRegression': LogisticRegression}
    return models[model_type](**params)


def fit_with_warnings(pipeline: Pipeline, frame: pd.DataFrame, config: dict) -> bool:
    """Fit the pipeline, log warnings and fit time to MLflow, and return True if it converged."""
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', ConvergenceWarning)
        pipeline.fit(frame[feature_columns(config)], frame[config['features']['target']])
    messages = [str(w.message) for w in caught]
    mlflow.log_param('converged', not any(issubclass(w.category, ConvergenceWarning) for w in caught))
    mlflow.log_text('\n'.join(messages) or 'No fit warnings', 'fit_warnings.txt')
    mlflow.log_metric('fit_seconds', time.perf_counter() - started)
    for message in messages:
        print(f'Fit warning: {message}', flush=True)
    return not any(issubclass(w.category, ConvergenceWarning) for w in caught)


def train_all_models(config: dict, commit_sha: str | None = None) -> dict:
    """Run the full training workflow and register the best model in MLflow.

    Steps: train a dummy baseline, grid-search every enabled model on the
    validation split, pick the best converged one, refit it on the full
    train year, evaluate once on the test year, verify the saved model
    reloads, then register it with the 'staging' and 'commit-<sha>' aliases.

    Returns:
        A summary dict, also saved to MLflow and to the local output dir.
    """
    commit = commit_sha or os.getenv('COMMIT_SHA')
    if not commit or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', commit):
        raise ValueError('COMMIT_SHA is required and must use letters, digits, dot, underscore or hyphen')
    validate_config(config)
    mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
    mlflow.set_experiment(config['mlflow']['experiment_name'])
    group = uuid.uuid4().hex
    cfg = config['data']
    classes = config['features']['target_classes']
    primary = config['training']['primary_metric']
    fingerprint = dataset_sha256(data_path(config))
    counts = inventory(config)
    print(f'Dataset SHA256: {fingerprint}\nPartition counts: {counts}', flush=True)
    train = sample_partition(config, 'train', counts['train'], cfg['sample_size'])
    validation = (sample_partition(config, 'validation', counts['validation'], cfg['validation_sample_size'])
                  if cfg['validation_sample_size'] is not None else None)

    def validation_batches() -> Iterator[pd.DataFrame]:
        """Use the in-memory sample if there is one, otherwise stream the full partition."""
        return frame_batches(validation, config) if validation is not None else iter_partition(config, 'validation')

    common_tags = {'commit_sha': commit, 'training_group': group, 'dataset_sha256': fingerprint}
    candidate_results = []
    dummy = build_pipeline(config, DummyClassifier(strategy='most_frequent'))
    with mlflow.start_run(run_name=f'dummy_{commit[:7]}', tags={**common_tags, 'role': 'reference'}) as run:
        fit_with_warnings(dummy, train, config)
        dummy_result = evaluate(dummy, validation_batches(), config)
        log_evaluation(dummy_result, 'validation', classes)
        dummy_run = run.info.run_id
    reference_score = dummy_result[0][primary]
    best = None
    for key in config['training']['enabled_models']:
        spec = config['models'][key]
        for params in ParameterGrid(spec['param_grid']):
            # One seed in config controls both sampling and the estimators.
            params = {**params, 'random_state': cfg['random_seed']}
            print(f'Training {key}: {params}', flush=True)
            with mlflow.start_run(run_name=f'{key}_{commit[:7]}', tags={**common_tags, 'role': 'candidate'}) as run:
                mlflow.log_params({'model_type': spec['type'], **params,
                                   'train_rows': len(train), 'validation_rows': sum(counts['validation'].values()) if validation is None else len(validation)})
                mlflow.log_dict(config, 'config.json')
                pipeline = build_pipeline(config, get_model_instance(spec['type'], params))
                converged = fit_with_warnings(pipeline, train, config)
                result = evaluate(pipeline, validation_batches(), config)
                log_evaluation(result, 'validation', classes)
                row = {'model': key, 'params': params, 'run_id': run.info.run_id,
                       'converged': converged, 'validation_metrics': result[0]}
                candidate_results.append(row)
                print(f'Validation {primary}: {result[0][primary]:.4f}; converged={converged}', flush=True)
                if converged and (best is None or result[0][primary] > best['validation_metrics'][primary]):
                    best = row
    if best is None:
        raise RuntimeError('No converged candidate. Inspect fit_warnings.txt in MLflow')
    improvement = best['validation_metrics'][primary] - reference_score
    if not np.isfinite(improvement) or improvement <= config['training']['minimum_improvement_over_dummy']:
        raise RuntimeError(f'No model registered: validation {primary} must improve on dummy ({reference_score:.4f})')

    # Select first, then refit on 2022 and evaluate once on 2023.
    del validation
    final_train = (sample_partition(config, 'refit', counts['refit'], cfg['sample_size'])
                   if config['training']['refit_on_full_train_year'] else train)
    final_model = build_pipeline(config, get_model_instance(config['models'][best['model']]['type'], best['params']))
    test = (sample_partition(config, 'test', counts['test'], cfg['test_sample_size'])
            if cfg['test_sample_size'] is not None else None)

    def test_batches() -> Iterator[pd.DataFrame]:
        """Use the in-memory sample if there is one, otherwise stream the full partition."""
        return frame_batches(test, config) if test is not None else iter_partition(config, 'test')

    with mlflow.start_run(run_name=f'final_{commit[:7]}', tags={**common_tags, 'role': 'final', 'selected_run_id': best['run_id']}) as run:
        final_run_id = run.info.run_id
        mlflow.log_params({'selected_model': best['model'], **best['params'], 'train_rows': len(final_train),
                           'test_rows': sum(counts['test'].values()) if test is None else len(test),
                           'refit_on_full_train_year': config['training']['refit_on_full_train_year']})
        if not fit_with_warnings(final_model, final_train, config):
            raise RuntimeError('Final refit did not converge; model will not be registered')
        final_result = evaluate(final_model, test_batches(), config)
        log_evaluation(final_result, 'test', classes)
        final_dummy = clone(dummy).fit(final_train[feature_columns(config)], final_train[config['features']['target']])
        log_evaluation(evaluate(final_dummy, test_batches(), config), 'dummy_test', classes)
        example = to_bts_frame(final_train.head(5), config)
        expected = final_model.predict(example)
        signature = mlflow.models.infer_signature(example, expected)
        # Ship only importable runtime code, never src/data or the training CSV.
        with tempfile.TemporaryDirectory() as directory:
            code_root = Path(directory) / 'src'
            (code_root / 'preprocessing').mkdir(parents=True)
            for relative in ['__init__.py', 'preprocessing/__init__.py', 'preprocessing/pipeline.py']:
                shutil.copy2(PROJECT_ROOT / 'src' / relative, code_root / relative)
            info = mlflow.sklearn.log_model(final_model, name='model', signature=signature,
                        input_example=example, code_paths=[str(code_root)],
                        serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
                        pip_requirements=[f'{name}=={importlib.metadata.version(name)}'
                                          for name in ['scikit-learn', 'pandas', 'numpy', 'cloudpickle']])
        # Verify both consumers before publishing registry aliases.
        loaded = mlflow.sklearn.load_model(info.model_uri)
        np.testing.assert_array_equal(loaded.predict(example), expected)
        np.testing.assert_array_equal(mlflow.pyfunc.load_model(info.model_uri).predict(example), expected)
        # Also prove the API can handle previously unseen categories after loading.
        unseen = example.copy()
        unseen['ORIGIN'] = 'ZZZ'
        unseen['OP_CARRIER_AIRLINE_ID'] = 999999.0
        unknown_predictions = loaded.predict(unseen)
        if not np.isin(unknown_predictions, classes).all():
            raise RuntimeError('Reloaded model returned invalid unknown-category predictions')
        if dataset_sha256(data_path(config)) != fingerprint:
            raise RuntimeError('Dataset changed during training; refusing to register')
        summary = {'commit_sha': commit, 'dataset_sha256': fingerprint, 'partition_counts': counts,
                   'sample_counts': {'train': len(train), 'refit': len(final_train),
                                     'validation': min(cfg['validation_sample_size'] or sum(counts['validation'].values()), sum(counts['validation'].values())),
                                     'test': sum(counts['test'].values()) if test is None else len(test)},
                   'selected': best, 'candidates': candidate_results,
                   'dummy_validation_run_id': dummy_run, 'dummy_validation_score': reference_score,
                   'final_run_id': final_run_id, 'test_metrics': final_result[0],
                   'model_uri': info.model_uri, 'reload_verified': True,
                   'versions': {name: importlib.metadata.version(name) for name in ['scikit-learn','pandas','numpy','mlflow']}}
        mlflow.log_dict(config, 'config.json')
        mlflow.log_dict(summary, 'training_summary.json')
        mlflow.log_text(json.dumps({'instances': example.to_dict(orient='records')}, indent=2), 'api_request_example.json')
        mlflow.log_text(json.dumps({'classifications': expected.tolist()}, indent=2), 'api_response_example.json')
        version = mlflow.register_model(info.model_uri, config['mlflow']['model_name'])
        client = MlflowClient()
        for alias in ['staging', f'commit-{commit}']:
            client.set_registered_model_alias(config['mlflow']['model_name'], alias, version.version)
        summary['registered_version'] = version.version
        summary['aliases'] = ['staging', f'commit-{commit}']
        mlflow.log_dict(summary, 'training_summary.json')
        out = PROJECT_ROOT / config['training']['output_dir'] / final_run_id
        out.mkdir(parents=True, exist_ok=True)
        (out / 'training_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        (out / 'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
        print(f'Model registered: {config["mlflow"]["model_name"]} v{version.version}\nSummary: {out / "training_summary.json"}', flush=True)
    return summary


def main() -> None:
    """CLI entry point: parse --config and run training."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, help='Alternative YAML config')
    args = parser.parse_args()
    train_all_models(load_config(args.config))


if __name__ == '__main__':
    main()