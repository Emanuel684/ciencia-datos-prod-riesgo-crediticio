from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ft_engineering import split_features_target
from model_deploy import ModelDeploymentService
from model_training import RANDOM_STATE, summarize_classification


@dataclass
class MonitoringConfig:
    data_path: str = "Base_de_datos.xlsx"
    target_col: str = "Pago_atiempo"
    monitor_dir: str = "artifacts/monitoring"
    deploy_endpoint_url: Optional[str] = None
    baseline_sample_size: int = 2000
    monitor_sample_size: int = 500
    period_seconds: int = 300


def _utc_now_iso() -> str:
    """Return current UTC time as an ISO 8601 formatted string.

    Returns:
        str: Current UTC datetime in ISO 8601 format with timezone offset,
            e.g. "2026-03-05T12:34:56.789012+00:00".

    Example:
        >>> isinstance(_utc_now_iso(), str)
        True
    """
    return datetime.now(timezone.utc).isoformat()


def _psi_numeric(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    """Compute Population Stability Index (PSI) between two numeric series.

    The function bins `expected` into quantile-based cut points (default `bins=10`),
    computes the proportion of observations in each bin for both `expected` and
    `actual`, applies a small epsilon to avoid division-by-zero, and returns the
    PSI defined as sum((actual_ratio - expected_ratio) * log(actual_ratio / expected_ratio)).

    Args:
        expected (pd.Series): Reference / baseline numeric series. Non-numeric values
            will be coerced to numeric and dropped.
        actual (pd.Series): Current / monitoring numeric series. Non-numeric values
            will be coerced to numeric and dropped.
        bins (int, optional): Number of quantile bins to use when computing cut points.
            Defaults to 10.

    Returns:
        float: PSI value (>= 0). Returns `np.nan` if either series has no valid numeric
            values after coercion. Returns `0.0` when there are insufficient unique
            cut points to form meaningful bins.

    Raises:
        None

    Example:
        >>> exp = pd.Series([1, 2, 3, 4, 5])
        >>> act = pd.Series([1, 2, 2, 4, 6])
        >>> _psi_numeric(exp, act)  # doctest: +SKIP
        0.0
    """
    expected = pd.to_numeric(expected, errors="coerce").dropna()
    actual = pd.to_numeric(actual, errors="coerce").dropna()

    if expected.empty or actual.empty:
        return np.nan

    quantiles = np.linspace(0, 1, bins + 1)
    cut_points = np.unique(np.quantile(expected, quantiles))
    if len(cut_points) < 3:
        return 0.0

    expected_counts, _ = np.histogram(expected, bins=cut_points)
    actual_counts, _ = np.histogram(actual, bins=cut_points)

    expected_ratio = expected_counts / max(expected_counts.sum(), 1)
    actual_ratio = actual_counts / max(actual_counts.sum(), 1)

    epsilon = 1e-6
    expected_ratio = np.clip(expected_ratio, epsilon, None)
    actual_ratio = np.clip(actual_ratio, epsilon, None)

    return float(
        np.sum((actual_ratio - expected_ratio) * np.log(actual_ratio / expected_ratio))
    )


def _psi_categorical(expected: pd.Series, actual: pd.Series) -> float:
    """Compute Population Stability Index (PSI) between two categorical series.

    The function computes PSI by comparing the category frequency distributions of
    `expected` (reference) and `actual` (current) series. Missing values are
    treated as the string "<NA>" and categories present in either series are
    included. A small epsilon is applied to frequencies to avoid division by zero
    and log-of-zero issues.

    Args:
        expected (pd.Series): Reference / baseline categorical series.
        actual (pd.Series): Current / monitoring categorical series.

    Returns:
        float: PSI value (non-negative). Returns np.nan if there are no categories
            to compare.

    Raises:
        None

    Example:
        >>> expected = pd.Series(["a", "b", "a", None])
        >>> actual = pd.Series(["a", "b", "c"])
        >>> _psi_categorical(expected, actual)  # doctest: +SKIP
        0.????  # numeric PSI value
    """
    expected = expected.astype("string").fillna("<NA>")
    actual = actual.astype("string").fillna("<NA>")

    categories = sorted(set(expected.unique()).union(set(actual.unique())))
    if not categories:
        return np.nan

    expected_freq = expected.value_counts(normalize=True).reindex(
        categories, fill_value=0.0
    )
    actual_freq = actual.value_counts(normalize=True).reindex(
        categories, fill_value=0.0
    )

    epsilon = 1e-6
    expected_freq = np.clip(expected_freq.values, epsilon, None)
    actual_freq = np.clip(actual_freq.values, epsilon, None)

    return float(
        np.sum((actual_freq - expected_freq) * np.log(actual_freq / expected_freq))
    )


def _compute_drift_table(
    reference_df: pd.DataFrame, current_df: pd.DataFrame
) -> pd.DataFrame:
    """Compute population stability index (PSI) per feature between reference and current data.

    Calculates a PSI value for each column present in both dataframes. Numeric
    features are compared using histogram-based binning via _psi_numeric, while
    non-numeric features use category frequency comparison via _psi_categorical.
    The returned table contains one row per feature with its type, computed PSI
    and a boolean drift_flag (True when PSI >= 0.2).

    Args:
        reference_df (pd.DataFrame): Reference / baseline dataframe used as the
            expected distribution.
        current_df (pd.DataFrame): Current / monitoring dataframe to compare
            against the reference distribution.

    Returns:
        pd.DataFrame: DataFrame with columns:
            - feature (str): feature name
            - feature_type (str): "numeric" or "categorical"
            - psi (float): population stability index (NaN if not computable)
            - drift_flag (bool): True if psi >= 0.2, False otherwise

        The DataFrame is sorted by "psi" in descending order, with NaN values
        placed last.

    Example:
        >>> ref = pd.DataFrame({"age": [20, 30, 40], "cat": ["a", "b", "a"]})
        >>> cur = pd.DataFrame({"age": [22, 35, 41], "cat": ["a", "b", "c"]})
        >>> table = _compute_drift_table(ref, cur)
        >>> table.columns
        Index(['feature', 'feature_type', 'psi', 'drift_flag'], dtype='object')
    """
    rows = []
    common_cols = [col for col in reference_df.columns if col in current_df.columns]

    for col in common_cols:
        ref = reference_df[col]
        cur = current_df[col]

        if pd.api.types.is_numeric_dtype(ref):
            psi = _psi_numeric(ref, cur)
            feature_type = "numeric"
        else:
            psi = _psi_categorical(ref, cur)
            feature_type = "categorical"

        rows.append(
            {
                "feature": col,
                "feature_type": feature_type,
                "psi": psi,
                "drift_flag": bool(psi >= 0.2) if pd.notna(psi) else False,
            }
        )

    return pd.DataFrame(rows).sort_values("psi", ascending=False, na_position="last")


def _call_endpoint(endpoint_url: str, records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Send a JSON POST request with records to a prediction endpoint and return the parsed JSON response.

    Args:
        endpoint_url (str): Full URL of the prediction endpoint (e.g. "http://host:port/predict").
        records (List[Dict[str, Any]]): List of input records to include in the request body under the
            "records" key (one dict per sample).

    Returns:
        Dict[str, Any]: Parsed JSON response returned by the endpoint.

    Raises:
        urllib.error.URLError: If a network-related error occurs when opening the URL.
        urllib.error.HTTPError: If the server returns an HTTP error status.
        json.JSONDecodeError: If the response body cannot be decoded as valid JSON.
        Exception: Propagates any other unexpected exceptions raised during the request/response handling.

    Example:
        >>> resp = _call_endpoint("http://localhost:8000/predict", [{"saldo_principal": 1000}])
        >>> resp.get("predictions")
    """
    payload = json.dumps({"records": records}).encode("utf-8")
    request = urllib.request.Request(
        endpoint_url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(request, timeout=90) as response:
        response_body = response.read().decode("utf-8")
    return json.loads(response_body)


def _predict_with_deploy(
    records: List[Dict[str, Any]],
    deploy_endpoint_url: Optional[str],
) -> Tuple[List[int], Optional[List[float]], str]:
    """Predict using a remote endpoint if available, otherwise fallback to the local service.

    Attempts to POST the provided records to the given deploy endpoint. If the call
    succeeds the function returns the endpoint predictions and probabilities and the
    source flag 'endpoint'. If a network error or timeout occurs, the function falls
    back to the local ModelDeploymentService and returns its predictions and the
    source flag 'local_service'.

    Args:
        records (List[Dict[str, Any]]): List of input records to score (one dict per sample).
        deploy_endpoint_url (Optional[str]): Full URL of the deployed prediction endpoint.
            If None, the function will always use the local ModelDeploymentService.

    Returns:
        Tuple[List[int], Optional[List[float]], str]:
            - predictions: List[int] of predicted class labels.
            - probabilities: Optional[List[float]] of predicted probabilities (or None).
            - source: str indicating the prediction source: 'endpoint' or 'local_service'.

    Raises:
        Exception: Any unexpected exception raised by the local ModelDeploymentService or
            by the endpoint call that is not explicitly caught will propagate.

    Example:
        >>> records = [{"saldo_principal": 1000, "edad_cliente": 30}, {"saldo_principal": 500, "edad_cliente": 45}]
        >>> preds, probs, src = _predict_with_deploy(records, "http://localhost:8000/predict")
    """
    if deploy_endpoint_url:
        try:
            payload = _call_endpoint(deploy_endpoint_url, records)
            return payload["predictions"], payload.get("probabilities"), "endpoint"
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError):
            pass

    service = ModelDeploymentService()
    response = service.predict_batch(records)
    return response.predictions, response.probabilities, "local_service"


def _append_csv(df: pd.DataFrame, csv_path: Path) -> None:
    """Append a DataFrame to a CSV file, creating parent directories when needed.

    If `csv_path` exists the dataframe is appended without writing a header.
    If it does not exist the dataframe is written including the header row.

    Args:
        df (pd.DataFrame): DataFrame to write or append to CSV.
        csv_path (Path): Destination path for the CSV file.

    Returns:
        None

    Raises:
        OSError: If the parent directory cannot be created.
        ValueError: If `df` is not a pandas DataFrame.

    Example:
        >>> from pathlib import Path
        >>> df = pd.DataFrame({"a": [1, 2]})
        >>> _append_csv(df, Path("artifacts/monitoring/prediction_log.csv"))
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if csv_path.exists():
        df.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        df.to_csv(csv_path, index=False)


def _build_prediction_log(
    input_df: pd.DataFrame,
    predictions: List[int],
    probabilities: Optional[List[float]],
    source: str,
    target_col: str,
) -> pd.DataFrame:
    """Create a prediction log DataFrame by combining inputs with model outputs.

    The returned DataFrame is a copy of `input_df` augmented with the model's
    prediction results and metadata useful for monitoring and auditing.

    Args:
        input_df (pd.DataFrame): Input features DataFrame used for scoring. May
            include the true target column.
        predictions (List[int]): Sequence of predicted class labels, aligned
            with the rows of `input_df`.
        probabilities (Optional[List[float]]): Sequence of predicted probabilities
            aligned with `predictions`, or None if not available.
        source (str): Identifier of the prediction source (e.g. 'endpoint',
            'local_service').
        target_col (str): Name of the target column in `input_df`. If present,
            its values are copied into the 'observed_target' column.

    Returns:
        pd.DataFrame: Copy of `input_df` with the following added columns:
            - prediction (int): model predicted label
            - prediction_probability (float): predicted probability or NaN
            - observed_target (int or NaN): value from `target_col` if present
            - prediction_source (str): value of `source`
            - scored_at_utc (str): ISO8601 UTC timestamp when log was built

    Raises:
        None

    Example:
        >>> df = pd.DataFrame({"x":[1,2], "Pago_atiempo":[1,0]})
        >>> log = _build_prediction_log(df, [1,0], [0.9, 0.2], "local", "Pago_atiempo")
        >>> "prediction" in log.columns
        True
    """
    log_df = input_df.copy()
    log_df["prediction"] = predictions
    if probabilities is not None:
        log_df["prediction_probability"] = probabilities
    else:
        log_df["prediction_probability"] = np.nan

    if target_col in log_df.columns:
        log_df["observed_target"] = log_df[target_col]
    else:
        log_df["observed_target"] = np.nan

    log_df["prediction_source"] = source
    log_df["scored_at_utc"] = _utc_now_iso()
    return log_df


def _compute_performance_if_available(
    log_df: pd.DataFrame,
) -> Optional[Dict[str, float]]:
    """Compute classification performance metrics when ground truth is available.

    This function extracts rows from `log_df` that contain an observed target
    value and computes classification metrics using `summarize_classification`.
    If no rows contain an observed target the function returns None.

    Args:
        log_df (pd.DataFrame): Prediction log produced by `_build_prediction_log`.
            Expected columns:
              - "observed_target" (may contain NaN for unlabeled rows)
              - "prediction" (model predicted labels)
              - "prediction_probability" (optional; may contain NaN)

    Returns:
        Optional[Dict[str, float]]: Dictionary of classification metrics as returned
        by `summarize_classification` (for example accuracy, precision, recall,
        f1, roc_auc). Returns None when there are no labeled rows to evaluate.

    Raises:
        None

    Example:
        >>> df = pd.DataFrame({
        ...     "observed_target": [1, 0, 1],
        ...     "prediction": [1, 0, 0],
        ...     "prediction_probability": [0.9, 0.2, 0.6],
        ... })
        >>> _compute_performance_if_available(df)  # doctest: +SKIP
        {'accuracy': 0.666..., 'precision': ..., 'recall': ..., 'f1': ..., 'roc_auc': ...}
    """
    valid = log_df.dropna(subset=["observed_target"])
    if valid.empty:
        return None

    y_true = valid["observed_target"].astype(int)
    y_pred = valid["prediction"].astype(int)
    y_proba = valid["prediction_probability"]
    if y_proba.isna().all():
        y_proba_arr = None
    else:
        y_proba_arr = y_proba.fillna(0.0).astype(float).values

    return summarize_classification(y_true, y_pred.values, y_proba_arr)


def run_monitoring_cycle(
    config: MonitoringConfig = MonitoringConfig(),
) -> Dict[str, Any]:
    """Execute a single monitoring cycle: score a sample, compute drift and performance, and persist artifacts.

    This function performs a full monitoring run:
    1. Loads the dataset from `config.data_path`.
    2. Splits features/target and creates a baseline and a monitoring sample.
    3. Scores the monitoring sample using the deployed endpoint (if configured)
       or the local ModelDeploymentService.
    4. Builds and appends a prediction log CSV.
    5. Computes per-feature PSI drift and appends a drift CSV.
    6. Computes performance metrics if ground truth is available and appends a performance CSV.
    7. Writes a JSON summary of the run to `config.monitor_dir`.

    Args:
        config (MonitoringConfig): Monitoring configuration. Fields used include:
            - data_path (str): Path to the source data file.
            - target_col (str): Name of the target column.
            - monitor_dir (str): Directory where artifacts are written.
            - deploy_endpoint_url (Optional[str]): Endpoint URL to call for predictions.
            - baseline_sample_size (int): Number of baseline records to sample.
            - monitor_sample_size (int): Number of current records to sample.
            - period_seconds (int): Period used by the periodic runner (not used here).

    Returns:
        Dict[str, Any]: Summary dictionary with at least the following keys:
            - run_at_utc (str): ISO8601 UTC timestamp of the run.
            - n_records_scored (int): Number of records scored.
            - prediction_log (str): Path to prediction_log.csv.
            - drift_metrics (str): Path to drift_metrics.csv.
            - performance_metrics (str): Path to performance_metrics.csv.
            - scoring_seconds (float): Time spent scoring (seconds).
            - drift_alert_features (int): Number of features flagged with drift.

    Raises:
        FileNotFoundError: If `config.data_path` does not exist or cannot be read.
        IOError: If writing artifact files fails (disk/permissions).
        Exception: Propagates unexpected errors from scoring, endpoint calls, or utilities.

    Side effects:
        - Reads `config.data_path`.
        - Writes artifacts under `config.monitor_dir`:
            prediction_log.csv, drift_metrics.csv, performance_metrics.csv, latest_run_summary.json

    Example:
        >>> cfg = MonitoringConfig(data_path="Base_de_datos.xlsx", monitor_dir="artifacts/monitoring")
        >>> summary = run_monitoring_cycle(cfg)
        >>> isinstance(summary, dict)
        True
    """
    monitor_dir = Path(config.monitor_dir)
    monitor_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_excel(config.data_path)
    x_raw, y = split_features_target(df, target_col=config.target_col)
    y = y.astype(int)

    x_baseline, x_pool, y_baseline, y_pool = train_test_split(
        x_raw,
        y,
        test_size=0.4,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    baseline_df = x_baseline.copy().head(config.baseline_sample_size)

    monitor_df = x_pool.copy().head(config.monitor_sample_size)
    monitor_df[config.target_col] = y_pool.head(config.monitor_sample_size).values

    records = monitor_df.drop(columns=[config.target_col]).to_dict(orient="records")

    start = time.perf_counter()
    predictions, probabilities, source = _predict_with_deploy(
        records, config.deploy_endpoint_url
    )
    scoring_seconds = time.perf_counter() - start

    scored_inputs = monitor_df.drop(columns=[config.target_col]).copy()
    scored_inputs[config.target_col] = monitor_df[config.target_col].values
    prediction_log = _build_prediction_log(
        input_df=scored_inputs,
        predictions=predictions,
        probabilities=probabilities,
        source=source,
        target_col=config.target_col,
    )

    prediction_log_path = monitor_dir / "prediction_log.csv"
    _append_csv(prediction_log, prediction_log_path)

    drift_table = _compute_drift_table(
        reference_df=baseline_df,
        current_df=monitor_df.drop(columns=[config.target_col]),
    )
    drift_table["run_at_utc"] = _utc_now_iso()
    drift_table_path = monitor_dir / "drift_metrics.csv"
    _append_csv(drift_table, drift_table_path)

    performance = _compute_performance_if_available(prediction_log)
    performance_path = monitor_dir / "performance_metrics.csv"
    if performance is not None:
        perf_df = pd.DataFrame(
            [
                {
                    "run_at_utc": _utc_now_iso(),
                    "n_samples": len(prediction_log),
                    "scoring_seconds": scoring_seconds,
                    **performance,
                }
            ]
        )
        _append_csv(perf_df, performance_path)

    run_summary = {
        "run_at_utc": _utc_now_iso(),
        "n_records_scored": len(prediction_log),
        "prediction_log": str(prediction_log_path),
        "drift_metrics": str(drift_table_path),
        "performance_metrics": str(performance_path),
        "scoring_seconds": scoring_seconds,
        "drift_alert_features": int((drift_table["drift_flag"] == True).sum()),
    }

    summary_path = monitor_dir / "latest_run_summary.json"
    summary_path.write_text(
        json.dumps(run_summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    return run_summary


def run_periodic_monitoring(
    config: MonitoringConfig = MonitoringConfig(),
    n_cycles: int = 3,
) -> List[Dict[str, Any]]:
    """Run the monitoring cycle repeatedly with a delay between runs.

    This helper runs `run_monitoring_cycle` `n_cycles` times, waits
    `config.period_seconds` between runs (except after the last run), and
    collects the per-run summaries into a list. Each summary is augmented with
    a "cycle" key indicating the run index (1-based).

    Args:
        config (MonitoringConfig): Monitoring configuration to use for each cycle.
            See `MonitoringConfig` for available fields (data_path, monitor_dir,
            baseline_sample_size, monitor_sample_size, deploy_endpoint_url, etc.).
        n_cycles (int): Number of monitoring cycles to execute. Defaults to 3.

    Returns:
        List[Dict[str, Any]]: List of run summary dictionaries returned by
        `run_monitoring_cycle`. Each dict contains the original summary keys and
        an additional "cycle" key with the 1-based cycle number.

    Raises:
        Exception: Propagates exceptions raised by `run_monitoring_cycle` or by
        `time.sleep` (e.g., KeyboardInterrupt).

    Example:
        >>> cfg = MonitoringConfig(monitor_dir="artifacts/monitoring", baseline_sample_size=100, monitor_sample_size=10)
        >>> results = run_periodic_monitoring(cfg, n_cycles=2)
        >>> isinstance(results, list) and len(results) == 2
        True
    """
    results = []
    for cycle in range(n_cycles):
        result = run_monitoring_cycle(config=config)
        result["cycle"] = cycle + 1
        results.append(result)
        if cycle < n_cycles - 1:
            time.sleep(config.period_seconds)
    return results


if __name__ == "__main__":
    summary = run_monitoring_cycle()
    print("Monitoreo ejecutado")
    for key, value in summary.items():
        print(f"- {key}: {value}")
