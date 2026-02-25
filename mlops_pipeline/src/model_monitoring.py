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
	return datetime.now(timezone.utc).isoformat()


def _psi_numeric(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
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

	return float(np.sum((actual_ratio - expected_ratio) * np.log(actual_ratio / expected_ratio)))


def _psi_categorical(expected: pd.Series, actual: pd.Series) -> float:
	expected = expected.astype("string").fillna("<NA>")
	actual = actual.astype("string").fillna("<NA>")

	categories = sorted(set(expected.unique()).union(set(actual.unique())))
	if not categories:
		return np.nan

	expected_freq = expected.value_counts(normalize=True).reindex(categories, fill_value=0.0)
	actual_freq = actual.value_counts(normalize=True).reindex(categories, fill_value=0.0)

	epsilon = 1e-6
	expected_freq = np.clip(expected_freq.values, epsilon, None)
	actual_freq = np.clip(actual_freq.values, epsilon, None)

	return float(np.sum((actual_freq - expected_freq) * np.log(actual_freq / expected_freq)))


def _compute_drift_table(reference_df: pd.DataFrame, current_df: pd.DataFrame) -> pd.DataFrame:
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


def run_monitoring_cycle(config: MonitoringConfig = MonitoringConfig()) -> Dict[str, Any]:
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
	predictions, probabilities, source = _predict_with_deploy(records, config.deploy_endpoint_url)
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
	summary_path.write_text(json.dumps(run_summary, indent=2, ensure_ascii=False), encoding="utf-8")

	return run_summary


def run_periodic_monitoring(
	config: MonitoringConfig = MonitoringConfig(),
	n_cycles: int = 3,
) -> List[Dict[str, Any]]:
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
