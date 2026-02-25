from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
	accuracy_score,
	balanced_accuracy_score,
	f1_score,
	precision_score,
	recall_score,
	roc_auc_score,
)
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline

from ft_engineering import pipeline_ml, split_features_target


RANDOM_STATE = 42


class HeuristicModel(BaseEstimator, ClassifierMixin):
	def __init__(
		self,
		puntaje_threshold: float = -0.4,
		deuda_ingreso_threshold: float = 0.7,
		carga_pago_threshold: float = 0.8,
		mora_threshold: float = 0.5,
	):
		self.puntaje_threshold = puntaje_threshold
		self.deuda_ingreso_threshold = deuda_ingreso_threshold
		self.carga_pago_threshold = carga_pago_threshold
		self.mora_threshold = mora_threshold

	def _find_col(self, columns: List[str], suffix: str) -> Optional[str]:
		matches = [col for col in columns if col.endswith(suffix)]
		return matches[0] if matches else None

	def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
		if y is not None:
			self.classes_ = np.unique(y)

		columns = X.columns.tolist()
		self.col_puntaje_ = self._find_col(columns, "puntaje_datacredito")
		self.col_deuda_ingreso_ = self._find_col(columns, "relacion_deuda_ingreso")
		self.col_carga_pago_ = self._find_col(columns, "carga_pago_mensual")
		self.col_mora_ = self._find_col(columns, "saldo_mora")
		self.col_empleado_ = self._find_col(columns, "tipo_laboral_Empleado")
		return self

	def predict(self, X: pd.DataFrame) -> np.ndarray:
		predictions = []

		for _, row in X.iterrows():
			puntaje = row[self.col_puntaje_] if self.col_puntaje_ else 0.0
			deuda_ingreso = row[self.col_deuda_ingreso_] if self.col_deuda_ingreso_ else 0.0
			carga_pago = row[self.col_carga_pago_] if self.col_carga_pago_ else 0.0
			mora = row[self.col_mora_] if self.col_mora_ else 0.0
			is_empleado = row[self.col_empleado_] == 1 if self.col_empleado_ else False

			if (
				puntaje < self.puntaje_threshold
				or mora > self.mora_threshold
				or deuda_ingreso > self.deuda_ingreso_threshold
				or carga_pago > self.carga_pago_threshold
			):
				predictions.append(0)
			elif is_empleado and puntaje >= self.puntaje_threshold and carga_pago <= 0:
				predictions.append(1)
			else:
				predictions.append(1)

		return np.array(predictions)


def summarize_classification(
	y_true: pd.Series,
	y_pred: np.ndarray,
	y_proba: Optional[np.ndarray] = None,
) -> Dict[str, float]:
	summary = {
		"accuracy": accuracy_score(y_true, y_pred),
		"precision": precision_score(y_true, y_pred, zero_division=0),
		"recall": recall_score(y_true, y_pred, zero_division=0),
		"f1": f1_score(y_true, y_pred, zero_division=0),
		"balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
	}

	if y_proba is not None:
		summary["roc_auc"] = roc_auc_score(y_true, y_proba)
	else:
		summary["roc_auc"] = np.nan

	return summary


def build_model(estimator: BaseEstimator) -> Pipeline:
	return Pipeline(
		steps=[
			("features", clone(pipeline_ml)),
			("model", estimator),
		]
	)


@dataclass
class TrainingResult:
	best_model_name: str
	best_model_pipeline: Pipeline
	summary_table: pd.DataFrame


def _score_for_selection(row: pd.Series) -> float:
	performance = row["test_f1"]
	consistency = 1 / (1 + row["cv_f1_std"])
	scalability = 1 / (1 + row["cv_fit_time_mean"])
	return 0.60 * performance + 0.25 * consistency + 0.15 * scalability


def _get_model_candidates() -> Dict[str, BaseEstimator]:
	return {
		"heuristic": HeuristicModel(),
		"logistic_regression": LogisticRegression(
			max_iter=2000,
			class_weight="balanced",
			random_state=RANDOM_STATE,
		),
		"random_forest": RandomForestClassifier(
			n_estimators=300,
			min_samples_leaf=2,
			class_weight="balanced",
			random_state=RANDOM_STATE,
			n_jobs=-1,
		),
	}


def _evaluate_candidate(
	model_name: str,
	estimator: BaseEstimator,
	x_train_raw: pd.DataFrame,
	y_train: pd.Series,
	x_test_raw: pd.DataFrame,
	y_test: pd.Series,
	cv: KFold,
) -> Dict[str, float]:
	model_pipeline = build_model(estimator)
	scoring = ["accuracy", "precision", "recall", "f1"]

	cv_output = cross_validate(
		model_pipeline,
		x_train_raw,
		y_train,
		cv=cv,
		scoring=scoring,
		return_train_score=True,
		n_jobs=-1,
	)

	model_pipeline.fit(x_train_raw, y_train)
	y_pred = model_pipeline.predict(x_test_raw)

	y_proba = None
	if hasattr(model_pipeline, "predict_proba"):
		y_proba = model_pipeline.predict_proba(x_test_raw)[:, 1]

	test_summary = summarize_classification(y_test, y_pred, y_proba)

	result = {
		"model": model_name,
		"cv_accuracy_mean": float(np.mean(cv_output["test_accuracy"])),
		"cv_precision_mean": float(np.mean(cv_output["test_precision"])),
		"cv_recall_mean": float(np.mean(cv_output["test_recall"])),
		"cv_f1_mean": float(np.mean(cv_output["test_f1"])),
		"cv_f1_std": float(np.std(cv_output["test_f1"])),
		"cv_fit_time_mean": float(np.mean(cv_output["fit_time"])),
		"test_accuracy": test_summary["accuracy"],
		"test_precision": test_summary["precision"],
		"test_recall": test_summary["recall"],
		"test_f1": test_summary["f1"],
		"test_balanced_accuracy": test_summary["balanced_accuracy"],
		"test_roc_auc": test_summary["roc_auc"],
	}

	result["selection_score"] = _score_for_selection(pd.Series(result))
	return result


def plot_model_comparison(summary_table: pd.DataFrame, output_dir: Path) -> Path:
	output_dir.mkdir(parents=True, exist_ok=True)

	data = summary_table.sort_values("selection_score", ascending=False).copy()

	fig, axes = plt.subplots(1, 3, figsize=(18, 5))

	axes[0].bar(data["model"], data["test_f1"], color="#1f77b4")
	axes[0].set_title("Performance (Test F1)")
	axes[0].set_ylabel("F1")
	axes[0].tick_params(axis="x", rotation=25)

	axes[1].bar(data["model"], data["cv_f1_std"], color="#ff7f0e")
	axes[1].set_title("Consistency (CV F1 std)")
	axes[1].set_ylabel("Std")
	axes[1].tick_params(axis="x", rotation=25)

	axes[2].bar(data["model"], data["cv_fit_time_mean"], color="#2ca02c")
	axes[2].set_title("Scalability (Fit time)")
	axes[2].set_ylabel("Seconds")
	axes[2].tick_params(axis="x", rotation=25)

	plt.tight_layout()
	chart_path = output_dir / "model_comparison.png"
	plt.savefig(chart_path, dpi=140)
	plt.close(fig)
	return chart_path


def train_and_select_model(
	data_path: str = "Base_de_datos.xlsx",
	output_dir: str = "artifacts",
) -> TrainingResult:
	df = pd.read_excel(data_path)
	x_features_raw, y_target = split_features_target(df, target_col="Pago_atiempo")
	y_target = y_target.astype(int)

	x_train_raw, x_test_raw, y_train, y_test = train_test_split(
		x_features_raw,
		y_target,
		test_size=0.25,
		stratify=y_target,
		random_state=RANDOM_STATE,
	)

	cv = KFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
	candidates = _get_model_candidates()

	results = []
	for model_name, estimator in candidates.items():
		results.append(
			_evaluate_candidate(
				model_name=model_name,
				estimator=estimator,
				x_train_raw=x_train_raw,
				y_train=y_train,
				x_test_raw=x_test_raw,
				y_test=y_test,
				cv=cv,
			)
		)

	summary_table = pd.DataFrame(results).sort_values(
		"selection_score", ascending=False
	)
	best_model_name = summary_table.iloc[0]["model"]

	best_pipeline = build_model(candidates[best_model_name])
	best_pipeline.fit(x_train_raw, y_train)

	output_path = Path(output_dir)
	output_path.mkdir(parents=True, exist_ok=True)

	summary_table.to_csv(output_path / "model_summary.csv", index=False)
	plot_model_comparison(summary_table, output_path)
	joblib.dump(best_pipeline, output_path / "best_model.joblib")

	return TrainingResult(
		best_model_name=best_model_name,
		best_model_pipeline=best_pipeline,
		summary_table=summary_table,
	)


if __name__ == "__main__":
	result = train_and_select_model()
	print("Modelo seleccionado:", result.best_model_name)
	print("\nTabla resumen:")
	print(result.summary_table.round(4).to_string(index=False))
