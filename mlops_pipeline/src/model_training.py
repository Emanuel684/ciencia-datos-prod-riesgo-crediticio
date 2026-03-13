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
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from ft_engineering import pipeline_ml, split_features_target


RANDOM_STATE = 42


class HeuristicModel(BaseEstimator, ClassifierMixin):
    """Simple rule-based classifier implemented as an sklearn estimator.

    The model applies a set of hand-crafted rules on derived numeric features
    (for example credit score, debt-to-income ratio, past-due balance) to
    decide a binary label. Designed to be used inside the same pipeline
    produced by build_model so that it receives preprocessed feature names.

    Args:
        puntaje_threshold (float): Threshold for the credit score-like feature.
        deuda_ingreso_threshold (float): Threshold for debt-to-income ratio.
        carga_pago_threshold (float): Threshold for monthly payment load.
        mora_threshold (float): Threshold for past-due balance.

    Attributes:
        classes_ (np.ndarray): Unique class labels seen during fit (if y provided).
        col_* (str): Resolved column names used by the heuristic after fit.

    Example:
        >>> from ft_engineering import pipeline_ml
        >>> svc = HeuristicModel()
        >>> svc.fit(pd.DataFrame({"puntaje_datacredito":[-1], "saldo_mora":[0]}))
        >>> svc.predict(pd.DataFrame({"puntaje_datacredito":[-1], "saldo_mora":[0]}))
        array([0])
    """

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
            deuda_ingreso = (
                row[self.col_deuda_ingreso_] if self.col_deuda_ingreso_ else 0.0
            )
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
    """Compute common classification metrics from predictions and (optional) scores.

    The function returns a dictionary with accuracy, precision, recall, f1-score,
    balanced accuracy and (when probabilities are provided) ROC AUC.

    Args:
        y_true (pd.Series): True binary labels.
        y_pred (np.ndarray): Predicted binary labels.
        y_proba (Optional[np.ndarray]): Predicted probability / score for the positive class.

    Returns:
        Dict[str, float]: Mapping with keys:
            - "accuracy", "precision", "recall", "f1", "balanced_accuracy", "roc_auc".

            If y_proba is None the "roc_auc" value will be np.nan.

    Raises:
        None

    Example:
        >>> summarize_classification(pd.Series([1,0]), np.array([1,0]), np.array([0.9,0.1]))
        {'accuracy': 1.0, 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'balanced_accuracy': 1.0, 'roc_auc': 1.0}
    """
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
    """Compose the feature pipeline with a final estimator into a sklearn Pipeline.

    The returned pipeline clones the global pipeline_ml instance and attaches the
    provided estimator as the final step. This ensures each candidate receives a
    fresh, independent feature-preprocessing pipeline.

    Args:
        estimator (BaseEstimator): Final estimator implementing fit/predict.

    Returns:
        sklearn.pipeline.Pipeline: New pipeline instance combining preprocessing and estimator.

    Example:
        >>> build_model(RandomForestClassifier())
        Pipeline(...)
    """
    return Pipeline(
        steps=[
            ("features", clone(pipeline_ml)),
            ("model", estimator),
        ]
    )


@dataclass
class TrainingResult:
    """Container for the result of the training and model selection run.

    Attributes:
        best_model_name (str): Name identifier of the selected best model.
        best_model_pipeline (Pipeline): Fitted sklearn Pipeline for the best model.
        summary_table (pd.DataFrame): DataFrame summarizing evaluation results for candidates.
    """

    best_model_name: str
    best_model_pipeline: Pipeline
    summary_table: pd.DataFrame


def _score_for_selection(row: pd.Series) -> float:
    """Compute a composite selection score used to rank model candidates.

    The function combines test F1 (performance), CV F1 standard deviation
    (stability/consistency) and mean CV fit time (scalability) into a single
    scalar used for model selection.

    Args:
        row (pd.Series): Row from the summary table containing keys:
            - "test_f1", "cv_f1_std", "cv_fit_time_mean"

    Returns:
        float: Composite score (higher is better).

    Example:
        >>> _score_for_selection(pd.Series({"test_f1":0.8,"cv_f1_std":0.1,"cv_fit_time_mean":0.2}))
        0.60*0.8 + 0.25*(1/(1+0.1)) + 0.15*(1/(1+0.2))
    """
    # performance = row["test_f1"]
    performance = row["cv_f1_mean"]
    consistency = 1 / (1 + row["cv_f1_std"])
    scalability = 1 / (1 + row["cv_fit_time_mean"])
    return 0.60 * performance + 0.25 * consistency + 0.15 * scalability
    # return  0.25 * consistency + 0.15 * scalability


def _get_model_candidates() -> Dict[str, BaseEstimator]:
    """Return a mapping of candidate model names to estimator instances.

    The candidates dictionary contains short string keys identifying each model
    and the corresponding sklearn-compatible estimator (or custom estimator).

    Returns:
        Dict[str, BaseEstimator]: Candidate estimators keyed by name.

    Example:
        >>> list(_get_model_candidates().keys())
        ['heuristic', 'logistic_regression', 'random_forest']
    """
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
    # cv: KFold,
    cv: StratifiedKFold,
) -> Dict[str, float]:
    """Evaluate a single candidate model with cross-validation and test-set metrics.

    The function trains a pipeline built with the provided estimator, runs
    cross-validation on the training set to collect CV metrics and fit times,
    fits on the full training set, and computes test-set performance.

    Args:
        model_name (str): Identifier for the candidate model.
        estimator (BaseEstimator): Estimator instance to evaluate.
        x_train_raw (pd.DataFrame): Training features (raw).
        y_train (pd.Series): Training labels.
        x_test_raw (pd.DataFrame): Test features (raw).
        y_test (pd.Series): Test labels.
        cv (KFold): Cross-validation splitter.

    Returns:
        Dict[str, float]: Dictionary containing aggregated CV statistics and test metrics.
            Keys include:
            - model, cv_accuracy_mean, cv_precision_mean, cv_recall_mean, cv_f1_mean,
              cv_f1_std, cv_fit_time_mean, test_accuracy, test_precision, test_recall,
              test_f1, test_balanced_accuracy, test_roc_auc, selection_score.

    Raises:
        Exception: Propagates exceptions from fitting or scoring.

    Example:
        >>> _evaluate_candidate("heuristic", HeuristicModel(), X_train, y_train, X_test, y_test, KFold(3))
        {'model': 'heuristic', 'cv_accuracy_mean': ..., ... }
    """
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
    """Train candidate models, compare them and persist the selected best pipeline.

    Steps performed:
      1. Load data from `data_path`.
      2. Split into train/test.
      3. For each candidate: run cross-validation on train, fit and evaluate on test.
      4. Rank candidates by a composite selection score and persist artifacts:
         model_summary.csv, model_comparison.png and best_model.joblib.

    Args:
        data_path (str): Path to the input Excel dataset.
        output_dir (str): Directory where artifacts will be written.

    Returns:
        TrainingResult: Dataclass with best_model_name, fitted best_model_pipeline,
            and summary_table DataFrame sorted by selection score.

    Raises:
        FileNotFoundError: If `data_path` cannot be read by pandas.
        Exception: Propagates unexpected errors from training, plotting, or joblib.

    Example:
        >>> res = train_and_select_model("Base_de_datos.xlsx", "artifacts")
        >>> isinstance(res.summary_table, pd.DataFrame)
        True
    """
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

    # cv = KFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
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
