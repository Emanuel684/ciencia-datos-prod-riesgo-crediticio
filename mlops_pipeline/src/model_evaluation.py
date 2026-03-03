from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

from ft_engineering import split_features_target
from model_deploy import ModelDeploymentService
from model_training import RANDOM_STATE, summarize_classification


@dataclass
class EvaluationConfig:
    data_path: str = "Base_de_datos.xlsx"
    output_dir: str = "artifacts/evaluation"
    target_col: str = "Pago_atiempo"
    test_size: float = 0.25
    deploy_endpoint_url: Optional[str] = None


def _post_batch_prediction(
    endpoint_url: str, records: List[Dict[str, Any]]
) -> Dict[str, Any]:
    payload = json.dumps({"records": records}).encode("utf-8")
    request = urllib.request.Request(
        endpoint_url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(request, timeout=60) as response:
        body = response.read().decode("utf-8")
    return json.loads(body)


def _predict_deployed(
    x_test_raw: pd.DataFrame,
    endpoint_url: Optional[str] = None,
) -> Dict[str, Any]:
    records = x_test_raw.to_dict(orient="records")

    if endpoint_url:
        try:
            return _post_batch_prediction(endpoint_url, records)
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError):
            pass

    local_service = ModelDeploymentService()
    response = local_service.predict_batch(records)
    return response.model_dump()


def _build_classification_report_table(metrics: Dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "metric": [
                "accuracy",
                "precision",
                "recall",
                "f1",
                "balanced_accuracy",
                "roc_auc",
            ],
            "value": [
                metrics.get("accuracy"),
                metrics.get("precision"),
                metrics.get("recall"),
                metrics.get("f1"),
                metrics.get("balanced_accuracy"),
                metrics.get("roc_auc"),
            ],
        }
    )


def _render_metrics_tab(
    output_html: Path,
    metrics_table: pd.DataFrame,
    metadata: Dict[str, Any],
    confusion_matrix_path: Path,
) -> None:
    metrics_rows = "\n".join(
        [
            f"<tr><td>{row.metric}</td><td>{row.value:.6f}</td></tr>"
            for row in metrics_table.itertuples(index=False)
        ]
    )

    html = f"""
<!DOCTYPE html>
<html lang=\"es\">
<head>
	<meta charset=\"UTF-8\" />
	<meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\" />
	<title>Model Evaluation Dashboard</title>
	<style>
		body {{ font-family: Arial, sans-serif; margin: 24px; }}
		.tabs {{ display: flex; border-bottom: 1px solid #ddd; gap: 8px; }}
		.tab-button {{ padding: 10px 14px; border: 1px solid #ddd; border-bottom: none; background: #f7f7f7; cursor: pointer; }}
		.tab-button.active {{ background: #ffffff; font-weight: bold; }}
		.tab-content {{ display: none; padding: 16px; border: 1px solid #ddd; }}
		.tab-content.active {{ display: block; }}
		table {{ border-collapse: collapse; min-width: 420px; }}
		th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
		th {{ background: #fafafa; }}
		code {{ background: #f2f2f2; padding: 2px 5px; }}
	</style>
</head>
<body>
	<h1>Evaluación del modelo desplegado</h1>
	<div class=\"tabs\">
		<button class=\"tab-button active\" onclick=\"openTab(event, 'tab-metrics')\">Pestaña métricas</button>
		<button class=\"tab-button\" onclick=\"openTab(event, 'tab-meta')\">Metadata</button>
	</div>

	<div id=\"tab-metrics\" class=\"tab-content active\">
		<h2>Métricas de desempeño</h2>
		<table>
			<thead><tr><th>Métrica</th><th>Valor</th></tr></thead>
			<tbody>
				{metrics_rows}
			</tbody>
		</table>
		<h3>Matriz de confusión</h3>
		<img src=\"{confusion_matrix_path.name}\" alt=\"Matriz de confusión\" width=\"640\" />
	</div>

	<div id=\"tab-meta\" class=\"tab-content\">
		<h2>Metadata de evaluación</h2>
		<ul>
			<li>Registros evaluados: <code>{metadata["n_records"]}</code></li>
			<li>Tiempo inferencia batch (segundos): <code>{metadata["inference_seconds"]:.6f}</code></li>
			<li>Endpoint usado: <code>{metadata["endpoint_used"]}</code></li>
			<li>Dataset: <code>{metadata["data_path"]}</code></li>
		</ul>
	</div>

	<script>
		function openTab(evt, tabId) {{
			const tabContents = document.getElementsByClassName('tab-content');
			for (let i = 0; i < tabContents.length; i++) {{
				tabContents[i].classList.remove('active');
			}}
			const tabButtons = document.getElementsByClassName('tab-button');
			for (let i = 0; i < tabButtons.length; i++) {{
				tabButtons[i].classList.remove('active');
			}}
			document.getElementById(tabId).classList.add('active');
			evt.currentTarget.classList.add('active');
		}}
	</script>
</body>
</html>
"""

    output_html.write_text(html, encoding="utf-8")


def evaluate_deployed_model(
    config: EvaluationConfig = EvaluationConfig(),
) -> Dict[str, Any]:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_excel(config.data_path)
    x_raw, y = split_features_target(df, target_col=config.target_col)
    y = y.astype(int)

    _, x_test_raw, _, y_test = train_test_split(
        x_raw,
        y,
        test_size=config.test_size,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    start = time.perf_counter()
    deploy_output = _predict_deployed(
        x_test_raw, endpoint_url=config.deploy_endpoint_url
    )
    inference_seconds = time.perf_counter() - start

    y_pred = deploy_output["predictions"]
    y_proba = deploy_output.get("probabilities")

    metrics = summarize_classification(y_test, y_pred, y_proba)
    metrics_table = _build_classification_report_table(metrics)

    summary_csv_path = output_dir / "deployed_model_metrics.csv"
    metrics_table.to_csv(summary_csv_path, index=False)

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax)
    ax.set_title("Matriz de confusión - Modelo desplegado")
    confusion_matrix_path = output_dir / "deployed_confusion_matrix.png"
    plt.tight_layout()
    plt.savefig(confusion_matrix_path, dpi=140)
    plt.close(fig)

    metadata = {
        "n_records": len(y_test),
        "inference_seconds": inference_seconds,
        "endpoint_used": config.deploy_endpoint_url or "local ModelDeploymentService",
        "data_path": config.data_path,
    }

    metadata_path = output_dir / "evaluation_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    dashboard_path = output_dir / "metrics_dashboard.html"
    _render_metrics_tab(
        output_html=dashboard_path,
        metrics_table=metrics_table,
        metadata=metadata,
        confusion_matrix_path=confusion_matrix_path,
    )

    return {
        "metrics_csv": str(summary_csv_path),
        "confusion_matrix": str(confusion_matrix_path),
        "metadata_json": str(metadata_path),
        "dashboard_html": str(dashboard_path),
        "metrics": metrics,
    }


if __name__ == "__main__":
    results = evaluate_deployed_model()
    print("Evaluación completada")
    for key, value in results.items():
        if key == "metrics":
            continue
        print(f"- {key}: {value}")
    print("- metrics:")
    for metric_name, metric_value in results["metrics"].items():
        print(f"    {metric_name}: {metric_value:.6f}")
