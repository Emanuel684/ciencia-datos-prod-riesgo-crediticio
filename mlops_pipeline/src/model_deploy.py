from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


DEFAULT_MODEL_PATH = Path("artifacts/best_model.joblib")


class BatchPredictRequest(BaseModel):
    records: List[Dict[str, Any]] = Field(..., min_length=1)


class BatchPredictResponse(BaseModel):
    n_records: int
    predictions: List[int]
    probabilities: Optional[List[float]] = None


class ModelDeploymentService:
    def __init__(self, model_path: Path = DEFAULT_MODEL_PATH):
        self.model_path = Path(model_path)
        self.model = self._load_model()

    def _load_model(self):
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"No se encontró el modelo en {self.model_path}. Ejecuta model_training.py primero."
            )
        return joblib.load(self.model_path)

    def predict_batch(self, records: List[Dict[str, Any]]) -> BatchPredictResponse:
        if not records:
            raise ValueError("La lista de registros está vacía.")

        input_df = pd.DataFrame(records)
        predictions = self.model.predict(input_df).tolist()

        probabilities = None
        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(input_df)[:, 1].tolist()

        return BatchPredictResponse(
            n_records=len(records),
            predictions=[int(pred) for pred in predictions],
            probabilities=probabilities,
        )


def create_app(model_path: Path = DEFAULT_MODEL_PATH) -> FastAPI:
    app = FastAPI(
        title="Credit Risk Batch Inference",
        version="1.0.0",
        description="Endpoint batch para predicción de Pago_atiempo usando el mejor modelo entrenado.",
    )

    try:
        service = ModelDeploymentService(model_path=model_path)
    except FileNotFoundError as error:
        service = None
        startup_error = str(error)
    else:
        startup_error = None

    @app.get("/health")
    def health() -> Dict[str, Any]:
        return {
            "status": "ok" if service is not None else "error",
            "model_path": str(model_path),
            "model_loaded": service is not None,
            "detail": startup_error,
        }

    @app.post("/predict/batch", response_model=BatchPredictResponse)
    def predict_batch(payload: BatchPredictRequest) -> BatchPredictResponse:
        if service is None:
            raise HTTPException(status_code=503, detail=startup_error)

        try:
            return service.predict_batch(payload.records)
        except Exception as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    return app


def write_image_artifacts(base_dir: Path = Path(".")) -> Dict[str, str]:
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    requirements_path = base_dir / "requirements.deploy.txt"
    dockerfile_path = base_dir / "Dockerfile"

    requirements_content = "\n".join(
        [
            "fastapi==0.116.1",
            "uvicorn[standard]==0.35.0",
            "pandas==2.3.2",
            "numpy==2.3.2",
            "scikit-learn==1.7.1",
            "joblib==1.5.2",
            "openpyxl==3.1.5",
        ]
    )

    dockerfile_content = "\n".join(
        [
            "FROM python:3.11-slim",
            "WORKDIR /app",
            "COPY requirements.deploy.txt /app/requirements.deploy.txt",
            "RUN pip install --no-cache-dir -r /app/requirements.deploy.txt",
            "COPY mlops_pipeline/src /app/mlops_pipeline/src",
            "COPY artifacts /app/artifacts",
            "ENV PYTHONPATH=/app/mlops_pipeline/src",
            "EXPOSE 8000",
            'CMD ["uvicorn", "model_deploy:app", "--host", "0.0.0.0", "--port", "8000"]',
        ]
    )

    requirements_path.write_text(requirements_content, encoding="utf-8")
    dockerfile_path.write_text(dockerfile_content, encoding="utf-8")

    return {
        "requirements": str(requirements_path),
        "dockerfile": str(dockerfile_path),
    }


app = create_app()


if __name__ == "__main__":
    generated = write_image_artifacts(Path("."))
    print("Archivos de imagen generados:")
    for key, value in generated.items():
        print(f"- {key}: {value}")
