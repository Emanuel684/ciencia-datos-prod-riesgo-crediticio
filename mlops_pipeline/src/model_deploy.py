from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ── Import the module that owns the custom classes BEFORE any joblib.load ──
# This guarantees pickle can resolve HeuristicModel, KNNColumnImputer, etc.
# regardless of which script is __main__.
import model_training  # noqa: E402  (must be importable from PYTHONPATH)
import ft_engineering  # noqa: E402

# Patch __main__ so pickle can find classes serialised when model_training
# was run as __main__ (the common case after `python model_training.py`).
for _attr in dir(model_training):
    if not _attr.startswith("__"):
        setattr(sys.modules["__main__"], _attr, getattr(model_training, _attr))
for _attr in dir(ft_engineering):
    if not _attr.startswith("__"):
        setattr(sys.modules["__main__"], _attr, getattr(ft_engineering, _attr))


DEFAULT_MODEL_PATH = Path("artifacts/best_model.joblib")


class BatchPredictRequest(BaseModel):
    """Pydantic model for a batch prediction request.

    Attributes:
        records (List[Dict[str, Any]]): List of input records to predict. Each
            record is a mapping feature_name -> value. At least one record is required.
    """

    records: List[Dict[str, Any]] = Field(..., min_length=1)


class BatchPredictResponse(BaseModel):
    """Pydantic model for a batch prediction response.

    Attributes:
        n_records (int): Number of records processed.
        predictions (List[int]): Predicted class labels for each input record.
        probabilities (Optional[List[float]]): Optional list of predicted
            probabilities for the positive class (if available).
    """

    n_records: int
    predictions: List[int]
    probabilities: Optional[List[float]] = None


class ModelDeploymentService:
    """Lightweight wrapper around a persisted model for batch prediction.

    This class loads a model from disk (joblib) and exposes a `predict_batch`
    method that accepts a list of input records and returns typed predictions.

    Args:
        model_path (Path): Path to the serialized model file. Defaults to
            DEFAULT_MODEL_PATH.

    Example:
        >>> svc = ModelDeploymentService(Path("artifacts/best_model.joblib"))
        >>> resp = svc.predict_batch([{"feature1": 1.0, "feature2": "A"}])
        >>> resp.n_records
        1
    """

    def __init__(self, model_path: Path = DEFAULT_MODEL_PATH):
        """Initialize the service and load the model from disk.

        The constructor attempts to load the model immediately. If the model file
        does not exist a FileNotFoundError is raised.

        Args:
            model_path (Path): Path to the serialized model file.

        Raises:
            FileNotFoundError: If the provided model_path does not exist.
        """
        self.model_path = Path(model_path)
        self.model = self._load_model()

    # def _load_model(self):
    #     """Load and return a joblib-serialized model from disk.

    #     Returns:
    #         Any: The deserialized model object.

    #     Raises:
    #         FileNotFoundError: If the model file cannot be found.
    #         Exception: Propagates exceptions from joblib.load for other I/O or
    #             deserialization errors.
    #     """
    #     if not self.model_path.exists():
    #         raise FileNotFoundError(
    #             f"No se encontró el modelo en {self.model_path}. Ejecuta model_training.py primero."
    #         )
    #     return joblib.load(self.model_path)
    def _load_model(self):
        """Load persisted pipeline.

        Custom classes (HeuristicModel, KNNColumnImputer, …) are already
        available because the module-level patch above registered them on
        sys.modules['__main__'] before this method is ever called.

        Returns:
            Any: Deserialized sklearn Pipeline.

        Raises:
            FileNotFoundError: If the model file cannot be found.
        """
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"No se encontró el modelo en {self.model_path}. "
                "Ejecuta model_training.py primero."
            )
        return joblib.load(self.model_path)

    def predict_batch(self, records: List[Dict[str, Any]]) -> BatchPredictResponse:
        """Predict a batch of records using the loaded model.

        The function converts the list of records to a pandas DataFrame and calls
        the model's `predict` method. If the model exposes `predict_proba`, the
        probability for the positive class (column index 1) is returned.

        Args:
            records (List[Dict[str, Any]]): List of input records to score.

        Returns:
            BatchPredictResponse: Structured response including predictions and
            optional probabilities.

        Raises:
            ValueError: If `records` is empty.
            Exception: Propagates errors raised by the model during prediction.
        """
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
    """Create and configure a FastAPI application that exposes batch prediction endpoints.

    The application exposes:
        - GET /health: basic health and model load status.
        - POST /predict/batch: batch prediction endpoint accepting JSON matching
          BatchPredictRequest and returning BatchPredictResponse.

    Args:
        model_path (Path): Path to the serialized model file used by the service.

    Returns:
        FastAPI: Configured FastAPI application instance.

    Example:
        >>> app = create_app(Path("artifacts/best_model.joblib"))
    """
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
        """Health check endpoint that reports model load status.

        Returns:
            Dict[str, Any]: Health payload with keys 'status', 'model_path',
            'model_loaded' and 'detail' (error message when model is not loaded).
        """
        return {
            "status": "ok" if service is not None else "error",
            "model_path": str(model_path),
            "model_loaded": service is not None,
            "detail": startup_error,
        }

    @app.post("/predict/batch", response_model=BatchPredictResponse)
    def predict_batch(payload: BatchPredictRequest) -> BatchPredictResponse:
        """Endpoint handler for batch predictions.

        Args:
            payload (BatchPredictRequest): Parsed request body with input records.

        Returns:
            BatchPredictResponse: Predictions and optional probabilities.

        Raises:
            HTTPException: Returns 503 if model not loaded; returns 400 on bad input
                or other errors raised during prediction.
        """
        if service is None:
            raise HTTPException(status_code=503, detail=startup_error)

        try:
            return service.predict_batch(payload.records)
        except Exception as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    return app


def write_image_artifacts(base_dir: Path = Path(".")) -> Dict[str, str]:
    """Generate Dockerfile and requirements file for deploying the model service.

    The function writes two files in `base_dir`:
      - requirements.deploy.txt: pinned runtime dependencies used by the image.
      - Dockerfile: simple Dockerfile that copies the code and artifacts into the image.

    Args:
        base_dir (Path): Directory where the files will be created. Defaults to current dir.

    Returns:
        Dict[str, str]: Mapping with keys 'requirements' and 'dockerfile' pointing to the
        created file paths as strings.

    Raises:
        OSError: If writing files fails due to filesystem permissions or disk issues.

    Example:
        >>> generated = write_image_artifacts(Path("docker"))
        >>> "requirements" in generated and "dockerfile" in generated
        True
    """
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
