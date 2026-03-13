FROM python:3.11-slim
WORKDIR /app
COPY requirements.deploy.txt /app/requirements.deploy.txt
RUN pip install --no-cache-dir -r /app/requirements.deploy.txt
COPY mlops_pipeline/src /app/mlops_pipeline/src
COPY artifacts /app/artifacts
ENV PYTHONPATH=/app/mlops_pipeline/src
EXPOSE 8000
CMD ["uvicorn", "model_deploy:app", "--host", "0.0.0.0", "--port", "8000"]