# MLOPS Assignment

This repository contains the solutions for all three MLOps assignments:

1. **my-fraud-pipeline (Assignment 1 — DVC)**
   - Versioning datasets and building reproducible pipelines using DVC.
   - Contains a pipeline to generate and clean synthetic fraud data.

2. **my-mlflow-experiments (Assignment 2 — MLflow)**
   - Tracking model experiments with MLflow.
   - Automates the selection of the best fraud detection model (by AUC-PR) and transitions it to the Staging phase in the Model Registry.

3. **my-fraud-api (Assignment 3 — FastAPI Serving)**
   - Serving a Random Forest fraud detection model via FastAPI.
   - Contains predictive endpoints `GET /model-info`, `POST /predict`, `POST /predict/batch` and `GET /metrics`.
