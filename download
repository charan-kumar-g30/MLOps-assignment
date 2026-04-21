import time
from typing import List
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# --- Model & Data Setup (Bootstrapping internal RF) ---
model = RandomForestClassifier(n_estimators=50, random_state=42)

# Generate synthetic data similar to workshop schema
n = 2000
df = pd.DataFrame({
    "amount": np.random.uniform(1, 1000, n),
    "num_transactions_24h": np.random.randint(1, 20, n),
    "distance_from_home_km": np.random.uniform(1, 100, n),
    "is_weekend": np.random.choice([0, 1], n),
})
# Target logic (Fraud if amount > 500 OR distance > 50)
df["is_fraud"] = (
    (df["amount"] > 500).astype(int) |
    (df["distance_from_home_km"] > 50).astype(int)
)
X = df.drop("is_fraud", axis=1)
y = df["is_fraud"]

model.fit(X, y)

# --- Global Metrics State (for Bonus) ---
metrics_state = {
    "total_predictions": 0,
    "total_latency_ms": 0.0,
    "fraud_flags": 0
}

app = FastAPI(title="Fraud Detection API")

# --- Schemas ---
class Transaction(BaseModel):
    amount: float
    num_transactions_24h: int
    distance_from_home_km: float
    is_weekend: int

class PredictionResult(BaseModel):
    probability: float
    is_fraud: bool
    risk_level: str

# --- Endpoints ---
@app.get("/health")
def health():
    return {"status": "ok", "model": "RandomForest"}

@app.get("/model-info")
def model_info():
    return {
        "model_type": "RandomForest",
        "version": "1.0.0",
        "features": list(X.columns)
    }

def process_prediction(txn: Transaction) -> PredictionResult:
    start_time = time.time()
    
    # Extract features matching the trained model's order
    features = pd.DataFrame([[txn.amount, txn.num_transactions_24h, txn.distance_from_home_km, txn.is_weekend]], columns=X.columns)
    
    prob = model.predict_proba(features)[0][1]
    
    # Calculate Risk Level Based on Probability
    risk_level = "LOW"
    if prob > 0.8:
        risk_level = "HIGH"
    elif prob > 0.5:
        risk_level = "MEDIUM"
        
    is_fraud = bool(prob > 0.5)
    
    # Update metrics
    latency_ms = (time.time() - start_time) * 1000
    metrics_state["total_predictions"] += 1
    metrics_state["total_latency_ms"] += latency_ms
    if is_fraud:
        metrics_state["fraud_flags"] += 1
        
    return PredictionResult(
        probability=round(prob, 4),
        is_fraud=is_fraud,
        risk_level=risk_level
    )

@app.post("/predict", response_model=PredictionResult)
def predict(txn: Transaction):
    return process_prediction(txn)

@app.post("/predict/batch", response_model=List[PredictionResult])
def predict_batch(txns: List[Transaction]):
    return [process_prediction(txn) for txn in txns]

@app.get("/metrics")
def get_metrics():
    total = metrics_state["total_predictions"]
    avg_latency = metrics_state["total_latency_ms"] / total if total > 0 else 0
    percent_fraud = (metrics_state["fraud_flags"] / total * 100) if total > 0 else 0
    return {
        "total_predictions": total,
        "average_latency_ms": round(avg_latency, 2),
        "percentage_fraud": round(percent_fraud, 2)
    }
