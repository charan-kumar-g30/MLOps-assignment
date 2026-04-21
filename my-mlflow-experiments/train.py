import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

# ----------------------
# Generate synthetic data
# ----------------------
n = 2000

df = pd.DataFrame({
    "amount": np.random.uniform(1, 1000, n),
    "num_transactions_24h": np.random.randint(1, 20, n),
    "distance_from_home_km": np.random.uniform(1, 100, n),
    "is_weekend": np.random.choice([0, 1], n),
})

df["is_fraud"] = (
    (df["amount"] > 500).astype(int) |
    (df["distance_from_home_km"] > 50).astype(int)
)

X = df.drop("is_fraud", axis=1)
y = df["is_fraud"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# ----------------------
# MLflow setup
# ----------------------
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("my_fraud_models")

def run_model(model, params):
    with mlflow.start_run():
        model.fit(X_train, y_train)

        y_probs = model.predict_proba(X_test)[:, 1]

        auc_roc = roc_auc_score(y_test, y_probs)
        auc_pr = average_precision_score(y_test, y_probs)

        # log params + metrics
        mlflow.log_params(params)
        mlflow.log_metric("AUC_ROC", auc_roc)
        mlflow.log_metric("AUC_PR", auc_pr)

        # log model
        mlflow.sklearn.log_model(model, "model",
                                registered_model_name="my_fraud_detector")

# ----------------------
# Run experiments
# ----------------------

run_model(LogisticRegression(C=0.1, max_iter=1000), {"model": "LR", "C": 0.1})
run_model(LogisticRegression(C=10.0, max_iter=1000), {"model": "LR", "C": 10.0})
run_model(RandomForestClassifier(n_estimators=50), {"model": "RF", "n_estimators": 50})
# Bonus model
run_model(RandomForestClassifier(n_estimators=200), {"model": "RF", "n_estimators": 200})
