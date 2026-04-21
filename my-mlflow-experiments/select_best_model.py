import mlflow
from mlflow.tracking import MlflowClient

# Connect directly to local sqlite backend
mlflow.set_tracking_uri("sqlite:///mlflow.db")
client = MlflowClient(tracking_uri="sqlite:///mlflow.db")

def select_best_model():
    # 1. Query all runs in the experiment
    experiment = client.get_experiment_by_name("my_fraud_models")
    if not experiment:
        print("Experiment 'my_fraud_models' not found.")
        return

    runs = client.search_runs(experiment.experiment_id)

    if not runs:
        print("No successful runs found to evaluate.")
        return

    # 2. Rank them by AUC-PR to find the winner
    # Sort runs in descending order based on AUC-PR
    ranked_runs = sorted(
        runs, 
        key=lambda r: r.data.metrics.get("AUC_PR", 0) if r.data.metrics.get("AUC_PR") is not None else 0, 
        reverse=True
    )
    
    print("--- Model Rankings (by AUC-PR) ---")
    for i, run in enumerate(ranked_runs):
        auc_pr = run.data.metrics.get("AUC_PR", 0)
        params = run.data.params
        print(f"Rank {i+1}: Run ID {run.info.run_id} | AUC-PR: {auc_pr:.4f} | Params: {params}")

    # The winner is the first one in the sorted list
    best_run = ranked_runs[0]
    best_score = best_run.data.metrics.get("AUC_PR", 0)

    # 3. Print the winner and its metrics
    print("\n--- Winner ---")
    print(f"Winner Run ID: {best_run.info.run_id}")
    print(f"Winner AUC-PR: {best_score}")
    print(f"Winner AUC-ROC: {best_run.data.metrics.get('AUC_ROC')}")
    print(f"Winner Params: {best_run.data.params}\n")

    # Find the specific model version registered for THIS winning run
    versions = client.search_model_versions(f"name='my_fraud_detector'")
    best_version = None
    for v in versions:
        if v.run_id == best_run.info.run_id:
            best_version = v.version
            break
            
    if best_version:
        # 4. Transition the winner to Staging stage in the registry
        client.transition_model_version_stage(
            name="my_fraud_detector",
            version=best_version,
            stage="Staging"
        )
        print(f"Model version {best_version} (Run {best_run.info.run_id}) successfully moved to Staging!")
    else:
        print("Warning: Could not find a registered model version for the winning run.")

if __name__ == "__main__":
    select_best_model()
