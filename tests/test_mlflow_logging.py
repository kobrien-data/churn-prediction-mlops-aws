"""
Smoke test: log a dummy run to local MLflow and print the run URL.
Run with: python tests/test_mlflow_logging.py
Then open the printed URL in your browser to verify the run appears.
"""

import mlflow

EXPERIMENT_NAME = "churn-prediction-smoke-test"
TRACKING_URI = "http://16.170.249.60:5000"  # EC2 MLflow server


def log_dummy_run():
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="dummy-smoke-test") as run:
        # Dummy hyperparameters
        mlflow.log_params({
            "model_type": "logistic_regression",
            "max_iter": 100,
            "C": 1.0,
            "solver": "lbfgs",
        })

        # Dummy metrics
        mlflow.log_metrics({
            "accuracy": 0.87,
            "precision": 0.84,
            "recall": 0.81,
            "f1_score": 0.82,
            "roc_auc": 0.91,
        })

        # Dummy tag
        mlflow.set_tag("stage", "smoke-test")

        run_id = run.info.run_id

    print(f"\nRun logged successfully!")
    print(f"  Run ID  : {run_id}")
    print(f"  UI URL  : {TRACKING_URI}/#/experiments/{mlflow.get_experiment_by_name(EXPERIMENT_NAME).experiment_id}/runs/{run_id}")
    print(f"\nOpen the URL above (or navigate to {TRACKING_URI}) to verify the run in the MLflow UI.")
    return run_id


if __name__ == "__main__":
    log_dummy_run()
