"""
register_model.py — Register a Colab-trained model in the local MLflow server.

Run from the backend/ directory after downloading the .pkl from Google Drive:

    uv run python notebooks/register_model.py \
        --model-path /path/to/SPY_regime_classifier_20250101_120000.pkl \
        --meta-path  /path/to/SPY_regime_classifier_20250101_120000.json

Prerequisites:
    - MLflow server running (make up or docker-compose ... up mlflow)
    - backend/.env has MLFLOW_TRACKING_URI=http://localhost:5000
"""

import argparse
import json
import pickle
import sys
from pathlib import Path


def register(model_path: Path, meta_path: Path) -> None:
    import mlflow
    import mlflow.sklearn
    from mlflow.tracking import MlflowClient

    # Load metadata
    with open(meta_path) as f:
        meta = json.load(f)

    symbol     = meta["symbol"]
    model_name = f"{symbol}-regime-classifier"

    print(f"Symbol:      {symbol}")
    print(f"Trained at:  {meta['trained_at']}")
    print(f"Samples:     {meta['n_samples']:,}")
    print(f"CV AUC:      {meta['cv_mean_roc_auc']}")
    print(f"CV F1:       {meta['cv_mean_f1']}")
    print()

    # Load model
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # Connect to MLflow
    from src.app.core.config import settings
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(f"regime-classifier-{symbol}")

    print(f"MLflow URI:  {settings.MLFLOW_TRACKING_URI}")
    print(f"Experiment:  regime-classifier-{symbol}")
    print()

    with mlflow.start_run(run_name=f"colab-{meta['trained_at']}") as run:
        # Log metadata as params
        mlflow.log_param("symbol",                 symbol)
        mlflow.log_param("source",                 "colab")
        mlflow.log_param("trained_at",             meta["trained_at"])
        mlflow.log_param("train_start",            meta["train_start"])
        mlflow.log_param("train_end",              meta["train_end"])
        mlflow.log_param("n_samples",              meta["n_samples"])
        mlflow.log_param("feature_schema_version", meta["feature_schema_version"])

        if meta.get("optuna_best_params"):
            mlflow.log_params({f"optuna_{k}": v for k, v in meta["optuna_best_params"].items()})

        # Log CV metrics
        mlflow.log_metrics({
            "cv_mean_roc_auc": meta["cv_mean_roc_auc"],
            "cv_mean_f1":      meta["cv_mean_f1"],
        })

        # Log the model and register it
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name=model_name,
        )

        run_id = run.info.run_id
        print(f"MLflow run:  {run_id}")

    # Promote newest version to @champion alias
    client   = MlflowClient()
    versions = client.get_latest_versions(model_name)
    if not versions:
        print("ERROR: No model versions found after registration")
        sys.exit(1)

    latest = max(versions, key=lambda v: int(v.version))
    client.set_registered_model_alias(model_name, "champion", latest.version)

    print()
    print(f"Registered:  {model_name} v{latest.version}")
    print(f"Alias set:   @champion → v{latest.version}")
    print()
    print("Restart ml-api to load the new model:")
    print("  docker-compose -f docker-compose.yaml -f docker-compose.app.yaml restart ml-api")


def main() -> None:
    parser = argparse.ArgumentParser(description="Register a Colab-trained model in MLflow")
    parser.add_argument("--model-path", required=True, help="Path to the .pkl model file")
    parser.add_argument("--meta-path",  required=True, help="Path to the .json metadata file")
    args = parser.parse_args()

    model_path = Path(args.model_path)
    meta_path  = Path(args.meta_path)

    if not model_path.exists():
        print(f"ERROR: model file not found: {model_path}")
        sys.exit(1)
    if not meta_path.exists():
        print(f"ERROR: meta file not found: {meta_path}")
        sys.exit(1)

    register(model_path, meta_path)


if __name__ == "__main__":
    main()
