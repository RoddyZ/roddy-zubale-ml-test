# TODO: Train, assert artifacts exist and ROC-AUC threshold
import os
import json
import pytest
import subprocess

ARTIFACTS_DIR = "artifacts"

def test_training_script_runs():
    """Ejecuta el script de entrenamiento y revisa que se generen artifacts."""
    # Borrar artifacts previos si existen
    if os.path.exists(ARTIFACTS_DIR):
        for f in os.listdir(ARTIFACTS_DIR):
            os.remove(os.path.join(ARTIFACTS_DIR, f))
    else:
        os.makedirs(ARTIFACTS_DIR)

    # Ejecutar entrenamiento con dataset oficial
    cmd = [
        "python",
        "-m",
        "src.train",
        "--data",
        "data/customer_churn_synth.csv",
        "--outdir",
        ARTIFACTS_DIR,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, f"Train script failed: {result.stderr}"

    # Checar archivos generados
    expected_files = [
        "model.pkl",
        "feature_pipeline.pkl",
        "metrics.json",
        "feature_importances.csv",
    ]
    for fname in expected_files:
        assert os.path.exists(os.path.join(ARTIFACTS_DIR, fname)), f"{fname} missing"

def test_metrics_quality():
    """Revisa que el ROC-AUC sea ≥ 0.83."""
    metrics_path = os.path.join(ARTIFACTS_DIR, "metrics.json")
    assert os.path.exists(metrics_path), "metrics.json not found"

    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    assert metrics["roc_auc"] >= 0.83, f"ROC-AUC too low: {metrics['roc_auc']}"
    assert 0.0 <= metrics["pr_auc"] <= 1.0
    assert 0.0 <= metrics["accuracy"] <= 1.0
