# TODO: Train, assert artifacts exist and ROC-AUC threshold
import json
import os
import subprocess
import sys

def test_training_creates_artifacts(tmp_path):
    """Ejecuta el script de entrenamiento y valida que genere los artifacts."""
    outdir = tmp_path / "artifacts"
    outdir.mkdir()

    # Ejecutar train.py como script
    result = subprocess.run(
        [
            sys.executable, "-m", "src.train",
            "--data", "data/customer_churn_synth.csv",
            "--outdir", str(outdir)
        ],
        capture_output=True,
        text=True,
    )
    # Debe terminar correctamente
    assert result.returncode == 0, f"train.py failed: {result.stderr}"

    # Archivos esperados
    expected_files = [
        "model.pkl",
        "feature_pipeline.pkl",
        "metrics.json",
        "feature_importances.csv",
    ]
    for fname in expected_files:
        path = outdir / fname
        assert path.exists(), f"{fname} was not created"

    # Revisar métricas
    metrics_path = outdir / "metrics.json"
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    assert "roc_auc" in metrics
    assert "pr_auc" in metrics
    assert "accuracy" in metrics

    # Validación mínima: ROC-AUC ≥ 0.83
    assert metrics["roc_auc"] >= 0.83, f"ROC-AUC too low: {metrics['roc_auc']}"
