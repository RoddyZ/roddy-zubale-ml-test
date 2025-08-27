# TODO: Boot API, call /predict using tests/sample.json
import json
import os
from fastapi.testclient import TestClient
from src.app import app

client = TestClient(app)

def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}

def test_predict_sample():
    # Cargar sample.json
    sample_path = os.path.join("tests", "sample.json")
    with open(sample_path, "r") as f:
        data = json.load(f)

    resp = client.post("/predict", json=data)
    assert resp.status_code == 200

    results = resp.json()
    # Debe devolver misma cantidad de filas que en el input
    assert isinstance(results, list)
    assert len(results) == len(data)

    for r in results:
        assert "probability" in r
        assert "class" in r
        # Probabilidad debe estar entre 0 y 1
        assert 0.0 <= r["probability"] <= 1.0
        # Clase debe ser 0 o 1
        assert r["class"] in (0, 1)
