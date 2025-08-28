# Mini-Prod ML Challenge (Starter)


This project is a **mini production ML pipeline** for customer churn prediction.  
It shows how to train, serve, monitor and package a binary classifier with modern MLOps practices.

---

## 📂 Project Structure

.
├── data/ # Example datasets (train, drift samples)

├── src/ # Source code

│ ├── train.py # Training script (Part A)

│ ├── app.py # FastAPI service (Part B)

│ ├── drift.py # Drift check CLI (Part D)

│ └── agent_monitor.py # Agentic monitor CLI (Part E)

├── artifacts/ # Saved models, metrics and reports

├── tests/ # Pytest tests

├── docker/ # Dockerfile

├── requirements.txt # Python dependencies

└── README.md


---

### 1. Setup Environment
```bash
python3 -m venv venv
source venv/bin/activate   # On Linux/Mac
venv\\Scripts\\activate    # On Windows

pip install -r requirements.txt
```

### 2. Train the Model

This will train a churn classifier, save metrics and model artifacts in artifacts/.

make train


Or directly:
```bash
python -m src.train --data data/customer_churn_synth.csv --outdir artifacts/
```

Artifacts produced:

artifacts/model.pkl

artifacts/feature_pipeline.pkl

artifacts/feature_importances.csv

artifacts/metrics.json

### 3. Run the API

Start the FastAPI service:

make serve


Or directly:
```bash
uvicorn src.app:app --host 0.0.0.0 --port 8000
```

Endpoints:

GET /health → check service status

POST /predict → send JSON rows and get churn probability + class

Example request:

curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '[{"plan_type":"Basic","contract_type":"Monthly","autopay":"No","is_promo_user":"No",
        "add_on_count":1,"tenure_months":30,"monthly_usage_gb":127.7,"avg_latency_ms":159.4,
        "support_tickets_30d":1,"discount_pct":12.2,"payment_failures_90d":0,"downtime_hours_30d":2.2}]'

### 4. Run Tests

Unit tests validate training and inference.

make test


Or directly:
```bash
pytest -q
```

### 5. Drift Check

Compare reference vs new dataset and save a drift report.

```bash
python -m src.drift --ref data/churn_ref_sample.csv --new data/churn_shifted_sample.csv
```

Output: artifacts/drift_report.json

### 6. Agent Monitor

Simulate a monitoring agent that checks metrics + drift and writes a plan.
```bash
python -m src.agent_monitor \
  --metrics data/metrics_history.jsonl \
  --drift data/drift_latest.json \
  --out artifacts/agent_plan.yaml
```

### 7. Docker

Build and run the API inside a container.
```bash
docker build -t churn-api -f docker/Dockerfile .
docker run -p 8000:8000 churn-api
```