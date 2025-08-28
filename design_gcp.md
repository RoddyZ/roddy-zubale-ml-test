# GCP Design

This document explains a possible design to deploy the churn prediction system on Google Cloud Platform (GCP).  
The goal is to keep the architecture **simple, scalable, and low-cost**.

---

## 1. Data Storage (BigQuery)
- **Source of truth:** Customer churn and usage data is stored in **BigQuery**.  
- **ETL / Ingestion:** Data is ingested daily from operational systems or batch files.  
- **Access:** BigQuery is used for:
  - Training datasets (export to Vertex AI or Cloud Storage).
  - Analytics and reporting (SQL dashboards).

---

## 2. Model Training (Vertex AI)
- **Vertex AI Pipelines** can run the training script (`src/train.py`) with data from BigQuery.  
- Trained model and artifacts are stored in **Cloud Storage**.  
- Optionally, schedule training weekly or when drift is detected.

---

## 3. Model Serving (Cloud Run)
- Package the FastAPI service (`src/app.py`) in Docker.  
- Deploy it to **Cloud Run** (serverless container).  
- Benefits:
  - Auto-scale on demand (pay only when requests come).
  - HTTPS endpoint exposed securely.
- Clients (web, backend systems) call the `/predict` endpoint.

---

## 4. Monitoring
- **Prediction logging:** All requests and responses are logged in **Cloud Logging** and can be exported to BigQuery.  
- **Drift monitoring:** Run the `src/drift.py` job on a schedule with **Cloud Scheduler + Cloud Run Job**.  
- **Agent monitor:** Run `src/agent_monitor.py` daily, check metrics and drift, and write decisions to Cloud Storage.  
- **Alerting:** Use **Cloud Monitoring** dashboards and alerts:
  - If ROC-AUC drops below threshold.
  - If drift exceeds threshold.
  - If API errors increase.

---

## 5. High-Level Architecture

    +-----------------+
    |   BigQuery      |
    | (Customer Data) |
    +--------+--------+
             |
             v
    +-----------------+
    |  Vertex AI      |
    |  Training       |
    +--------+--------+
             |
      Model + Artifacts
             |
             v
    +-----------------+       +-----------------+
    |   Cloud Run     | <---- |  Client Apps    |
    | (FastAPI Model) |       |  (Web / Mobile) |
    +--------+--------+       +-----------------+
             |
      Logs / Metrics
             v
    +-----------------+
    | Cloud Logging & |
    | Cloud Monitoring|
    +-----------------+




---

## 6. Summary
- **BigQuery**: scalable storage and analytics for churn data.  
- **Vertex AI**: training and retraining pipeline.  
- **Cloud Run**: lightweight and scalable inference service.  
- **Monitoring**: drift + quality checks with Cloud Logging and Monitoring.  

This design is simple to start, and can evolve with more automation (CI/CD, Vertex Feature Store, Pub/Sub streaming) in the future.

