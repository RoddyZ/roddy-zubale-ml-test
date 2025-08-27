# TODO: Implement Agentic Monitor (LLM-optional)
# CLI: python -m src.agent_monitor --metrics data/metrics_history.jsonl --drift data/drift_latest.json --out artifacts/agent_plan.yaml
import argparse
import json
import statistics
import yaml

def load_metrics(path):
    metrics = []
    with open(path, "r") as f:
        for line in f:
            metrics.append(json.loads(line))
    return metrics

def load_drift(path):
    with open(path, "r") as f:
        return json.load(f)

def analyze(metrics, drift):
    findings = {}
    status = "healthy"

    # --- ROC-AUC drop ---
    roc_values = [m["roc_auc"] for m in metrics[-7:]]  # últimos 7 días
    median_auc = statistics.median(roc_values)
    latest_auc = metrics[-1]["roc_auc"]
    drop_pct = 100 * (median_auc - latest_auc) / median_auc
    if drop_pct >= 3:
        findings["roc_auc_drop_pct"] = round(drop_pct, 2)
        status = "warn"
    if drop_pct >= 6:
        findings["roc_auc_drop_pct"] = round(drop_pct, 2)
        status = "critical"

    # --- Latency ---
    latencies = [m["latency_p95_ms"] for m in metrics[-2:]]  # últimos 2
    if all(l > 400 for l in latencies):
        findings["latency_p95_ms"] = latencies[-1]
        status = "warn"

    # --- Drift + PR-AUC ---
    latest_pr = metrics[-1]["pr_auc"]
    median_pr = statistics.median([m["pr_auc"] for m in metrics[-7:]])
    pr_drop = 100 * (median_pr - latest_pr) / median_pr
    if drift.get("overall_drift", False) and pr_drop >= 5:
        findings["drift_overall"] = True
        findings["pr_auc_drop_pct"] = round(pr_drop, 2)
        status = "critical"

    # --- Actions ---
    actions = []
    if status == "healthy":
        actions = ["do_nothing"]
    elif status == "warn":
        actions = ["trigger_retraining", "raise_thresholds"]
    elif status == "critical":
        actions = ["open_incident", "roll_back_model", "page_oncall=false"]

    rationale = []
    if "roc_auc_drop_pct" in findings:
        rationale.append(f"AUC fell {findings['roc_auc_drop_pct']}% vs 7-day median")
    if "latency_p95_ms" in findings:
        rationale.append(f"p95 latency {findings['latency_p95_ms']}ms > 400ms")
    if findings.get("drift_overall"):
        rationale.append("Overall drift detected with PR-AUC drop ≥5%")

    return {
        "status": status,
        "findings": findings,
        "actions": actions,
        "rationale": " ; ".join(rationale)
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--drift", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    metrics = load_metrics(args.metrics)
    drift = load_drift(args.drift)
    plan = analyze(metrics, drift)

    with open(args.out, "w") as f:
        yaml.dump(plan, f)

if __name__ == "__main__":
    main()
