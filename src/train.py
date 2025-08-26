#!/usr/bin/env python3
"""
Mini-Prod ML Challenge — Part A (Train)

Usage:
  python -m src.train --data data/customer_churn_synth.csv --outdir artifacts/

Outputs in --outdir:
  - model.pkl                  (trained classifier)
  - feature_pipeline.pkl       (sklearn preprocessing pipeline)
  - feature_importances.csv    (importance per engineered feature)
  - metrics.json               (ROC-AUC, PR-AUC, Accuracy, timestamp, git_sha)

Notes:
  * Deterministic split via fixed seed
  * Offline friendly (no external calls)
  * Optional randomized HPO via --hpo-trials N (10–20 recommended)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from dataclasses import dataclass
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Prefer xgboost; fall back to logistic regression if not available
try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except Exception:  # pragma: no cover
    from sklearn.linear_model import LogisticRegression
    _HAS_XGB = False
    warnings.warn("xgboost not available — falling back to LogisticRegression.")

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

CAT_COLS: List[str] = [
    "plan_type",           # {Basic, Standard, Pro}
    "contract_type",       # {Monthly, Annual}
    "autopay",             # {Yes, No}
    "is_promo_user",       # {Yes, No}
]
NUM_COLS: List[str] = [
    "add_on_count",
    "tenure_months",
    "monthly_usage_gb",
    "avg_latency_ms",
    "support_tickets_30d",
    "discount_pct",
    "payment_failures_90d",
    "downtime_hours_30d",
]
TARGET_COL = "churned"

@dataclass
class TrainConfig:
    data_path: str
    outdir: str
    test_size: float = 0.2
    random_state: int = RANDOM_SEED
    hpo_trials: int = 0


def _git_sha() -> str | None:
    """Return short git SHA if available, else None."""
    import subprocess

    try:
        sha = (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
        return sha
    except Exception:
        return None


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    expected = set(CAT_COLS + NUM_COLS + [TARGET_COL])
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns in dataset: {sorted(missing)}")
    return df


def build_preprocessor() -> ColumnTransformer:
    # Handle sklearn version differences
    skl_version = tuple(map(int, sklearn.__version__.split(".")[:2]))
    if skl_version >= (1, 2):
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
    else:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False, dtype=np.float32)

    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", ohe),
        ]
    )
    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ]
    )
    pre = ColumnTransformer(
        transformers=[
            ("cat", cat_pipe, CAT_COLS),
            ("num", num_pipe, NUM_COLS),
        ],
        remainder="drop",
    )
    return pre


def build_model() -> object:
    if _HAS_XGB:
        return XGBClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=RANDOM_SEED,
            objective="binary:logistic",
            n_jobs=0,
            tree_method="hist",
            eval_metric="logloss",
        )
    else:
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(
            max_iter=2000,
            solver="lbfgs",
        )


def train_eval(
    df: pd.DataFrame,
    cfg: TrainConfig,
) -> Tuple[Pipeline, dict, np.ndarray, List[str]]:
    X = df[CAT_COLS + NUM_COLS].copy()
    y = df[TARGET_COL].astype(int).values

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=cfg.test_size, stratify=y, random_state=cfg.random_state
    )

    pre = build_preprocessor()
    model = build_model()

    pipe = Pipeline(steps=[("pre", pre), ("model", model)])

    if _HAS_XGB and cfg.hpo_trials and cfg.hpo_trials > 0:
        param_dist = {
            "model__n_estimators": [200, 300, 400, 600],
            "model__max_depth": [4, 6, 8],
            "model__learning_rate": np.linspace(0.03, 0.15, 5),
            "model__subsample": [0.7, 0.8, 0.9, 1.0],
            "model__colsample_bytree": [0.7, 0.8, 0.9, 1.0],
            "model__reg_lambda": [0.5, 1.0, 2.0],
        }
        search = RandomizedSearchCV(
            pipe,
            param_distributions=param_dist,
            n_iter=int(cfg.hpo_trials),
            scoring="roc_auc",
            n_jobs=-1,
            cv=3,
            random_state=cfg.random_state,
            verbose=1,
        )
        search.fit(X_train, y_train)
        pipe = search.best_estimator_
    else:
        pipe.fit(X_train, y_train)

    # Evaluate
    y_proba = pipe.predict_proba(X_val)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    roc_auc = float(roc_auc_score(y_val, y_proba))
    pr_auc = float(average_precision_score(y_val, y_proba))
    acc = float(accuracy_score(y_val, y_pred))

    metrics = {
        "roc_auc": round(roc_auc, 6),
        "pr_auc": round(pr_auc, 6),
        "accuracy": round(acc, 6),
        "timestamp": int(time.time()),
        "git_sha": _git_sha(),
        "test_size": cfg.test_size,
        "random_state": cfg.random_state,
        "model": "xgboost" if _HAS_XGB else "logistic_regression",
    }

    pre_fitted = pipe.named_steps["pre"]
    feature_names = list(pre_fitted.get_feature_names_out(CAT_COLS + NUM_COLS))

    try:
        importances = pipe.named_steps["model"].feature_importances_
        feat_importances = np.vstack([feature_names, importances]).T
    except Exception:
        try:
            coefs = pipe.named_steps["model"].coef_.ravel()
            feat_importances = np.vstack([feature_names, np.abs(coefs)]).T
        except Exception:
            feat_importances = np.vstack([feature_names, np.zeros(len(feature_names))]).T

    return pipe, metrics, feat_importances, feature_names


def save_artifacts(
    pipe: Pipeline,
    metrics: dict,
    feat_importances: np.ndarray,
    outdir: str,
) -> None:
    os.makedirs(outdir, exist_ok=True)

    model_path = os.path.join(outdir, "model.pkl")
    joblib.dump(pipe, model_path)

    feat_pipe_path = os.path.join(outdir, "feature_pipeline.pkl")
    joblib.dump(pipe.named_steps["pre"], feat_pipe_path)

    metrics_path = os.path.join(outdir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    fi_df = pd.DataFrame(feat_importances, columns=["feature", "importance"]) \
        .assign(importance=lambda d: pd.to_numeric(d["importance"], errors="coerce")) \
        .sort_values("importance", ascending=False)
    fi_path = os.path.join(outdir, "feature_importances.csv")
    fi_df.to_csv(fi_path, index=False)

    print(f"Saved: {model_path}\nSaved: {feat_pipe_path}\nSaved: {metrics_path}\nSaved: {fi_path}")


def parse_args(argv: List[str]) -> TrainConfig:
    p = argparse.ArgumentParser(description="Train churn classifier (Part A)")
    p.add_argument("--data", required=True, help="Path to CSV dataset (with 'churned' target)")
    p.add_argument("--outdir", required=True, help="Directory to write artifacts")
    p.add_argument("--test-size", type=float, default=0.2, help="Validation fraction (default 0.2)")
    p.add_argument(
        "--hpo-trials",
        type=int,
        default=0,
        help="Optional randomized HPO trials (e.g., 10–20). 0 disables.",
    )
    args = p.parse_args(argv)
    return TrainConfig(
        data_path=args.data,
        outdir=args.outdir,
        test_size=args.test_size,
        hpo_trials=args.hpo_trials,
    )


def main(argv: List[str] | None = None) -> int:
    cfg = parse_args(sys.argv[1:] if argv is None else argv)
    df = load_data(cfg.data_path)
    pipe, metrics, feat_importances, _ = train_eval(df, cfg)

    if metrics["roc_auc"] < 0.83:
        warnings.warn(
            f"ROC-AUC {metrics['roc_auc']:.4f} < 0.83 — consider enabling --hpo-trials or revisiting features."
        )

    save_artifacts(pipe, metrics, feat_importances, cfg.outdir)

    print("\n=== Validation Metrics ===")
    for k in ("roc_auc", "pr_auc", "accuracy"):
        print(f"{k}: {metrics[k]:.6f}")
    print("========================\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
