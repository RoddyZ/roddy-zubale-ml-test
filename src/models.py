# TODO: Train/save/load utilities
# src/models.py
import joblib
import warnings
from sklearn.linear_model import LogisticRegression

try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False
    warnings.warn("xgboost not available — falling back to LogisticRegression.")

RANDOM_SEED = 42

def build_model():
    """Return default classifier (XGB or LogisticRegression)."""
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
    return LogisticRegression(max_iter=2000, solver="lbfgs")

def save_model(model, path: str):
    joblib.dump(model, path)

def load_model(path: str):
    return joblib.load(path)
