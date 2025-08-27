# TODO: Implement sklearn ColumnTransformer
# src/feature.py
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import sklearn

CAT_COLS = [
    "plan_type",
    "contract_type",
    "autopay",
    "is_promo_user",
]
NUM_COLS = [
    "add_on_count",
    "tenure_months",
    "monthly_usage_gb",
    "avg_latency_ms",
    "support_tickets_30d",
    "discount_pct",
    "payment_failures_90d",
    "downtime_hours_30d",
]

def build_preprocessor() -> ColumnTransformer:
    """Return sklearn ColumnTransformer with cat + num pipelines."""
    skl_version = tuple(map(int, sklearn.__version__.split(".")[:2]))
    if skl_version >= (1, 2):
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=float)
    else:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False, dtype=float)

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", ohe),
    ])
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    return ColumnTransformer(
        transformers=[
            ("cat", cat_pipe, CAT_COLS),
            ("num", num_pipe, NUM_COLS),
        ]
    )
