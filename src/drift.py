# TODO: Implement PSI/KS drift calc.
# CLI: python -m src.drift --ref data/churn_ref_sample.csv --new data/churn_shifted_sample.csv
import argparse
import json
import os
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

THRESHOLD = 0.2

def psi(expected, actual, buckets=10):
    """Population Stability Index (PSI) para dos distribuciones."""
    expected = np.array(expected)
    actual = np.array(actual)

    # Si todos los valores son iguales, PSI = 0
    if np.all(expected == expected[0]) and np.all(actual == actual[0]):
        return 0.0

    # Dividir en quantiles
    breakpoints = np.percentile(expected, np.arange(0, 100, 100 / buckets))
    breakpoints = np.unique(breakpoints)

    expected_counts, _ = np.histogram(expected, bins=np.append(breakpoints, [expected.max() + 1]))
    actual_counts, _ = np.histogram(actual, bins=np.append(breakpoints, [expected.max() + 1]))

    expected_perc = expected_counts / len(expected)
    actual_perc = actual_counts / len(actual)

    # Evitar log(0)
    expected_perc = np.where(expected_perc == 0, 1e-6, expected_perc)
    actual_perc = np.where(actual_perc == 0, 1e-6, actual_perc)

    psi_value = np.sum((expected_perc - actual_perc) * np.log(expected_perc / actual_perc))
    return float(psi_value)


def drift_check(ref_path, new_path, out_path):
    df_ref = pd.read_csv(ref_path)
    df_new = pd.read_csv(new_path)

    features = {}
    for col in df_ref.columns:
        if col == "churned":  # target no se usa
            continue

        if pd.api.types.is_numeric_dtype(df_ref[col]):
            # PSI
            value = psi(df_ref[col].values, df_new[col].values)
        else:
            # Categóricas: comparar distribuciones con PSI
            ref_counts = df_ref[col].value_counts(normalize=True)
            new_counts = df_new[col].value_counts(normalize=True)
            all_cats = set(ref_counts.index) | set(new_counts.index)
            exp = [ref_counts.get(cat, 0) for cat in all_cats]
            act = [new_counts.get(cat, 0) for cat in all_cats]
            value = psi(exp, act, buckets=len(all_cats))

        features[col] = round(value, 3)

    overall_drift = any(v > THRESHOLD for v in features.values())

    report = {
        "threshold": THRESHOLD,
        "overall_drift": overall_drift,
        "features": features,
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Drift report saved to {out_path}")
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Drift detection CLI")
    parser.add_argument("--ref", required=True, help="Path to reference CSV")
    parser.add_argument("--new", required=True, help="Path to new CSV")
    parser.add_argument("--out", default="artifacts/drift_report.json", help="Output JSON path")
    args = parser.parse_args()

    drift_check(args.ref, args.new, args.out)


if __name__ == "__main__":
    main()
