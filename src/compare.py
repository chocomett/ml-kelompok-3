import json
import pandas as pd

from pathlib import Path


# =========================================================
# PROJECT PATH
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

METRICS_DIR = (
    BASE_DIR
    / "outputs"
    / "evaluation"
    / "metrics"
)

OUTPUT_DIR = (
    BASE_DIR
    / "outputs"
    / "evaluation"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# LOAD ALL METRICS
# =========================================================

print("Loading evaluation results...")

results = []

for json_file in METRICS_DIR.glob("*_metrics.json"):

    with open(json_file, "r") as file:
        metrics = json.load(file)

    results.append(metrics)


# =========================================================
# CREATE DATAFRAME
# =========================================================

df = pd.DataFrame(results)


# =========================================================
# SORT MODEL
# =========================================================

df = df.sort_values(
    by="f1_score",
    ascending=False
)


# =========================================================
# DISPLAY RESULTS
# =========================================================

print("\n" + "=" * 80)
print("MODEL COMPARISON")
print("=" * 80)

print(
    df.to_string(
        index=False
    )
)


# =========================================================
# SAVE CSV
# =========================================================

output_path = (
    OUTPUT_DIR
    / "model_comparison.csv"
)

df.to_csv(
    output_path,
    index=False
)

print("\nComparison saved to:")
print(output_path)


# =========================================================
# BEST MODEL
# =========================================================

best_model = df.iloc[0]

print("\n" + "=" * 80)
print("BEST MODEL BASED ON F1 SCORE")
print("=" * 80)

print(f"Model    : {best_model['model']}")
print(f"F1 Score : {best_model['f1_score']:.4f}")
print(f"Precision: {best_model['precision']:.4f}")
print(f"Recall   : {best_model['recall']:.4f}")
print(f"ROC-AUC  : {best_model['roc_auc']:.4f}")