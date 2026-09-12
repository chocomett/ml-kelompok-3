import pandas as pd
import joblib
import json

from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

import matplotlib.pyplot as plt


# =========================================================
# PROJECT PATH
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models" / "experiments"

OUTPUT_DIR = BASE_DIR / "outputs" / "evaluation"

METRICS_DIR = OUTPUT_DIR / "metrics"
CONFUSION_DIR = OUTPUT_DIR / "confusion_matrix"


METRICS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

CONFUSION_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# LOAD TEST DATA
# =========================================================

print("Loading test dataset...")

test_df = pd.read_csv(
    DATA_DIR / "test.csv"
)

X_test = test_df.drop(columns=["Class"])
y_test = test_df["Class"]

print("Test data:", X_test.shape)

print("\nClass distribution:")
print(y_test.value_counts())


# =========================================================
# EVALUATE FUNCTION
# =========================================================

def evaluate_model(model_name, model):

    print("\n" + "=" * 60)
    print(f"EVALUATING: {model_name}")
    print("=" * 60)

    # -----------------------------------------------------
    # Prediction
    # -----------------------------------------------------

    y_pred = model.predict(X_test)

    # Probability untuk ROC-AUC
    y_prob = model.predict_proba(X_test)[:, 1]


    # -----------------------------------------------------
    # Metrics
    # -----------------------------------------------------

    metrics = {

        "model": model_name,

        "accuracy": float(
            accuracy_score(y_test, y_pred)
        ),

        "precision": float(
            precision_score(
                y_test,
                y_pred,
                zero_division=0
            )
        ),

        "recall": float(
            recall_score(
                y_test,
                y_pred,
                zero_division=0
            )
        ),

        "f1_score": float(
            f1_score(
                y_test,
                y_pred,
                zero_division=0
            )
        ),

        "roc_auc": float(
            roc_auc_score(
                y_test,
                y_prob
            )
        )
    }


    # -----------------------------------------------------
    # Print Metrics
    # -----------------------------------------------------

    print("\nRESULT:")

    for metric, value in metrics.items():

        if metric != "model":

            print(
                f"{metric}: {value:.4f}"
            )


    # -----------------------------------------------------
    # Save Metrics JSON
    # -----------------------------------------------------

    metrics_path = (
        METRICS_DIR /
        f"{model_name}_metrics.json"
    )

    with open(
        metrics_path,
        "w"
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4
        )


    print(
        f"\nMetrics saved: {metrics_path.name}"
    )


    # -----------------------------------------------------
    # Confusion Matrix
    # -----------------------------------------------------

    cm = confusion_matrix(
        y_test,
        y_pred
    )


    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Normal", "Fraud"]
    )

    display.plot()

    plt.title(
        f"Confusion Matrix - {model_name}"
    )

    plt.savefig(
        CONFUSION_DIR /
        f"{model_name}.png",
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()


    print(
        f"Confusion Matrix saved: {model_name}.png"
    )

    return metrics


# =========================================================
# LOAD AND EVALUATE ALL MODELS
# =========================================================

print("\nLoading models...")


for model_path in MODEL_DIR.glob("*.joblib"):

    model_name = model_path.stem

    model = joblib.load(
        model_path
    )

    evaluate_model(
        model_name,
        model
    )


# =========================================================
# FINISHED
# =========================================================

print("\n" + "=" * 60)
print("ALL MODELS EVALUATED")
print("=" * 60)