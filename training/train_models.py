import pandas as pd
import joblib
from pathlib import Path

from train_logistic import train_logistic
from train_random_forest import train_random_forest


# =========================================================
# PATH PROJECT
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models" / "experiments"

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# =========================================================
# LOAD DATASET
# =========================================================

print("Loading datasets...")

train_original = pd.read_csv(
    DATA_DIR / "train_original.csv"
)

train_smote = pd.read_csv(
    DATA_DIR / "train_smote.csv"
)


# =========================================================
# FUNCTION UNTUK MEMISAHKAN FEATURE DAN TARGET
# =========================================================

def split_features_target(df):

    X = df.drop(columns=["Class"])
    y = df["Class"]

    return X, y


X_original, y_original = split_features_target(
    train_original
)

X_smote, y_smote = split_features_target(
    train_smote
)


print("\nDataset Original:")
print("X:", X_original.shape)
print("y:", y_original.shape)

print("\nDataset SMOTE:")
print("X:", X_smote.shape)
print("y:", y_smote.shape)


# =========================================================
# TRAIN LOGISTIC REGRESSION
# =========================================================

print("\n" + "=" * 50)
print("TRAINING LOGISTIC REGRESSION")
print("=" * 50)


print("\nTraining Logistic Regression - Original...")

logistic_original = train_logistic(
    X_original,
    y_original
)

joblib.dump(
    logistic_original,
    MODEL_DIR / "logistic_original.joblib"
)

print("Saved: logistic_original.joblib")


print("\nTraining Logistic Regression - SMOTE...")

logistic_smote = train_logistic(
    X_smote,
    y_smote
)

joblib.dump(
    logistic_smote,
    MODEL_DIR / "logistic_smote.joblib"
)

print("Saved: logistic_smote.joblib")


# =========================================================
# TRAIN RANDOM FOREST
# =========================================================

print("\n" + "=" * 50)
print("TRAINING RANDOM FOREST")
print("=" * 50)


print("\nTraining Random Forest - Original...")

random_forest_original = train_random_forest(
    X_original,
    y_original
)

joblib.dump(
    random_forest_original,
    MODEL_DIR / "random_forest_original.joblib"
)

print("Saved: random_forest_original.joblib")


print("\nTraining Random Forest - SMOTE...")

random_forest_smote = train_random_forest(
    X_smote,
    y_smote
)

joblib.dump(
    random_forest_smote,
    MODEL_DIR / "random_forest_smote.joblib"
)

print("Saved: random_forest_smote.joblib")


# =========================================================
# SELESAI
# =========================================================

print("\n" + "=" * 50)
print("SEMUA MODEL BERHASIL DILATIH")
print("=" * 50)

print("\nModel yang dibuat:")

for model_file in MODEL_DIR.glob("*.joblib"):
    print("-", model_file.name)