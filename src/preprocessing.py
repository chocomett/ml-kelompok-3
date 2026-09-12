"""
=====================================================
PREPROCESSING - Credit Card Fraud Detection
=====================================================

Proses:
1. Load raw dataset
2. Bersihkan data corrupt
3. Bersihkan Class
4. Hapus duplicate
5. Pisahkan X dan y
6. Train-test split
7. Scaling Time dan Amount
8. SMOTE pada training data
9. Simpan processed dataset
=====================================================
"""

import pandas as pd

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from imblearn.over_sampling import SMOTE


# =====================================================
# CONFIGURATION
# =====================================================

DATA_PATH = Path("data/raw/creditcard_csv.csv")

OUTPUT_PATH = Path("data/processed")

OUTPUT_PATH.mkdir(
    parents=True,
    exist_ok=True
)


# =====================================================
# STEP 1: LOAD DATA
# =====================================================

print("\n" + "=" * 50)
print("STEP 1: LOAD DATA")
print("=" * 50)


df = pd.read_csv(DATA_PATH)

print("Dataset awal:", df.shape)


# =====================================================
# STEP 2: CLEAN NON-NUMERIC / CORRUPT DATA
# =====================================================

print("\n" + "=" * 50)
print("STEP 2: CLEAN CORRUPT DATA")
print("=" * 50)


feature_columns = df.columns.drop("Class")


# Ubah seluruh feature menjadi numeric
for col in feature_columns:

    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )


# Cari baris corrupt
corrupt_mask = (
    df[feature_columns]
    .isna()
    .any(axis=1)
)


print(
    "Jumlah baris corrupt:",
    corrupt_mask.sum()
)


# Hapus data corrupt
df = df.loc[
    ~corrupt_mask
].copy()


print(
    "Dataset setelah cleaning:",
    df.shape
)


# =====================================================
# STEP 3: CLEAN CLASS
# =====================================================

print("\n" + "=" * 50)
print("STEP 3: CLEAN CLASS")
print("=" * 50)


df["Class"] = (
    df["Class"]
    .astype(str)
    .str.replace("'", "")
    .astype(int)
)


print(
    "Class unik:",
    df["Class"].unique()
)


# =====================================================
# STEP 4: REMOVE DUPLICATES
# =====================================================

print("\n" + "=" * 50)
print("STEP 4: REMOVE DUPLICATES")
print("=" * 50)


print(
    "Sebelum:",
    df.shape
)


df = df.drop_duplicates()


print(
    "Sesudah:",
    df.shape
)


# =====================================================
# STEP 5: CHECK FINAL DATA
# =====================================================

print("\n" + "=" * 50)
print("STEP 5: FINAL DATA CHECK")
print("=" * 50)


print("\nMissing value:")

print(
    df.isnull()
    .sum()
    .sum()
)


print("\nTipe data:")

print(df.dtypes)


# =====================================================
# STEP 6: SPLIT FEATURE & TARGET
# =====================================================

print("\n" + "=" * 50)
print("STEP 6: SPLIT FEATURE & TARGET")
print("=" * 50)


X = df.drop(
    columns=["Class"]
)

y = df["Class"]


print(
    "Shape X:",
    X.shape
)

print(
    "Shape y:",
    y.shape
)


print(
    "\nDistribusi Class:"
)

print(
    y.value_counts()
)


# =====================================================
# STEP 7: TRAIN TEST SPLIT
# =====================================================

print("\n" + "=" * 50)
print("STEP 7: TRAIN TEST SPLIT")
print("=" * 50)


X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.2,

    random_state=42,

    # Sangat penting karena dataset imbalance
    stratify=y
)


print(
    "Train:",
    X_train.shape
)

print(
    "Test:",
    X_test.shape
)


# =====================================================
# STEP 8: SCALING
# =====================================================

print("\n" + "=" * 50)
print("STEP 8: SCALING")
print("=" * 50)


scaler = StandardScaler()


columns_to_scale = [
    "Amount",
    "Time"
]


# Fit hanya pada TRAIN
X_train = X_train.copy()
X_test = X_test.copy()


X_train[columns_to_scale] = (

    scaler.fit_transform(
        X_train[columns_to_scale]
    )
)


# Test hanya transform
X_test[columns_to_scale] = (

    scaler.transform(
        X_test[columns_to_scale]
    )
)


print(
    "Scaling selesai."
)


# =====================================================
# STEP 9: SMOTE
# =====================================================

print("\n" + "=" * 50)
print("STEP 9: SMOTE")
print("=" * 50)


print(
    "\nSebelum SMOTE:"
)

print(
    y_train.value_counts()
)


smote = SMOTE(
    random_state=42
)


X_train_res, y_train_res = (

    smote.fit_resample(
        X_train,
        y_train
    )
)


print(
    "\nSesudah SMOTE:"
)

print(
    pd.Series(
        y_train_res
    ).value_counts()
)


# =====================================================
# STEP 10: SAVE PROCESSED DATA
# =====================================================

print("\n" + "=" * 50)
print("STEP 10: SAVE DATA")
print("=" * 50)


# -------------------------
# TRAIN ORIGINAL
# -------------------------

train_original = X_train.copy()

train_original["Class"] = y_train


train_original.to_csv(

    OUTPUT_PATH / "train_original.csv",

    index=False
)


# -------------------------
# TRAIN SMOTE
# -------------------------

train_smote = X_train_res.copy()

train_smote["Class"] = y_train_res


train_smote.to_csv(

    OUTPUT_PATH / "train_smote.csv",

    index=False
)


# -------------------------
# TEST DATA
# -------------------------

test_final = X_test.copy()

test_final["Class"] = y_test


test_final.to_csv(

    OUTPUT_PATH / "test.csv",

    index=False
)


print("\nFile berhasil disimpan:")

print("- train_original.csv")
print("- train_smote.csv")
print("- test.csv")


print("\n" + "=" * 50)
print("PREPROCESSING SELESAI")
print("=" * 50)