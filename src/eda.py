"""
=====================================================
EDA - Credit Card Fraud Detection
=====================================================

Tujuan:
Melakukan Exploratory Data Analysis (EDA) untuk memahami
struktur, kualitas, distribusi, dan hubungan antar data.
=====================================================
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


# =====================================================
# CONFIGURATION
# =====================================================

DATA_PATH = Path("data/raw/creditcard_csv.csv")
OUTPUT_PATH = Path("outputs/eda")

OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

pd.set_option("display.max_columns", None)


# =====================================================
# STEP 1: LOAD DATA
# =====================================================

print("\n" + "=" * 50)
print("STEP 1: LOAD DATA")
print("=" * 50)

df = pd.read_csv(DATA_PATH)

print("Jumlah baris & kolom:", df.shape)


# =====================================================
# STEP 2: CEK STRUKTUR DATA
# =====================================================

print("\n" + "=" * 50)
print("STEP 2: STRUKTUR DATA")
print("=" * 50)

print("\nTipe Data:")
print(df.dtypes)

print("\n5 Data Pertama:")
print(df.head())


# =====================================================
# STEP 3: CEK DATA NON-NUMERIK
# =====================================================

print("\n" + "=" * 50)
print("STEP 3: CEK DATA NON-NUMERIK")
print("=" * 50)

for col in df.columns:

    # Class ditangani sebagai target
    if col == "Class":
        continue

    invalid_mask = (
        pd.to_numeric(
            df[col],
            errors="coerce"
        ).isna()
        &
        df[col].notna()
    )

    if invalid_mask.any():

        print(f"\n⚠️ DATA BERMASALAH DI KOLOM: {col}")

        print(
            df.loc[
                invalid_mask,
                [col]
            ]
        )


# =====================================================
# STEP 4: MISSING VALUE & DUPLICATE
# =====================================================

print("\n" + "=" * 50)
print("STEP 4: MISSING VALUE & DUPLICATE")
print("=" * 50)

print(
    "\nJumlah Missing Value:",
    df.isnull().sum().sum()
)

print(
    "Jumlah Duplikat:",
    df.duplicated().sum()
)


# =====================================================
# STEP 5: DISTRIBUSI CLASS
# =====================================================

print("\n" + "=" * 50)
print("STEP 5: DISTRIBUSI CLASS")
print("=" * 50)

class_counts = df["Class"].value_counts()

print(class_counts)

fraud_pct = (
    class_counts.min()
    /
    class_counts.sum()
    *
    100
)

print(f"\nPersentase Fraud: {fraud_pct:.3f}%")


plt.figure(figsize=(6, 4))

sns.countplot(
    x="Class",
    data=df
)

plt.title("Distribusi Class")

# Karena dataset sangat imbalance
plt.yscale("log")

plt.savefig(
    OUTPUT_PATH / "01_distribusi_class.png",
    dpi=100,
    bbox_inches="tight"
)

plt.close()


# =====================================================
# STEP 6: DISTRIBUSI AMOUNT & TIME
# =====================================================

print("\n" + "=" * 50)
print("STEP 6: DISTRIBUSI AMOUNT & TIME")
print("=" * 50)

print(
    df[
        ["Time", "Amount"]
    ].describe()
)


fig, axes = plt.subplots(
    1,
    2,
    figsize=(12, 4)
)


# Amount
sns.histplot(
    df["Amount"],
    bins=50,
    ax=axes[0]
)

axes[0].set_xlim(0, 2000)
axes[0].set_title("Distribusi Amount")


# Time
sns.histplot(
    df["Time"],
    bins=50,
    ax=axes[1]
)

axes[1].set_title("Distribusi Time")


plt.savefig(
    OUTPUT_PATH / "02_distribusi_amount_time.png",
    dpi=100,
    bbox_inches="tight"
)

plt.close()


# =====================================================
# STEP 7: AMOUNT FRAUD VS NORMAL
# =====================================================

print("\n" + "=" * 50)
print("STEP 7: AMOUNT FRAUD VS NORMAL")
print("=" * 50)


# Bersihkan Class sementara untuk analisis
df_temp = df.copy()

df_temp["Class"] = (
    df_temp["Class"]
    .astype(str)
    .str.replace("'", "")
    .astype(int)
)


fraud = df_temp[
    df_temp["Class"] == 1
]

normal = df_temp[
    df_temp["Class"] == 0
]


print(
    "\nRata-rata Amount FRAUD:",
    fraud["Amount"].mean()
)

print(
    "Rata-rata Amount NORMAL:",
    normal["Amount"].mean()
)


plt.figure(figsize=(6, 4))

sns.boxplot(
    x="Class",
    y="Amount",
    data=df_temp,
    showfliers=False
)

plt.title("Amount Berdasarkan Class")

plt.savefig(
    OUTPUT_PATH / "03_boxplot_amount_class.png",
    dpi=100,
    bbox_inches="tight"
)

plt.close()


# =====================================================
# STEP 8: CORRELATION HEATMAP
# =====================================================

print("\n" + "=" * 50)
print("STEP 8: KORELASI ANTAR FITUR")
print("=" * 50)


df_corr = df.copy()


# Ubah seluruh feature menjadi numerik untuk correlation
feature_columns = df_corr.columns.drop("Class")

for col in feature_columns:

    df_corr[col] = pd.to_numeric(
        df_corr[col],
        errors="coerce"
    )


# Bersihkan Class
df_corr["Class"] = (
    df_corr["Class"]
    .astype(str)
    .str.replace("'", "")
)

df_corr["Class"] = pd.to_numeric(
    df_corr["Class"],
    errors="coerce"
)


# Hilangkan data invalid sementara untuk correlation
df_corr = df_corr.dropna()


corr = df_corr.corr(
    numeric_only=True
)


plt.figure(figsize=(14, 10))

sns.heatmap(
    corr,
    cmap="coolwarm",
    center=0
)

plt.title("Correlation Heatmap")

plt.savefig(
    OUTPUT_PATH / "04_correlation_heatmap.png",
    dpi=100,
    bbox_inches="tight"
)

plt.close()


# =====================================================
# TOP FEATURE CORRELATION
# =====================================================

corr_class = (
    corr["Class"]
    .drop("Class")
    .sort_values(
        key=abs,
        ascending=False
    )
)


print(
    "\nTop 10 fitur paling berkorelasi dengan Class:"
)

print(
    corr_class.head(10)
)


# =====================================================
# FINISH
# =====================================================

print("\n" + "=" * 50)
print("EDA SELESAI")
print("=" * 50)

print(
    f"\nVisualisasi disimpan di: {OUTPUT_PATH}"
)