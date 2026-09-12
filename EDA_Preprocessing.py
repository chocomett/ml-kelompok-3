"""
=====================================================
EDA + PREPROCESSING - Credit Card Fraud Detection
=====================================================
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', None)

# --- STEP 1: Load data & lihat struktur dasar ---
df = pd.read_csv('creditcard_csv.csv')

# --- CEK TIPE DATA NUMERIK ---

print("\nCEK DATA NON-NUMERIK")

for col in df.columns:

    # Class ditangani terpisah
    if col == 'Class':
        continue

    invalid_mask = pd.to_numeric(
        df[col],
        errors='coerce'
    ).isna() & df[col].notna()

    if invalid_mask.any():

        print(f"\n⚠️ DATA BERMASALAH DI KOLOM: {col}")

        print(
            df.loc[
                invalid_mask,
                [col]
            ]
        )

print("STEP 1: Struktur dasar")
print("Jumlah baris & kolom:", df.shape)
print(df.dtypes)
print(df.head())

# --- CLEAN DATA CORRUPT / NON-NUMERIK ---

print("\nCEK DATA CORRUPT")

feature_columns = df.columns.drop('Class')

# Ubah seluruh feature menjadi numeric
for col in feature_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')


# Cari baris yang memiliki nilai invalid
corrupt_mask = df[feature_columns].isna().any(axis=1)

print("Jumlah baris corrupt:", corrupt_mask.sum())

if corrupt_mask.any():
    print("\nData corrupt:")
    print(df.loc[corrupt_mask])


# Hapus baris corrupt
df = df.loc[~corrupt_mask].copy()

print("\nDataset setelah membersihkan data corrupt:")
print(df.shape)

print("\nTipe data setelah cleaning:")
print(df.dtypes)

# --- STEP 2: Cek missing value & duplikat ---
print("\nSTEP 2: Missing value & duplikat")
print(df.isnull().sum().sum(), "missing value")
print("Duplikat:", df.duplicated().sum())

# --- STEP 3: Distribusi Class ---
print("\nSTEP 3: Distribusi Class")
class_counts = df['Class'].value_counts()
print(class_counts)
fraud_pct = class_counts.min() / class_counts.sum() * 100
print(f"Persentase fraud: {fraud_pct:.3f}%")

plt.figure(figsize=(5, 4))
sns.countplot(x='Class', data=df)
plt.title('Distribusi Class')
plt.yscale('log')
plt.savefig('01_distribusi_class.png', dpi=100, bbox_inches='tight')
plt.close()

# --- STEP 4: Distribusi Amount & Time ---
print("\nSTEP 4: Distribusi Amount & Time")
print(df[['Time', 'Amount']].describe())

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.histplot(df['Amount'], bins=50, ax=axes[0])
axes[0].set_xlim(0, 2000)
axes[0].set_title('Distribusi Amount')
sns.histplot(df['Time'], bins=50, ax=axes[1])
axes[1].set_title('Distribusi Time')
plt.savefig('02_distribusi_amount_time.png', dpi=100, bbox_inches='tight')
plt.close()

# --- STEP 5: Amount fraud vs normal ---
print("\nSTEP 5: Amount - Fraud vs Normal")
fraud = df[df['Class'].astype(str).str.contains('1')]
normal = df[df['Class'].astype(str).str.contains('0')]
print("Rata-rata Amount FRAUD :", fraud['Amount'].mean())
print("Rata-rata Amount NORMAL:", normal['Amount'].mean())

plt.figure(figsize=(6, 4))
sns.boxplot(x='Class', y='Amount', data=df, showfliers=False)
plt.savefig('03_boxplot_amount_class.png', dpi=100, bbox_inches='tight')
plt.close()

# --- STEP 6: Correlation heatmap ---
print("\nSTEP 6: Korelasi antar fitur")
df_corr = df.copy()
df_corr['Class'] = df_corr['Class'].astype(str).str.replace("'", "").astype(int)
corr = df_corr.corr(numeric_only=True)

plt.figure(figsize=(14, 10))
sns.heatmap(corr, cmap='coolwarm', center=0)
plt.savefig('04_correlation_heatmap.png', dpi=100, bbox_inches='tight')
plt.close()

corr_class = corr['Class'].drop('Class').sort_values(key=abs, ascending=False)
print("Top 10 fitur paling berkorelasi dengan Class:")
print(corr_class.head(10))

# --- STEP 7: Bersihin kolom Class ---
print("\nSTEP 7: Bersihin kolom Class")

df['Class'] = (
    df['Class']
    .astype(str)
    .str.replace("'", "")
    .astype(int)
)

print("Class unik:", df['Class'].unique())


# --- STEP 8: Hapus duplikat ---
print("\nSTEP 8: Hapus duplikat")

print("Sebelum:", df.shape)

df = df.drop_duplicates()

print("Sesudah:", df.shape)


# --- STEP 9: Pisah fitur (X) & target (y) ---
print("\nSTEP 9: Pisah fitur dan target")

X = df.drop(columns=['Class'])
y = df['Class']

print("Shape X:", X.shape)
print("Shape y:", y.shape)

print("Distribusi Class:")
print(y.value_counts())


# --- STEP 10: Train-test split ---
print("\nSTEP 10: Train-test split")

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Train:", X_train.shape)
print("Test :", X_test.shape)


# --- STEP 11: Scaling Amount & Time ---
print("\nSTEP 11: Scaling")

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# Scaling hanya berdasarkan data training
X_train[['Amount', 'Time']] = scaler.fit_transform(
    X_train[['Amount', 'Time']]
)

# Data test hanya ditransform menggunakan scaler dari train
X_test[['Amount', 'Time']] = scaler.transform(
    X_test[['Amount', 'Time']]
)

print("Scaling selesai.")


# --- STEP 12: SMOTE (hanya di data train) ---
print("\nSTEP 12: SMOTE")

from imblearn.over_sampling import SMOTE

print(
    "Sebelum SMOTE:",
    y_train.value_counts().to_dict()
)

smote = SMOTE(random_state=42)

X_train_res, y_train_res = smote.fit_resample(
    X_train,
    y_train
)

print(
    "Sesudah SMOTE:",
    pd.Series(y_train_res).value_counts().to_dict()
)


# --- STEP 13: Simpan hasil akhir ---
print("\nSTEP 13: Simpan file")

# Data training setelah SMOTE
train_smote = X_train_res.copy()
train_smote['Class'] = y_train_res

train_smote.to_csv(
    'train_smote.csv',
    index=False
)

# Data training sebelum SMOTE
train_original = X_train.copy()
train_original['Class'] = y_train

train_original.to_csv(
    'train_original.csv',
    index=False
)

# Data testing
test_final = X_test.copy()
test_final['Class'] = y_test

test_final.to_csv(
    'test.csv',
    index=False
)

print("Selesai! File tersimpan:")
print("- train_smote.csv")
print("- train_original.csv")
print("- test.csv")