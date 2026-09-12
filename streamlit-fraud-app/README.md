# Streamlit App — Credit Card Fraud Detection (Kelompok 3)

Dashboard untuk menyajikan hasil project: EDA, perbandingan model, dan prediksi fraud interaktif.

## Cara Menjalankan

```bash
pip install -r requirements.txt
streamlit run app.py
```

Aplikasi akan terbuka otomatis di browser (`http://localhost:8501`).

## Struktur Folder

```
.
├── app.py                     # Aplikasi utama Streamlit
├── requirements.txt
└── assets/
    ├── models/
    │   ├── fraud_model.joblib   # Model produksi: Random Forest (data original)
    │   └── scaler.joblib        # StandardScaler untuk kolom Time & Amount
    ├── sample_transactions.csv  # Sampel 1.500 transaksi untuk demo/EDA
    ├── dataset_info.json        # Statistik ringkas dataset
    ├── eda/                     # 4 chart hasil EDA (src/eda.py)
    └── evaluation/
        ├── model_comparison.csv
        └── confusion_matrix/    # Confusion matrix ke-4 model
```

## Halaman Aplikasi

1. **Beranda** — ringkasan project & statistik dataset
2. **Dataset & EDA** — contoh data + visualisasi EDA yang sudah dibuat kelompok
3. **Perbandingan Model** — tabel metrik & confusion matrix ke-4 model (Logistic Regression / Random Forest × Original/SMOTE)
4. **Prediksi Transaksi** — coba model produksi (Random Forest Original) dengan 3 mode:
   - Ambil contoh dari dataset
   - Input manual (Time, Amount, dan fitur V1–V28)
   - Upload CSV untuk prediksi batch

## Catatan Teknis

- Model produksi (`fraud_model.joblib`) dilatih pada fitur `Time` dan `Amount` yang sudah di-*scale* dengan `StandardScaler`. Karena file `scaler.joblib` sebelumnya tidak disimpan di repo asli (hanya dataset `.csv` yang di-*gitignore*), scaler ini **dihitung ulang** mengikuti persis langkah `src/preprocessing.py` (split 80/20, `random_state=42`, `stratify=y`) sehingga hasilnya konsisten dengan model yang sudah dilatih.
- Untuk ke depannya, disarankan menyimpan `scaler.joblib` langsung saat preprocessing supaya tidak perlu dihitung ulang.
