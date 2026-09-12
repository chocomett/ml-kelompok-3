"""
=====================================================
STREAMLIT APP - Credit Card Fraud Detection
Kelompok 3
=====================================================

Menyajikan hasil project (EDA, perbandingan model, dan
prediksi fraud) yang sudah dikerjakan kelompok ke dalam
sebuah dashboard interaktif.

Jalankan dengan:
    streamlit run app.py
=====================================================
"""

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# =====================================================
# CONFIG & PATH
# =====================================================

BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "assets"

MODEL_PATH = ASSETS_DIR / "models" / "fraud_model.joblib"
SCALER_PATH = ASSETS_DIR / "models" / "scaler.joblib"
SAMPLE_PATH = ASSETS_DIR / "sample_transactions.csv"
INFO_PATH = ASSETS_DIR / "dataset_info.json"
EDA_DIR = ASSETS_DIR / "eda"
EVAL_DIR = ASSETS_DIR / "evaluation"

FEATURE_ORDER = (
    ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
)

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide",
)


# =====================================================
# CACHED LOADERS
# =====================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_resource
def load_scaler():
    return joblib.load(SCALER_PATH)


@st.cache_data
def load_sample():
    return pd.read_csv(SAMPLE_PATH)


@st.cache_data
def load_dataset_info():
    with open(INFO_PATH) as f:
        return json.load(f)


@st.cache_data
def load_model_comparison():
    return pd.read_csv(EVAL_DIR / "model_comparison.csv")


MODEL_LABELS = {
    "random_forest_original": "Random Forest (Original)",
    "random_forest_smote": "Random Forest (SMOTE)",
    "logistic_original": "Logistic Regression (Original)",
    "logistic_smote": "Logistic Regression (SMOTE)",
}


def predict_transaction(row: dict, model, scaler):
    """row: dict fitur mentah (belum di-scale) -> (label, proba_fraud)"""
    X = pd.DataFrame([row])[FEATURE_ORDER]
    X_scaled = X.copy()
    X_scaled[["Amount", "Time"]] = scaler.transform(X_scaled[["Amount", "Time"]])
    pred = model.predict(X_scaled)[0]
    proba = model.predict_proba(X_scaled)[0, 1]
    return int(pred), float(proba)


# =====================================================
# SIDEBAR NAVIGATION
# =====================================================

st.sidebar.title("💳 Fraud Detection")
st.sidebar.caption("Kelompok 3 — Machine Learning")

page = st.sidebar.radio(
    "Navigasi",
    [
        "🏠 Beranda",
        "📊 Dataset & EDA",
        "🧪 Perbandingan Model",
        "🔍 Prediksi Transaksi",
    ],
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Model produksi:** Random Forest\n\n"
    "Dilatih pada data asli (tanpa SMOTE) — performa terbaik "
    "berdasarkan F1-score pada data uji."
)


# =====================================================
# PAGE: BERANDA
# =====================================================

if page == "🏠 Beranda":

    st.title("💳 Credit Card Fraud Detection")
    st.markdown(
        "Dashboard ini menyajikan hasil project **klasifikasi transaksi "
        "kartu kredit** (normal vs. fraud) mulai dari eksplorasi data, "
        "perbandingan model, hingga prediksi interaktif."
    )

    info = load_dataset_info()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Transaksi", f"{info['n_rows']:,}")
    col2.metric("Jumlah Fitur", info["n_cols"] - 1)
    col3.metric("Transaksi Fraud", f"{info['class_counts']['1']:,}"
                if "1" in info["class_counts"] else info["class_counts"].get(1))
    col4.metric("Persentase Fraud", f"{info['fraud_pct']:.3f}%")

    st.markdown("### Alur Project")
    st.markdown(
        """
1. **Data preprocessing** — membersihkan data korup, duplikat, dan scaling `Time`/`Amount`
2. **Exploratory Data Analysis (EDA)** — memahami distribusi & korelasi fitur
3. **Training model** — Logistic Regression & Random Forest, masing-masing pada data asli dan data hasil SMOTE
4. **Evaluasi** — membandingkan Accuracy, Precision, Recall, F1-score, dan ROC-AUC
5. **Deployment** — model terbaik disajikan lewat aplikasi Streamlit ini
        """
    )

    st.info(
        "Dataset sangat *imbalanced* (fraud < 0,2% dari seluruh transaksi), "
        "sehingga **Precision, Recall, F1-score, dan ROC-AUC** jauh lebih "
        "bermakna dibanding Accuracy saja.",
        icon="⚠️",
    )


# =====================================================
# PAGE: DATASET & EDA
# =====================================================

elif page == "📊 Dataset & EDA":

    st.title("📊 Dataset & Exploratory Data Analysis")

    info = load_dataset_info()
    sample = load_sample()

    st.markdown(
        "Dataset berisi transaksi kartu kredit dengan fitur `Time`, `Amount`, "
        "dan `V1`–`V28` (hasil transformasi PCA, dianonimkan untuk menjaga "
        "kerahasiaan data nasabah), serta target `Class` (0 = normal, 1 = fraud)."
    )

    st.markdown("#### Contoh Data")
    st.caption("Sampel 1.500 transaksi (150 fraud + 1.350 normal) dari dataset asli.")
    st.dataframe(sample.head(20), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Rata-rata Amount**")
        st.write(f"Normal: `{info['amount_mean_normal']:.2f}`  |  Fraud: `{info['amount_mean_fraud']:.2f}`")
    with c2:
        st.markdown("**Median Amount**")
        st.write(f"Normal: `{info['amount_median_normal']:.2f}`  |  Fraud: `{info['amount_median_fraud']:.2f}`")

    st.markdown("---")
    st.markdown("### Visualisasi EDA")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Distribusi Class", "Amount & Time", "Amount vs Class", "Korelasi Fitur"]
    )

    with tab1:
        st.image(str(EDA_DIR / "01_distribusi_class.png"), use_container_width=True)
        st.caption(
            "Distribusi kelas sangat timpang (skala log) — fraud hanya "
            f"sekitar {info['fraud_pct']:.3f}% dari seluruh transaksi."
        )

    with tab2:
        st.image(str(EDA_DIR / "02_distribusi_amount_time.png"), use_container_width=True)
        st.caption("Sebagian besar transaksi bernominal kecil; `Time` mencakup ±48 jam data.")

    with tab3:
        st.image(str(EDA_DIR / "03_boxplot_amount_class.png"), use_container_width=True)
        st.caption("Perbandingan sebaran nominal transaksi antara kelas normal dan fraud.")

    with tab4:
        st.image(str(EDA_DIR / "04_correlation_heatmap.png"), use_container_width=True)
        st.caption("Korelasi antar fitur V1–V28 (hasil PCA) terhadap kelas transaksi.")


# =====================================================
# PAGE: PERBANDINGAN MODEL
# =====================================================

elif page == "🧪 Perbandingan Model":

    st.title("🧪 Perbandingan Model")

    st.markdown(
        "Dua algoritma diuji, masing-masing pada **data asli** (imbalanced) "
        "dan **data hasil SMOTE** (oversampling kelas minoritas), lalu "
        "dievaluasi pada data uji yang sama."
    )

    comparison = load_model_comparison().copy()
    comparison["model"] = comparison["model"].map(lambda m: MODEL_LABELS.get(m, m))
    comparison = comparison.rename(columns={
        "model": "Model",
        "accuracy": "Accuracy",
        "precision": "Precision",
        "recall": "Recall",
        "f1_score": "F1-Score",
        "roc_auc": "ROC-AUC",
    })

    st.dataframe(
        comparison.style.format({
            "Accuracy": "{:.4f}", "Precision": "{:.4f}", "Recall": "{:.4f}",
            "F1-Score": "{:.4f}", "ROC-AUC": "{:.4f}",
        }).highlight_max(subset=["F1-Score"], color="#c6f6d5"),
        use_container_width=True,
    )

    best = comparison.sort_values("F1-Score", ascending=False).iloc[0]
    st.success(
        f"**Model terbaik (F1-score): {best['Model']}** — dipilih sebagai "
        f"model produksi untuk halaman prediksi.",
        icon="🏆",
    )

    st.markdown("#### Confusion Matrix per Model")
    model_files = {
        "random_forest_original": "Random Forest (Original)",
        "random_forest_smote": "Random Forest (SMOTE)",
        "logistic_original": "Logistic Regression (Original)",
        "logistic_smote": "Logistic Regression (SMOTE)",
    }
    cols = st.columns(2)
    for i, (fname, label) in enumerate(model_files.items()):
        with cols[i % 2]:
            st.image(
                str(EVAL_DIR / "confusion_matrix" / f"{fname}.png"),
                caption=label,
                use_container_width=True,
            )

    st.markdown(
        """
**Catatan interpretasi:**
- *Random Forest (Original)* memberi keseimbangan terbaik antara Precision
  dan Recall (F1-score tertinggi) — dipakai sebagai model produksi.
- *Logistic Regression (SMOTE)* punya Recall tinggi tapi Precision sangat
  rendah → banyak transaksi normal salah ditandai sebagai fraud (false alarm).
- SMOTE membantu Recall pada beberapa model, tapi tidak selalu memperbaiki
  F1-score karena menambah false positive.
        """
    )


# =====================================================
# PAGE: PREDIKSI
# =====================================================

elif page == "🔍 Prediksi Transaksi":

    st.title("🔍 Prediksi Transaksi (Fraud atau Normal)")
    st.markdown(
        "Model yang dipakai: **Random Forest (Original)** — model produksi "
        "hasil training kelompok, disimpan dengan `joblib`."
    )

    model = load_model()
    scaler = load_scaler()
    sample = load_sample()

    mode = st.radio(
        "Pilih mode input",
        ["Ambil contoh dari dataset", "Input manual", "Upload CSV (batch)"],
        horizontal=True,
    )

    # -------------------------------------------------
    # MODE 1: Contoh dari dataset
    # -------------------------------------------------
    if mode == "Ambil contoh dari dataset":

        idx = st.selectbox(
            "Pilih baris contoh transaksi",
            options=sample.index,
            format_func=lambda i: (
                f"Baris {i} — Amount: {sample.loc[i, 'Amount']:.2f} — "
                f"Label asli: {'FRAUD' if sample.loc[i, 'Class'] == 1 else 'Normal'}"
            ),
        )

        row = sample.loc[idx, FEATURE_ORDER].to_dict()
        actual_label = int(sample.loc[idx, "Class"])

        st.dataframe(sample.loc[[idx], FEATURE_ORDER], use_container_width=True)

        if st.button("Prediksi", type="primary"):
            pred, proba = predict_transaction(row, model, scaler)
            st.markdown("#### Hasil Prediksi")
            c1, c2, c3 = st.columns(3)
            c1.metric("Prediksi Model", "🚨 FRAUD" if pred == 1 else "✅ Normal")
            c2.metric("Probabilitas Fraud", f"{proba:.2%}")
            c3.metric("Label Sebenarnya", "🚨 FRAUD" if actual_label == 1 else "✅ Normal")
            if pred == actual_label:
                st.success("Prediksi model sesuai dengan label sebenarnya.")
            else:
                st.warning("Prediksi model berbeda dari label sebenarnya.")

    # -------------------------------------------------
    # MODE 2: Input manual
    # -------------------------------------------------
    elif mode == "Input manual":

        st.caption(
            "`Time` dan `Amount` adalah fitur asli yang mudah dipahami. "
            "`V1`–`V28` adalah hasil transformasi PCA (anonim) — biarkan "
            "default 0 jika tidak tahu nilainya, atau isi manual bila kamu "
            "punya nilai dari data uji."
        )

        c1, c2 = st.columns(2)
        with c1:
            time_val = st.number_input("Time (detik sejak transaksi pertama)", value=50000.0, step=100.0)
        with c2:
            amount_val = st.number_input("Amount", value=100.0, step=10.0, min_value=0.0)

        v_values = {}
        with st.expander("Fitur PCA V1–V28 (opsional, default 0)"):
            v_cols = st.columns(4)
            for i in range(1, 29):
                with v_cols[(i - 1) % 4]:
                    v_values[f"V{i}"] = st.number_input(f"V{i}", value=0.0, step=0.1, key=f"v_{i}")

        if st.button("Prediksi", type="primary"):
            row = {"Time": time_val, "Amount": amount_val, **v_values}
            pred, proba = predict_transaction(row, model, scaler)
            st.markdown("#### Hasil Prediksi")
            c1, c2 = st.columns(2)
            c1.metric("Prediksi Model", "🚨 FRAUD" if pred == 1 else "✅ Normal")
            c2.metric("Probabilitas Fraud", f"{proba:.2%}")
            st.progress(min(max(proba, 0.0), 1.0))

    # -------------------------------------------------
    # MODE 3: Upload CSV batch
    # -------------------------------------------------
    else:
        st.caption(
            "Upload file CSV berisi kolom `Time`, `V1`–`V28`, `Amount` "
            "(boleh tanpa kolom `Class`). Setiap baris akan diprediksi."
        )

        uploaded = st.file_uploader("Upload CSV", type=["csv"])

        if uploaded is not None:
            try:
                df_upload = pd.read_csv(uploaded)
                missing = [c for c in FEATURE_ORDER if c not in df_upload.columns]
                if missing:
                    st.error(f"Kolom berikut tidak ditemukan di file: {missing}")
                else:
                    X = df_upload[FEATURE_ORDER].copy()
                    X_scaled = X.copy()
                    X_scaled[["Amount", "Time"]] = scaler.transform(X_scaled[["Amount", "Time"]])

                    preds = model.predict(X_scaled)
                    probas = model.predict_proba(X_scaled)[:, 1]

                    result = df_upload.copy()
                    result["Prediksi"] = ["FRAUD" if p == 1 else "Normal" for p in preds]
                    result["Probabilitas Fraud"] = probas

                    n_fraud = int((preds == 1).sum())
                    st.success(f"Selesai memprediksi {len(result)} transaksi — {n_fraud} terdeteksi FRAUD.")
                    st.dataframe(result, use_container_width=True)

                    st.download_button(
                        "Download hasil prediksi (CSV)",
                        data=result.to_csv(index=False).encode("utf-8"),
                        file_name="hasil_prediksi_fraud.csv",
                        mime="text/csv",
                    )
            except Exception as e:
                st.error(f"Gagal memproses file: {e}")
