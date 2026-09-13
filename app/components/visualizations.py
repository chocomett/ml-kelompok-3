import streamlit as st

from helpers import format_model_name


def render_confusion_matrices(confusion_dir):
    st.markdown('<div class="section-title">🔍 Confusion Matrix Analysis</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="section-description">
    Confusion Matrix menunjukkan bagaimana setiap model membedakan
    transaksi Normal dan Fraud, termasuk prediksi yang benar dan kesalahan klasifikasi.
    </div>
    """, unsafe_allow_html=True)

    confusion_models = [
        "logistic_original",
        "logistic_smote",
        "random_forest_original",
        "random_forest_smote",
    ]

    cm_col1, cm_col2 = st.columns(2)
    for i, model_name in enumerate(confusion_models):
        image_path = confusion_dir / f"{model_name}.png"
        target_column = cm_col1 if i % 2 == 0 else cm_col2

        with target_column:
            st.markdown(f"### {format_model_name(model_name)}")
            if image_path.exists():
                st.image(str(image_path), width="stretch")
            else:
                st.warning(f"Confusion matrix not found: {image_path.name}")


def render_eda(eda_dir):
    st.markdown('<div class="section-title">📈 Exploratory Data Analysis</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="section-description">
    Sebelum training, dataset dianalisis untuk memahami
    class imbalance, distribusi transaksi, dan hubungan antar fitur.
    </div>
    """, unsafe_allow_html=True)

    eda_images = [
        ("Class Distribution", eda_dir / "01_distribusi_class.png"),
        ("Amount & Time Distribution", eda_dir / "02_distribusi_amount_time.png"),
        ("Amount by Class", eda_dir / "03_boxplot_amount_class.png"),
        ("Correlation Heatmap", eda_dir / "04_correlation_heatmap.png"),
    ]

    eda_col1, eda_col2 = st.columns(2)
    for i, (title, image_path) in enumerate(eda_images):
        target_column = eda_col1 if i % 2 == 0 else eda_col2
        with target_column:
            st.markdown(f"### {title}")
            if image_path.exists():
                st.image(str(image_path), width="stretch")
            else:
                st.warning(f"EDA image not found: {image_path.name}")
