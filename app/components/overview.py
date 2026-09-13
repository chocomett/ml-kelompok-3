import streamlit as st


def render_overview():
    st.markdown('<div class="section-title">Project Overview</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="section-description">
    Project ini tidak hanya melatih satu model. Dataset diproses,
    diseimbangkan menggunakan SMOTE pada data training tertentu,
    kemudian beberapa model dibandingkan menggunakan metrik klasifikasi.
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="card">
        <div class="card-title">📊 Data Exploration</div>
        <div class="card-text">
        Dataset dianalisis untuk memahami distribusi class,
        transaction amount, time, dan korelasi antar fitur.
        </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="card">
        <div class="card-title">⚙️ Data Preprocessing</div>
        <div class="card-text">
        Data dibersihkan, duplicate dihapus,
        dilakukan train-test split, scaling,
        dan SMOTE hanya pada training data.
        </div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="card">
        <div class="card-title">🧠 Model Evaluation</div>
        <div class="card-text">
        Empat eksperimen model dibandingkan
        menggunakan Accuracy, Precision,
        Recall, F1 Score, dan ROC-AUC.
        </div>
        </div>
        """, unsafe_allow_html=True)
