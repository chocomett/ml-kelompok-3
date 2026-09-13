import streamlit as st


def render_experiments():
    st.markdown('<div class="section-title">🧠 Model Experiments</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="section-description">
    Untuk menentukan pendekatan terbaik, kami tidak langsung memilih satu model.
    Kami menguji <strong>4 eksperimen</strong> dengan kombinasi algoritma
    dan kondisi data training yang berbeda.
    </div>
    """, unsafe_allow_html=True)

    cards = [
        ("Logistic Regression", "Original Training Data"),
        ("Logistic Regression", "SMOTE Training Data"),
        ("Random Forest", "Original Training Data"),
        ("Random Forest", "SMOTE Training Data"),
    ]

    cols = st.columns(4)
    for col, (title, subtitle) in zip(cols, cards):
        with col:
            st.markdown(
                f"""
                <div class="card">
                <div class="card-title">{title}</div>
                <div class="card-text">{subtitle}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
