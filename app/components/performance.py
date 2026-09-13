import streamlit as st

from helpers import format_model_name, get_best_model


def render_performance(comparison_df):
    st.markdown('<div class="section-title">📊 Model Performance</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="section-description">
    Setiap model diuji menggunakan test dataset yang sama.
    Pemilihan model utama menggunakan <strong>F1 Score</strong>
    karena dataset fraud memiliki class imbalance dan kita perlu
    mempertimbangkan keseimbangan antara Precision dan Recall.
    </div>
    """, unsafe_allow_html=True)

    best = get_best_model(comparison_df)
    best_col1, best_col2 = st.columns([1.2, 2])

    with best_col1:
        st.markdown(
            f"""
            <div class="card">
            <div class="card-title">🏆 Best Model</div>
            <h2>{format_model_name(best["model"])}</h2>
            <div class="card-text">Selected based on highest F1 Score.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with best_col2:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy", f'{best["accuracy"]:.4f}')
        m2.metric("Precision", f'{best["precision"]:.4f}')
        m3.metric("Recall", f'{best["recall"]:.4f}')
        m4.metric("F1 Score", f'{best["f1_score"]:.4f}')

    st.markdown("### Full Model Comparison")
    display_comparison = comparison_df.copy()
    display_comparison["model"] = display_comparison["model"].apply(format_model_name)
    st.dataframe(display_comparison, width="stretch", hide_index=True)

    return best
