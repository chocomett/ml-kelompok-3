import streamlit as st


def render_hero():
    st.markdown("""
    <div class="hero">
    <div class="hero-badge">
    Machine Learning Classification Project
    </div>
    <div class="hero-title">
    💳 Credit Card Fraud Detection
    </div>
    <div class="hero-description">
    Sistem klasifikasi transaksi kartu kredit untuk membedakan
    <strong>Normal Transaction</strong> dan <strong>Fraud Transaction</strong>
    menggunakan beberapa eksperimen machine learning dan evaluasi performa model.
    </div>
    </div>
    """, unsafe_allow_html=True)
