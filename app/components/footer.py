import streamlit as st


def render_footer():
    st.markdown("---")
    st.markdown("""
    <div style="text-align:center; color:#6b7280; padding:1rem;">
    Credit Card Fraud Detection • Machine Learning Project
    </div>
    """, unsafe_allow_html=True)
