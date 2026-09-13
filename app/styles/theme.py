import streamlit as st


def load_styles():
    st.markdown("""
    <style>
    .stApp {
        background:
            radial-gradient(
                circle at top left,
                rgba(36, 99, 235, 0.12),
                transparent 30%
            ),
            #0e1117;
    }

    .block-container {
        max-width: 1250px;
        padding-top: 3rem;
        padding-bottom: 5rem;
    }

    .hero {
        padding: 3rem 0 2.5rem 0;
    }

    .hero-badge {
        display: inline-block;
        padding: 0.4rem 0.9rem;
        border-radius: 999px;
        background: rgba(59, 130, 246, 0.15);
        border: 1px solid rgba(59, 130, 246, 0.35);
        color: #93c5fd;
        font-size: 0.85rem;
        margin-bottom: 1rem;
    }

    .hero-title {
        font-size: 3.5rem;
        font-weight: 800;
        letter-spacing: -2px;
        margin-bottom: 1rem;
    }

    .hero-description {
        font-size: 1.1rem;
        color: #9ca3af;
        max-width: 800px;
        line-height: 1.7;
    }

    .section-title {
        font-size: 2rem;
        font-weight: 750;
        margin-top: 4rem;
        margin-bottom: 0.5rem;
    }

    .section-description {
        color: #9ca3af;
        font-size: 1rem;
        margin-bottom: 1.8rem;
    }

    .card {
        background: rgba(255,255,255,0.035);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 18px;
        padding: 1.5rem;
        min-height: 150px;
    }

    .card-title {
        font-size: 1.05rem;
        font-weight: 700;
        margin-bottom: 0.7rem;
    }

    .card-text {
        color: #9ca3af;
        line-height: 1.6;
    }

    .result-normal {
        background: rgba(34, 197, 94, 0.10);
        border: 1px solid rgba(34, 197, 94, 0.35);
        border-radius: 18px;
        padding: 1.5rem;
    }

    .result-fraud {
        background: rgba(239, 68, 68, 0.10);
        border: 1px solid rgba(239, 68, 68, 0.35);
        border-radius: 18px;
        padding: 1.5rem;
    }

    .metric-card {
        background: rgba(255,255,255,0.035);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 1.2rem;
        text-align: center;
    }

    .metric-label {
        color: #9ca3af;
        font-size: 0.85rem;
    }

    .metric-value {
        font-size: 1.7rem;
        font-weight: 750;
        margin-top: 0.3rem;
    }

    hr {
        border-color: rgba(255,255,255,0.08);
    }
    </style>
    """, unsafe_allow_html=True)
