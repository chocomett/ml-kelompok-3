import streamlit as st

from components.experiments import render_experiments
from components.footer import render_footer
from components.hero import render_hero
from components.overview import render_overview
from components.performance import render_performance
from components.tester import render_tester
from components.visualizations import render_confusion_matrices, render_eda
from data_loader import (
    COMPARISON_PATH,
    CONFUSION_DIR,
    EDA_DIR,
    load_model_comparison,
    load_test_data,
)
from helpers import get_best_model
from styles.theme import load_styles


st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="collapsed",
)

load_styles()

try:
    test_df = load_test_data()
    comparison_df = load_model_comparison()
except FileNotFoundError as exc:
    st.error(f"Required project file was not found:\n\n{exc}")
    st.stop()

best_model_name = get_best_model(comparison_df)["model"]

render_hero()
render_overview()
render_tester(test_df, best_model_name)
render_experiments()
render_performance(comparison_df)
render_confusion_matrices(CONFUSION_DIR)
render_eda(EDA_DIR)
render_footer()
