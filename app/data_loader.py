from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models" / "experiments"
OUTPUT_DIR = BASE_DIR / "outputs"
EDA_DIR = OUTPUT_DIR / "eda"
EVALUATION_DIR = OUTPUT_DIR / "evaluation"
CONFUSION_DIR = EVALUATION_DIR / "confusion_matrix"

TEST_PATH = DATA_DIR / "test.csv"
COMPARISON_PATH = EVALUATION_DIR / "model_comparison.csv"


@st.cache_data
def load_test_data():
    return pd.read_csv(TEST_PATH)


@st.cache_data
def load_model_comparison():
    return pd.read_csv(COMPARISON_PATH)


@st.cache_resource
def load_model(model_name):
    model_path = MODEL_DIR / f"{model_name}.joblib"
    return joblib.load(model_path)
