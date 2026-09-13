import random

import pandas as pd
import streamlit as st


def get_random_transaction(df: pd.DataFrame, transaction_type: str = "Random"):
    if "Class" not in df.columns:
        raise ValueError("Dataset tester harus memiliki kolom 'Class'.")

    if transaction_type == "Normal":
        filtered = df[df["Class"] == 0]
    elif transaction_type == "Fraud":
        filtered = df[df["Class"] == 1]
    else:
        filtered = df

    if filtered.empty:
        raise ValueError(f"Tidak ada transaksi untuk tipe '{transaction_type}'.")

    # Hindari langsung mendapatkan transaksi yang sama pada mode yang sama.
    state_key = f"last_transaction_{transaction_type.lower()}_index"
    previous_index = st.session_state.get(state_key)

    candidates = filtered
    if previous_index is not None and len(filtered) > 1:
        candidates = filtered.drop(index=previous_index, errors="ignore")

    selected_index = random.choice(candidates.index.tolist())
    st.session_state[state_key] = selected_index

    return candidates.loc[selected_index]


def get_best_model(comparison: pd.DataFrame):
    return comparison.loc[comparison["f1_score"].idxmax()]


def format_model_name(name):
    return str(name).replace("_", " ").title()
