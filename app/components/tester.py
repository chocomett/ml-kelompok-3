import pandas as pd
import streamlit as st

from data_loader import load_model
from helpers import format_model_name, get_random_transaction


def render_tester(test_df, best_model_name):
    st.markdown('<div class="section-title">🧪 Test a Transaction</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="section-description">
    Uji model menggunakan transaksi nyata dari sample test dataset.
    Tidak perlu memilih index secara manual. Sistem akan mengambil
    transaksi secara acak.
    </div>
    """, unsafe_allow_html=True)

    test_col1, test_col2 = st.columns([1, 2])

    with test_col1:
        transaction_mode = st.radio(
            "Transaction sample",
            ["Random Transaction", "Random Normal", "Random Fraud"],
            index=0,
        )

    with test_col2:
        st.markdown(
            f"""
            <div class="card">
            <div class="card-title">🏆 Prediction Model</div>
            <div class="card-text">
            Live prediction menggunakan model terbaik berdasarkan
            <strong>F1 Score</strong>:
            <br><br>
            <strong>{format_model_name(best_model_name)}</strong>
            </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if "current_transaction" not in st.session_state:
        st.session_state.current_transaction = None

    if st.button("🎲 Generate Transaction", width="stretch"):
        if transaction_mode == "Random Normal":
            selected_type = "Normal"
        elif transaction_mode == "Random Fraud":
            selected_type = "Fraud"
        else:
            selected_type = "Random"

        try:
            st.session_state.current_transaction = get_random_transaction(
                test_df, selected_type
            )
        except ValueError as exc:
            st.error(str(exc))
            return

    if st.session_state.current_transaction is None:
        return

    transaction = st.session_state.current_transaction
    X_transaction = transaction.drop(labels=["Class"], errors="ignore")
    X_transaction = pd.DataFrame([X_transaction])

    try:
        model = load_model(best_model_name)
        prediction = model.predict(X_transaction)[0]

        probability = None
        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(X_transaction)[0][1]

        st.markdown("<br>", unsafe_allow_html=True)

        if prediction == 1:
            st.markdown("""
            <div class="result-fraud">
            <h2>🚨 FRAUD DETECTED</h2>
            Model memprediksi transaksi ini sebagai
            <strong>Fraud Transaction</strong>.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="result-normal">
            <h2>✅ NORMAL TRANSACTION</h2>
            Model memprediksi transaksi ini sebagai
            <strong>Normal Transaction</strong>.
            </div>
            """, unsafe_allow_html=True)

        metric1, metric2, metric3 = st.columns(3)
        actual_label = "Fraud" if transaction["Class"] == 1 else "Normal"
        predicted_label = "Fraud" if prediction == 1 else "Normal"

        with metric1:
            st.metric("Actual Class", actual_label)
        with metric2:
            st.metric("Model Prediction", predicted_label)
        with metric3:
            st.metric("Fraud Probability", f"{probability:.2%}" if probability is not None else "N/A")

        if prediction == transaction["Class"]:
            st.success("Prediction Correct — model prediction matches the actual label.")
        else:
            st.warning("Prediction Incorrect — model prediction does not match the actual label.")

        st.markdown("### Transaction Information")
        info_columns = [col for col in ["Time", "Amount"] if col in transaction.index]
        if info_columns:
            display_info = transaction[info_columns].to_frame().T
            st.dataframe(display_info, width="stretch", hide_index=True)

        with st.expander("View all transaction features"):
            full_transaction = transaction.to_frame().T
            st.dataframe(full_transaction, width="stretch", hide_index=True)

    except FileNotFoundError:
        st.error(f"Model file '{best_model_name}.joblib' was not found.")
    except Exception as exc:
        st.error(f"Prediction gagal: {exc}")
