import streamlit as st
import pandas as pd
import joblib
import time

# =========================================================
# 1. PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="💳 Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

# =========================================================
# 2. LOAD MODEL
# =========================================================
@st.cache_resource
def load_model():
    return joblib.load("THE_xgb_Champion_model.pkl")

model = load_model()

# =========================================================
# 3. SIDEBAR
# =========================================================
st.sidebar.title("💳 Fraud Detection App")
st.sidebar.markdown("""
Upload a CSV file with transaction data.

**Important**
- Same features used during training
- Target column not required
- Fraud threshold: **0.29**
""")

st.sidebar.markdown("### 🧠 Model Info")
st.sidebar.write("Model: XGBoost Classifier")
st.sidebar.write("Metric: ROC-AUC ≈ 0.94")
st.sidebar.write("Decision Threshold: 0.29")

uploaded_file = st.sidebar.file_uploader("📂 Upload CSV", type="csv")

# =========================================================
# 4. MAIN TITLE
# =========================================================
st.title("💳 Credit Card Fraud Detection System")
st.write(
    "This application detects **potentially fraudulent transactions** using "
    "a trained machine learning model and converts predictions into "
    "**actionable business insights**."
)

# =========================================================
# 5. PROCESS FILE
# =========================================================
if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("📋 Uploaded Data Preview")
    st.dataframe(df.head(), height=200)

    # =====================================================
    # 6. FEATURE VALIDATION
    # =====================================================
    expected_features = model.feature_names_in_
    missing_features = set(expected_features) - set(df.columns)

    if missing_features:
        st.error(f"❌ Missing required columns: {missing_features}")
        st.stop()

    df = df[expected_features]

    # =====================================================
    # 7. PREDICTION LOADING
    # =====================================================
    st.subheader("⚡ Running Fraud Detection...")
    progress = st.progress(0)
    for i in range(100):
        time.sleep(0.01)
        progress.progress(i + 1)

    # =====================================================
    # 8. MODEL PREDICTIONS
    # =====================================================
    fraud_probability = model.predict_proba(df)[:, 1]
    threshold = 0.29
    predicted_class = (fraud_probability >= threshold).astype(int)

    df["fraud_probability"] = fraud_probability
    df["predicted_class"] = predicted_class

    # =====================================================
    # 9. LABEL GENERATION
    # =====================================================
    def risk_level(p):
        if p >= 0.6:
            return "High Risk"
        elif p >= threshold:
            return "Medium Risk"
        else:
            return "Low Risk"

    def confidence_level(p):
        if p >= 0.7:
            return "High Confidence"
        elif p >= 0.4:
            return "Medium Confidence"
        else:
            return "Low Confidence"

    def severity_level(p):
        if p >= 0.85:
            return "Critical"
        elif p >= 0.6:
            return "High"
        elif p >= threshold:
            return "Moderate"
        else:
            return "Low"

    df["risk_level"] = df["fraud_probability"].apply(risk_level)
    df["confidence_level"] = df["fraud_probability"].apply(confidence_level)
    df["fraud_severity"] = df["fraud_probability"].apply(severity_level)

    # =====================================================
    # 10. BUSINESS ACTION
    # =====================================================
    def recommended_action(row):
        if row["fraud_severity"] in ["Critical", "High"]:
            return "🚫 Block Transaction"
        elif row["fraud_severity"] == "Moderate":
            return "🔍 Manual Review"
        else:
            return "✅ Approve Transaction"

    df["recommended_action"] = df.apply(recommended_action, axis=1)

    # =====================================================
    # 11. SUMMARY METRICS
    # =====================================================
    st.subheader("📊 Fraud Detection Summary")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Transactions", len(df))
    col2.metric("Fraud Detected", int(df["predicted_class"].sum()))
    col3.metric("High / Critical Risk", int(df["fraud_severity"].isin(["High", "Critical"]).sum()))
    col4.metric("Manual Review Needed", int((df["recommended_action"] == "🔍 Manual Review").sum()))

    # =====================================================
    # 12. VISUALS
    # =====================================================
    st.subheader("📈 Risk Distribution")
    st.bar_chart(df["fraud_severity"].value_counts())

    # =====================================================
    # 13. HIGHLIGHT ROWS
    # =====================================================
    st.subheader("🔍 Detailed Predictions")

    def highlight_rows(row):
        if row["fraud_severity"] == "Critical":
            return ["background-color: #B71C1C; color: white"] * len(row)
        elif row["fraud_severity"] == "High":
            return ["background-color: #FF6F61"] * len(row)
        elif row["fraud_severity"] == "Moderate":
            return ["background-color: #FFE082"] * len(row)
        else:
            return [""] * len(row)

    st.dataframe(
        df.style.apply(highlight_rows, axis=1),
        height=450
    )

    # =====================================================
    # 14. DOWNLOAD RESULTS
    # =====================================================
    st.download_button(
        "⬇️ Download Prediction Results",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="fraud_detection_results.csv",
        mime="text/csv"
    )

else:
    st.info("📌 Upload a CSV file to start fraud detection.")

# =========================================================
# 15. FOOTER
# =========================================================
st.markdown("---")
st.markdown("Built by **Uday Kumar** | Data Science & Machine Learning Project")
