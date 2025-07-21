import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_diabetes
from src.preprocessing import clean_data, preprocess_features
from src.training import train_models
from src.model_identifier import identify_model

st.set_page_config(page_title="ML Model Predictor", layout="wide")

#Header
st.markdown("## ML Model Predictor")
st.caption("Identify which trained model produced uploaded predictions")
st.markdown("---")
st.info("✨**More features coming soon!** We’re working on improving accuracy with every update.")

#Sidebar
st.sidebar.header("Settings")
task_type = st.sidebar.radio("Select task type", ["Classification", "Regression"])

dataset_name = st.sidebar.selectbox("Choose a dataset", ["Iris", "Wine", "Breast Cancer", "Diabetes", "Titanic"])

#Load Dataset
datasets = {
    "Iris": load_iris(as_frame=True),
    "Wine": load_wine(as_frame=True),
    "Breast Cancer": load_breast_cancer(as_frame=True),
    "Diabetes": load_diabetes(as_frame=True) }

df = datasets[dataset_name].frame.copy()

#Clean & Preprocess
df = clean_data(df)
target_col = st.sidebar.selectbox("Select target column", df.columns)
X, y = preprocess_features(df, target_col)

# Train Models 
with st.spinner("Training models..."):
    trained_models, X_test, y_test = train_models(X, y, task_type)

if len(trained_models)==0:
    st.error("❌ Choose an appropriate column for the selected task!")
else:
    st.success("✅ Models trained successfully! Upload your predictions below.")
    st.subheader("📂 Upload Predictions")
    uploaded_file =st.file_uploader("Upload a CSV file with `y_true`, `y_pred`, and features", type="csv")
    st.write("**y_true**: actual values")
    st.write("**y_pred**: predicted values")
    st.write("**features**: other columns")
    if uploaded_file:
        user_df =pd.read_csv(uploaded_file)
        if "y_true" not in user_df.columns or "y_pred" not in user_df.columns:
            st.error("CSV must contain `y_true` and `y_pred` columns.")
        else:
            y_true =user_df["y_true"].values
            y_pred_upload =user_df["y_pred"].values
            X_user =user_df.drop(columns=["y_true", "y_pred"], errors="ignore").values

            # Identify model
            predicted_model, probabilities = identify_model(
                trained_models, X_user, y_true, y_pred_upload, task_type, temperature=1.0
            )
            st.success(f"**Predicted model:** {predicted_model}")

            # Visualization
            col1, col2 = st.columns(2)
            with col1:
                fig, ax = plt.subplots(figsize=(6, 3))
                ax.bar(probabilities.keys(), probabilities.values(), color="dodgerblue")
                ax.set_title("Model Prediction Probabilities (Bar Chart)")
                ax.set_ylabel("Probability")
                plt.xticks(rotation=30, ha='right')
                st.pyplot(fig)

            with col2:
                fig2, ax2 = plt.subplots()
                ax2.pie(probabilities.values(), labels=probabilities.keys(), autopct='%1.1f%%')
                ax2.set_title("Model Prediction Probabilities (Pie Chart)")
                st.pyplot(fig2)

st.markdown("---")
st.caption("ML Model Predictor BY : Nikunj")
