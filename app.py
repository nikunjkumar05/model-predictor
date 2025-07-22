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
st.sidebar.markdown("### ⚙️ Settings")
task_type = st.sidebar.radio("Task Type", ["Classification", "Regression"], help="Choose model type")

dataset_name = st.sidebar.selectbox(
    "Dataset",
    ["Iris", "Wine", "Breast Cancer", "Diabetes"],
    help="Select a dataset for training"
)

#Load Dataset
datasets = {
    "Iris": load_iris(as_frame=True),
    "Wine": load_wine(as_frame=True),
    "Breast Cancer": load_breast_cancer(as_frame=True),
    "Diabetes": load_diabetes(as_frame=True)
}

df = datasets[dataset_name].frame.copy()

#Clean & Preprocess
df = clean_data(df)
target_col = st.sidebar.selectbox("Select target column", df.columns)
X, y = preprocess_features(df, target_col)


#Demo CSV Downloads
st.subheader("📂 Download Demo Files")
cols = st.columns(2)
demo_files = {
    "Iris (SVC)[target]": "demo_files/iris_SVC.csv",
    "Wine (RandomForest)[target]": "demo_files/wine_randomforest_classfier.csv",
    "Diabetes (LinearRegression)[target]": "demo_files/diabetes_linearregression.csv"
}
for idx, (name, path) in enumerate(demo_files.items()):
    with cols[idx % 2]:
            with open(path, "rb") as f:
                st.download_button(label=name,data=f,file_name=path.split("/")[-1],mime="text/csv")

#Train Models
with st.spinner("Training models..."):
    trained_models, X_test, y_test = train_models(X, y, task_type)

if len(trained_models) == 0:
    st.error("❌ Choose an appropriate column for the selected task!")
else:
    st.success("✅ Models trained successfully! Upload your predictions below.")

    #Upload Section
    st.subheader("📤 Upload Predictions")
    uploaded_file = st.file_uploader("Upload a CSV with `y_true`, `y_pred`, and features", type="csv")
    st.caption("**y_true**: actual values | **y_pred**: predicted values | other columns: features")

    if uploaded_file:
        user_df = pd.read_csv(uploaded_file)
        st.write("### Preview of uploaded data:")
        st.dataframe(user_df.head(), use_container_width=True)

        if "y_true" not in user_df.columns or "y_pred" not in user_df.columns:
            st.error("CSV must contain `y_true` and `y_pred` columns.")
        else:
            y_true = user_df["y_true"].values
            y_pred_upload = user_df["y_pred"].values
            X_user = user_df.drop(columns=["y_true", "y_pred"], errors="ignore").values

            # Identify model
            if(task_type!="Classification"):
                st.write("Beta Version")
            predicted_model, probabilities = identify_model(
                trained_models, X_user, y_true, y_pred_upload, task_type, temperature=1.0
            )
            st.success(f"**Predicted model:** {predicted_model}")

            # Visualization
            col1, col2 = st.columns(2)
            with col1:
                fig, ax = plt.subplots(figsize=(6, 3))
                ax.bar(probabilities.keys(), probabilities.values(), color="skyblue")
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
st.caption("ML Model Predictor © 2025 – Developed by Nikunj")
