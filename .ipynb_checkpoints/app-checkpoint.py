import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import (
    load_iris, load_wine, load_breast_cancer, load_digits,
    fetch_california_housing, load_diabetes
)
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier,
    RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score, log_loss, mean_squared_error
import warnings
warnings.filterwarnings("ignore")

st.title("ML Model Predictor - GridSearch & Cross-Validation Edition")

task_type = st.radio("Select task type:", ["Classification", "Regression"])

# ----------------- Model parameter grids -----------------
classification_param_grids = {
    "LogisticRegression": {"C": [0.1, 1, 10]},
    "SVM (RBF)": {"C": [0.1, 1, 10], "gamma": ["scale", "auto"]},
    "Linear SVM": {"C": [0.1, 1, 10]},
    "RandomForest": {"n_estimators": [50, 100], "max_depth": [None, 10]},
    "GradientBoosting": {"n_estimators": [50, 100], "learning_rate": [0.05, 0.1]},
    "AdaBoost": {"n_estimators": [50, 100]},
    "KNN": {"n_neighbors": [3, 5, 7]},
    "NaiveBayes": {},
    "DecisionTree": {"max_depth": [None, 5, 10]},
    "XGBoost": {"n_estimators": [50, 100], "learning_rate": [0.05, 0.1]}
}

regression_param_grids = {
    "LinearRegression": {},
    "SVR": {"C": [0.1, 1, 10], "kernel": ["rbf", "linear"]},
    "RandomForestRegressor": {"n_estimators": [50, 100], "max_depth": [None, 10]},
    "GradientBoostingRegressor": {"n_estimators": [50, 100], "learning_rate": [0.05, 0.1]},
    "AdaBoostRegressor": {"n_estimators": [50, 100]},
    "KNNRegressor": {"n_neighbors": [3, 5, 7]},
    "DecisionTreeRegressor": {"max_depth": [None, 5, 10]},
    "XGBRegressor": {"n_estimators": [50, 100], "learning_rate": [0.05, 0.1]}
}

# ----------------- Datasets -----------------
datasets = {
    "Iris": load_iris(as_frame=True),
    "Wine": load_wine(as_frame=True),
    "Breast Cancer": load_breast_cancer(as_frame=True),
    "Digits": load_digits(as_frame=True),
    "California Housing": fetch_california_housing(as_frame=True),
    "Diabetes": load_diabetes(as_frame=True),
}

dataset_name = st.selectbox("Choose a dataset", list(datasets.keys()))
dataset = datasets[dataset_name]
df = dataset.frame.copy()

# For regression datasets, bin targets if task = Classification
if task_type == "Classification" and dataset_name in ["California Housing", "Diabetes"]:
    df["target"] = pd.qcut(df["target"], q=3, labels=False)

target_col = st.selectbox("Select target column", df.columns)
X = df.drop(columns=[target_col]).values
y = df[target_col].values

# Preprocess
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ----------------- Train Models with GridSearch + CV -----------------
trained_models = {}
if task_type == "Classification":
    for name, params in classification_param_grids.items():
        base_model = eval(name.replace(" ", ""))() if name != "NaiveBayes" else GaussianNB()
        grid = GridSearchCV(base_model, params, cv=3, n_jobs=-1) if params else base_model
        try:
            if params:
                grid.fit(X_train, y_train)
                best_model = grid.best_estimator_
            else:
                base_model.fit(X_train, y_train)
                best_model = base_model
            cv_score = cross_val_score(best_model, X_train, y_train, cv=3).mean()
            trained_models[name] = best_model
        except:
            continue
    st.success(f"{len(trained_models)} classification models trained with GridSearchCV!")

else:
    for name, params in regression_param_grids.items():
        base_model = eval(name)()
        grid = GridSearchCV(base_model, params, cv=3, n_jobs=-1) if params else base_model
        try:
            if params:
                grid.fit(X_train, y_train)
                best_model = grid.best_estimator_
            else:
                base_model.fit(X_train, y_train)
                best_model = base_model
            cv_score = cross_val_score(best_model, X_train, y_train, cv=3).mean()
            trained_models[name] = best_model
        except:
            continue
    st.success(f"{len(trained_models)} regression models trained with GridSearchCV!")

# ----------------- Upload Predictions -----------------
uploaded_file = st.file_uploader("Upload CSV with predictions")

if uploaded_file is not None:
    user_df = pd.read_csv(uploaded_file)
    y_true = user_df["y_true"].values
    y_pred = user_df["y_pred"].values
    X_user = user_df.drop(columns=["y_true", "y_pred"] + [c for c in user_df.columns if c.startswith("prob_")], errors="ignore").values

    scores = {}
    if task_type == "Classification":
        for name, model in trained_models.items():
            model_preds = model.predict(X_user)
            model_probs = model.predict_proba(X_user)
            acc = accuracy_score(y_true, model_preds)
            try:
                ll = log_loss(y_true, model_probs)
            except:
                ll = np.inf
            scores[name] = acc - 0.1 * ll

    else:  # Regression
        for name, model in trained_models.items():
            model_preds = model.predict(X_user)
            rmse = np.sqrt(mean_squared_error(y_true, model_preds))
            scores[name] = -rmse  # lower RMSE = better

    total = sum(np.exp(list(scores.values())))
    probabilities = {k: np.exp(v) / total for k, v in scores.items()}
    predicted_model = max(probabilities, key=probabilities.get)
    st.success(f"Predicted {task_type.lower()} model: **{predicted_model}**")

    fig, ax = plt.subplots()
    ax.pie(probabilities.values(), labels=probabilities.keys(), autopct='%1.1f%%')
    ax.set_title("Model Prediction Probabilities")
    st.pyplot(fig)
