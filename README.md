# 🤖 ML Model Predictor

A Streamlit web application that helps identify which trained Machine Learning model most likely produced a given set of predictions.  
This tool is designed for **Classification** and **Regression** tasks and supports both built-in datasets and your own uploaded predictions(upcoming)
---

## ✨ Key Features

- **Model Identification**: Predicts which trained model matches your uploaded predictions.
- **Built-in Datasets**: Iris, Wine, Breast Cancer, Diabetes, and Titanic datasets included.
- **Automatic Data Processing**:
  - Cleans missing values.
  - Encodes categorical features.
- **Interactive Training**:
  - Trains multiple ML models (classification or regression).
  - Displays evaluation metrics.
- **CSV Upload Support**:
  - Upload a file with:
    - `y_true`: ground-truth values.
    - `y_pred`: predicted values.
    - Additional columns: feature values used for prediction.
- **Probability Visualization**:
  - Bar chart and pie chart of model probabilities.
- **Continuous Updates**:
  - *"More features coming soon 🚀 — we're improving accuracy with every update!"*

---

## 🛠️ Tech Stack

- **Frontend**: [Streamlit](https://streamlit.io/)
- **Data Processing**: pandas, seaborn
- **Machine Learning**: scikit-learn
- **Visualization**: matplotlib, seaborn

---

## 📂 Project Structure

project-folder/
│
├─ app.py                 # Main Streamlit app
├─ requirements.txt       # List of dependencies
├─ README.md              # Project documentation
└─ src/
    ├─ preprocessing.py   # Data cleaning & preprocessing functions
    ├─ training.py        # Functions to train multiple ML models
    └─ model_identifier.py# Logic to identify most probable model



📊 Usage Guide
1. Choose Task Type
Select Classification or Regression from the sidebar.

2. Pick a Dataset
Choose from built-in datasets or load Titanic via seaborn.

3. Select Target Column
Pick the column you want to predict.
Models will be trained automatically.

4. Upload Predictions
Prepare a CSV with:
y_true: actual values.
y_pred: predicted values.
Any additional feature columns.



---

## 📥 Installation

1. **Clone this repository**:
```bash
git clone https://github.com/nikunjkumar05/model-predictor.git
cd ml-model-predictor


"We’re constantly improving this app — stay tuned for more updates!"
