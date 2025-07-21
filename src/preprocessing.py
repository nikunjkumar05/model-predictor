import pandas as pd
from sklearn.preprocessing import StandardScaler


def clean_data(df):
    # Handle missing values and drop duplicates.
    df=df.drop_duplicates()
    df=df.fillna(df.mean(numeric_only=True))
    return df

def preprocess_features(df, target_col):
    # Split features and target, then scale features.
    X=df.drop(columns=[target_col])
    y=df[target_col]

    scaler=StandardScaler()
    X_scaled=scaler.fit_transform(X)
    return X_scaled,y
