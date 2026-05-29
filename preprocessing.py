import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os


def load_and_preprocess(filepath="data/heart_disease.csv"):
    df = pd.read_csv(filepath)
    df = df.dropna().drop_duplicates().reset_index(drop=True)

    X = df.drop("heart_disease", axis=1)
    y = df["heart_disease"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_sc = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_sc  = pd.DataFrame(scaler.transform(X_test),  columns=X_test.columns)

    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/scaler.pkl")

    return (
        X_train_sc, X_test_sc,
        y_train.reset_index(drop=True),
        y_test.reset_index(drop=True),
    )
