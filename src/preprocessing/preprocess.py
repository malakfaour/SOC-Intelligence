import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


def load_data(path, nrows=50000):  # 🔥 LIMIT DATA
    df = pd.read_csv(path, nrows=nrows)
    return df

def handle_missing_values(df):
    # Fill numeric with median
    for col in df.select_dtypes(include=np.number).columns:
        df[col] = df[col].fillna(df[col].median())

    # Fill categorical with "unknown"
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].fillna("unknown")

    return df

def encode_categorical(df, target_column):
    encoders = {}
    
    for col in df.select_dtypes(include='object').columns:
        if col == target_column:
            continue  # skip target
        
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    return df, encoders



def scale_features(df):
    scaler = StandardScaler()
    
    numeric_cols = df.select_dtypes(include=np.number).columns
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    return df, scaler

def split_features_labels(df, target_columns):
    X = df.drop(columns=target_columns)
    y = df[target_columns]

    return X, y

def split_data(X, y, test_size=0.3):
    return train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)

def preprocess_pipeline(path, target_columns):
    df = load_data(path, nrows=50000)

    df = handle_missing_values(df)

    # 🔥 Separate BEFORE encoding/scaling
    X, y = split_features_labels(df, target_columns)

    X, encoders = encode_categorical(X, target_columns)

    scaler = None

    return X, y, encoders, scaler
