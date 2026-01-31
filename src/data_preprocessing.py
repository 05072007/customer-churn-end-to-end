import pandas as pd

def load_data(path):
    return pd.read_csv(path)

def preprocess_data(df):
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(inplace=True)

    if 'customerID' in df.columns:
        df.drop('customerID', axis=1, inplace=True)

    df = pd.get_dummies(df, drop_first=True)

    return df
