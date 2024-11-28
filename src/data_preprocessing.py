import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os

def load_data(file_path):
    """ Load dataset from a CSV file. """
    return pd.read_csv(file_path, sep='\t')

def handle_missing_values(df):
    """ Fill missing values with mean (numerical) or mode (categorical). """
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].mean())
    return df

def encode_categorical_data(df):
    """ Encode categorical columns using LabelEncoder. """
    label_encoders = {}
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
    return df, label_encoders

def scale_data(df):
    """ Scale numerical columns using StandardScaler. """
    numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
    if len(numerical_columns) == 0:
        print("No numerical columns to scale.")
        return df
    scaler = StandardScaler()
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
    return df

if __name__ == "__main__":
    data_path = "data/customers.csv"
    df = load_data(data_path)

    if not df.empty:
        df = handle_missing_values(df)
        df, encoders = encode_categorical_data(df)
        df = scale_data(df)

        os.makedirs("data", exist_ok=True)
        df.to_csv("data/customers_preprocessed.csv", index=False)
        print("Data preprocessing completed successfully.")
        print(df.head())
    else:
        print("The loaded dataset is empty.")