import pandas as pd

def save_data(df, file_path):
    """ Save DataFrame to CSV file. """
    df.to_csv(file_path, index=False)

def load_processed_data(file_path):
    """ Load preprocessed data. """
    return pd.read_csv(file_path)