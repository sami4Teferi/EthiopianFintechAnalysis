import pandas as pd
import os

def load_data(file_paths):
    """Load and concatenate multiple review CSV files."""
    if isinstance(file_paths, str):
        file_paths = [file_paths]
    dataframes = [pd.read_csv(path) for path in file_paths]
    return pd.concat(dataframes, ignore_index=True)

def show_duplicates(df):
    """Show duplicate reviews (based on 'review' column)."""
    if 'review' not in df.columns:
        raise ValueError("Column 'review' not found in the DataFrame.")
    duplicates = df[df.duplicated(subset='review', keep=False)]
    return duplicates

def remove_duplicates(df):
    """Remove duplicate reviews (based on 'review' column)."""
    return df.drop_duplicates(subset='review')

def show_missing(df):
    """Show rows with missing values in 'review', 'rating', or 'date'."""
    return df[df[['review', 'rating', 'date']].isnull().any(axis=1)]

def handle_missing(df):
    """Remove rows with missing 'review', 'rating', or 'date'."""
    return df.dropna(subset=['review', 'rating', 'date'])

def normalize_dates(df):
    """Normalize 'date' column to YYYY-MM-DD format and drop invalid dates."""
    df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.strftime('%Y-%m-%d')
    return df.dropna(subset=['date'])

def add_metadata(df, bank_name, source_name='Telegram'):
    """Add 'bank' and 'source' columns."""
    df['bank'] = bank_name
    df['source'] = source_name
    return df
