

import pandas as pd
import os

def load_data(file_paths):
    """Load and concatenate multiple review CSV files."""
    if isinstance(file_paths, str):
        file_paths = [file_paths]
    dataframes = [pd.read_csv(path) for path in file_paths]
    return pd.concat(dataframes, ignore_index=True) if len(dataframes) > 1 else dataframes[0]

def show_duplicates(df):
    """Show duplicate reviews (based on 'review_text' column)."""
    if 'review_text' not in df.columns:
        raise ValueError("Column 'review_text' not found in the DataFrame.")
    duplicates = df[df.duplicated(subset='review_text', keep=False)]
    return duplicates

def remove_duplicates(df):
    """Remove duplicate reviews (based on 'review_text' column)."""
    if 'review_text' not in df.columns:
        raise ValueError("Column 'review_text' not found in the DataFrame.")
    return df.drop_duplicates(subset='review_text')

def show_missing(df):
    """Show rows with missing values in 'review_text', 'rating', or 'date'."""
    return df[df[['review_text', 'rating', 'date']].isnull().any(axis=1)]

def handle_missing(df):
    """Remove rows with missing 'review_text', 'rating', or 'date'."""
    return df.dropna(subset=['review_text', 'rating', 'date'])

def normalize_dates(df):
    """Normalize 'date' column to YYYY-MM-DD format and drop invalid dates."""
    df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.strftime('%Y-%m-%d')
    return df.dropna(subset=['date'])

def add_metadata(df, bank_name, source_name='Telegram'):
    """Add 'bank' and 'source' columns."""
    df['bank'] = bank_name
    df['source'] = source_name
    return df

def save_cleaned_reviews(df, output_path):
    """Save cleaned DataFrame to a CSV file."""
    df.to_csv(output_path, index=False)
