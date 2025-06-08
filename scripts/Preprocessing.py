import pandas as pd

def load_data(csv_path):
    """
    Load review CSV into DataFrame.
    """
    df = pd.read_csv(csv_path)
    df.columns = [col.lower().strip() for col in df.columns]
    return df

def show_duplicates(df):
    """
    Show duplicate reviews (based on 'review' column).
    """
    if 'review' not in df.columns:
        raise ValueError("Column 'review' not found in the DataFrame.")
    duplicates = df[df.duplicated(subset='review')]
    return duplicates

def remove_duplicates(df):
    """
    Remove duplicate reviews (based on 'review' column).
    """
    return df.drop_duplicates(subset='review')

def show_missing(df):
    """
    Show rows with missing values in 'review', 'rating', or 'date'.
    """
    return df[df[['review', 'rating', 'date']].isnull().any(axis=1)]

def handle_missing(df):
    """
    Remove rows with missing 'review', 'rating', or 'date'.
    """
    return df.dropna(subset=['review', 'rating', 'date'])

def normalize_dates(df):
    """
    Normalize 'date' column to YYYY-MM-DD format and drop invalid dates.
    """
    df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.date
    return df.dropna(subset=['date'])

def add_metadata(df, bank_name, source_name='Telegram'):
    """
    Add 'bank' and 'source' columns.
    """
    df['bank'] = bank_name
    df['source'] = source_name
    return df

def save_cleaned_reviews(df, output_path):
    """
    Save cleaned DataFrame to a CSV file.
    """
    df.to_csv(output_path, index=False)
