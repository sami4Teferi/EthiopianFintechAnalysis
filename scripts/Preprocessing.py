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

def save_cleaned_reviews(df, output_path):
    """Save cleaned DataFrame to a CSV file."""
    df.to_csv(output_path, index=False)

def preprocess_multiple_files(bank_file_dict):
    """
    Load multiple CSVs from bank_file_dict {bank_name: file_path},
    add metadata, concatenate, and run full cleaning pipeline.
    
    Returns cleaned DataFrame.
    """
    dfs = []
    for bank, file_path in bank_file_dict.items():
        if not os.path.exists(file_path):
            print(f"Warning: File not found for {bank}: {file_path}. Skipping.")
            continue
        df = pd.read_csv(file_path)
        df = add_metadata(df, bank_name=bank)
        dfs.append(df)

    if not dfs:
        raise ValueError("No valid files loaded.")
    
    combined_df = pd.concat(dfs, ignore_index=True)

    print(f"Initial combined data shape: {combined_df.shape}")
    print(f"Showing duplicates:")
    duplicates = show_duplicates(combined_df)
    print(duplicates)

    combined_df = remove_duplicates(combined_df)
    print(f"Shape after removing duplicates: {combined_df.shape}")

    print(f"Showing missing values:")
    missing = show_missing(combined_df)
    print(missing)

    combined_df = handle_missing(combined_df)
    print(f"Shape after handling missing values: {combined_df.shape}")

    combined_df = normalize_dates(combined_df)
    print(f"Shape after normalizing dates: {combined_df.shape}")

    return combined_df
