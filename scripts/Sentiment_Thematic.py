# Imports
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
import spacy
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

# Append scripts path for custom functions if needed
sys.path.append('../scripts')

# Load your custom functions (make sure these are implemented properly in Sentiment_Thematic.py)
from Sentiment_Thematic import load_cleaned_data

# Load Spacy model
nlp = spacy.load("en_core_web_sm")

# 1. Load cleaned data from CSV files
def load_data(cleaned_files):
    print("Loading cleaned data...")
    bank_dfs = load_cleaned_data(cleaned_files)
    for bank, df in bank_dfs.items():
        print(f"Loaded {bank} cleaned data: {df.shape}")
    return bank_dfs

# 2. Sentiment analysis using DistilBERT pipeline
def sentiment_analysis(df, model):
    print("Running sentiment analysis...")
    results = model(list(df['review_text']), truncation=True)
    df['sentiment_label'] = [res['label'] for res in results]
    # Convert scores to positive sentiment probability (assuming binary classification: POSITIVE/NEGATIVE)
    df['sentiment_score'] = [res['score'] if res['label']=='POSITIVE' else 1 - res['score'] for res in results]
    return df

# 3. Extract top keywords with TF-IDF
def extract_keywords(df, top_n=20):
    print("Extracting keywords using TF-IDF...")
    # Preprocess text for vectorization
    processed = df['review_text'].apply(preprocess_text)
    vectorizer = TfidfVectorizer(max_features=top_n, stop_words='english', ngram_range=(1,2))
    X = vectorizer.fit_transform(processed)
    keywords = vectorizer.get_feature_names_out()
    scores = X.sum(axis=0).A1
    keyword_scores = sorted(zip(keywords, scores), key=lambda x: x[1], reverse=True)
    return keyword_scores

# Text preprocessing (lemmatization, lowercasing, removing stop words and non-alpha tokens)
def preprocess_text(text):
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    return " ".join(tokens)

# 4. Assign themes to each review based on keywords
def assign_themes(text, themes):
    assigned = []
    text_lower = text.lower()
    for theme, keywords in themes.items():
        if any(kw in text_lower for kw in keywords):
            assigned.append(theme)
    return assigned if assigned else ["Other"]

# 5. Save final DataFrame to CSV with required columns
def save_to_csv(df, filename):
    print(f"Saving results to {filename}...")
    df_to_save = df[['review_text', 'sentiment_label', 'sentiment_score', 'identified_themes']]
    df_to_save.to_csv(filename, index=False)
    print("Saved successfully.")

# 6. Aggregate sentiment score by bank and rating
def aggregate_by_bank_rating(df):
    print("Aggregating sentiment scores by bank and rating...")
    agg_df = df.groupby(['bank_name', 'rating'])['sentiment_score'].mean().reset_index()
    return agg_df

#