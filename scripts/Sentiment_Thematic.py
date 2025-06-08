# Sentiment_Thematic.py

import os
import pandas as pd
import spacy
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import re

# Load spaCy model globally (only once)
nlp = spacy.load("en_core_web_sm")

def load_cleaned_data(file_paths):
    """Load cleaned CSV files into a dict of DataFrames."""
    dfs = {}
    for bank, path in file_paths.items():
        dfs[bank] = pd.read_csv(path)
        print(f"Loaded {bank} cleaned data: {dfs[bank].shape}")
    return dfs

def sentiment_analysis(df, text_col='review_text', model=None):
    """Run sentiment analysis and add label & score columns."""
    if model is None:
        model = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    texts = df[text_col].fillna('').tolist()
    sentiments = model(texts)
    df['sentiment_label'] = [s['label'] for s in sentiments]
    df['sentiment_score'] = [s['score'] for s in sentiments]
    return df

def preprocess_text(text):
    """Preprocess text: lowercase, lemmatize, remove stopwords/punctuation."""
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
    return " ".join(tokens)

def extract_keywords(df, text_col='review_text', top_n=20):
    """Extract top TF-IDF keywords from text column."""
    df['processed_text'] = df[text_col].apply(preprocess_text)
    vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1,2))
    X = vectorizer.fit_transform(df['processed_text'])
    keywords_scores = sorted(zip(vectorizer.get_feature_names_out(), X.mean(axis=0).A1), key=lambda x: x[1], reverse=True)
    return keywords_scores[:top_n]

def assign_themes(df, keywords_by_theme, text_col='processed_text'):
    """Assign one or more themes to each review based on presence of theme keywords."""
    themes = []
    for text in df[text_col]:
        matched_themes = []
        for theme, keywords in keywords_by_theme.items():
            if any(re.search(rf'\b{re.escape(k)}\b', text) for k in keywords):
                matched_themes.append(theme)
        themes.append(matched_themes if matched_themes else ['Other'])
    df['themes'] = themes
    return df

def save_data(df, filepath):
    """Save DataFrame to CSV."""
    df.to_csv(filepath, index=False)
    print(f"Saved data to {filepath}")
