# scripts/text_utils.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

def compute_headline_length(df, text_col='headline'):
    df['headline_length'] = df[text_col].astype(str).apply(len)
    return df

def plot_headline_length_distribution(df):
    plt.figure(figsize=(8, 5))
    sns.histplot(df['headline_length'], bins=30, kde=True)
    plt.title("Headline Length Distribution")
    plt.xlabel("Length")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_articles_per_publisher(df):
    top_publishers = df['publisher'].value_counts().head(10)
    plt.figure(figsize=(10, 5))
    sns.barplot(x=top_publishers.index, y=top_publishers.values)
    plt.title("Top 10 Publishers by Article Count")
    plt.ylabel("Article Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_articles_over_time(df, date_col='date'):
    df[date_col] = pd.to_datetime(df[date_col], format='mixed', utc=True, errors='raise')
    time_series = df.groupby(df[date_col].dt.date).size()
    plt.figure(figsize=(12, 5))
    plt.plot(time_series.index, time_series.values)
    plt.title('Articles over Time')
    plt.xlabel('Date')
    plt.ylabel('Number of Articles')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def extract_keywords(df, text_col='headline', method='count', top_n=10):
    """Extracts top N keywords from a text column using CountVectorizer or TfidfVectorizer."""
    if text_col not in df.columns:
        raise ValueError(f"Column '{text_col}' not found in DataFrame.")
    texts = df[text_col].fillna('').astype(str)

    if method == 'count':
        vectorizer = CountVectorizer(stop_words='english', max_features=150000) # Added max_features as a precaution
    elif method == 'tfidf':
        vectorizer = TfidfVectorizer(stop_words='english', max_features=150000) # Added max_features as a precaution
    else:
        raise ValueError("Method must be 'count' or 'tfidf'.")

    X = vectorizer.fit_transform(texts)
    
    sums = np.asarray(X.sum(axis=0)).ravel()
    
    # Create a DataFrame/Series for keywords and their scores
    keywords_df = pd.DataFrame({'keyword': vectorizer.get_feature_names_out(), 'score': sums})
    
    # Sort by score and get top N
    top_keywords = keywords_df.sort_values(by='score', ascending=False).head(top_n)
    
    # Return as a Series (keyword: score) for consistency with plot_top_keywords if it expects that
    return pd.Series(top_keywords.score.values, index=top_keywords.keyword)

def plot_top_keywords(keywords, method='TF-IDF'):
    plt.figure(figsize=(10, 5))
    sns.barplot(x=keywords.values, y=keywords.index)
    plt.title(f"Top Keywords ({method})")
    plt.xlabel("Score")
    plt.tight_layout()
    plt.show()
