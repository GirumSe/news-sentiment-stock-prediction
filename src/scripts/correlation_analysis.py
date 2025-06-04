import pandas as pd
from pandarallel import pandarallel
import seaborn as sns
import matplotlib.pyplot as plt
import os
from textblob import TextBlob
from scipy.stats import pearsonr

pandarallel.initialize()

def load_news_data(path):
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'], errors='coerce', infer_datetime_format=True)
    df['date'] = df['date'].dt.tz_localize(None).dt.date
    return df

def compute_sentiment(df):
    df['sentiment'] = df['headline'].parallel_apply(lambda x: TextBlob(str(x)).sentiment.polarity)
    return df

def aggregate_daily_sentiment(df):
    return df.groupby(['date', 'stock'])['sentiment'].mean().reset_index()

def load_all_stock_data(data_dir):
    stock_dfs = []
    for filename in os.listdir(data_dir):
        if filename.endswith('_historical_data.csv'):
            symbol = filename.split('_')[0]
            df = pd.read_csv(os.path.join(data_dir, filename), parse_dates=['Date'])
            df['Date'] = df['Date'].dt.date
            df['stock'] = symbol
            df.sort_values('Date', inplace=True)
            df['daily_return'] = df['Close'].pct_change()
            stock_dfs.append(df[['Date', 'stock', 'daily_return']].dropna())
    return pd.concat(stock_dfs, ignore_index=True)

def merge_sentiment_with_returns(sentiment_df, stock_df):
    merged = pd.merge(sentiment_df, stock_df, left_on=['date', 'stock'], right_on=['Date', 'stock'])
    return merged.dropna(subset=['daily_return', 'sentiment'])

def compute_correlation_per_stock(merged_df):
    def compute_corr(group):
        # Make sure to drop any rows with missing values in the columns of interest
        valid = group[['sentiment', 'daily_return']].dropna()
        n_samples = len(valid)
        if n_samples < 2:
            return pd.Series({'n_samples': n_samples, 'correlation': None})
        else:
            return pd.Series({
                'n_samples': n_samples,
                'correlation': pearsonr(valid['sentiment'], valid['daily_return'])[0]
            })
    return merged_df.groupby('stock').apply(compute_corr).reset_index()
    
def plot_sentiment_vs_return(df, stock):
    sub = df[df['stock'] == stock]
    sns.scatterplot(data=sub, x='sentiment', y='daily_return')
    plt.title(f'Sentiment vs Return for {stock}')
    plt.xlabel('Sentiment Score')
    plt.ylabel('Daily Return')
    plt.grid(True)
    plt.show()