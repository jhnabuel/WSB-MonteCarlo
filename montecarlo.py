import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import pandas as pd
import statistics
import math
import os
from scipy.stats import norm
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def get_stock_price():
    # Folder name
    folder_path = "/Users/johnabuel/Desktop/stock data"

    tickers = ["TSLA", "NVDA", "MSTR"]

    start_date = '2023-01-01'
    end_date = '2025-04-30'

    for ticker in tickers:
        data = yf.download(ticker, start=start_date, end=end_date)

         # Reset index to turn the datetime index into a 'Date' column
        data.reset_index(inplace=True)


        data = data[["Date", "Open", "High", "Low", "Close"]]

        file_path = os.path.join(folder_path, f"{ticker}.csv")
        data.to_csv(file_path)
        print(f"Saved {ticker} data to {file_path}")

'''
def read_stock_price(stock):
    folder_path = "/Users/johnabuel/Desktop/stock data"

    file_path = os.path.join(folder_path, f"{stock}.csv")

    file =  pd.read_csv(file_path, parse_dates=["Date"]).drop(0)
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(file["Date"], file["Close"], label=f"{stock} Historical Price", color="red")

   
    # Clean y-axis with fixed number of ticks
    ax.yaxis.set_major_locator(MaxNLocator(nbins=8))  # Adjust nbins to change spacing

    ax.set_title(f"{stock} Stock Price Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.legend()
    ax.grid(True)
    plt.show()

''' 


def cdf(array, b, a):
    mean = statistics.mean(array)
    std = statistics.stdev(array)
    p1 = 0.5 * (1 + math.erf((a - mean) / math.sqrt(2 * std **2)))
    p2= 0.5 * (1 + math.erf((a - mean) / math.sqrt(2 * std **2)))

    return (p2-p1)

def plot_final_price_histogram(simulated, bins=50):
    final_day_prices = simulated[-1, :]  # Last row: day 7 prices
    print(f"Number of final day prices being plotted: {len(final_day_prices)}")
    plt.figure(figsize=(12, 6))
    sns.histplot(final_day_prices, bins=bins, stat="frequency")
    plt.xlabel("Close price (NVDA)")
    plt.ylabel("Frequency")
    plt.title("Distribution of Simulated Prices (Day 7)")
    plt.grid(True)
    plt.show()

def monte_carlo_simulation(stock):
    print("Starting simulation:")
    folder_path =  "/Users/johnabuel/Desktop/stock data"
    file_path = os.path.join(folder_path, f"{stock}.csv")

    # Load historical price data
    df = pd.read_csv(file_path, parse_dates=["Date"]).drop(0)

    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")

    prices = df["Close"].reset_index(drop=True)
    dates = df["Date"].reset_index(drop=True)

    # Compute log returns
    log_returns = np.log(1 + prices.pct_change().dropna())

    # After calculating log returns
    dates = dates[1:].reset_index(drop=True)
    prices = prices[1:].reset_index(drop=True)


    # Calculate the drift and volatility
    u = log_returns.mean()
    var = log_returns.var()
    drift = u - (0.5 * var)
    stdev = log_returns.std()

    # Generate random daily returns using Geometric Brownian Movement
    days = 200
    simulations = 1

    # Monte Carlo Simulations
    Z = norm.ppf(np.random.rand(days, simulations))
    daily_returns = np.exp(drift + stdev * Z)
    simulated_prices = np.zeros((days + 1, simulations))
    simulated_prices[0] = prices.iloc[-1]

    for t in range(1, days + 1):
        simulated_prices[t] = simulated_prices[t-1] * daily_returns[t-1]

    # Future prices
    last_date = dates.iloc[-1]
    future_dates = pd.date_range(start=last_date, periods=days + 1)

    # Plot historical
    fig, ax = plt.subplots(figsize=(12,6))
    ax.plot(dates[-14:], prices[-14:], label="Recent Price", color="blue", linestyle="dashed")

        # Create a colormap with as many distinct colors as simulations (or the first N)
    N = min(100, simulations)  # Limit for clarity on the plot
    cmap = plt.get_cmap("nipy_spectral", N)  # Choose a colorful colormap

    # Plot forecast
    for i in range(N):
        ax.plot(future_dates, simulated_prices[:, i], color=cmap(i), linewidth=0.8, alpha=0.9)

    ax.set_title(f"{stock} 7-Day Monte Carlo Forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.grid(True)
    ax.legend()
    plt.show()

    print(f"Shape of simulated_prices: {simulated_prices.shape}")
    # Plot final day prices histogramj
    plot_final_price_histogram(simulated_prices)
     # Optional: Return results
    return {
        "simulated_prices": simulated_prices,  # Exclude initial seed
        "future_dates": future_dates,
        "last_price": prices.iloc[-1]
            }

def sentiment_analysis():
        # Load your CSV
    df = pd.read_csv("/Users/johnabuel/Desktop/stock data/nvda_hot_posts.csv")

    # Combine title and post into one text field (optional but helpful)
    df['text'] = df['Title'].fillna('') + ". " + df['Body'].fillna('')

    # Initialize VADER
    analyzer = SentimentIntensityAnalyzer()

    # Apply sentiment analysis to the title + post
    def analyze(text):
        return analyzer.polarity_scores(text)['compound']
    
    sentiment_scores = df['text'].apply(analyze)
    
    '''
    df['sentiment_score'] = sentiment_scores
    for title, score in zip(df['Title'], df['sentiment_score']):
        print(f"Title: {title}\nSentiment Score: {score:.3f}\n{'-'*60}")
    '''
    df['sentiment_score'] = sentiment_scores
    for title, score in zip(df['Title'], df['sentiment_score']):
        if score == 0.0:
            print(f"[Neutral] Text: {title}")

   



# def monte_carlo_with_sentiment():


if __name__ == "__main__":
    # sentiment_analysis()
   # get_stock_price()
    monte_carlo_simulation("NVDA")
