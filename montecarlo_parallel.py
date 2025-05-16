import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
import seaborn as sns
import yfinance as yf
import pandas as pd
import os
from scipy.stats import norm
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from tqdm import tqdm
import time
import multiprocessing as mp
from multiprocessing import Pool

def get_stock_price():
    # Folder name
    folder_path = "/Users/johnabuel/Desktop/stock data"

    tickers = ["NVDA", "MSTR"]

    start_date = '2022-01-01'
    end_date = '2025-04-30'

    for ticker in tickers:
        data = yf.download(ticker, start=start_date, end=end_date)

         # Reset index to turn the datetime index into a 'Date' column
        data.reset_index(inplace=True)


        data = data[["Date", "Open", "High", "Low", "Close"]]

        file_path = os.path.join(folder_path, f"{ticker}.csv")
        data.to_csv(file_path)
        print(f"Saved {ticker} data to {file_path}")

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



def plot_final_price_histogram(simulated, actual_price=None, bins=100):
    final_day_prices = simulated[-1, :]
    mean_price = np.mean(final_day_prices)

    print(f"Number of final day prices being plotted: {len(final_day_prices)}")
    print("Sum of frequencies per bin ≈", len(final_day_prices))

    plt.figure(figsize=(12, 6))
    sns.histplot(final_day_prices, bins=bins, stat="count", kde=True, color="skyblue", edgecolor="black")

    # Mean line
    plt.axvline(mean_price, color='blue', linestyle='--', linewidth=2, label=f"Mean Simulated Price: ${mean_price:.2f}")

    # Actual price (if available)
    if actual_price is not None:
        plt.axvline(actual_price, color='red', linestyle='-', linewidth=2, label=f"Actual Price: ${actual_price:.2f}")

    plt.xlabel("Close Price of Stock on April 30, 2025")
    plt.ylabel("Frequency")
    plt.title("Distribution of Simulated Final Prices on April 30, 2025")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def simulate_single_path(last_price, total_days, base_drift, stdev, future_dates, 
                         weekly_sentiment=None, sentiment_scaling_factor=0.005, sentiment_decay=0.4):
    prices = np.zeros(total_days + 1)
    prices[0] = last_price

    current_sentiment = 0
    current_week = None
    MAX_DAILY_IMPACT = 0.003  # Max 0.3% daily price impact from sentiment

 

    for day_idx in range(1, total_days + 1):
        current_date = future_dates[day_idx - 1]

         
        # Skip weekends (Saturday=5, Sunday=6)
        if current_date.weekday() >= 5:  # If it's a weekend
            prices[day_idx] = prices[day_idx - 1]  # Price stays the same as previous day
            continue  # Skip to next iteration

        week_key = current_date.strftime('%Y-%W')

        if week_key != current_week:
            current_week = week_key
            new_sentiment = weekly_sentiment.get(week_key, 0) if weekly_sentiment else 0
            current_sentiment = new_sentiment * sentiment_scaling_factor


         
        # Calculate effective sentiment with decay
        # Using business days in week (Monday=0 to Friday=4)
        weekday = current_date.weekday()  # 0-4 for business days
        effective_sentiment = current_sentiment * (1 - sentiment_decay * (weekday/4))


        # Asymmetric impact and capping
        if effective_sentiment > 0:
            effective_sentiment = min(effective_sentiment * 0.7, MAX_DAILY_IMPACT)  # Reduce positive impact
        else:
            effective_sentiment = max(effective_sentiment * 1.3, -MAX_DAILY_IMPACT)  # Amplify negative impact
        
        Z = norm.ppf(np.random.rand())  # single random shock
        daily_return = np.exp(base_drift + effective_sentiment + stdev * Z)
        prices[day_idx] = prices[day_idx - 1] * daily_return

        # Daily decay
        current_sentiment *= (1 - sentiment_decay/5)
    return prices

def monte_carlo_simulation_weekly_sentiment(stock, simulations, sentiment_scaling_factor=0.005, use_sentiment=True):
    start_time = time.time()  # Start timing
    # Set forecast window
    forecast_start = pd.to_datetime("2024-11-01")
    forecast_end = pd.to_datetime("2025-04-30")
    
    # Generate both date ranges
    calendar_dates = pd.date_range(start=forecast_start, end=forecast_end)  # All days
    business_dates = pd.bdate_range(start=forecast_start, end=forecast_end)  # Trading days
    total_days = len(calendar_dates)

    print("Starting simulation:")
    folder_path = "/Users/johnabuel/Desktop/stock data"
    file_path = os.path.join(folder_path, f"{stock}.csv")

    # Load historical price data
    df = pd.read_csv(file_path, parse_dates=["Date"]).drop(0)
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    prices = df["Close"].reset_index(drop=True)
    dates = df["Date"].reset_index(drop=True)

    # Limit historical data to 3 years ending April 30, 2025
    df = df[df["Date"] <= pd.to_datetime("2025-04-30")]
    df = df[df["Date"] >= pd.to_datetime("2022-05-01")]

    # Compute log returns
    log_returns = np.log(1 + prices.pct_change().dropna())
    dates = dates[1:].reset_index(drop=True)
    prices = prices[1:].reset_index(drop=True)

    # Calculate the drift and volatility
    u = log_returns.mean() 
    var = log_returns.var()
    stdev = log_returns.std()  
    base_drift = u - (0.5 * var)  
    
    # Load weekly sentiment (if enabled)
    weekly_sentiment = {}
    if use_sentiment:
        sentiment_csv = os.path.join(folder_path, f"{stock.lower()}_top_relevant_posts.csv")
        weekly_sentiment = sentiment_analysis_weekly(sentiment_csv)
        normalized_weekly_sentiment = normalize_sentiment(weekly_sentiment)
    
    # Find the last available price before the forecast starts
    last_hist_idx = df[df["Date"] < forecast_start].index[-1]
    last_price = df.loc[last_hist_idx, "Close"]

    # Prepare arguments for multiprocessing
    args_list = [
        (
            last_price,
            total_days,
            base_drift,
            stdev,
            calendar_dates,
            normalized_weekly_sentiment if use_sentiment else None,
            sentiment_scaling_factor
        )
        for _ in range(simulations)
    ]

    print(f"\nUsing {mp.cpu_count()} cores for parallel simulation...")

    # Run simulations in parallel
    with mp.Pool() as pool:
        results = list(tqdm(pool.starmap(simulate_single_path, args_list), total=simulations))

    # Reshape results into a 2D array (days+1, simulations)
    simulated_prices = np.column_stack(results)

    # Plot historical and simulated forecast
    fig, ax = plt.subplots(figsize=(12, 6))

    # Find the indices where dates are before forecast_start
    indices = dates[dates < forecast_start].index
    last_21_indices = indices[-21:]

    # Plot historical data
    ax.plot(dates[last_21_indices], prices[last_21_indices], label="Historical", color="blue")

    # Convert to business days for plotting
    is_business_day = calendar_dates.to_series().dt.dayofweek < 5
    business_day_prices = simulated_prices[1:][is_business_day]  # Skip seed

    # Simulated price paths
    N = min(100, simulations)
    cmap = plt.get_cmap("nipy_spectral", N)
    for i in range(N):
        ax.plot(business_dates, business_day_prices[:, i], color=cmap(i), linewidth=0.7, alpha=0.3)

    # Titles and labels
    ax.set_title(f"{stock} 6-Month Monte Carlo Forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    plt.show()

    print(f"Shape of simulated_prices: {simulated_prices.shape}")

    # Plot final day prices histogram
    plot_final_price_histogram(simulated_prices)

    end_time = time.time()  # End timing
    elapsed = end_time - start_time
    print(f"\nExecution time for {simulations} simulations: {elapsed:.2f} seconds")
    
    return {
        "simulated_prices": simulated_prices,  # Includes all days (calendar)
        "calendar_dates": calendar_dates,
        "business_dates": business_dates,
        "last_price": prices.iloc[-1]
    }

def plot_simulation_comparison(result, actual_df, stock, title_suffix):
    sim_prices = result["simulated_prices"]
    calendar_dates = result["calendar_dates"]
    business_dates = result["business_dates"]

    # Convert to business days 
    is_business_day = calendar_dates.to_series().dt.dayofweek < 5
    business_prices = sim_prices[1:][is_business_day]  # Skip seed
    

    # Load actual price data
    actual_df = actual_df[actual_df["Date"].isin(business_dates)].sort_values("Date")
    actual_prices = actual_df["Close"].values
    
    # Ensure same length (trim if necessary)
    min_length = min(len(business_dates), len(actual_prices))
    plot_dates = business_dates[:min_length]
    plot_prices = business_prices[:min_length, :]
    actual_prices = actual_prices[:min_length]

    # Calculate statistics on BUSINESS DAYS only
    mean_path = np.mean(plot_prices, axis=1)
    ci_upper = np.percentile(plot_prices, 97.5, axis=1)
    ci_lower = np.percentile(plot_prices, 2.5, axis=1)


    # ---- Calculate Metrics ----
    # Root Mean Square Error (RMSE)
    rmse = np.sqrt(np.mean((actual_prices - mean_path) ** 2))
    # Mean Absolute Percentage Error (MAPE)
    mape = np.mean(np.abs((actual_prices - mean_path) / actual_prices)) * 100
    # 95% CI of Final Price
    final_prices = plot_prices[-1, :]  # All simulated final prices
    ci_low_final = np.percentile(final_prices, 2.5)
    ci_high_final = np.percentile(final_prices, 97.5)
    
     # Print Metrics
    print("\n--- Model Performance Metrics ---")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAPE: {mape:.2f}%")
    print(f"95% CI of Final Price: [${ci_low_final:.2f}, ${ci_high_final:.2f}]")
    print(f"Actual Final Price: ${actual_prices[-1]:.2f}")


    # ---- Plotting ----

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot 100 sample paths
    for i in range(min(100, sim_prices.shape[1])):
        ax.plot(plot_dates, plot_prices[:, i], color="gray", alpha=0.1)

    # Mean and confidence intervals
    ax.plot(plot_dates, mean_path, color="blue", linewidth=2, label="Mean Predicted Price")
    ax.fill_between(plot_dates, ci_lower, ci_upper, color="lightblue", alpha=0.3, label="95% Confidence Interval")


    # Actual price plot
    ax.plot(plot_dates, actual_prices, color="red", label="Actual Price", linewidth=1)


    ax.set_title(f"{stock} - Monte Carlo Simulation ({title_suffix})")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()

def load_actual_prices(stock):
    # Load and filter actual prices to match simulation dates
    df = pd.read_csv(f"/Users/johnabuel/Desktop/stock data/{stock}.csv", parse_dates=["Date"])
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df.dropna()
    df = df[(df["Date"] >= "2024-11-01") & (df["Date"] <= "2025-04-30")]  # Critical fix
    return df


def sentiment_analysis_weekly(posts_csv):
    df = pd.read_csv(posts_csv)

    # Combine title and post into one text field (optional but helpful)
    df['text'] = df['Title'].fillna('') + ". " + df['Body'].fillna('')
    df['Date'] = pd.to_datetime(df['Date'])

    # Initialize VADER
    analyzer = SentimentIntensityAnalyzer()

    # Apply sentiment analysis to the title + post
    def analyze(text):
        return analyzer.polarity_scores(text)['compound']
    
    sentiment_scores = df['text'].apply(analyze)
    df['week'] = df['Date'].dt.strftime('%Y-%W')  # Year-week format
    df['sentiment_score'] = sentiment_scores

     # Group by week and compute average
    weekly_sentiment = df.groupby('week')['sentiment_score'].mean().to_dict()

    # Print weekly sentiment scores
    print("\nWeekly Sentiment Scores:")
    for week, score in weekly_sentiment.items():
        print(f"Week {week}: {score:.4f}")

    return weekly_sentiment

def normalize_sentiment(weekly_sentiment):
    values = np.array(list(weekly_sentiment.values()))
    mean = np.mean(values)
    std = np.std(values)
    return {k: (v-mean)/std for k, v in weekly_sentiment.items()}

# def monte_carlo_with_sentiment():


if __name__ == "__main__":
    stocks = ["MSTR"]
    results = {}

    for stock in stocks:
        print(f"\n=== Processing {stock} ===")
        
        # Load actual data once (outside simulations)
        actual_df = load_actual_prices(stock)

        # Run simulations
        print("Running STANDARD GBM...")
        results[(stock, "standard")] = monte_carlo_simulation_weekly_sentiment(
            stock, simulations=10000, use_sentiment=False
        )

        print("Running SENTIMENT-ADJUSTED GBM...")
        results[(stock, "sentiment")] = monte_carlo_simulation_weekly_sentiment(
            stock, simulations=10000, use_sentiment=True
        )

        # Compare results
        print("\n--- Results ---")
        plot_simulation_comparison(results[(stock, "standard")], actual_df, stock, "Standard GBM")
        plot_simulation_comparison(results[(stock, "sentiment")], actual_df, stock, "Sentiment-Adjusted GBM")