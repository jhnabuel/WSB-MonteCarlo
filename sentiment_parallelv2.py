import pandas as pd
import numpy as np
import time
from datetime import datetime
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import psutil
from multiprocessing import Pool, cpu_count


def analyze_post_group(post_group):
    analyzer = SentimentIntensityAnalyzer()
    main_posts = post_group[~post_group['Title'].str.startswith('[Comment on]', na=False)]

    if main_posts.empty:
        return 0  # No main post

    post = main_posts.iloc[0]
    title_sent = analyzer.polarity_scores(str(post['Title']))['compound']
    body_sent = analyzer.polarity_scores(str(post['Body']))['compound'] if pd.notna(post['Body']) else 0

    # Comments
    comments = post_group[post_group['Title'].str.startswith('[Comment on]', na=False)]
    comment_scores = [
        analyzer.polarity_scores(str(row['Body']))['compound']
        for _, row in comments.iterrows() if pd.notna(row['Body'])
    ]
    comment_avg = np.mean(comment_scores) if comment_scores else 0

    return 0.2 * title_sent + 0.5 * body_sent + 0.3 * comment_avg


def process_week(group):
    week, df_week = group
    post_groups = []
    current_group = []

    for _, row in df_week.iterrows():
        if not str(row['Title']).startswith('[Comment on]'):
            if current_group:
                post_groups.append(pd.DataFrame(current_group))
            current_group = [row]
        else:
            current_group.append(row)

    if current_group:
        post_groups.append(pd.DataFrame(current_group))

    scores = [analyze_post_group(pg) for pg in post_groups if not pg.empty]
    week_score = np.mean(scores) if scores else 0
    return (week, week_score)


def parallel_sentiment_analysis(csv_path):
    start_time = time.time()
    mem_before = psutil.virtual_memory().percent
    cpu_before = psutil.cpu_percent(interval=1)

    print("Loading data...")
    df = pd.read_csv(csv_path)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])
    df['week'] = df['Date'].dt.strftime('%Y-%W')

    grouped = list(df.groupby('week'))

    print(f"Processing {len(grouped)} weeks using {cpu_count()} cores...")

    with Pool() as pool:
        results = pool.map(process_week, grouped)

    final_results = dict(results)

    # --- Performance summary ---
    elapsed = time.time() - start_time
    cpu_after = psutil.cpu_percent(interval=1)
    mem_after = psutil.virtual_memory().percent

    print("\nResults:")
    for week, score in sorted(final_results.items()):
        print(f"{week}: {score:.4f}")

    print(f"\nCompleted in {elapsed:.2f}s")
    print(f"CPU Usage: {cpu_before}% → {cpu_after}%")
    print(f"RAM Usage: {mem_before}% → {mem_after}%")

    return final_results


if __name__ == "__main__":
    print("\nRunning parallel sentiment analysis for NVDA")
    nvda_result = parallel_sentiment_analysis("/Users/johnabuel/Desktop/stock data/nvda_top_posts.csv")

    print("\nRunning parallel sentiment analysis for MSTR")
    mstr_result = parallel_sentiment_analysis("/Users/johnabuel/Desktop/stock data/mstr_top_posts.csv")
