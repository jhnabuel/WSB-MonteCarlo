from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import time
from datetime import datetime
import psutil
from time import sleep

def sequential_sentiment_analysis(csv_path):
    """Process posts and comments sequentially"""
    start_time = time.time()
    mem_before = psutil.virtual_memory().percent
    cpu_before = psutil.cpu_percent(interval=1)
    
    print("Loading data...")
    df = pd.read_csv(csv_path)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])
    df['week'] = df['Date'].dt.strftime('%Y-%W')
    
    analyzer = SentimentIntensityAnalyzer()
    weekly_results = {}
    current_post = None
    post_counter = 0

    print("Processing posts sequentially...")
    for _, row in df.iterrows():
        # Start new post group
        if not str(row['Title']).startswith('[Comment on]'):
            if current_post:  # Save previous post
                weekly_results = update_weekly_results(weekly_results, current_post)
                post_counter += 1
                print(f"Processed post {post_counter}", end='\r')
            
            # Initialize new post
            current_post = {
                'week': row['week'],
                'title': row['Title'],
                'title_sent': analyzer.polarity_scores(str(row['Title']))['compound'],
                'body_sent': analyzer.polarity_scores(str(row['Body']))['compound'] if pd.notna(row['Body']) else 0,
                'comments': []
            }
        else:  # Add comment to current post
            if current_post:  # Ensure we have a parent post
                current_post['comments'].append(
                    analyzer.polarity_scores(str(row['Body']))['compound'] if pd.notna(row['Body']) else 0
                )
    
    # Process the last post
    if current_post:
        weekly_results = update_weekly_results(weekly_results, current_post)
    
    # Calculate weekly averages
    final_results = {}
    for week, data in weekly_results.items():
        final_results[week] = sum(data['scores'])/len(data['scores']) if data['scores'] else 0
    
    # Performance stats
    elapsed = time.time() - start_time
    cpu_after = psutil.cpu_percent(interval=1)
    mem_after = psutil.virtual_memory().percent
    
    print("\n Results:")
    for week, score in final_results.items():
        print(f"{week}: {score:.4f}")
    
    print(f"\n Completed in {elapsed:.2f}s")
    print(f"CPU Usage: {cpu_before}% → {cpu_after}%")
    print(f"RAM Usage: {mem_before}% → {mem_after}%")
    
    return final_results

def update_weekly_results(weekly_results, post_data):
    """Helper to aggregate post scores by week"""
    week = post_data['week']
    comment_avg = sum(post_data['comments'])/len(post_data['comments']) if post_data['comments'] else 0
    post_score = 0.2*post_data['title_sent'] + 0.5*post_data['body_sent'] + 0.3*comment_avg
    
    if week not in weekly_results:
        weekly_results[week] = {'scores': [], 'posts': 0}
    weekly_results[week]['scores'].append(post_score)
    weekly_results[week]['posts'] += 1
    
    return weekly_results

if __name__ == "__main__":
    print("\nRunning sequential sentiment analysis for NVDA...")
    nvda_results = sequential_sentiment_analysis("/Users/johnabuel/Desktop/stock data/nvda_top_posts.csv")

    print("\nRunning sequential sentiment analysis for MSTR...")
    mstr_results = sequential_sentiment_analysis("/Users/johnabuel/Desktop/stock data/mstr_top_posts.csv")
