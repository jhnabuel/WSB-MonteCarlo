import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import psutil
import time
import numpy as np
from datetime import datetime

def analyze_sentiment(text, analyzer):
    """Thread-safe sentiment analysis"""
    return analyzer.polarity_scores(text)['compound']

def process_dataset(csv_path, week_workers=4, post_workers=8):
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
    
    # Pre-group by week
    weekly_groups = {week: group for week, group in df.groupby('week')}
    
    def process_post(post_group):
        try:
            # SAFER POST EXTRACTION
            main_posts = post_group[~post_group['Title'].str.startswith('[Comment on]', na=False)]
            if len(main_posts) == 0:
                return 0  # Skip if no main post found
            
            post = main_posts.iloc[0]
            comments = post_group[post_group['Title'].str.startswith('[Comment on]', na=False)]
            
            # SENTIMENT ANALYSIS
            analyzer = SentimentIntensityAnalyzer()
            title_sent = analyzer.polarity_scores(post['Title'])['compound']
            body_sent = analyzer.polarity_scores(post['Body'])['compound'] if pd.notna(post['Body']) else 0
            comment_sents = [analyzer.polarity_scores(c['Body'])['compound'] 
                            for _, c in comments.iterrows() if pd.notna(c['Body'])]
            comment_avg = np.mean(comment_sents) if comment_sents else 0
            
            return 0.2*title_sent + 0.5*body_sent + 0.3*comment_avg
        
        except Exception as e:
            print(f"Error processing post: {str(e)[:100]}...")
            return 0  # Neutral score on failure

    def process_week(week, week_group):
        try:
            # SAFER POST GROUPING
            post_groups = []
            current_group = []
            
            for _, row in week_group.iterrows():
                if not str(row['Title']).startswith('[Comment on]'):
                    if current_group:  # Save previous group
                        post_groups.append(pd.DataFrame(current_group))
                    current_group = [row]  # Start new group
                else:
                    current_group.append(row)
            
            if current_group:  # Add last group
                post_groups.append(pd.DataFrame(current_group))

            # PROCESS POSTS
            with ThreadPoolExecutor(max_workers=8) as executor:
                futures = [executor.submit(process_post, g) for g in post_groups]
                return np.mean([f.result() for f in as_completed(futures)])
                
        except Exception as e:
            print(f"Error processing week {week}: {str(e)[:100]}...")
            return 0
    
    # Process weeks concurrently
    print("Processing weeks...")
    with ThreadPoolExecutor(max_workers=week_workers) as executor:
        week_futures = {executor.submit(process_week, week, group): week 
                        for week, group in weekly_groups.items()}
        
        # Main progress bar for weeks
        with tqdm(as_completed(week_futures), total=len(week_futures),
               desc="Overall Progress", unit="week") as pbar:
            for future in pbar:
                week = week_futures[future]
                weekly_results[week] = future.result()
                pbar.set_postfix_str(f"CPU: {psutil.cpu_percent()}% RAM: {psutil.virtual_memory().percent}%")
    
    # Performance stats
    elapsed = time.time() - start_time
    cpu_after = psutil.cpu_percent(interval=1)
    mem_after = psutil.virtual_memory().percent
    
    print(f"\nCompleted in {elapsed:.2f}s")
    print(f"CPU Usage: {cpu_before}% → {cpu_after}%")
    print(f"RAM Usage: {mem_before}% → {mem_after}%")
    
    return weekly_results

if __name__ == "__main__":
    print("\nRunning parallel sentiment analysis for NVDA...")
    nvda_results = process_dataset("/Users/johnabuel/Desktop/stock data/nvda_top_posts.csv")

    print("\nRunning parallel sentiment analysis for MSTR...")
    mstr_results = process_dataset("/Users/johnabuel/Desktop/stock data/mstr_top_posts.csv")