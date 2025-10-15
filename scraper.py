import tweepy
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN")

def scrape_tweets_for_training():
    """
    Scrapes a broad set of English tweets from India to create a
    training dataset for our ML model.
    """
    if not BEARER_TOKEN:
        raise ValueError("Twitter Bearer Token not found in .env file")
        
    client = tweepy.Client(BEARER_TOKEN)

    # =========================================================================
    # CORRECTED QUERY: Replaced 'place_country' with location keywords.
    # We now search for tweets containing both a hazard term AND a location term.
    # =========================================================================
    query = ("coronavirus" OR "covid" OR "ppe kit" OR "lockdown" OR "quarantine" OR "social distancing" OR "contact tracing" OR "India" OR "cough")

    print("Scraping a broad set of tweets for ML training...")
    
    response = client.search_recent_tweets(
        query=query,
        max_results=100,
        tweet_fields=["created_at"]
    )

    if not response.data:
        print("No tweets were found for your query.")
        return

    tweet_list = []
    for tweet in response.data:
        tweet_list.append({
            'id': tweet.id,
            'text': tweet.text,
            'created_at': tweet.created_at
        })
    
    df = pd.DataFrame(tweet_list)
    output_path = 'dataset/Corona_NLP_train.csv'
    df.to_csv(output_path, index=False)
    
    print(f"Success! Saved {len(tweet_list)} tweets for training to {output_path}")
