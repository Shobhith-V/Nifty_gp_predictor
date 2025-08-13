# src/news_fetcher.py
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from transformers import pipeline
import streamlit as st
# Initialize sentiment analysis model
sentiment_model = pipeline("sentiment-analysis")

def fetch_nifty_news(start_date, end_date):
    """
    Fetch Nifty-related news between start_date and end_date.
    Uses Google News search.
    """
    query = f"Nifty 50 site:moneycontrol.com OR site:economictimes.indiatimes.com"
    url = f"https://news.google.com/rss/search?q={query}+after:{start_date}+before:{end_date}"

    response = requests.get(url)
    soup = BeautifulSoup(response.content, "xml")
    items = soup.find_all("item")

    news_list = []
    for item in items:
        news_list.append({
            "title": item.title.text,
            "link": item.link.text
        })

    return news_list

def get_sentiment_score(news_list):

    st.write(f"**Debug Info:** Number of news items received: `{len(news_list)}`")

    if not news_list:
        return 0.0
    
    scores = []
    # 'news' is a tuple like ('headline text', 'url')
    for news in news_list:
        headline_text = news[0]
        
        # Make sure the headline is not empty
        if not headline_text or not isinstance(headline_text, str):
            continue

        result = sentiment_model(headline_text)[0]
        
        # --- DEBUG 2: See the raw output from the model for each headline ---
        #st.write(f"**Headline:** `{headline_text}` | **Model Output:** `{result}`")
        
        label = result['label']
        score = result['score']

        if label == "POSITIVE":
            scores.append(score)
        elif label == "NEGATIVE":
            scores.append(-score)
        else:
            # Handle NEUTRAL or any other unrecognized labels as 0
            scores.append(0)
    

    if not scores:
        return 0.0

    # Return the average score
    return sum(scores) / len(scores)
