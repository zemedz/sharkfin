from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import yfinance as yf
import requests
from transformers import pipeline
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Market Insight AI is running!"}

NEWS_API_KEY = os.getenv("NEWS_API_KEY", "your_newsapi_key_here")

def get_sentiment_model():
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=-1
    )

def get_live_news():
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": "(politics OR economy OR war OR crypto OR market)",
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": 5,
        "apiKey": NEWS_API_KEY
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        articles = response.json().get("articles", [])
        return [article['title'] for article in articles if 'title' in article]
    return []

def map_event_to_asset(text):
    text = text.lower()
    if "oil" in text or "middle east" in text:
        return "XLE"
    elif "bitcoin" in text or "crypto" in text:
        return "BTC-USD"
    elif "china" in text or "trade" in text:
        return "BABA"
    elif "green" in text or "energy" in text:
        return "ICLN"
    elif "interest rates" in text or "inflation" in text:
        return "TLT"
    else:
        return "SPY"

def get_asset_price(ticker):
    stock = yf.Ticker(ticker)
    hist = stock.history(period="5d")
    if hist.empty:
        return None
    return hist['Close'][-1]

class HeadlineInput(BaseModel):
    headline: str

@app.get("/news-insights")
def news_insights():
    model = get_sentiment_model()
    headlines = get_live_news()
    insights = []
    for headline in headlines:
        sentiment = model(headline)[0]
        asset = map_event_to_asset(headline)
        price = get_asset_price(asset)
        insights.append({
            "headline": headline,
            "sentiment": sentiment['label'],
            "confidence": sentiment['score'],
            "linked_asset": asset,
            "latest_price": price
        })
    return {"insights": insights}

@app.post("/analyze-headline")
def analyze_custom_headline(input: HeadlineInput):
    model = get_sentiment_model()
    sentiment = model(input.headline)[0]
    asset = map_event_to_asset(input.headline)
    price = get_asset_price(asset)
    return {
        "headline": input.headline,
        "sentiment": sentiment['label'],
        "confidence": sentiment['score'],
        "linked_asset": asset,
        "latest_price": price
    }
