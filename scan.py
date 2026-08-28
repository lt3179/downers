import os
import json
import time
from datetime import datetime, timedelta
import requests
import yfinance as yf
from google import genai
from google.genai import errors

NEWSDATA_API_KEY = os.environ["NEWSDATA_API_KEY"]
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]

client = genai.Client(api_key=GEMINI_API_KEY)

SECONDS_BETWEEN_CALLS = 5

def fetch_headlines():
    url = "https://newsdata.io/api/1/news"
    params = {
        "apikey": NEWSDATA_API_KEY,
        "category": "business",
        "language": "en",
        "country": "us",
        "q": "scandal OR backlash OR lawsuit OR recall OR controversy OR breach"
    }
    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()
    headlines = []
    for article in data.get("results", []):
        headlines.append({
            "title": article.get("title", ""),
            "description": article.get("description", ""),
            "url": article.get("link", ""),
            "source": article.get("source_id", ""),
            "pubDate": article.get("pubDate", "")
        })
    return headlines

CLASSIFICATION_PROMPT = """You are screening business headlines for a project called Downers, which looks for stocks that drop due to short-term reputational or confidence shocks (scandals, PR gaffes, executive controversies) rather than real fundamental business problems (earnings misses, guidance cuts, structural competitive damage).

For the headline below, respond with ONLY a JSON object (no markdown, no extra text) with these fields:
- "company": company name if identifiable, else null
- "ticker": stock ticker if you know it, else null
- "category": one of ["pr_scandal", "executive_controversy", "data_breach_or_cyberattack", "product_recall_scare", "legal_news", "fundamental_issue", "not_relevant"]
- "fundamental_impact_score": integer 1-5, where 1 = pure confidence shock with no real business impact, 5 = this is a genuine fundamentals problem (earnings, guidance, structural)
- "reasoning": 2-3 sentences explaining your reasoning in plain language, written so a reader can evaluate your logic and disagree with it if they want
- "confident": true or false — false if you don't have enough information to classify this reliably

Headline: {title}
Description: {description}
"""

def classify_headline(headline, max_retries=3):
    prompt = CLASSIFICATION_PROMPT.format(
        title=headline["title"],
        description=headline["description"] or "(no description)"
    )
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model="gemini-3.5-flash-lite",
                contents=prompt
            )
            text = response.text.strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return {"error": "Could not parse response", "raw": text}
        except errors.APIError as e:
            if attempt < max_retries - 1:
                print(f"API error ({e}), waiting 30s before retry {attempt + 1}...")
                time.sleep(30)
            else:
                return {"error": f"API error after retries: {str(e)}"}
        except Exception as e:
            return {"error": f"Unexpected error: {str(e)}"}

def parse_pub_date(pub_date_str):
    try:
        return datetime.strptime(pub_date_str, "%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError):
        return datetime.utcnow()

def get_price_data(ticker, pub_date_str):
    """
    Returns two baselines:
      - base_price: the single last close strictly before the event date
      - avg_base_price_3d: the average close over the 3 trading days
        before the event (smooths single-day noise)
    Plus the current price for comparison.
    """
    event_date = parse_pub_date(pub_date_str)

    try:
        stock = yf.Ticker(ticker)
        start = event_date - timedelta(days=15)
        end = datetime.utcnow() + timedelta(days=1)
        hist = stock.history(start=start, end=end)

        if hist.empty:
            return {"base_price": None, "current_price": None,
                    "price_error": "No price history returned for this ticker"}

        hist.index = hist.index.tz_localize(None) if hist.index.tz is not None else hist.index
        event_date_only = event_date.replace(hour=0, minute=0, second=0, microsecond=0)

        before_event = hist[hist.index < event_date_only]
        if before_event.empty:
            return {"base_price": None, "current_price": None,
                    "price_error": "No trading data before the event date"}

        base_price = round(float(before_event["Close"].iloc[-1]), 2)
        base_price_date = before_event.index[-1].strftime("%Y-%m-%d")

        last_n = before_event["Close"].tail(3)
        avg_base_price_3d = round(float(last_n.mean()), 2)
        avg_base_days_used = int(len(last_n))

        current_price = round(float(hist["Close"].iloc[-1]), 2)
        current_price_date = hist.index[-1].strftime("%Y-%m-%d")

        return {
            "base_price": base_price,
            "base_price_date": base_price_date,
            "avg_base_price_3d": avg_base_price_3d,
            "avg_base_days_used": avg_base_days_used,
            "current_price": current_price,
            "current_price_date": current_price_date
        }
    except Exception as e:
        return {"base_price": None, "current_price": None,
                "price_error": f"Price lookup failed: {str(e)}"}

def get_five_year_trend(ticker):
    """
    Monthly closing prices over the last 5 years, for a lightweight
    trendline. Downsampled to ~60 points instead of ~1,250 daily points.
    """
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="5y", interval="1mo")
        if hist.empty:
            return {"trend_5y": None, "trend_5y_error": "No 5-year history available"}
        points = [
            {"date": idx.strftime("%Y-%m"), "close": round(float(row["Close"]), 2)}
            for idx, row in hist.iterrows()
        ]
        return {"trend_5y": points}
    except Exception as e:
        return {"trend_5y": None, "trend_5y_error": f"5-year trend lookup failed: {str(e)}"}

def main():
    headlines = fetch_headlines()
    results = []
    for i, h in enumerate(headlines):
        print(f"Classifying {i + 1}/{len(headlines)}: {h['title'][:60]}")
        classification = classify_headline(h)
        merged = {**h, **classification}

        is_flagged = (
            merged.get("confident") is True
            and merged.get("category") not in (None, "not_relevant")
            and isinstance(merged.get("fundamental_impact_score"), int)
            and merged["fundamental_impact_score"] <= 3
            and merged.get("ticker")
        )
        if is_flagged:
            price_data = get_price_data(merged["ticker"], h["pubDate"])
            merged.update(price_data)
            trend_data = get_five_year_trend(merged["ticker"])
            merged.update(trend_data)

        results.append(merged)
        if i < len(headlines) - 1:
            time.sleep(SECONDS_BETWEEN_CALLS)

    with open("events.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"Processed {len(headlines)} headlines. Saved to events.json")

if __name__ == "__main__":
    main()
