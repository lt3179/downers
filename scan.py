import os
import json
import time
from datetime import datetime, timedelta
from datetime import datetime as dt
from urllib.parse import quote
import requests
import feedparser
import yfinance as yf
from google import genai
from google.genai import errors

NEWSDATA_API_KEY = os.environ["NEWSDATA_API_KEY"]
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]

client = genai.Client(api_key=GEMINI_API_KEY)

SECONDS_BETWEEN_CALLS = 5

def fetch_headlines():
    headlines = []

    # Source 1: NewsData.io (fewer articles, but real descriptions)
    try:
        url = "https://newsdata.io/api/1/latest"
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
        for article in data.get("results", []):
            headlines.append({
                "title": article.get("title", ""),
                "description": article.get("description", ""),
                "url": article.get("link", ""),
                "source": article.get("source_id", ""),
                "pubDate": article.get("pubDate", "")
            })
        print(f"NewsData.io: {len(data.get('results', []))} articles")
    except Exception as e:
        print(f"NewsData.io fetch failed: {e}")

    # Source 2: Google News RSS (many more articles, title-only context)
    try:
        query = "(scandal OR backlash OR lawsuit OR recall OR controversy OR breach) when:2d"
        encoded_query = quote(query)
        rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"
        feed = feedparser.parse(rss_url)

        for entry in feed.entries:
            pub_date = ""
            if hasattr(entry, "published_parsed") and entry.published_parsed:
                pub_date = dt(*entry.published_parsed[:6]).strftime("%Y-%m-%d %H:%M:%S")
            source_name = ""
            if hasattr(entry, "source") and hasattr(entry.source, "title"):
                source_name = entry.source.title

            headlines.append({
                "title": entry.get("title", ""),
                "description": "",  # Google's RSS doesn't provide real snippets
                "url": entry.get("link", ""),
                "source": source_name,
                "pubDate": pub_date
            })
        print(f"Google News RSS: {len(feed.entries)} articles")
    except Exception as e:
        print(f"Google News RSS fetch failed: {e}")

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

SUBSTITUTE_PROMPT = """The following news story involves a company that either isn't publicly traded, or whose stock ticker couldn't be identified.

Story: {title}
Description: {description}
Company involved: {company}

Is there a publicly traded company — a parent company, franchisor, major competitor, or key supplier/customer — that would plausibly see a similar short-term stock dip from this same news, due to shared brand exposure or investor confusion? This needs to be a genuine, defensible connection, not a stretch. If nothing reasonable comes to mind, say so.

Respond with ONLY a JSON object (no markdown, no extra text):
- "has_substitute": true or false
- "substitute_company": name of the substitute company, or null
- "substitute_ticker": its stock ticker, or null
- "substitute_reasoning": 1-2 sentences explaining the connection, or null if has_substitute is false
"""

def find_substitute_ticker(headline, company_name, max_retries=2):
    prompt = SUBSTITUTE_PROMPT.format(
        title=headline["title"],
        description=headline["description"] or "(no description)",
        company=company_name or "unknown"
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
                return {"has_substitute": False}
        except errors.APIError:
            if attempt < max_retries - 1:
                time.sleep(30)
            else:
                return {"has_substitute": False}
        except Exception:
            return {"has_substitute": False}
    return {"has_substitute": False}

def get_price_data(ticker):
    """
    Detects the pre-dip baseline directly from price action instead of
    trusting the article's publish date (which is sometimes missing or
    imprecise, especially from RSS sources). Finds the single largest
    day-over-day decline in the recent trading window and uses the close
    right before that drop as the baseline.
    """
    try:
        stock = yf.Ticker(ticker)
        end = datetime.utcnow() + timedelta(days=1)
        start = end - timedelta(days=45)
        hist = stock.history(start=start, end=end)

        if hist.empty or len(hist) < 3:
            return {"base_price": None, "current_price": None,
                    "price_error": "Not enough price history for this ticker"}

        hist.index = hist.index.tz_localize(None) if hist.index.tz is not None else hist.index
        closes = hist["Close"]

        recent = closes.tail(16)
        daily_changes = recent.pct_change().dropna()

        if daily_changes.empty:
            return {"base_price": None, "current_price": None,
                    "price_error": "Not enough recent trading days to detect a dip"}

        worst_day = daily_changes.idxmin()
        worst_change = daily_changes.loc[worst_day]

        if worst_change > -0.015:
            return {"base_price": None, "current_price": None,
                    "price_error": "No clear single-day price drop detected in the recent window"}

        worst_day_pos = recent.index.get_loc(worst_day)
        if worst_day_pos == 0:
            return {"base_price": None, "current_price": None,
                    "price_error": "Drop day is at the edge of the lookback window"}

        base_price = round(float(recent.iloc[worst_day_pos - 1]), 2)
        base_price_date = recent.index[worst_day_pos - 1].strftime("%Y-%m-%d")

        window = closes[closes.index <= recent.index[worst_day_pos - 1]].tail(15)
        avg_base_price_15d = round(float(window.mean()), 2)
        avg_base_days_used = int(len(window))

        current_price = round(float(closes.iloc[-1]), 2)
        current_price_date = hist.index[-1].strftime("%Y-%m-%d")

        return {
            "base_price": base_price,
            "base_price_date": base_price_date,
            "avg_base_price_15d": avg_base_price_15d,
            "avg_base_days_used": avg_base_days_used,
            "current_price": current_price,
            "current_price_date": current_price_date,
            "detected_drop_date": worst_day.strftime("%Y-%m-%d"),
            "detected_drop_pct": round(worst_change * 100, 1)
        }
    except Exception as e:
        return {"base_price": None, "current_price": None,
                "price_error": f"Price lookup failed: {str(e)}"}

def get_year_volatility(ticker):
    """
    A convenient 1-year volatility proxy using yfinance's built-in 52-week
    high/low (from ticker.info) rather than computing standard deviation
    ourselves — the spread from low to high, as a percent of the low.
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        high = info.get("fiftyTwoWeekHigh")
        low = info.get("fiftyTwoWeekLow")
        if high is None or low is None or low == 0:
            return {"volatility_pct": None, "volatility_error": "52-week range unavailable"}
        volatility_pct = round((high - low) / low * 100, 1)
        return {
            "volatility_pct": volatility_pct,
            "volatility_52w_high": round(float(high), 2),
            "volatility_52w_low": round(float(low), 2)
        }
    except Exception as e:
        return {"volatility_pct": None, "volatility_error": f"Volatility lookup failed: {str(e)}"}

def get_five_year_trend(ticker):
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
    flagged_count = 0

    for i, h in enumerate(headlines):
        print(f"Classifying {i + 1}/{len(headlines)}: {h['title'][:60]}")
        classification = classify_headline(h)
        merged = {**h, **classification}

        is_flagged = (
            merged.get("confident") is True
            and merged.get("category") not in (None, "not_relevant")
            and isinstance(merged.get("fundamental_impact_score"), int)
            and merged["fundamental_impact_score"] <= 3
        )

        if is_flagged:
            flagged_count += 1
            lookup_ticker = merged.get("ticker")

            if not lookup_ticker:
                time.sleep(SECONDS_BETWEEN_CALLS)
                sub = find_substitute_ticker(h, merged.get("company"))
                if sub.get("has_substitute") and sub.get("substitute_ticker"):
                    merged["is_substitute"] = True
                    merged["substitute_company"] = sub.get("substitute_company")
                    merged["substitute_ticker"] = sub.get("substitute_ticker")
                    merged["substitute_reasoning"] = sub.get("substitute_reasoning")
                    lookup_ticker = sub["substitute_ticker"]
                else:
                    merged["no_investment_angle"] = True

            if lookup_ticker:
                price_data = get_price_data(lookup_ticker)
                merged.update(price_data)
                trend_data = get_five_year_trend(lookup_ticker)
                merged.update(trend_data)
                vol_data = get_year_volatility(lookup_ticker)
                merged.update(vol_data)

        results.append(merged)
        if i < len(headlines) - 1:
            time.sleep(SECONDS_BETWEEN_CALLS)

    output = {
        "generated_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "headlines_reviewed": len(headlines),
        "flagged_count": flagged_count,
        "events": results
    }

    with open("events.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"Processed {len(headlines)} headlines, {flagged_count} flagged. Saved to events.json")

if __name__ == "__main__":
    main()
