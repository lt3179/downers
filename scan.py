import os
import json
import requests
import google.generativeai as genai

NEWSDATA_API_KEY = os.environ["NEWSDATA_API_KEY"]
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.5-flash")

def fetch_headlines():
    url = "https://newsdata.io/api/1/news"
    params = {
        "apikey": NEWSDATA_API_KEY,
        "category": "business",
        "language": "en",
        "country": "us"
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
            "source": article.get("source_id", "")
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

def classify_headline(headline):
    prompt = CLASSIFICATION_PROMPT.format(
        title=headline["title"],
        description=headline["description"] or "(no description)"
    )
    response = model.generate_content(prompt)
    text = response.text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"error": "Could not parse response", "raw": text}

def main():
    headlines = fetch_headlines()
    results = []
    for h in headlines:
        classification = classify_headline(h)
        results.append({**h, **classification})

    with open("events.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"Processed {len(headlines)} headlines. Saved to events.json")

if __name__ == "__main__":
    main()
