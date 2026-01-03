from flask import Flask, request, jsonify, send_from_directory
import pandas as pd
import re
import random
from ntscraper import Nitter
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

app = Flask(__name__)

analyzer = SentimentIntensityAnalyzer()
scraper = Nitter()

def clean_text(text):
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'https?://\S+', '', text)
    return text.strip()

def get_sentiment(text):
    score = analyzer.polarity_scores(text)['compound']
    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    return "Neutral"

@app.route("/")
def home():
    return send_from_directory(".", "index.html")

@app.route("/analyze_topic", methods=["POST"])
def analyze_topic():
    topic = request.form.get("topic", "General")
    tweets = []
    status = "live"

    try:
        # This is where your IndexError happened. We now catch it.
        scraped = scraper.get_tweets(topic, mode="term", number=15)
        if scraped and scraped.get("tweets"):
            tweets = [{"text": t["text"]} for t in scraped["tweets"]]
        else:
            raise Exception("No live tweets found")
    except Exception as e:
        # FALLBACK: If Nitter is down/blocked, generate simulated data
        status = "simulated"
        tweets = [
            {"text": f"I think {topic} is absolutely amazing and game-changing!"},
            {"text": f"Using {topic} has been a terrible experience so far."},
            {"text": f"Just another day talking about {topic}. It's okay."},
            {"text": f"The latest news about {topic} is very positive!"},
            {"text": f"I'm so frustrated with how {topic} works."},
            {"text": f"Neutral feelings about {topic} for now."},
            {"text": f"Best decision ever to switch to {topic}!"},
            {"text": f"Avoid {topic} if you can. Not worth it."},
        ]

    df = pd.DataFrame(tweets)
    df["clean"] = df["text"].apply(clean_text)
    df["sentiment"] = df["clean"].apply(get_sentiment)

    counts = df["sentiment"].value_counts().to_dict()
    for cat in ["Positive", "Negative", "Neutral"]:
        if cat not in counts: counts[cat] = 0
            
    return jsonify({
        "counts": counts,
        "status": status,
        "topic": topic
    })

@app.route("/analyze_text", methods=["POST"])
def analyze_text():
    text = request.form.get("text", "")
    if not text: return jsonify({"error": "Empty text"}), 400
    
    scores = analyzer.polarity_scores(text)
    return jsonify({
        "sentiment": get_sentiment(text),
        "scores": scores
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)