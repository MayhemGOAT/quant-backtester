"""FinBERT-based financial news sentiment analysis for FinPulse."""

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf


class FinBERTAnalyzer:
    """Score financial headlines with ProsusAI/finbert."""

    MODEL_NAME = "ProsusAI/finbert"

    def __init__(self):
        self._pipeline = None

    @property
    def pipeline(self):
        if self._pipeline is None:
            from transformers import pipeline

            print("Loading FinBERT sentiment model...")
            self._pipeline = pipeline(
                "sentiment-analysis",
                model=self.MODEL_NAME,
                tokenizer=self.MODEL_NAME,
                top_k=None,
            )
            print("FinBERT loaded.")
        return self._pipeline

    def score_headline(self, headline):
        """Return positive, negative, neutral probabilities for one headline."""
        if not headline or not str(headline).strip():
            return {"positive": 0.0, "negative": 0.0, "neutral": 1.0}

        results = self.pipeline(str(headline)[:512])[0]
        if results and isinstance(results[0], dict):
            label_scores = results
        else:
            label_scores = results[0] if results else []
        scores = {item["label"]: item["score"] for item in label_scores}
        return {
            "positive": scores.get("positive", 0.0),
            "negative": scores.get("negative", 0.0),
            "neutral": scores.get("neutral", 0.0),
        }

    def score_headlines(self, headlines):
        """Score multiple headlines and return aggregate probabilities."""
        if not headlines:
            return {"positive": 0.0, "negative": 0.0, "neutral": 1.0, "count": 0}

        totals = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
        for headline in headlines:
            scores = self.score_headline(headline)
            for label in totals:
                totals[label] += scores[label]

        count = len(headlines)
        return {
            "positive": totals["positive"] / count,
            "negative": totals["negative"] / count,
            "neutral": totals["neutral"] / count,
            "count": count,
        }

    @staticmethod
    def _parse_article(article):
        """Extract headline metadata from current and legacy yfinance news formats."""
        content = article.get("content", article)

        title = (content.get("title") or article.get("title") or "").strip()
        if not title:
            return None

        publisher = content.get("provider", {}).get("displayName")
        if not publisher:
            publisher = article.get("publisher", "unknown")

        published = content.get("pubDate") or content.get("displayTime")
        if published:
            published = pd.Timestamp(published).date()
        else:
            timestamp = article.get("providerPublishTime")
            if timestamp:
                published = datetime.fromtimestamp(timestamp, tz=timezone.utc).date()
            else:
                published = datetime.now(timezone.utc).date()

        return {
            "date": published,
            "headline": title,
            "publisher": publisher,
        }

    def fetch_news(self, ticker):
        """Fetch recent Yahoo Finance headlines for a ticker."""
        stock = yf.Ticker(ticker)
        articles = stock.news or []

        rows = []
        for article in articles:
            parsed = self._parse_article(article)
            if parsed:
                rows.append(parsed)

        if not rows:
            print(f"No recent news found for {ticker}.")
            return pd.DataFrame(columns=["date", "headline", "publisher"])

        news_df = pd.DataFrame(rows).drop_duplicates(subset=["headline"])
        print(f"Fetched {len(news_df)} unique headlines for {ticker}.")
        return news_df

    def build_daily_sentiment(self, news_df, start_date=None, end_date=None):
        """Aggregate FinBERT scores into one row per calendar day."""
        if news_df.empty:
            return pd.DataFrame(
                columns=[
                    "Sentiment_Score",
                    "Sentiment_Positive",
                    "Sentiment_Negative",
                    "Sentiment_Neutral",
                    "News_Count",
                ]
            ), {}

        daily_rows = []
        for date, group in news_df.groupby("date"):
            scores = self.score_headlines(group["headline"].tolist())
            daily_rows.append(
                {
                    "date": pd.Timestamp(date),
                    "Sentiment_Score": scores["positive"] - scores["negative"],
                    "Sentiment_Positive": scores["positive"],
                    "Sentiment_Negative": scores["negative"],
                    "Sentiment_Neutral": scores["neutral"],
                    "News_Count": scores["count"],
                }
            )

        sentiment_df = pd.DataFrame(daily_rows).set_index("date").sort_index()

        if start_date is not None and end_date is not None:
            range_end = max(pd.Timestamp(end_date), sentiment_df.index.max())
            full_index = pd.date_range(start=start_date, end=range_end, freq="D")
            sentiment_df = sentiment_df.reindex(full_index)

        sentiment_df["News_Count"] = sentiment_df["News_Count"].fillna(0)
        for column in [
            "Sentiment_Score",
            "Sentiment_Positive",
            "Sentiment_Negative",
            "Sentiment_Neutral",
        ]:
            sentiment_df[column] = sentiment_df[column].ffill(limit=3).fillna(0.0)

        aggregate = self.score_headlines(news_df["headline"].tolist())
        return sentiment_df, aggregate


def add_sentiment_features(df, ticker, analyzer=None):
    """Merge FinBERT daily sentiment features into the price dataframe."""
    analyzer = analyzer or FinBERTAnalyzer()

    print(f"\nRunning FinBERT sentiment analysis for {ticker}...")
    news_df = analyzer.fetch_news(ticker)
    sentiment_df, aggregate = analyzer.build_daily_sentiment(
        news_df,
        start_date=df.index.min(),
        end_date=df.index.max(),
    )

    merged = df.copy()
    merged.index = pd.to_datetime(merged.index).normalize()
    merged = merged.join(sentiment_df, how="left")

    merged["News_Count"] = pd.to_numeric(merged["News_Count"], errors="coerce").fillna(0)
    for column in [
        "Sentiment_Score",
        "Sentiment_Positive",
        "Sentiment_Negative",
        "Sentiment_Neutral",
    ]:
        merged[column] = pd.to_numeric(merged[column], errors="coerce").ffill(limit=3).fillna(0.0)

    if aggregate.get("count", 0) > 0:
        trailing_days = merged.index[-5:]
        merged.loc[trailing_days, "Sentiment_Score"] = (
            aggregate["positive"] - aggregate["negative"]
        )
        merged.loc[trailing_days, "Sentiment_Positive"] = aggregate["positive"]
        merged.loc[trailing_days, "Sentiment_Negative"] = aggregate["negative"]
        merged.loc[trailing_days, "Sentiment_Neutral"] = aggregate["neutral"]
        merged.loc[trailing_days, "News_Count"] = aggregate["count"]

    merged["Sentiment_MA_5"] = merged["Sentiment_Score"].rolling(5).mean().fillna(0.0)
    merged["Sentiment_MA_20"] = merged["Sentiment_Score"].rolling(20).mean().fillna(0.0)
    merged["Sentiment_Momentum_5"] = merged["Sentiment_Score"].diff(5).fillna(0.0)
    merged["Sentiment_Volatility_10"] = merged["Sentiment_Score"].rolling(10).std().fillna(0.0)

    merged["Sentiment_Bullish"] = (merged["Sentiment_Score"] > 0.1).astype(int)
    merged["Sentiment_Bearish"] = (merged["Sentiment_Score"] < -0.1).astype(int)

    merged = merged.dropna()
    scored_days = int((sentiment_df["News_Count"] > 0).sum()) if not sentiment_df.empty else 0
    matched_days = int(merged["News_Count"].gt(0).sum())
    print(
        "Sentiment features added: "
        f"{scored_days} headline days scored, "
        f"{matched_days} trading days aligned to news."
    )
    return merged, analyzer
