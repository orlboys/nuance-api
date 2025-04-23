# This module provides a class for sentiment analysis using VADER Sentiment Analysis.
# It includes a method to analyze the sentiment of a given text and return the sentiment score.

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# This module provides a class for sentiment analysis using VADER Sentiment Analysis.
# It includes a method to analyze the sentiment of a given text and return the sentiment score.
class SentimentDetector:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()

    def analyze_sentiment(self, text):
        sentiment_score = self.analyzer.polarity_scores(text)
        return sentiment_score

# This function is a wrapper around the SentimentDetector class to provide a simple interface for sentiment analysis.
# It takes a text input and returns the sentiment score.
def analyze(text: str | None = None) -> dict[str, float]:
    if text is None:
        return {"error": "No text provided for sentiment analysis."}
    
    sentiment_detector = SentimentDetector()
    result = sentiment_detector.analyze_sentiment(text)
    print(f"Sentiment analysis result: {result}")
    # Return the sentiment score as a dictionary
    return result