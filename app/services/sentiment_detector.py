# This module provides a class for sentiment analysis using VADER Sentiment Analysis.
# It includes a method to analyze the sentiment of a given text and return the sentiment score.

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import spacy

nlp = spacy.load("en_core_web_sm")

# This module provides a class for sentiment analysis using VADER Sentiment Analysis.
# It includes a method to analyze the sentiment of a given text and return the sentiment score.
class SentimentDetector:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()

    def preprocess_text(self, text: str) -> str:
        # Preprocess the text if needed (e.g., remove special characters, convert to lowercase, convert to lemma, etc.)
        doc = nlp(text)
        tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and token.is_alpha]
        return " ".join(tokens)

    def analyze_sentiment(self, text) -> dict[str, dict[str, float]]:
        # Preprocess the text
        cleaned_text = self.preprocess_text(text)
        # Analyze the sentiment of the text using VADER
        sentiment_score = self.analyzer.polarity_scores(cleaned_text)
        return sentiment_score

# This function is a wrapper around the SentimentDetector class to provide a simple interface for sentiment analysis.
# It takes a text input and returns the sentiment score.
async def analyze(text: str | None = None) -> dict[str, float]:
    if text is None:
        return {"error": "No text provided for sentiment analysis."}
    
    sentiment_detector = SentimentDetector()
    # Analyze the sentiment of the text
    result = sentiment_detector.analyze_sentiment(text)
    print(f"Sentiment analysis result: {result}")
    # Return the sentiment score as a dictionary
    return result