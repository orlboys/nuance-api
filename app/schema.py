# This module provdies the schemaas for the FastAPI application.
# It defines the data models used for request and response validation.
# It uses Pydantic for data validation and serialization.

from pydantic import BaseModel

## Sentiment Analysis Schema

### Sentiment Response Schema
class SentimentScores(BaseModel):
    neg: float
    neu: float
    pos: float
    compound: float

### Sentiment Response Schema (How the response will look like)
class SentimentResponse(BaseModel):
    result: SentimentScores # The result of the sentiment analysis - a dictionary with sentiment scores
    error: str | None = None # Optional error message if any error occurs during analysis

### Simple Sentiment Response Schema (a simplified version of the SentimentResponse for only +/-/0 sentiment)
class SimpleSentimentResponse(BaseModel):
    sentiment: bool | None = None # True for positive, False for negative, None for neutral
    error: str | None = None # Optional error message if any error occurs during analysis

### Sentiment Request Schema
class SentimentRequest(BaseModel):
    text: str # The text to be analyzed for sentiment
    model: str | None = None # Optional model name to be used for analysis
    threshold: float | None = None # Optional threshold for sentiment classification

### Bias Detection Request Schema
class BiasDetectionRequest(BaseModel):
    text: str # The text to be analyzed for bias
    model: str | None = None # Optional model name to be used for analysis
    threshold: float | None = None # Optional threshold for bias classification

class BiasDetectionResponse(BaseModel):
    result: dict # The result of the bias detection - a dictionary with bias scores
    error: str | None = None # Optional error message if any error occurs during analysis