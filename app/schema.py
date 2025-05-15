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
    error: str | None = None # Optional error message if any error occurs during analysis

### Sentiment Response Schema (How the response will look like)
class SentimentResponse(BaseModel):
    result: SentimentScores # The result of the sentiment analysis - an instance of SentimentScores
    error: str | None = None # Optional error message if any error occurs during analysis

### Simple Sentiment Response Schema (a simplified version of the SentimentResponse for only +/-/0 sentiment)
class SimpleSentimentResponse(BaseModel):
    sentiment: bool | None = None # True for positive, False for negative, None for neutral
    error: str | None = None # Optional error message if any error occurs during analysis