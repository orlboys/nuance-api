from fastapi import APIRouter
from ..services.sentiment_detector import analyze
from ..schema import SentimentResponse, SimpleSentimentResponse

router = APIRouter()

async def analyze_text(text: str) -> dict:
    try:
        result = await analyze(text)
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}

@router.post("/analyze_sentiment/full_detail", response_model=SentimentResponse)
async def return_analyzed_text_detailed(text: str) -> SentimentResponse:
    # Endpoint to analyze the sentiment of a given text. GIVES FULL DETAIL - neg, neu, pos, compound
    # :param text: The text to be analyzed.
    # :return: A dictionary containing the sentiment scores (neg, neu, pos, compound) or an error message.
    # :NOTE: Check schema.py for the SentimentResponse model.

    # The analyze_text function is called within the route handler to perform the sentiment analysis.
    return await analyze_text(text)

@router.post("/analyze_sentiment", response_model=SimpleSentimentResponse)
async def return_analyzed_text_simplified(text: str) -> SimpleSentimentResponse:
    # Endpoint to analyze the sentiment of a given text. GIVES ONLY +, -, or 0
    # :param text: The text to be analyzed.
    # :return: A positive (True), negative (False), or neutral (None) sentiment score.
    # :NOTE: Check schema.py for the SimpleSentimentResponse model.

    analyzed_text = await analyze_text(text)
    # Handle error response
    if "error" in analyzed_text:
        return {"sentiment": None, "error": analyzed_text["error"]}
    
    compound = analyzed_text["result"]["compound"]
    if compound > 0.02:
        return {"sentiment": True}
    elif compound < -0.02:
        return {"sentiment": False}
    else:
        return {"sentiment": None}
