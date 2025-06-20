from fastapi import APIRouter
from ..services.sentiment_detector import analyze
from ..schema import SentimentRequest, SentimentResponse, SimpleSentimentResponse, SentimentRequest

router = APIRouter()

async def analyze_text(text: str) -> dict:
    try:
        result = await analyze(text)
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}

@router.post("/analyze_sentiment/full_detail")
async def return_analyzed_text_detailed(request: SentimentRequest) -> SentimentResponse:
    # Endpoint to analyze the sentiment of a given text. GIVES FULL DETAIL - neg, neu, pos, compound
    # :param text: The text to be analyzed.
    # :return: A dictionary containing the sentiment scores (neg, neu, pos, compound) or an error message.
    # :NOTE: Check schema.py for the SentimentResponse model.

    analyzed_text = await analyze_text(request.text)
    # Handle error response
    if "error" in analyzed_text:
        return {"result": None, "error": analyzed_text["error"]}
    return {"sentiment": analyzed_text["result"]}


    # The analyze_text function is called within the route handler to perform the sentiment analysis.

    analyzed_text = await analyze_text(request.text)
    # Handle error response
    if "error" in analyzed_text:
        return {"result": None, "error": analyzed_text["error"]}
    return {"sentiment": analyzed_text["result"]}

@router.post("/analyze_sentiment")
async def return_analyzed_text_simplified(request: SentimentRequest) -> SimpleSentimentResponse:
    # Endpoint to analyze the sentiment of a given text. GIVES ONLY +, -, or 0
    # :param text: The text to be analyzed.
    # :return: A positive (True), negative (False), or neutral (None) sentiment score.
    # :NOTE: Check schema.py for the SimpleSentimentResponse model.

    analyzed_text = await analyze_text(request.text)
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
