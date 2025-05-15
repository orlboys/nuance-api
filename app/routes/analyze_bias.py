from fastapi import APIRouter
from ..serives.bias_detector import analyze
from ..schema import BiasResponse, BiasRequest

router = APIRouter()

async def analyze_text(text: str) -> dict:
    try:
        result = await analyze(text)
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}

@router.post("/analyze_bias", response_model=BiasResponse, request_model=BiasRequest)
async def return_analyzed_text(text: str) -> BiasResponse:
    # Endpoint to analyze the bias of a given text.
    # :param text: The text to be analyzed.
    # :return: A dictionary containing the bias scores (left, right, compound) or an error message.
    # :NOTE: Check schema.py for the BiasResponse model.

    analyzed_text = await analyze_text(text)
    # Handle error response
    if "error" in analyzed_text:
        return {"bias": None, "error": analyzed_text["error"]}
    
    return {"result": analyzed_text["result"]}