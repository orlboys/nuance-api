from fastapi import APIRouter
from ..services.bias_detector import analyze
from ..schema import BiasResponse, BiasRequest

router = APIRouter()

async def analyze_text(text: str) -> dict:
    # Function to analyze the bias of a given text.
    # :param text: The text to be analyzed.
    # :return: A dictionary containing the bias score or an error message.
    try:
        result = await analyze(text)
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}

@router.post("/analyze_bias", response_model:BiasResponse, request_model:BiasRequest)
async def return_analyzed_text(text: str) -> BiasResponse:
    # Endpoint to analyze the bias of a given text.
    # :param text: The text to be analyzed.
    # :return: A dictionary containing the bias score or an error message.
    # :NOTE: Check schema.py for the BiasResponse and BiasRequest models.

    try:
        result = await analyze(text)
        return result
    except Exception as e:
        return {"error": str(e)}