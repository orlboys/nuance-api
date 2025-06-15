from fastapi import APIRouter
from ..services.bias_detector import predict_bias
from ..schema import BiasResponse, BiasRequest

router = APIRouter()

async def analyze_text(text: str) -> dict:
    try:
        result = await predict_bias(text)
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}

@router.post("/analyze_bias", response_model=BiasResponse)
async def return_analyzed_text(request: BiasRequest) -> BiasResponse:
    # Extract text from the request
    analyzed_text = await analyze_text(request.text)

    # Handle error response
    if "error" in analyzed_text:
        return {"bias": None, "error": analyzed_text["error"]}
    
    return {"bias": analyzed_text["result"], "error": None}
