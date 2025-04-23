# This is the main entry point for the FastAPI application.
# It initializes the FastAPI app and sets up the API endpoint for sentiment analysis.
from dotenv import load_dotenv
import os
from fastapi import FastAPI
from services.sentiment_detector import analyze

load_dotenv()
app = FastAPI()

@app.post("/analyze")
async def analyze_text(text: str) -> dict[str, dict[str, float]]:
    # Call the analyze function from the analysis module
    try:
        result = analyze(text)
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}