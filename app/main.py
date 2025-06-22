# This is the main entry point for the FastAPI application.
# It initializes the FastAPI app and sets up the API endpoint for sentiment analysis.
from dotenv import load_dotenv
import os
from fastapi import FastAPI
from .routes import routers
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()
app = FastAPI(
    title="Nuance API",
    description="API for sentiment and bias analysis using NLP models.",
    version="0.1.0",
)

# CORS Middleware to allow cross-origin requests (aka let my frontend talk to my backend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[ # be specific here! don't allow all origins in production
        "http://localhost:3000",  # localhost for development
        "http://127.0.0.1:3000",  # Localhost for development
        "https://frontend-domain.com",  # frontend production domain - UNKNOWN AT THE MOMENT SO AS A PLACEHOLDER
    ],  # Allows all origins, adjust as needed
    allow_credentials=True,
    allow_methods=["GET", "POST"],  # Allows only GET and POST methods - retrieve data and request data
    allow_headers=["*"],  # Allows all headers
)

# Includes all routers from the routes package
# This allows for easy addition of new routes in the future
for router in routers:
    app.include_router(router)