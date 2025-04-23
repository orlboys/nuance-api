# This is the main entry point for the FastAPI application.
# It initializes the FastAPI app and sets up the API endpoint for sentiment analysis.
from dotenv import load_dotenv
import os
from fastapi import FastAPI
from .routes import routers

load_dotenv()
app = FastAPI()

# Includes all routers from the routes package
# This allows for easy addition of new routes in the future
for router in routers:
    app.include_router(router)