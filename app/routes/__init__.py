# __init__.py
# This module imports and aggregates all the route modules for the API as a single package.

from .analyze_sentiment import router as sentiment_router

# In the future, import other routes here
# from .auth import router as auth_router

# Aggregate all routes in one place
routers = [
    sentiment_router,
    # Add more routers as you go
]
