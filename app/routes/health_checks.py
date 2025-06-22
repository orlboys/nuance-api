from fastapi import APIRouter

router = APIRouter()

@router.get("/")
async def root():
    # This endpoint is used to check if the Nuance API is running.
    return {"message": "Nuance API is running!"}

@router.get("/health")
async def health_check():
    return {"status": "healthy", "message": "Nuance API is healthy!"}