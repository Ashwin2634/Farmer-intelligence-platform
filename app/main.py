from fastapi import FastAPI

from disease_detection.router import router as disease_detection_router
from crop_recommendation.router import router as crop_recommendation_router

app = FastAPI(
    title="PolyHouse AI Service",
    description="""
    AI Service for PolyHouse India

    Modules:
    • Plant Disease Detection
    • Crop Recommendation
    """,
    version="2.0.0",
)

# Register Routers
app.include_router(disease_detection_router)
app.include_router(crop_recommendation_router)


@app.get("/", tags=["System"])
def root():
    return {
        "message": "PolyHouse AI Service is running",
        "version": "2.0.0",
        "status": "online",
        "modules": {
            "disease_detection": "/disease-detection",
            "crop_recommendation": "/crop-recommendation",
        },
        "docs": "/docs",
    }


@app.get("/health", tags=["System"])
def health():
    return {
        "status": "healthy",
        "service": "PolyHouse AI Service",
        "version": "2.0.0",
    }