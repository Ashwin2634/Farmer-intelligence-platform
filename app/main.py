from fastapi import FastAPI

from disease_detection.router import router as disease_detection_router
from crop_recommendation.router import router as crop_recommendation_router

app = FastAPI(
    title="PolyHouse AI Service",
    description="AI service for plant disease detection and crop recommendation",
    version="2.0.0",
)

# Mount module routers
app.include_router(disease_detection_router)
app.include_router(crop_recommendation_router)


@app.get("/")
def root():
    return {
        "message": "PolyHouse AI Service is running",
        "modules": [
            "disease-detection",
            "crop-recommendation",
        ],
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
    }