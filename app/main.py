from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from disease_detection.router import router as disease_detection_router
from crop_recommendation.router import router as crop_recommendation_router

app = FastAPI(
    title="AlexxaFarms AI Service",
    description="""
    AI Service for AlexxaFarms

    Modules:
    • Plant Disease Detection
    • Crop Recommendation
    """,
    version="2.0.0",
)

# ── CORS — allow the local HTML frontend to call the API ──────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register Routers
app.include_router(disease_detection_router)
app.include_router(crop_recommendation_router)


@app.get("/", tags=["System"])
def root():
    return {
        "message": "AlexxaFarms AI Service is running",
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
        "service": "AlexxaFarms AI Service",
        "version": "2.0.0",
    }