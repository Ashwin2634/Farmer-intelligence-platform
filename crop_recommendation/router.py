from fastapi import APIRouter

from crop_recommendation.schemas import (
    CropRecommendationRequest,
    CropRecommendationResponse,
)
from crop_recommendation.inference.service import predict_crop

router = APIRouter(
    prefix="/crop-recommendation",
    tags=["Crop Recommendation"],
)


@router.get("/")
def crop_recommendation_root():
    return {
        "message": "Crop Recommendation Service Running"
    }


@router.post(
    "/predict",
    response_model=CropRecommendationResponse,
    summary="Recommend suitable crops",
    description="Returns the Top-3 recommended crops based on soil and environmental conditions."
)
def predict(request: CropRecommendationRequest):
    return predict_crop(request)