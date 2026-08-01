from fastapi import APIRouter

router = APIRouter(
    prefix="/crop-recommendation",
    tags=["Crop Recommendation"],
)


@router.get("/")
def crop_recommendation_root():
    return {
        "message": "Crop Recommendation module - coming soon",
        "status": "placeholder",
    }
