from pydantic import BaseModel, Field, ConfigDict


class CropRecommendationRequest(BaseModel):
    """
    Request schema for Crop Recommendation API
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "nitrogen": 90,
                "phosphorus": 42,
                "potassium": 43,
                "temperature": 25,
                "humidity": 82,
                "ph": 6.5,
            }
        }
    )

    nitrogen: float = Field(
        ...,
        ge=0,
        le=500,
        description="Available Nitrogen (kg/ha)"
    )

    phosphorus: float = Field(
        ...,
        ge=0,
        le=300,
        description="Available Phosphorus (kg/ha)"
    )

    potassium: float = Field(
        ...,
        ge=0,
        le=500,
        description="Available Potassium (kg/ha)"
    )

    temperature: float = Field(
        ...,
        ge=5,
        le=50,
        description="Ambient Temperature (°C)"
    )

    humidity: float = Field(
        ...,
        ge=0,
        le=100,
        description="Relative Humidity (%)"
    )

    ph: float = Field(
        ...,
        ge=3.5,
        le=9.5,
        description="Soil pH"
    )

class CropRecommendationItem(BaseModel):
    """
    Single crop recommendation
    """

    crop: str
    confidence: float = Field(
        ...,
        ge=0,
        le=100,
        description="Prediction confidence (%)"
    )


class CropRecommendationResponse(BaseModel):
    """
    API response schema
    """

    recommendations: list[CropRecommendationItem]