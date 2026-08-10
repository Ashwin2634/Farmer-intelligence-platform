from typing import Optional, Literal
from pydantic import BaseModel, Field, ConfigDict, model_validator


class CropRecommendationRequest(BaseModel):
    """
    Request schema for Crop Recommendation API supporting both full names
    (nitrogen, phosphorus, potassium) and shorthand/aliases (N, P, K), as well as
    optional category filtering (vegetable | fruit | flower | herb).
    """

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "category": "vegetable",
                "nitrogen": 90,
                "phosphorus": 42,
                "potassium": 43,
                "temperature": 25,
                "humidity": 82,
                "ph": 6.5,
            }
        }
    )

    category: Optional[Literal["vegetable", "fruit", "flower", "herb"]] = Field(
        default=None,
        description="Optional category filter: vegetable | fruit | flower | herb"
    )

    nitrogen: Optional[float] = Field(
        None,
        alias="N",
        ge=0,
        le=500,
        description="Available Nitrogen (kg/ha)"
    )

    phosphorus: Optional[float] = Field(
        None,
        alias="P",
        ge=0,
        le=300,
        description="Available Phosphorus (kg/ha)"
    )

    potassium: Optional[float] = Field(
        None,
        alias="K",
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

    @model_validator(mode="before")
    @classmethod
    def check_npk_fields(cls, values):
        if isinstance(values, dict):
            # Check nitrogen / N
            if values.get("nitrogen") is None and values.get("N") is not None:
                values["nitrogen"] = values.get("N")
            elif values.get("N") is None and values.get("nitrogen") is not None:
                values["N"] = values.get("nitrogen")

            # Check phosphorus / P
            if values.get("phosphorus") is None and values.get("P") is not None:
                values["phosphorus"] = values.get("P")
            elif values.get("P") is None and values.get("phosphorus") is not None:
                values["P"] = values.get("phosphorus")

            # Check potassium / K
            if values.get("potassium") is None and values.get("K") is not None:
                values["potassium"] = values.get("K")
            elif values.get("K") is None and values.get("potassium") is not None:
                values["K"] = values.get("potassium")

            if values.get("nitrogen") is None:
                raise ValueError("Nitrogen field ('nitrogen' or 'N') is required.")
            if values.get("phosphorus") is None:
                raise ValueError("Phosphorus field ('phosphorus' or 'P') is required.")
            if values.get("potassium") is None:
                raise ValueError("Potassium field ('potassium' or 'K') is required.")

        return values

class CropRecommendationItem(BaseModel):
    """
    Single crop recommendation
    """

    crop: str
    category: str = Field(..., description="Crop category: vegetable, fruit, flower, herb")
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