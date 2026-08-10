import json
import pandas as pd
from crop_recommendation.config import BASE_DIR
from crop_recommendation.inference.model_loader import model, encoder

# Load crop metadata at startup
METADATA_PATH = BASE_DIR / "crop_metadata.json"
with open(METADATA_PATH, "r", encoding="utf-8") as f:
    CROP_METADATA = json.load(f)

FEATURE_ORDER = [
    "N",
    "P",
    "K",
    "temperature",
    "humidity",
    "ph",
]


def predict_crop(data):

    sample = pd.DataFrame(
        [[
            data.nitrogen,
            data.phosphorus,
            data.potassium,
            data.temperature,
            data.humidity,
            data.ph,
        ]],
        columns=FEATURE_ORDER,
    )

    probabilities = model.predict_proba(sample)[0]

    all_predictions = []
    for idx, prob in enumerate(probabilities):
        crop = encoder.classes_[idx]
        crop_info = CROP_METADATA.get(crop, {})
        category = crop_info.get("category", "unknown")

        all_predictions.append(
            {
                "crop": crop,
                "category": category,
                "confidence": round(float(prob * 100), 2),
            }
        )

    # Filter by category if requested
    if data.category:
        target_category = data.category.lower()
        all_predictions = [
            p for p in all_predictions if p["category"] == target_category
        ]

    # Sort descending by confidence
    all_predictions.sort(key=lambda x: x["confidence"], reverse=True)

    # Return Top-3 predictions
    recommendations = all_predictions[:3]

    return {
        "recommendations": recommendations
    }