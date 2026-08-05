import pandas as pd

from crop_recommendation.inference.model_loader import model, encoder


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

    top3_indices = probabilities.argsort()[-3:][::-1]

    recommendations = []

    for idx in top3_indices:

        recommendations.append(
            {
                "crop": encoder.classes_[idx],
                "confidence": round(float(probabilities[idx] * 100), 2),
            }
        )

    return {
        "recommendations": recommendations
    }