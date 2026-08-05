import joblib

from crop_recommendation.config import MODEL_DIR

MODEL_PATH = MODEL_DIR / "xgboost_crop_model_v3.pkl"
ENCODER_PATH = MODEL_DIR / "label_encoder_v3.pkl"

model = joblib.load(MODEL_PATH)

encoder = joblib.load(ENCODER_PATH)