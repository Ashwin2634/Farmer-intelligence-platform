from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
import io

from app.services.yolo_detector import YOLODetector


app = FastAPI(
    title="PolyHouse AI Service",
    description="AI service for plant disease detection",
    version="1.0.0",
)

detector = YOLODetector()


@app.get("/")
def root():
    return {
        "message": "PolyHouse AI Service is running",
        "model": "YOLO11 Segmentation",
        "classes": detector.class_names,
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
    }


@app.post("/detect")
async def detect_image(file: UploadFile = File(...)):

    if not file.content_type:
        raise HTTPException(
            status_code=400,
            detail="No content type found."
        )

    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Uploaded file must be an image."
        )

    try:

        image_bytes = await file.read()

        image = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")

        prediction = detector.detect(image)

        return {
            "success": True,
            "filename": file.filename,
            **prediction,
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Detection failed: {str(e)}"
        )