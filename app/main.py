from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
import io

from app.services.yolo_detector import YOLODetector

app = FastAPI(
    title="PolyHouse AI Service",
    description="AI service for plant disease detection",
    version="2.0.0",
)

detector = YOLODetector()


@app.get("/")
def root():
    return {
        "message": "PolyHouse AI Service is running",
        "model": "YOLO11 Segmentation (V4)",
        "healthy_detection": "No detections = Healthy",
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

        detections = prediction.get("detections", [])

        # ------------------------------------------------------------------
        # Healthy leaf (no lesion detected)
        # ------------------------------------------------------------------
        if len(detections) == 0:
            return {
                "success": True,
                "filename": file.filename,
                "model": prediction.get("model", "YOLO11 Segmentation"),
                "image": prediction.get("image"),
                "inference_time_ms": prediction.get("inference_time_ms"),
                "status": "healthy",
                "diagnosis": "Healthy",
                "message": "No disease symptoms detected.",
                "total_detections": 0,
                "detections": [],
            }

        # ------------------------------------------------------------------
        # Diseased leaf
        # ------------------------------------------------------------------
        top_detection = max(
            detections,
            key=lambda d: d.get("confidence", 0)
        )

        return {
            "success": True,
            "filename": file.filename,
            "model": prediction.get("model", "YOLO11 Segmentation"),
            "image": prediction.get("image"),
            "inference_time_ms": prediction.get("inference_time_ms"),
            "status": "diseased",
            "diagnosis": top_detection.get("class"),
            "confidence": round(top_detection.get("confidence", 0), 4),
            "total_detections": len(detections),
            "detections": detections,
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Detection failed: {str(e)}"
        )