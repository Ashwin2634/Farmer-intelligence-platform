from fastapi import APIRouter, UploadFile, File, HTTPException
from PIL import Image
import io

from disease_detection.services.yolo_detector import YOLODetector

router = APIRouter(
    prefix="/disease-detection",
    tags=["Disease Detection"],
)

detector = YOLODetector()


@router.post("/detect")
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

        crop_name = top_detection.get("crop", "")
        disease_name = top_detection.get("disease", "")
        diagnosis_str = f"{crop_name} - {disease_name}".strip(" -")

        return {
            "success": True,
            "filename": file.filename,
            "model": prediction.get("model", "YOLO11 Segmentation"),
            "crop": crop_name,
            "image": prediction.get("image"),
            "inference_time_ms": prediction.get("inference_time_ms"),
            "status": "diseased",
            "diagnosis": diagnosis_str or "Diseased",
            "confidence": round(top_detection.get("confidence", 0), 4),
            "total_detections": len(detections),
            "detections": detections,
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Detection failed: {str(e)}"
        )
