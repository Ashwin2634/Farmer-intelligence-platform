import io
import os
import tempfile

from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from PIL import Image

from disease_detection.services.yolo_detector import EfficientNetDetector
from disease_detection.services.video_processor import (
    VideoDiseaseAnalyzer,
    VideoProcessingError,
)

router = APIRouter(
    prefix="/disease-detection",
    tags=["Disease Detection"],
)

detector = EfficientNetDetector()
video_analyzer = VideoDiseaseAnalyzer(detector, max_duration_s=60)

MAX_VIDEO_BYTES = 100 * 1024 * 1024  # 100 MB
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".3gp", ".m4v"}


def _media_kind(file: UploadFile):
    content_type = (file.content_type or "").lower()
    ext = os.path.splitext(file.filename or "")[1].lower()

    if content_type.startswith("image/"):
        return "image"
    if content_type.startswith("video/") or ext in VIDEO_EXTS:
        return "video"
    return None


async def _save_upload_to_temp(file: UploadFile, max_bytes: int) -> str:
    """Stream the upload to disk in chunks (OpenCV needs a file path)."""
    suffix = os.path.splitext(file.filename or "")[1] or ".mp4"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    size = 0
    try:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            if size > max_bytes:
                raise HTTPException(
                    status_code=413,
                    detail=f"Video too large. Max size is {max_bytes // (1024 * 1024)} MB.",
                )
            tmp.write(chunk)
    except BaseException:
        tmp.close()
        os.unlink(tmp.name)
        raise
    tmp.close()
    return tmp.name


def _build_response(prediction: dict, filename: str, media_type: str) -> dict:
    treatment = prediction.get("treatment")
    is_healthy = prediction.get("is_healthy", False)

    response = {
        "success": True,
        "filename": filename,
        "media_type": media_type,
        "model": prediction.get("model"),
        "inference_time_ms": prediction.get("inference_time_ms"),
        "confidence": prediction.get("confidence"),
        "class_name": prediction.get("class_name"),
    }

    if media_type == "image":
        response["image"] = prediction.get("image")
    else:
        response["video"] = prediction.get("video")
        response["processing_time_ms"] = prediction.get("processing_time_ms")
        response["analysis"] = prediction.get("analysis")

    if is_healthy:
        response.update({
            "status": "healthy",
            "diagnosis": "Healthy",
            "message": "No disease symptoms detected.",
            "treatment": None,
        })

        # Video only: overall result is healthy, but some frames showed a disease
        issues = (prediction.get("analysis") or {}).get("detected_issues", [])
        if issues:
            names = ", ".join(i["class_name"] for i in issues)
            response["warning"] = (
                f"Overall result is healthy, but possible disease was seen in part of the video: {names}."
            )
        return response

    # Diseased
    response.update({
        "status": "diseased",
        "crop": prediction.get("crop"),
        "diagnosis": f"{prediction.get('crop')} - {prediction.get('disease')}".strip(" -"),
        "treatment": treatment,
    })

    if treatment:
        response["treatment_summary"] = {
            "pathogen": treatment.get("pathogen"),
            "organic_treatment": treatment.get("organic_treatment", []),
            "chemical_treatment": treatment.get("chemical_treatment", []),
            "prevention": treatment.get("prevention", []),
            "polyhouse_specific": treatment.get("polyhouse_specific", []),
            "severity_levels": treatment.get("severity_levels", {}),
        }

    return response


@router.post("/detect")
async def detect(
    file: UploadFile = File(...),
    num_frames: int = Query(16, ge=4, le=64, description="Frames to sample (video only)"),
    include_frames: bool = Query(False, description="Return per-frame results (video only)"),
    crop_mode: str = Query("center", pattern="^(squash|center|multi)$", description="Video crop mode"),
    min_view_confidence: float = Query(0.5, ge=0.0, le=0.95),
    debug: bool = Query(False, description="Save analyzed crops to ./debug_frames (video only)"),
):
    kind = _media_kind(file)
    if kind is None:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file must be an image or a video.",
        )

    # ---------------- IMAGE ----------------
    if kind == "image":
        try:
            image_bytes = await file.read()
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            prediction = await run_in_threadpool(detector.predict, image)
            return _build_response(prediction, file.filename, "image")
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

    # ---------------- VIDEO ----------------
    
    tmp_path = None
    try:
        tmp_path = await _save_upload_to_temp(file, MAX_VIDEO_BYTES)
        debug_dir = None
        if debug:
            debug_dir = os.path.join("debug_frames", datetime.now().strftime("%Y%m%d_%H%M%S"))

        prediction = await run_in_threadpool(
            video_analyzer.analyze,
            tmp_path,
            num_frames=num_frames,
            include_frames=include_frames,
            crop_mode=crop_mode,
            min_view_confidence=min_view_confidence,
            debug_dir=debug_dir,
        )
        return _build_response(prediction, file.filename, "video")
    except HTTPException:
        raise
    except VideoProcessingError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Video detection failed: {str(e)}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)