# disease_detection/services/video_processor.py
import os
import time
from collections import Counter
from typing import Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image

from disease_detection.services.yolo_detector import EfficientNetDetector

CROP_MODES = ("squash", "center", "multi")


class VideoProcessingError(Exception):
    """Raised when a video cannot be read or has no usable frames."""


def probe_video(path: str) -> Dict:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise VideoProcessingError("Could not open the video file. It may be corrupted or unsupported.")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()

    duration = round(total / fps, 2) if fps > 0 and total > 0 else None
    return {
        "total_frames": total,
        "fps": round(fps, 2) if fps > 0 else None,
        "duration_s": duration,
        "width": width,
        "height": height,
    }


def _sharpness(frame_bgr: np.ndarray) -> float:
    """Variance of Laplacian on the centre square at a fixed size."""
    h, w = frame_bgr.shape[:2]
    s = min(h, w)
    y0, x0 = (h - s) // 2, (w - s) // 2
    gray = cv2.cvtColor(frame_bgr[y0:y0 + s, x0:x0 + s], cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (256, 256), interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _downscale(frame: np.ndarray, max_side: int) -> np.ndarray:
    h, w = frame.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1:
        return frame
    return cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def extract_frames(
    path: str,
    num_frames: int = 16,
    filter_blur: bool = True,
    max_side: int = 1280,
) -> List[Tuple[int, Image.Image]]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise VideoProcessingError("Could not open the video file.")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0) or 30.0

    n_candidates = num_frames * 2 if filter_blur else num_frames
    candidates: List[Tuple[int, np.ndarray]] = []

    if total > 0:
        n_candidates = min(n_candidates, total)
        targets = set(np.linspace(0, total - 1, n_candidates).astype(int).tolist())
        last = max(targets)
        idx = 0
        while idx <= last:
            if not cap.grab():
                break
            if idx in targets:
                ok, frame = cap.retrieve()
                if ok and frame is not None:
                    candidates.append((idx, _downscale(frame, max_side)))
            idx += 1
    else:
        step = max(1, int(round(fps)))
        idx = 0
        while len(candidates) < n_candidates:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % step == 0:
                candidates.append((idx, _downscale(frame, max_side)))
            idx += 1

    cap.release()

    if not candidates:
        raise VideoProcessingError("No readable frames found in the video.")

    if filter_blur and len(candidates) > num_frames:
        scored = sorted(candidates, key=lambda c: _sharpness(c[1]), reverse=True)[:num_frames]
        candidates = sorted(scored, key=lambda c: c[0])

    return [
        (idx, Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        for idx, frame in candidates
    ]


def make_views(img: Image.Image, mode: str) -> List[Image.Image]:
    """Turn one frame into 1 or 3 model inputs without distorting the aspect ratio."""
    w, h = img.size
    if mode == "squash" or w == h:
        return [img]

    s = min(w, h)
    fractions = (0.5,) if mode == "center" else (0.0, 0.5, 1.0)
    views = []
    for frac in fractions:
        if w > h:
            left, top = int((w - s) * frac), 0
        else:
            left, top = 0, int((h - s) * frac)
        views.append(img.crop((left, top, left + s, top + s)))
    return views


class VideoDiseaseAnalyzer:
    def __init__(
        self,
        detector: EfficientNetDetector,
        max_duration_s: float = 60.0,
        filter_blur: bool = True,
        max_side: int = 1280,
        batch_size: int = 8,
    ):
        self.detector = detector
        self.max_duration_s = max_duration_s
        self.filter_blur = filter_blur
        self.max_side = max_side
        self.batch_size = batch_size

    def analyze(
        self,
        video_path: str,
        num_frames: int = 16,
        min_issue_ratio: float = 0.25,
        include_frames: bool = False,
        crop_mode: str = "center",
        min_view_confidence: float = 0.5,
        debug_dir: str = None,
    ) -> Dict:
        if crop_mode not in CROP_MODES:
            raise VideoProcessingError(f"crop_mode must be one of {CROP_MODES}")

        t_start = time.perf_counter()
        names = self.detector.class_names

        info = probe_video(video_path)
        if info["duration_s"] and info["duration_s"] > self.max_duration_s:
            raise VideoProcessingError(
                f"Video is too long ({info['duration_s']}s). Maximum allowed is {int(self.max_duration_s)}s."
            )

        frames = extract_frames(
            video_path,
            num_frames=num_frames,
            filter_blur=self.filter_blur,
            max_side=self.max_side,
        )
        frame_indices = [i for i, _ in frames]
        n_frames = len(frames)

        # One frame -> 1 or 3 square views
        views: List[Image.Image] = []
        view_frame: List[int] = []
        for k, (_, img) in enumerate(frames):
            for v in make_views(img, crop_mode):
                views.append(v)
                view_frame.append(k)
        view_frame = np.array(view_frame)

        t_infer = time.perf_counter()
        probs = self.detector.predict_probs(views, batch_size=self.batch_size)  # [V, C]
        inference_ms = round((time.perf_counter() - t_infer) * 1000, 2)

        conf = probs.max(axis=1)
        top_ids = probs.argmax(axis=1)

        # Drop views where the model is unsure (background, hands, blur).
        # Always keep at least a few so we can still answer.
        keep = conf >= min_view_confidence
        min_keep = min(len(views), max(3, int(np.ceil(0.25 * len(views)))))
        if keep.sum() < min_keep:
            order = np.argsort(conf)[::-1][:min_keep]
            keep = np.zeros(len(views), dtype=bool)
            keep[order] = True

        kp, kc, kt = probs[keep], conf[keep], top_ids[keep]
        n_used = len(kp)

        # Confidence-weighted average of probabilities
        weighted = (kp * kc[:, None]).sum(axis=0) / kc.sum()
        final_id = int(weighted.argmax())
        agreement = float((kt == final_id).mean())

        # ---------- Debug: save every view with its prediction ----------
        if debug_dir:
            os.makedirs(debug_dir, exist_ok=True)
            for j, v in enumerate(views):
                tag = names[int(top_ids[j])].replace("/", "_")[:45]
                status = "keep" if keep[j] else "drop"
                fname = f"f{int(view_frame[j]):02d}_v{j:02d}_{status}_{tag}_{conf[j]:.2f}.jpg"
                v.save(os.path.join(debug_dir, fname), quality=90)

        top3_ids = np.argsort(weighted)[::-1][:3]
        top_predictions = [
            {"class_name": names[int(i)], "mean_confidence": round(float(weighted[i]), 4)}
            for i in top3_ids
        ]

        counts = Counter(kt.tolist())
        class_distribution = [
            {"class_name": names[cid], "frames": cnt, "ratio": round(cnt / n_used, 3)}
            for cid, cnt in counts.most_common()
        ]

        detected_issues = []
        for cid, cnt in counts.most_common():
            ratio = cnt / n_used
            if ratio < min_issue_ratio:
                continue
            desc = self.detector._describe_class(cid)
            if desc["is_healthy"]:
                continue
            mask = kt == cid
            detected_issues.append({
                "class_name": desc["class_name"],
                "crop": desc["crop"],
                "disease": desc["disease"],
                "frames": cnt,
                "frame_ratio": round(ratio, 3),
                "mean_confidence": round(float(kp[mask, cid].mean()), 4),
                "treatment": desc["treatment"],
            })

        final_info = self.detector._describe_class(final_id)

        analysis = {
            "frames_analyzed": n_frames,
            "views_analyzed": len(views),
            "views_used": n_used,
            "crop_mode": crop_mode,
            "agreement": round(agreement, 3),
            "top_predictions": top_predictions,
            "class_distribution": class_distribution,
            "detected_issues": detected_issues,
        }
        if debug_dir:
            analysis["debug_dir"] = os.path.abspath(debug_dir)

        if include_frames:
            fps = info["fps"]
            frames_out = []
            for k in range(n_frames):
                idxs = np.where(view_frame == k)[0]
                best = int(idxs[np.argmax(conf[idxs])])
                t3 = np.argsort(probs[best])[::-1][:3]
                frames_out.append({
                    "frame_index": frame_indices[k],
                    "timestamp_s": round(frame_indices[k] / fps, 2) if fps else None,
                    "class_name": names[int(top_ids[best])],
                    "confidence": round(float(conf[best]), 4),
                    "top3": [
                        {"class_name": names[int(i)], "confidence": round(float(probs[best, i]), 4)}
                        for i in t3
                    ],
                })
            analysis["frames"] = frames_out

        return {
            "model": "Ashwin's_AI_Model",
            "video": info,
            "class_id": final_id,
            "class_name": final_info["class_name"],
            "crop": final_info["crop"],
            "disease": final_info["disease"],
            "confidence": round(float(weighted[final_id]), 4),
            "is_healthy": final_info["is_healthy"],
            "treatment": final_info["treatment"],
            "inference_time_ms": inference_ms,
            "processing_time_ms": round((time.perf_counter() - t_start) * 1000, 2),
            "analysis": analysis,
        }