#!/usr/bin/env python3
"""
curate_healthy_dataset.py
=========================
Production-quality pipeline to curate healthy plant leaf images.

Steps:
  1. Corruption filter   - remove unreadable / zero-byte files
  2. Exact duplicate     - SHA-256 hash dedup only
  3. Quality filter      - lenient (blur, black, white, tiny)
  4. Crop verification   - move bad crops to review/
  5. Disease detection   - HSV-based analysis, move suspects to review/
  6. Diversity sampling  - k-means on features, select 500 diverse Tomatoes

Output:
  healthy_dataset_clean/tomato/     500 images
  healthy_dataset_clean/cucumber/   <=340 images
  healthy_dataset_clean/grape/      <=470 images
  review/{crop}/crop_issues/
  review/{crop}/disease_suspected/
  reports/
"""

import os
import sys
import json
import shutil
import hashlib
import logging
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import cv2
from PIL import Image, UnidentifiedImageError

warnings.filterwarnings("ignore")

# -- Paths --------------------------------------------------------------------
BASE_DIR      = Path(r"e:\AI_Service\datasets")
RAW_DIR       = BASE_DIR / "healthy_dataset_raw"
CLEAN_DIR     = BASE_DIR / "healthy_dataset_clean"
REVIEW_DIR    = BASE_DIR / "review"
REPORTS_DIR   = Path(r"e:\AI_Service\reports")

CROPS = ["tomato", "cucumber", "grape"]

TARGETS = {
    "tomato":   500,
    "cucumber": 340,
    "grape":    470,
}

# -- Quality thresholds (intentionally lenient) --------------------------------
BLUR_THRESHOLD        = 15.0   # Laplacian variance - remove ONLY if < 15 (very lenient)
BLACK_MEAN_THRESHOLD  = 15     # mean pixel value - mostly black
WHITE_MEAN_THRESHOLD  = 240    # mean pixel value - mostly white
MIN_DIMENSION         = 50     # pixels - extremely tiny images

# -- Disease detection thresholds ---------------------------------------------
DISEASE_HUE_FRACTION  = 0.40   # >40% non-green hue pixels -> suspect

# -- Logging ------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("curate")


# =============================================================================
# Utility helpers
# =============================================================================

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def try_open_image(path: Path) -> Tuple[bool, Optional[np.ndarray], str]:
    """Return (ok, cv2_bgr_or_None, reason)."""
    # Zero-byte check
    if path.stat().st_size == 0:
        return False, None, "zero_byte"
    # PIL verify (catches truncated / corrupt headers)
    try:
        with Image.open(path) as img:
            img.verify()
    except Exception as e:
        return False, None, f"pil_verify_failed: {e}"
    # Re-open for pixel data (verify() closes the file)
    try:
        with Image.open(path) as img:
            img.convert("RGB")
    except Exception as e:
        return False, None, f"pil_open_failed: {e}"
    # OpenCV read
    bgr = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        return False, None, "cv2_decode_failed"
    return True, bgr, ""


def laplacian_variance(bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def mean_brightness(bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return float(gray.mean())


def extract_feature_vector(bgr: np.ndarray) -> np.ndarray:
    """24-dim HSV histogram + sharpness + brightness + aspect ratio = 27 dims."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    feats = []
    for ch in range(3):
        hist = cv2.calcHist([hsv], [ch], None, [8], [0, 256])
        hist = hist.flatten()
        hist = hist / (hist.sum() + 1e-6)
        feats.extend(hist.tolist())
    lv = laplacian_variance(bgr) / 5000.0          # normalise
    mb = mean_brightness(bgr) / 255.0
    h, w = bgr.shape[:2]
    ar = w / max(h, 1)
    feats.extend([lv, mb, ar])
    return np.array(feats, dtype=np.float32)


# -- HSV disease detection ----------------------------------------------------

def is_disease_suspected(bgr: np.ndarray) -> bool:
    """
    Healthy leaves are predominantly green / yellow-green in HSV.
    Hue range covers yellow-green to green (OpenCV hue: 0-180 for 0-360 degrees).
    If >DISEASE_HUE_FRACTION of non-background pixels fall outside
    [20, 95] (generous range), flag as suspect.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    # Mask out very dark (shadow) and very bright (specular) pixels
    mask = cv2.inRange(hsv, np.array([0, 30, 30]), np.array([180, 255, 250]))
    total = int(mask.sum() / 255)
    if total < 100:
        return False  # Too little leaf content to judge
    # Green / yellow-green hue mask (generous)
    green_mask = cv2.inRange(hsv, np.array([20, 30, 30]), np.array([95, 255, 250]))
    green_mask = cv2.bitwise_and(green_mask, mask)
    green_count = int(green_mask.sum() / 255)
    non_green_frac = 1.0 - (green_count / max(total, 1))
    return non_green_frac > DISEASE_HUE_FRACTION


# -- Crop quality check -------------------------------------------------------

def is_bad_crop(bgr: np.ndarray) -> bool:
    """
    Bad crop = image is nearly a solid color (background fill).
    Detect: std of all pixel values is very low across all channels.
    Also flag images where the leaf occupies <0.5% of frame (nearly empty frame).
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    # If std dev is tiny across the whole image -> solid background
    if gray.std() < 8.0:
        return True
    # Detect if leaf (non-background) covers almost nothing
    # Using Canny edges as proxy for content
    edges = cv2.Canny(gray, 30, 100)
    edge_frac = edges.sum() / 255 / max(gray.size, 1)
    if edge_frac < 0.005:
        return True
    return False


# =============================================================================
# Diversity-aware k-means sampling
# =============================================================================

def kmeans_diversity_sample(
    entries: List[dict],   # each has "path" and "features" (np.ndarray)
    k: int,
    random_state: int = 42,
) -> List[dict]:
    """
    Run k-means on feature vectors.
    From each cluster, pick the image with median sharpness (not best/worst).
    Returns exactly k entries.
    """
    from sklearn.cluster import MiniBatchKMeans  # type: ignore

    n = len(entries)
    if n <= k:
        return entries

    log.info(f"  k-means: fitting {n} images -> {k} clusters ...")
    X = np.stack([e["features"] for e in entries])
    # Normalise each feature dimension
    X_min = X.min(axis=0, keepdims=True)
    X_max = X.max(axis=0, keepdims=True)
    X_norm = (X - X_min) / (X_max - X_min + 1e-8)

    km = MiniBatchKMeans(n_clusters=k, random_state=random_state,
                         batch_size=2048, max_iter=300, n_init=5)
    labels = km.fit_predict(X_norm)
    log.info("  k-means fitting done.")

    selected = []
    for cluster_id in range(k):
        idxs = np.where(labels == cluster_id)[0]
        if len(idxs) == 0:
            continue
        cluster_entries = [entries[i] for i in idxs]
        # Pick the image whose sharpness is closest to the cluster median
        sharpnesses = np.array([e["sharpness"] for e in cluster_entries])
        median_sharpness = float(np.median(sharpnesses))
        closest_idx = int(np.argmin(np.abs(sharpnesses - median_sharpness)))
        selected.append(cluster_entries[closest_idx])

    # If some clusters were empty, we may have < k; fill greedily from unused
    if len(selected) < k:
        selected_paths = {str(e["path"]) for e in selected}
        remaining = [e for e in entries if str(e["path"]) not in selected_paths]
        # Sort remaining by sharpness (most useful first)
        remaining.sort(key=lambda e: e["sharpness"], reverse=True)
        shortfall = k - len(selected)
        log.info(f"  Filling {shortfall} missing slots from unused images ...")
        selected.extend(remaining[:shortfall])

    log.info(f"  Final selected: {len(selected)} images")
    return selected[:k]


# =============================================================================
# Main pipeline per crop
# =============================================================================

def process_crop(crop: str) -> dict:
    """Run the full 6-step pipeline for one crop. Returns report dict."""

    raw_crop_dir   = RAW_DIR / crop
    clean_crop_dir = CLEAN_DIR / crop
    review_crop    = REVIEW_DIR / crop
    review_crop_issues  = review_crop / "crop_issues"
    review_crop_disease = review_crop / "disease_suspected"

    # Ensure output dirs exist
    clean_crop_dir.mkdir(parents=True, exist_ok=True)
    review_crop_issues.mkdir(parents=True, exist_ok=True)
    review_crop_disease.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    target = TARGETS[crop]
    all_files = sorted(raw_crop_dir.glob("*"))
    all_files = [f for f in all_files if f.is_file()]

    log.info(f"\n{'='*60}")
    log.info(f"  Crop: {crop.upper()}  |  Raw: {len(all_files)}  |  Target: {target}")
    log.info(f"{'='*60}")

    report = {
        "crop": crop,
        "target": target,
        "raw_count": len(all_files),
        "step1_corrupted": [],
        "step2_exact_duplicates": [],
        "step3_low_quality": [],
        "step4_crop_issues": [],
        "step5_disease_suspected": [],
        "step6_sampling": {},
        "final_count": 0,
        "warnings": [],
    }

    # -- STEP 1: Corruption filter --------------------------------------------
    log.info("  Step 1: Corruption filter ...")
    valid_entries = []
    for i, path in enumerate(all_files):
        if (i + 1) % 200 == 0:
            log.info(f"    ... processing {i+1}/{len(all_files)}")
        ok, bgr, reason = try_open_image(path)
        if not ok:
            report["step1_corrupted"].append({"file": path.name, "reason": reason})
        else:
            valid_entries.append({"path": path, "bgr": bgr})

    log.info(f"    Corrupted removed: {len(report['step1_corrupted'])}")
    log.info(f"    Valid after Step 1: {len(valid_entries)}")

    # -- STEP 2: Exact duplicate removal (SHA-256) ----------------------------
    log.info("  Step 2: Exact duplicate removal (SHA-256) ...")
    seen_hashes: Dict[str, str] = {}
    deduped_entries = []
    for entry in valid_entries:
        h = sha256(entry["path"])
        if h in seen_hashes:
            report["step2_exact_duplicates"].append({
                "file": entry["path"].name,
                "duplicate_of": seen_hashes[h],
                "hash": h,
            })
        else:
            seen_hashes[h] = entry["path"].name
            deduped_entries.append(entry)

    log.info(f"    Exact duplicates removed: {len(report['step2_exact_duplicates'])}")
    log.info(f"    Valid after Step 2: {len(deduped_entries)}")

    # -- STEP 3: Quality filter -----------------------------------------------
    log.info("  Step 3: Quality filtering (lenient thresholds) ...")
    quality_passed = []
    for entry in deduped_entries:
        bgr = entry["bgr"]
        path = entry["path"]
        h, w = bgr.shape[:2]

        reason = None
        if min(h, w) < MIN_DIMENSION:
            reason = f"too_tiny: {w}x{h}"
        elif mean_brightness(bgr) < BLACK_MEAN_THRESHOLD:
            reason = "mostly_black"
        elif mean_brightness(bgr) > WHITE_MEAN_THRESHOLD:
            reason = "mostly_white"
        elif laplacian_variance(bgr) < BLUR_THRESHOLD:
            reason = "completely_blurred"

        if reason:
            report["step3_low_quality"].append({"file": path.name, "reason": reason})
        else:
            quality_passed.append(entry)

    log.info(f"    Low-quality removed: {len(report['step3_low_quality'])}")
    log.info(f"    Valid after Step 3: {len(quality_passed)}")

    # -- STEP 4: Crop verification --------------------------------------------
    log.info("  Step 4: Crop verification ...")
    crop_ok_entries = []
    for entry in quality_passed:
        bgr = entry["bgr"]
        path = entry["path"]
        if is_bad_crop(bgr):
            dest = review_crop_issues / path.name
            shutil.copy2(path, dest)
            report["step4_crop_issues"].append({
                "file": path.name,
                "moved_to": str(dest),
            })
        else:
            crop_ok_entries.append(entry)

    log.info(f"    Bad crops moved to review: {len(report['step4_crop_issues'])}")
    log.info(f"    Valid after Step 4: {len(crop_ok_entries)}")

    # -- STEP 5: Disease detection --------------------------------------------
    log.info("  Step 5: Disease verification (HSV) ...")
    healthy_entries = []
    for entry in crop_ok_entries:
        bgr = entry["bgr"]
        path = entry["path"]
        if is_disease_suspected(bgr):
            dest = review_crop_disease / path.name
            shutil.copy2(path, dest)
            report["step5_disease_suspected"].append({
                "file": path.name,
                "moved_to": str(dest),
            })
        else:
            healthy_entries.append(entry)

    log.info(f"    Disease suspects moved to review: {len(report['step5_disease_suspected'])}")
    log.info(f"    Valid after Step 5: {len(healthy_entries)}")

    # -- Check against target -------------------------------------------------
    n_valid = len(healthy_entries)
    if n_valid < target:
        msg = (f"WARNING: Only {n_valid} valid images remain after cleaning, "
               f"but target is {target}. Keeping all {n_valid} images.")
        log.warning(f"  !!  {msg}")
        report["warnings"].append(msg)

    # -- STEP 6: Diversity sampling (if needed) --------------------------------
    if n_valid <= target:
        log.info(f"  Step 6: No sampling needed ({n_valid} <= target {target}). Keeping all.")
        final_entries = healthy_entries
        report["step6_sampling"] = {
            "needed": False,
            "reason": f"{n_valid} valid images <= target {target}",
        }
    else:
        log.info(f"  Step 6: Diversity-aware sampling {n_valid} -> {target} ...")
        for entry in healthy_entries:
            entry["features"] = extract_feature_vector(entry["bgr"])
            entry["sharpness"] = laplacian_variance(entry["bgr"])

        final_entries = kmeans_diversity_sample(healthy_entries, k=target)
        report["step6_sampling"] = {
            "needed": True,
            "pre_sample_count": n_valid,
            "post_sample_count": len(final_entries),
            "method": "MiniBatchKMeans on 27-dim HSV+sharpness+brightness+aspect features, median-sharpness selection per cluster",
        }

    # -- Copy final images to clean dir ----------------------------------------
    log.info(f"  Copying {len(final_entries)} images to {clean_crop_dir} ...")
    for entry in final_entries:
        src = entry["path"]
        dst = clean_crop_dir / src.name
        shutil.copy2(src, dst)

    report["final_count"] = len(final_entries)
    log.info(f"  DONE: {crop.upper()} -> Final count: {report['final_count']}")

    return report


# =============================================================================
# Report generation
# =============================================================================

def save_reports(all_reports: List[dict]) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # removed_exact_duplicates.json
    dup_report = {r["crop"]: r["step2_exact_duplicates"] for r in all_reports}
    (REPORTS_DIR / "removed_exact_duplicates.json").write_text(
        json.dumps(dup_report, indent=2), encoding="utf-8"
    )

    # removed_low_quality.json
    lq_report = {}
    for r in all_reports:
        lq_report[r["crop"]] = {
            "corrupted": r["step1_corrupted"],
            "low_quality": r["step3_low_quality"],
        }
    (REPORTS_DIR / "removed_low_quality.json").write_text(
        json.dumps(lq_report, indent=2), encoding="utf-8"
    )

    # crop_verification.json
    crop_report = {r["crop"]: r["step4_crop_issues"] for r in all_reports}
    (REPORTS_DIR / "crop_verification.json").write_text(
        json.dumps(crop_report, indent=2), encoding="utf-8"
    )

    # disease_review.json
    dis_report = {r["crop"]: r["step5_disease_suspected"] for r in all_reports}
    (REPORTS_DIR / "disease_review.json").write_text(
        json.dumps(dis_report, indent=2), encoding="utf-8"
    )

    # final_counts.json
    final_counts = {}
    for r in all_reports:
        final_counts[r["crop"]] = {
            "raw": r["raw_count"],
            "target": r["target"],
            "final": r["final_count"],
            "target_met": r["final_count"] == r["target"],
            "warnings": r["warnings"],
        }
    (REPORTS_DIR / "final_counts.json").write_text(
        json.dumps(final_counts, indent=2), encoding="utf-8"
    )

    # dataset_summary.md
    lines = [
        "# Healthy Dataset Curation Report",
        f"\nGenerated: {ts}\n",
        "## Final Counts\n",
        "| Crop | Raw | Corrupted | Exact Dups | Low Quality | Bad Crops | Disease Suspect | Final | Target | Status |",
        "|------|-----|-----------|------------|-------------|-----------|-----------------|-------|--------|--------|",
    ]
    all_ok = True
    for r in all_reports:
        met = r["final_count"] == r["target"]
        if not met:
            all_ok = False
        ok_sym = "OK" if met else "WARNING"
        lines.append(
            f"| {r['crop'].capitalize()} "
            f"| {r['raw_count']} "
            f"| {len(r['step1_corrupted'])} "
            f"| {len(r['step2_exact_duplicates'])} "
            f"| {len(r['step3_low_quality'])} "
            f"| {len(r['step4_crop_issues'])} "
            f"| {len(r['step5_disease_suspected'])} "
            f"| **{r['final_count']}** "
            f"| {r['target']} "
            f"| {ok_sym} |"
        )

    lines.append("\n## Step-by-Step Removal Summary\n")
    for r in all_reports:
        lines.append(f"### {r['crop'].capitalize()}\n")
        lines.append(f"- Raw images: {r['raw_count']}")
        lines.append(f"- Step 1 - Corrupted removed: {len(r['step1_corrupted'])}")
        lines.append(f"- Step 2 - Exact duplicates removed: {len(r['step2_exact_duplicates'])}")
        lines.append(f"- Step 3 - Low quality removed: {len(r['step3_low_quality'])}")
        lines.append(f"- Step 4 - Bad crops (moved to review): {len(r['step4_crop_issues'])}")
        lines.append(f"- Step 5 - Disease suspected (moved to review): {len(r['step5_disease_suspected'])}")
        if r["step6_sampling"].get("needed"):
            lines.append(f"- Step 6 - Diversity sampling: {r['step6_sampling']['pre_sample_count']} -> {r['step6_sampling']['post_sample_count']}")
        else:
            lines.append(f"- Step 6 - No sampling needed ({r['final_count']} <= target {r['target']})")
        lines.append(f"- **Final: {r['final_count']} / {r['target']} (target)**")
        if r["warnings"]:
            for w in r["warnings"]:
                lines.append(f"\n> WARNING: {w}")
        lines.append("")

    lines.append("\n## Review Folders\n")
    lines.append("Images moved to `review/` are NOT deleted from the source dataset.")
    lines.append("They require manual inspection.\n")
    for r in all_reports:
        total_review = len(r["step4_crop_issues"]) + len(r["step5_disease_suspected"])
        lines.append(f"- **{r['crop'].capitalize()}**: {total_review} images in review")

    status_line = "All target counts satisfied." if all_ok else "One or more crops did not reach their target. See warnings above."
    lines.append(f"\n---\n{status_line}\n")

    (REPORTS_DIR / "dataset_summary.md").write_text("\n".join(lines), encoding="utf-8")
    log.info(f"  Reports saved to {REPORTS_DIR}")


# =============================================================================
# Entry point
# =============================================================================

def main():
    # Check sklearn availability (needed for k-means step 6)
    try:
        from sklearn.cluster import MiniBatchKMeans  # noqa: F401
    except ImportError:
        log.error("scikit-learn not found. Install with: pip install scikit-learn")
        sys.exit(1)

    log.info("=" * 60)
    log.info("  Healthy Dataset Curation Pipeline")
    log.info(f"  Raw dir:   {RAW_DIR}")
    log.info(f"  Clean dir: {CLEAN_DIR}")
    log.info("=" * 60)

    all_reports = []
    for crop in CROPS:
        report = process_crop(crop)
        all_reports.append(report)

    log.info("\n  Saving reports ...")
    save_reports(all_reports)

    log.info("\n  -- FINAL SUMMARY ------------------------------------------")
    all_ok = True
    for r in all_reports:
        met = r["final_count"] == r["target"]
        if not met:
            all_ok = False
        sym = "OK" if met else "!!"
        log.info(f"  {sym}  {r['crop'].capitalize():10s}: {r['final_count']:4d} / {r['target']} (target)")
    if all_ok:
        log.info("  All targets met!")
    else:
        log.info("  Some targets not met - see reports/dataset_summary.md")
    log.info("  -----------------------------------------------------------")


if __name__ == "__main__":
    main()
