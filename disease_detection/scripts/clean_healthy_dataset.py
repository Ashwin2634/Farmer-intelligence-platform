#!/usr/bin/env python3
import os
import sys
import json
import csv
import shutil
import hashlib
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Any

try:
    from PIL import Image, ImageOps
    import numpy as np
    import cv2
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# Constants and targets
CROP_TARGETS = {
    "tomato": 500,
    "cucumber": 340,
    "grape": 470
}

SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def compute_sha256(path: Path) -> str:
    """Computes SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

def compute_phash(path: Path, hash_size: int = 8) -> str:
    """Compute 8x8 pHash for an image using PIL."""
    if not PIL_AVAILABLE:
        return ""
    try:
        with Image.open(path) as img:
            img = img.convert("L").resize((hash_size + 2, hash_size + 2), Image.Resampling.LANCZOS)
            # Simple 8x8 DCT calculation or resizing approximation
            # To be efficient and stable, resize to hash_size x hash_size
            img_small = img.resize((hash_size, hash_size), Image.Resampling.LANCZOS)
            pixels = list(img_small.getdata())
            avg = sum(pixels) / len(pixels)
            bits = "".join("1" if p >= avg else "0" for p in pixels)
            val = int(bits, 2)
            hex_len = (hash_size * hash_size + 3) // 4
            return format(val, f"0{hex_len}x")
    except Exception:
        return ""

def hamming_distance(h1: str, h2: str) -> int:
    """Compute Hamming distance between two hex pHash strings."""
    if len(h1) != len(h2) or not h1 or not h2:
        return 999
    try:
        v1 = int(h1, 16)
        v2 = int(h2, 16)
        return bin(v1 ^ v2).count("1")
    except ValueError:
        return 999

def estimate_blur(path: Path) -> float:
    """Estimate blur using variance of Laplacian."""
    if not PIL_AVAILABLE:
        return 200.0  # Default to passing if OpenCV is not working/available
    try:
        # Load image via OpenCV for image processing checks
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return 0.0
        # If image is very small, scale it to avoid tiny Laplacian variance
        if img.shape[0] < 50 or img.shape[1] < 50:
            return 0.0
        val = cv2.Laplacian(img, cv2.CV_64F).var()
        return val
    except Exception:
        return 0.0

def analyze_exposure_and_text(path: Path) -> Tuple[float, float, bool]:
    """
    Analyze image brightness, contrast, and flag if it likely contains solid text margins or collages.
    Returns: (brightness_avg, contrast_std, is_suspicious)
    """
    if not PIL_AVAILABLE:
        return 127.0, 50.0, False
    try:
        # OpenCV read
        img = cv2.imread(str(path))
        if img is None:
            return 0.0, 0.0, True
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        brightness = float(np.mean(gray))
        contrast = float(np.std(gray))
        
        # Heuristic for UI/borders/text: if there are completely flat solid areas at edges
        # check the variance of the outer borders
        h, w = gray.shape
        border_size = max(5, int(min(h, w) * 0.02))
        top_border = gray[0:border_size, :]
        bottom_border = gray[h-border_size:h, :]
        left_border = gray[:, 0:border_size]
        right_border = gray[:, w-border_size:w]
        
        # If any border is extremely uniform (like a solid white or black border common in collages/screenshots)
        border_stds = [np.std(top_border), np.std(bottom_border), np.std(left_border), np.std(right_border)]
        is_suspicious = False
        if any(std < 1.5 for std in border_stds) and (brightness > 240 or brightness < 15):
            # Likely has a solid white or black frame border typical of screenshots or edited collages
            is_suspicious = True
            
        return brightness, contrast, is_suspicious
    except Exception:
        return 0.0, 0.0, True

def evaluate_quality_score(info: Dict[str, Any]) -> float:
    """
    Compute a normalized composite quality score between 0.0 and 1.0.
    Considers resolution, contrast, sharpness, and brightness.
    """
    # Resolution score (higher is better, cap at 2000px width/height)
    w = info["width"]
    h = info["height"]
    res_score = min(1.0, (w * h) / (1920 * 1080))
    
    # Sharpness score (variance of Laplacian, log-scaled or capped)
    blur_val = info["blur_variance"]
    # 0 -> 0.0, 100 -> 0.4, 500+ -> 1.0
    sharp_score = min(1.0, blur_val / 500.0) if blur_val > 0 else 0.0
    
    # Brightness score (optimum around 100-180)
    bright = info["brightness"]
    if bright < 40 or bright > 220:
        bright_score = 0.2
    elif 100 <= bright <= 180:
        bright_score = 1.0
    else:
        # linear falloff
        bright_score = 1.0 - abs(bright - 140) / 100.0
        bright_score = max(0.2, bright_score)
        
    # Contrast score (optimum standard deviation > 30)
    contrast = info["contrast"]
    contrast_score = min(1.0, contrast / 50.0)
    
    # Composite score (weighted average)
    score = (res_score * 0.2) + (sharp_score * 0.4) + (bright_score * 0.2) + (contrast_score * 0.2)
    return round(score, 4)

def check_greenness_and_spots(path: Path) -> Tuple[bool, str]:
    """
    Checks if the leaf image has disease spots or is not predominantly green/plant-colored.
    Returns: (is_healthy, reason)
    """
    if not PIL_AVAILABLE:
        return True, ""
    try:
        img = cv2.imread(str(path))
        if img is None:
            return False, "unreadable"
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Define range for green (healthy leaves)
        lower_green = np.array([30, 20, 20])
        upper_green = np.array([90, 255, 255])
        
        # Range for yellow/dry/brown (potential disease/spots/decay)
        lower_diseased = np.array([5, 30, 20])
        upper_diseased = np.array([28, 255, 255])
        
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        mask_diseased = cv2.inRange(hsv, lower_diseased, upper_diseased)
        
        green_pixels = np.count_nonzero(mask_green)
        diseased_pixels = np.count_nonzero(mask_diseased)
        total_pixels = img.shape[0] * img.shape[1]
        
        green_ratio = green_pixels / total_pixels
        diseased_ratio = diseased_pixels / total_pixels
        
        # If green is very low, it might not be a leaf image, or it is a highly dried/dead leaf
        if green_ratio < 0.08 and diseased_ratio < 0.08:
            return False, f"low_vegetation_content (green_ratio: {green_ratio:.3f})"
            
        # If the ratio of yellow/brown/diseased pixels is high relative to green, it might be diseased
        if diseased_pixels > 0 and green_pixels > 0:
            spot_ratio = diseased_pixels / (green_pixels + diseased_pixels)
            if spot_ratio > 0.25:
                return False, f"suspicious_spots_or_decay (spot_ratio: {spot_ratio:.3f})"
                
        return True, ""
    except Exception as e:
        return False, f"error_checking_health: {str(e)}"

def run_cleaning_pipeline(raw_dir: Path, clean_dir: Path, reports_dir: Path):
    raw_dir = Path(raw_dir).resolve()
    clean_dir = Path(clean_dir).resolve()
    reports_dir = Path(reports_dir).resolve()
    
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. SCAN THE ENTIRE DATASET
    print("Step 1 & 2: Scanning and validating raw dataset...")
    
    all_raw_images = []
    crop_counts_raw = defaultdict(int)
    
    # Track statistics
    stats = {
        "total_scanned": 0,
        "by_crop_raw": {},
        "corrupted_files": 0,
        "zero_byte_files": 0,
        "unreadable_images": 0,
        "unsupported_formats": 0,
        "extremely_small_images": 0,
        "file_formats": defaultdict(int),
        "resolutions": [],
    }
    
    removed_duplicates_log = []
    removed_low_quality_log = []
    manual_review_log = []
    
    # Keep track of records for metadata
    all_records = []
    
    for crop in ["tomato", "cucumber", "grape"]:
        crop_path = raw_dir / crop
        if not crop_path.exists():
            continue
        
        for file_path in sorted(crop_path.iterdir()):
            if file_path.is_dir():
                continue
            stats["total_scanned"] += 1
            crop_counts_raw[crop] += 1
            
            ext = file_path.suffix.lower()
            stats["file_formats"][ext] += 1
            
            # Check unsupported formats
            if ext not in SUPPORTED_FORMATS:
                stats["unsupported_formats"] += 1
                removed_low_quality_log.append({
                    "original_path": str(file_path.relative_to(raw_dir)),
                    "crop": crop,
                    "reason": "unsupported_format",
                    "details": f"Extension {ext} not supported"
                })
                continue
                
            # Check zero byte files
            file_size = file_path.stat().st_size
            if file_size == 0:
                stats["zero_byte_files"] += 1
                removed_low_quality_log.append({
                    "original_path": str(file_path.relative_to(raw_dir)),
                    "crop": crop,
                    "reason": "zero_byte_file",
                    "details": "0 bytes size"
                })
                continue
                
            # Check readability and dimensions
            try:
                with Image.open(file_path) as img:
                    w, h = img.size
                    img_format = img.format
            except Exception:
                stats["unreadable_images"] += 1
                removed_low_quality_log.append({
                    "original_path": str(file_path.relative_to(raw_dir)),
                    "crop": crop,
                    "reason": "unreadable_or_corrupted",
                    "details": "PIL failed to open"
                })
                continue
                
            stats["resolutions"].append((w, h))
            
            # Check extremely small images (threshold: < 128 in any dimension)
            if w < 128 or h < 128:
                stats["extremely_small_images"] += 1
                removed_low_quality_log.append({
                    "original_path": str(file_path.relative_to(raw_dir)),
                    "crop": crop,
                    "reason": "extremely_small",
                    "details": f"Dimensions {w}x{h} are below 128px threshold"
                })
                continue
                
            # Compute hashes
            sha = compute_sha256(file_path)
            ph = compute_phash(file_path)
            
            # Run image quality check metrics
            blur_val = estimate_blur(file_path)
            brightness, contrast, is_suspicious_border = analyze_exposure_and_text(file_path)
            
            # Assemble initial info
            all_raw_images.append({
                "path": file_path,
                "rel_path": file_path.relative_to(raw_dir),
                "filename": file_path.name,
                "crop": crop,
                "width": w,
                "height": h,
                "file_size": file_size,
                "sha256": sha,
                "phash": ph,
                "blur_variance": blur_val,
                "brightness": brightness,
                "contrast": contrast,
                "suspicious_border": is_suspicious_border
            })
            
    stats["by_crop_raw"] = dict(crop_counts_raw)
    print(f"Scanned {stats['total_scanned']} raw images. Valid/readable count: {len(all_raw_images)}")

    # 3. DETECT EXACT DUPLICATES (SHA256)
    print("Step 3: Detecting exact duplicates...")
    unique_sha_images = []
    sha_seen = {}
    for img in all_raw_images:
        sha = img["sha256"]
        if sha in sha_seen:
            # Duplicate found
            first_seen = sha_seen[sha]
            removed_duplicates_log.append({
                "original_path": str(img["rel_path"]),
                "kept_path": str(first_seen["rel_path"]),
                "crop": img["crop"],
                "type": "exact_sha256"
            })
        else:
            sha_seen[sha] = img
            unique_sha_images.append(img)
            
    print(f"After exact duplicate removal: {len(unique_sha_images)} images remaining.")

    # 4. DETECT NEAR-DUPLICATES (pHash distance <= 5)
    print("Step 4: Detecting near-duplicates...")
    crop_groups = defaultdict(list)
    for img in unique_sha_images:
        crop_groups[img["crop"]].append(img)
        
    near_dup_filtered_images = []
    
    for crop, imgs in crop_groups.items():
        n = len(imgs)
        parent = list(range(n))
        
        def find_parent(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
            
        def union_sets(x, y):
            px = find_parent(x)
            py = find_parent(y)
            if px != py:
                parent[px] = py
                
        for i in range(n):
            for j in range(i+1, n):
                dist = hamming_distance(imgs[i]["phash"], imgs[j]["phash"])
                if dist <= 5:
                    union_sets(i, j)
                    
        clusters = defaultdict(list)
        for idx in range(n):
            clusters[find_parent(idx)].append(imgs[idx])
            
        for root, cluster_imgs in clusters.items():
            if len(cluster_imgs) == 1:
                near_dup_filtered_images.append(cluster_imgs[0])
            else:
                best_img = max(cluster_imgs, key=lambda x: (x["width"] * x["height"], x["blur_variance"]))
                near_dup_filtered_images.append(best_img)
                for c_img in cluster_imgs:
                    if c_img["path"] != best_img["path"]:
                        removed_duplicates_log.append({
                            "original_path": str(c_img["rel_path"]),
                            "kept_path": str(best_img["rel_path"]),
                            "crop": c_img["crop"],
                            "type": "near_duplicate_phash",
                            "phash_distance": hamming_distance(c_img["phash"], best_img["phash"])
                        })
                        
    print(f"After near-duplicate removal: {len(near_dup_filtered_images)} images remaining.")

    # 5. REMOVE LOW-QUALITY & suspicious images
    print("Step 5 & 6 & 7: Quality and health validation checks...")
    validated_images = []
    
    for img in near_dup_filtered_images:
        path = img["path"]
        
        if img["blur_variance"] < 80.0:
            removed_low_quality_log.append({
                "original_path": str(img["rel_path"]),
                "crop": img["crop"],
                "reason": "blurry",
                "details": f"Laplacian variance {img['blur_variance']:.1f} < 80.0"
            })
            continue
            
        if img["brightness"] < 30.0 or img["brightness"] > 235.0:
            removed_low_quality_log.append({
                "original_path": str(img["rel_path"]),
                "crop": img["crop"],
                "reason": "poor_exposure",
                "details": f"Average brightness {img['brightness']:.1f} out of range [30, 235]"
            })
            continue
            
        if img["suspicious_border"]:
            manual_review_log.append({
                "original_path": str(img["rel_path"]),
                "crop": img["crop"],
                "reason": "suspicious_border_or_collage",
                "details": "Identified solid framing border typical of screenshots/collages/text pages"
            })
            continue
            
        is_healthy, health_reason = check_greenness_and_spots(path)
        if not is_healthy:
            manual_review_log.append({
                "original_path": str(img["rel_path"]),
                "crop": img["crop"],
                "reason": health_reason,
                "details": "Automatic health check flagged potential disease symptoms or low vegetation index"
            })
            continue
            
        img["quality_score"] = evaluate_quality_score(img)
        validated_images.append(img)
        
    print(f"After low-quality removal and health routing: {len(validated_images)} validated healthy images remain.")
    print(f"Moved {len(manual_review_log)} suspicious files to manual review log.")

    # 8. DIVERSITY-AWARE DOWNSAMPLING
    print("Step 8 & 9: Performing quality-aware, diversity-preserving downsampling...")
    final_images = []
    
    validated_by_crop = defaultdict(list)
    for img in validated_images:
        validated_by_crop[img["crop"]].append(img)
        
    for crop in ["tomato", "cucumber", "grape"]:
        imgs = validated_by_crop[crop]
        target = CROP_TARGETS[crop]
        
        if len(imgs) <= target:
            print(f"Crop '{crop}': count {len(imgs)} is <= target {target}. Keeping all.")
            final_images.extend(imgs)
        else:
            print(f"Crop '{crop}': count {len(imgs)} exceeds target {target}. Downsampling using pHash diversity clustering...")
            imgs_sorted = sorted(imgs, key=lambda x: x["quality_score"], reverse=True)
            
            selected_indices = [0]
            selected_phash_list = [imgs_sorted[0]["phash"]]
            min_distances = np.array([hamming_distance(x["phash"], selected_phash_list[0]) for x in imgs_sorted])
            
            while len(selected_indices) < target:
                best_val = -1.0
                best_idx = -1
                
                for idx, img_cand in enumerate(imgs_sorted):
                    if idx in selected_indices:
                        continue
                    dist = min_distances[idx]
                    candidate_score = dist + 16.0 * img_cand["quality_score"]
                    if candidate_score > best_val:
                        best_val = candidate_score
                        best_idx = idx
                        
                selected_indices.append(best_idx)
                new_phash = imgs_sorted[best_idx]["phash"]
                for idx, img_cand in enumerate(imgs_sorted):
                    if idx not in selected_indices:
                        d = hamming_distance(img_cand["phash"], new_phash)
                        if d < min_distances[idx]:
                            min_distances[idx] = d
                            
            for idx in selected_indices:
                final_images.append(imgs_sorted[idx])
                
            for idx, img_cand in enumerate(imgs_sorted):
                if idx not in selected_indices:
                    removed_low_quality_log.append({
                        "original_path": str(img_cand["rel_path"]),
                        "crop": img_cand["crop"],
                        "reason": "diversity_downsampled",
                        "details": f"Excluded during diversity-aware selection (Quality: {img_cand['quality_score']:.3f})"
                    })

    # 10. RENAME & COPY FILES TO CLEAN DATASET
    print("Step 10 & 13: Creating final cleaned dataset and manual review folders...")
    clean_dir.mkdir(parents=True, exist_ok=True)
    for crop in ["tomato", "cucumber", "grape"]:
        (clean_dir / crop).mkdir(parents=True, exist_ok=True)
        
    manual_review_dir = clean_dir.parent / "manual_review"
    if manual_review_log:
        manual_review_dir.mkdir(parents=True, exist_ok=True)
        for crop in ["tomato", "cucumber", "grape"]:
            (manual_review_dir / crop).mkdir(parents=True, exist_ok=True)

    final_by_crop = defaultdict(list)
    for img in final_images:
        final_by_crop[img["crop"]].append(img)
        
    final_counts = {}
    
    for crop in ["tomato", "cucumber", "grape"]:
        imgs = sorted(final_by_crop[crop], key=lambda x: x["filename"])
        final_counts[crop] = len(imgs)
        
        for idx, img in enumerate(imgs, 1):
            new_filename = f"{crop}_healthy_{idx:05d}{img['path'].suffix.lower()}"
            dest_path = clean_dir / crop / new_filename
            
            shutil.copy2(img["path"], dest_path)
            
            img["new_filename"] = new_filename
            img["clean_path"] = dest_path
            img["review_status"] = "approved"
            all_records.append(img)
            
    for idx, item in enumerate(manual_review_log, 1):
        src_path = raw_dir / item["original_path"]
        crop = item["crop"]
        dest_filename = f"{crop}_suspicious_{idx:05d}{src_path.suffix.lower()}"
        
        if manual_review_dir.exists():
            shutil.copy2(src_path, manual_review_dir / crop / dest_filename)
            
        item["new_filename"] = dest_filename
        item["review_status"] = "flagged_for_review"
        item["quality_score"] = 0.0
        item["width"] = 0
        item["height"] = 0
        
        try:
            with Image.open(src_path) as im:
                item["width"], item["height"] = im.size
        except Exception:
            pass

    # 11. GENERATE METADATA (CSV & JSON)
    print("Step 11: Generating metadata.csv and metadata.json...")
    metadata_entries = []
    
    for rec in all_records:
        metadata_entries.append({
            "filename": rec["filename"],
            "crop": rec["crop"],
            "resolution": f"{rec['width']}x{rec['height']}",
            "quality_score": rec["quality_score"],
            "duplicate_status": "unique",
            "review_status": rec["review_status"],
            "original_filename": rec["filename"],
            "new_filename": rec["new_filename"]
        })
        
    for mr in manual_review_log:
        metadata_entries.append({
            "filename": Path(mr["original_path"]).name,
            "crop": mr["crop"],
            "resolution": f"{mr['width']}x{mr['height']}",
            "quality_score": 0.0,
            "duplicate_status": "unique",
            "review_status": mr["review_status"],
            "original_filename": Path(mr["original_path"]).name,
            "new_filename": mr["new_filename"]
        })
        
    for dup in removed_duplicates_log:
        metadata_entries.append({
            "filename": Path(dup["original_path"]).name,
            "crop": dup["crop"],
            "resolution": "unknown",
            "quality_score": 0.0,
            "duplicate_status": "duplicate",
            "review_status": "removed_as_duplicate",
            "original_filename": Path(dup["original_path"]).name,
            "new_filename": f"duplicate_of_{Path(dup['kept_path']).name}"
        })
        
    csv_path = clean_dir / "metadata.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "filename", "crop", "resolution", "quality_score",
            "duplicate_status", "review_status", "original_filename", "new_filename"
        ])
        writer.writeheader()
        writer.writerows(metadata_entries)
        
    json_path = clean_dir / "metadata.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metadata_entries, f, indent=2)

    # 12. GENERATE REPORTS
    print("Step 12: Saving reports...")
    
    stats_output = {
        "initial_counts": stats["by_crop_raw"],
        "removed_counts": {
            "corrupted_files": stats["corrupted_files"],
            "zero_byte_files": stats["zero_byte_files"],
            "unreadable_images": stats["unreadable_images"],
            "unsupported_formats": stats["unsupported_formats"],
            "extremely_small_images": stats["extremely_small_images"],
            "exact_duplicates": len([x for x in removed_duplicates_log if x["type"] == "exact_sha256"]),
            "near_duplicates": len([x for x in removed_duplicates_log if x["type"] == "near_duplicate_phash"]),
            "blurry": len([x for x in removed_low_quality_log if x["reason"] == "blurry"]),
            "poor_exposure": len([x for x in removed_low_quality_log if x["reason"] == "poor_exposure"]),
            "diversity_downsampled": len([x for x in removed_low_quality_log if x["reason"] == "diversity_downsampled"]),
        },
        "manual_review_counts": {
            "tomato": len([x for x in manual_review_log if x["crop"] == "tomato"]),
            "cucumber": len([x for x in manual_review_log if x["crop"] == "cucumber"]),
            "grape": len([x for x in manual_review_log if x["crop"] == "grape"]),
        },
        "final_counts": final_counts
    }
    
    with open(reports_dir / "dataset_statistics.json", "w", encoding="utf-8") as f:
        json.dump(stats_output, f, indent=2)
        
    with open(reports_dir / "removed_duplicates.json", "w", encoding="utf-8") as f:
        json.dump(removed_duplicates_log, f, indent=2)
        
    with open(reports_dir / "removed_low_quality.json", "w", encoding="utf-8") as f:
        json.dump(removed_low_quality_log, f, indent=2)
        
    with open(reports_dir / "manual_review.json", "w", encoding="utf-8") as f:
        json.dump(manual_review_log, f, indent=2)
        
    quality_dist = [x["quality_score"] for x in all_records]
    with open(reports_dir / "quality_distribution.json", "w", encoding="utf-8") as f:
        json.dump(quality_dist, f, indent=2)
        
    with open(reports_dir / "crop_distribution.json", "w", encoding="utf-8") as f:
        json.dump(final_counts, f, indent=2)

    avg_quality = np.mean(quality_dist) if quality_dist else 0.0
    std_quality = np.std(quality_dist) if quality_dist else 0.0

    summary_md = f"""# Dataset Summary - healthy_dataset_clean

This report summarizes the dataset cleaning process.

## Crop-Wise Image Counts
- **Tomato**: {final_counts.get('tomato', 0)} (Target: 500)
- **Cucumber**: {final_counts.get('cucumber', 0)} (Target: 340)
- **Grape**: {final_counts.get('grape', 0)} (Target: 470)

## Quality Metrics
- **Average Quality Score**: {avg_quality:.3f}
- **Standard Deviation of Quality**: {std_quality:.3f}
"""
    with open(reports_dir / "dataset_summary.md", "w", encoding="utf-8") as f:
        f.write(summary_md)

    cleaning_report = f"""# Dataset Cleaning Report

## Pipeline Metrics Summary

| Crop | Initial Count | Duplicate Removed | Quality Removed | Manual Review | Final Count | Target |
|---|---|---|---|---|---|---|
| Tomato | {stats['by_crop_raw'].get('tomato', 0)} | {len([x for x in removed_duplicates_log if x['crop'] == 'tomato'])} | {len([x for x in removed_low_quality_log if x['crop'] == 'tomato'])} | {len([x for x in manual_review_log if x['crop'] == 'tomato'])} | {final_counts.get('tomato', 0)} | 500 |
| Cucumber | {stats['by_crop_raw'].get('cucumber', 0)} | {len([x for x in removed_duplicates_log if x['crop'] == 'cucumber'])} | {len([x for x in removed_low_quality_log if x['crop'] == 'cucumber'])} | {len([x for x in manual_review_log if x['crop'] == 'cucumber'])} | {final_counts.get('cucumber', 0)} | 340 |
| Grape | {stats['by_crop_raw'].get('grape', 0)} | {len([x for x in removed_duplicates_log if x['crop'] == 'grape'])} | {len([x for x in removed_low_quality_log if x['crop'] == 'grape'])} | {len([x for x in manual_review_log if x['crop'] == 'grape'])} | {final_counts.get('grape', 0)} | 470 |
| **Total** | {stats['total_scanned']} | {len(removed_duplicates_log)} | {len(removed_low_quality_log)} | {len(manual_review_log)} | {sum(final_counts.values())} | 1310 |

## Removal and Review Causes

- **Exact Duplicate Count**: {len([x for x in removed_duplicates_log if x['type'] == 'exact_sha256'])}
- **Near Duplicate Count**: {len([x for x in removed_duplicates_log if x['type'] == 'near_duplicate_phash'])}
- **Blurry Images**: {len([x for x in removed_low_quality_log if x['reason'] == 'blurry'])}
- **Poor Exposure**: {len([x for x in removed_low_quality_log if x['reason'] == 'poor_exposure'])}
- **Flagged for Manual Review**: {len(manual_review_log)}
- **Downsampled for Balance**: {len([x for x in removed_low_quality_log if x['reason'] == 'diversity_downsampled'])}
"""
    with open(reports_dir / "cleaning_report.md", "w", encoding="utf-8") as f:
        f.write(cleaning_report)
        
    print("\n== Pipeline completed successfully! ==")
    print(f"Cleaned dataset location: {clean_dir}")
    print(f"Reports saved to: {reports_dir}")

if __name__ == "__main__":
    run_cleaning_pipeline(
        raw_dir=Path("datasets/healthy_dataset_raw"),
        clean_dir=Path("datasets/healthy_dataset_clean"),
        reports_dir=Path("reports")
    )
