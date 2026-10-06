#!/usr/bin/env python3
import os
import sys
import json
import csv
import shutil
import random
from pathlib import Path
import numpy as np
import cv2
from ultralytics import SAM

# Class Mapping
CROP_CLASSES = {
    "tomato": 0,
    "cucumber": 1,
    "grape": 2
}

# Directories
# WORKSPACE_ROOT = Path("e:/AI_Service")
WORKSPACE_ROOT = Path("/content/drive/MyDrive/AI_Service")
INPUT_DIR = WORKSPACE_ROOT / "datasets/healthy_dataset_clean"
YOLO_DIR = WORKSPACE_ROOT / "healthy_dataset_yolo"
REPORTS_DIR = WORKSPACE_ROOT / "reports"
VIS_DIR = REPORTS_DIR / "visualizations"
REVIEW_DIR = WORKSPACE_ROOT / "review"
MASKS_DIR = WORKSPACE_ROOT / "generated_masks"
OVERLAY_DIR = WORKSPACE_ROOT / "overlay_visualizations"

# Create directories
for d in [YOLO_DIR, REPORTS_DIR, VIS_DIR, REVIEW_DIR, MASKS_DIR, OVERLAY_DIR]:
    d.mkdir(parents=True, exist_ok=True)

def post_process_mask(mask: np.ndarray) -> np.ndarray:
    """
    Applies automatic post-processing:
    - Fills tiny holes
    - Removes isolated blobs (retains only the largest connected component)
    - Smooths tiny jagged artifacts using morphological operations
    """
    if mask is None or np.sum(mask) == 0:
        return np.zeros_like(mask) if mask is not None else None

    # Step 1: Smooth jagged edges and remove tiny isolated pixels/noise with morph open
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_refined = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Step 2: Keep only the single largest connected component (isolated blobs removal)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_refined)
    if num_labels <= 1:
        return np.zeros_like(mask)
        
    # Find index of the largest foreground component
    largest_idx = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
    mask_largest = (labels == largest_idx).astype(np.uint8) * 255

    # Step 3: Fill internal holes
    # Find all contours and draw them filled on a new canvas
    contours, hierarchy = cv2.findContours(mask_largest, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    mask_filled = np.zeros_like(mask_largest)
    cv2.drawContours(mask_filled, contours, -1, 255, -1)
    
    # Final smoothing: closing morph
    mask_final = cv2.morphologyEx(mask_filled, cv2.MORPH_CLOSE, kernel)

    return mask_final

def validate_mask(raw_mask: np.ndarray, refined_mask: np.ndarray, img_shape: tuple) -> tuple:
    """
    Validates the mask using specific reject criteria:
    - coverage < 5% or > 95%
    - multiple disconnected regions (in raw prediction)
    - extremely noisy mask
    - mask touches every border (all 4 edges)
    - obvious background inclusion (touches corner areas)
    - leaf mostly missing
    
    Returns: (is_valid, reason_string)
    """
    H, W = img_shape[:2]
    total_area = H * W
    
    # Refined mask metrics
    mask_pixels = np.sum(refined_mask > 0)
    coverage = mask_pixels / total_area
    
    # Rule: coverage < 5% or > 95%
    if coverage < 0.05:
        return False, f"Low coverage ({coverage*100:.2f}%)"
    if coverage > 0.95:
        return False, f"High coverage ({coverage*100:.2f}%)"
        
    # Rule: Multiple disconnected regions
    # If the raw prediction had multiple significant disjoint regions
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats((raw_mask > 0).astype(np.uint8))
    if num_labels > 2: # Background + at least 2 foreground components
        areas = stats[1:, cv2.CC_STAT_AREA]
        sorted_areas = sorted(areas, reverse=True)
        # If second largest region is significant (>5% of the largest region or >0.5% of total image)
        if sorted_areas[1] > (sorted_areas[0] * 0.05) or sorted_areas[1] > (total_area * 0.005):
            return False, f"Multiple significant disconnected regions (2nd size: {sorted_areas[1]} px)"

    # Rule: mask touches every border
    touches_top = np.any(refined_mask[0, :] > 0)
    touches_bottom = np.any(refined_mask[H - 1, :] > 0)
    touches_left = np.any(refined_mask[:, 0] > 0)
    touches_right = np.any(refined_mask[:, W - 1] > 0)
    if touches_top and touches_bottom and touches_left and touches_right:
        return False, "Mask touches all 4 borders"

    # Rule: obvious background inclusion
    # Check if any corner region is heavily included
    corner_margin = min(15, min(H, W) // 20)
    corners = [
        refined_mask[0:corner_margin, 0:corner_margin],
        refined_mask[0:corner_margin, W-corner_margin:W],
        refined_mask[H-corner_margin:H, 0:corner_margin],
        refined_mask[H-corner_margin:H, W-corner_margin:W]
    ]
    for idx, corner in enumerate(corners):
        if np.mean(corner > 0) > 0.3: # If more than 30% of corner region is filled
            return False, f"Obvious background inclusion in corner {idx}"

    # Rule: extremely noisy mask (high perimeter to area ratio)
    contours, _ = cv2.findContours(refined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        main_contour = max(contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(main_contour, True)
        area = cv2.contourArea(main_contour)
        if area > 0:
            compactness = (perimeter ** 2) / area
            if compactness > 150: # Standard circle is ~12.6, highly jagged/noisy is >150
                return False, f"Extremely noisy / jagged mask (compactness: {compactness:.2f})"
        else:
            return False, "Zero contour area"
    else:
        return False, "No contours found"

    return True, "Valid"

def make_clockwise(pts: np.ndarray) -> np.ndarray:
    """Ensures polygon coordinates are ordered clockwise in screen space."""
    signed_area = 0.0
    num_pts = len(pts)
    for i in range(num_pts):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % num_pts]
        signed_area += (x2 - x1) * (y2 + y1)
    if signed_area < 0:
        return pts[::-1]
    return pts

def run_pipeline():
    print("Initializing SAM2 model...")
    model = SAM('sam2_t.pt')
    
    # Scan images
    supported_extensions = {".jpg", ".jpeg", ".png"}
    all_images = []
    
    for crop in CROP_CLASSES.keys():
        crop_dir = INPUT_DIR / crop
        if not crop_dir.exists():
            print(f"Warning: {crop_dir} does not exist. Skipping.")
            continue
        for f in crop_dir.iterdir():
            if f.suffix.lower() in supported_extensions:
                all_images.append((f, crop))
                
    print(f"Found {len(all_images)} images to process.")
    
    results_db = []
    review_db = []
    failed_db = []
    
    # Process images
    for idx, (img_path, crop) in enumerate(all_images, 1):
        print(f"[{idx}/{len(all_images)}] Processing {img_path.name} ({crop})...")
        
        # Load image details
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  Error: Failed to load image {img_path}")
            failed_db.append({"path": str(img_path), "reason": "Could not read image"})
            continue
            
        orig_h, orig_w, _ = img.shape
        
        # Resize to max 1024 for fast CPU prediction
        max_dim = 1024
        scale = min(max_dim / orig_w, max_dim / orig_h)
        if scale < 1.0:
            w_new = int(orig_w * scale)
            h_new = int(orig_h * scale)
            img_resized = cv2.resize(img, (w_new, h_new), interpolation=cv2.INTER_AREA)
        else:
            w_new = orig_w
            h_new = orig_h
            img_resized = img
            scale = 1.0
            
        # Convert BGR to RGB for model input
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        
        # Define 5 Prompt Strategies scaled to resized dimensions
        strategies = [
            {"type": "center_point", "points": [[w_new // 2, h_new // 2]], "labels": [1]},
            {"type": "bbox_center_third", "bboxes": [[int(w_new * 0.15), int(h_new * 0.15), int(w_new * 0.85), int(h_new * 0.85)]]},
            {"type": "bbox_center_half", "bboxes": [[int(w_new * 0.25), int(h_new * 0.25), int(w_new * 0.75), int(h_new * 0.75)]]},
            {"type": "multi_point_cross", "points": [
                [w_new // 2, h_new // 2],
                [w_new // 2 - int(w_new * 0.1), h_new // 2],
                [w_new // 2 + int(w_new * 0.1), h_new // 2],
                [w_new // 2, h_new // 2 - int(h_new * 0.1)],
                [w_new // 2, h_new // 2 + int(h_new * 0.1)]
            ], "labels": [1, 1, 1, 1, 1]},
            {"type": "auto_fallback", "auto": True}
        ]
        
        selected_mask_full = None
        selected_strategy = None
        validation_reason = ""
        
        for strategy_idx, strat in enumerate(strategies, 1):
            try:
                if strat.get("auto"):
                    # Call without prompt (acts as automatic mask generator)
                    pred_res = model.predict(img_rgb, verbose=False)
                elif "bboxes" in strat:
                    pred_res = model.predict(img_rgb, bboxes=strat["bboxes"], verbose=False)
                else:
                    pred_res = model.predict(img_rgb, points=strat["points"], labels=strat["labels"], verbose=False)
                
                if not pred_res or len(pred_res) == 0 or pred_res[0].masks is None or len(pred_res[0].masks) == 0:
                    continue
                
                masks_tensor = pred_res[0].masks.data
                
                # Evaluate all output masks from this strategy
                best_strat_mask_full = None
                best_strat_reason = ""
                
                for m_idx in range(len(masks_tensor)):
                    raw_mask_resized = (masks_tensor[m_idx].cpu().numpy() > 0).astype(np.uint8) * 255
                    
                    # Upsample raw mask back to original resolution
                    raw_mask_full = cv2.resize(raw_mask_resized, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
                    refined_mask_full = post_process_mask(raw_mask_full)
                    
                    is_valid, reason = validate_mask(raw_mask_full, refined_mask_full, (orig_h, orig_w))
                    if is_valid:
                        best_strat_mask_full = refined_mask_full
                        break
                    else:
                        best_strat_reason = reason
                        
                if best_strat_mask_full is not None:
                    selected_mask_full = best_strat_mask_full
                    selected_strategy = strat["type"]
                    break
                else:
                    validation_reason = best_strat_reason
            except Exception as ex:
                validation_reason = f"Error in inference: {str(ex)}"
                continue
                
        # If successfully segmented
        if selected_mask_full is not None:
            print(f"  ✓ Success using strategy '{selected_strategy}'")
            
            # Save raw binary mask
            mask_filename = img_path.name
            # requirement specifies saving as png binary mask or matching filename. Let's make it png.
            mask_filename_png = img_path.stem + ".png"
            mask_save_path = MASKS_DIR / mask_filename_png
            cv2.imwrite(str(mask_save_path), selected_mask_full)
            
            # Extract main contour
            contours, _ = cv2.findContours(selected_mask_full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            main_contour = max(contours, key=cv2.contourArea)
            
            # Simplify polygon to preserve serrations but keep it clean
            perimeter = cv2.arcLength(main_contour, True)
            approx = cv2.approxPolyDP(main_contour, 0.0005 * perimeter, True)
            
            # Ensure coordinates are in clockwise order
            poly_points = approx.reshape(-1, 2)
            poly_clockwise = make_clockwise(poly_points)
            
            # Calculate metrics
            mask_area = int(np.sum(selected_mask_full > 0))
            leaf_area_pct = (mask_area / (orig_h * orig_w)) * 100.0
            
            x, y, bbox_w, bbox_h = cv2.boundingRect(main_contour)
            bbox_area = int(bbox_w * bbox_h)
            aspect_ratio = float(bbox_w / bbox_h) if bbox_h > 0 else 0.0
            
            hull = cv2.convexHull(poly_clockwise)
            hull_area = float(cv2.contourArea(hull))
            solidity = float(mask_area / hull_area) if hull_area > 0 else 0.0
            
            hull_perimeter = float(cv2.arcLength(hull, True))
            poly_perimeter = float(cv2.arcLength(poly_clockwise, True))
            convexity = float(hull_perimeter / poly_perimeter) if poly_perimeter > 0 else 0.0
            
            # Keep normalized polygon coordinates
            normalized_poly = []
            for pt in poly_clockwise:
                nx = max(0.0, min(1.0, float(pt[0] / orig_w)))
                ny = max(0.0, min(1.0, float(pt[1] / orig_h)))
                normalized_poly.append((nx, ny))
                
            results_db.append({
                "filename": img_path.name,
                "crop": crop,
                "class_id": CROP_CLASSES[crop],
                "img_path": str(img_path),
                "mask_path": str(mask_save_path),
                "width": orig_w,
                "height": orig_h,
                "leaf_area_pct": leaf_area_pct,
                "mask_area": mask_area,
                "bbox_area": bbox_area,
                "aspect_ratio": aspect_ratio,
                "solidity": solidity,
                "convexity": convexity,
                "polygon_perimeter": poly_perimeter,
                "vertices_count": len(normalized_poly),
                "polygon": normalized_poly
            })
        else:
            # Move image to review/
            print(f"  ✗ Failed. Moving to review/ (Reason: {validation_reason})")
            review_dest = REVIEW_DIR / img_path.name
            shutil.copy2(img_path, review_dest)
            review_db.append({
                "filename": img_path.name,
                "crop": crop,
                "src_path": str(img_path),
                "dest_path": str(review_dest),
                "reason": validation_reason
            })
            
    # --- STRATIFIED TRAIN / VAL / TEST SPLIT ---
    print("\nSplitting dataset (80% Train / 10% Val / 10% Test)...")
    grouped_by_crop = {crop: [] for crop in CROP_CLASSES.keys()}
    for res in results_db:
        grouped_by_crop[res["crop"]].append(res)
        
    split_counts = {"train": 0, "val": 0, "test": 0}
    split_details = {crop: {"train": 0, "val": 0, "test": 0} for crop in CROP_CLASSES.keys()}
    
    # Fix seed for reproducibility
    random.seed(42)
    
    for crop, items in grouped_by_crop.items():
        random.shuffle(items)
        n = len(items)
        n_train = int(n * 0.8)
        n_val = int(n * 0.1)
        
        train_items = items[:n_train]
        val_items = items[n_train:n_train + n_val]
        test_items = items[n_train + n_val:]
        
        for split_name, split_items in [("train", train_items), ("val", val_items), ("test", test_items)]:
            split_img_dir = YOLO_DIR / split_name / "images"
            split_lbl_dir = YOLO_DIR / split_name / "labels"
            split_img_dir.mkdir(parents=True, exist_ok=True)
            split_lbl_dir.mkdir(parents=True, exist_ok=True)
            
            for item in split_items:
                # Copy Image
                shutil.copy2(item["img_path"], split_img_dir / item["filename"])
                
                # Export YOLO segmentation polygon label
                # format: class_id x1 y1 x2 y2 ...
                poly_coords = []
                for pt in item["polygon"]:
                    poly_coords.append(f"{pt[0]:.6f} {pt[1]:.6f}")
                label_str = f"{item['class_id']} " + " ".join(poly_coords)
                
                label_filename = Path(item["filename"]).stem + ".txt"
                with open(split_lbl_dir / label_filename, "w", encoding="utf-8") as f_lbl:
                    f_lbl.write(label_str + "\n")
                    
                split_counts[split_name] += 1
                split_details[crop][split_name] += 1
                
    # --- VISUAL QA OVERLAY GENERATION ---
    print("\nGenerating visual QA overlays...")
    # Generate overlay: green polygon original image, transparent mask
    # 50 random samples overall
    # 25 samples from every crop specifically
    all_sample_candidates = results_db.copy()
    random.shuffle(all_sample_candidates)
    
    sampled_set = set()
    visualizations_generated = []
    
    def generate_and_save_overlay(item, out_dir):
        img = cv2.imread(item["img_path"])
        mask = cv2.imread(item["mask_path"], cv2.IMREAD_GRAYSCALE)
        
        # Color mask: green overlay (B=0, G=255, R=0)
        overlay = img.copy()
        overlay[mask > 0] = [0, 255, 0]
        
        # Weighted add for transparency (alpha = 0.7, beta = 0.3)
        blended = cv2.addWeighted(img, 0.7, overlay, 0.3, 0)
        
        # Draw outline
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(blended, contours, -1, (0, 180, 0), 2)
        
        out_path = out_dir / item["filename"]
        cv2.imwrite(str(out_path), blended)
        return out_path
        
    # Generate 50 random samples
    random_samples = all_sample_candidates[:min(50, len(all_sample_candidates))]
    for sample in random_samples:
        path = generate_and_save_overlay(sample, VIS_DIR)
        shutil.copy2(path, OVERLAY_DIR / sample["filename"])
        sampled_set.add(sample["filename"])
        visualizations_generated.append(sample["filename"])
        
    # Generate 25 samples per crop
    for crop in CROP_CLASSES.keys():
        crop_candidates = [r for r in results_db if r["crop"] == crop]
        random.shuffle(crop_candidates)
        crop_samples = crop_candidates[:min(25, len(crop_candidates))]
        for sample in crop_samples:
            path = generate_and_save_overlay(sample, VIS_DIR)
            shutil.copy2(path, OVERLAY_DIR / sample["filename"])
            if sample["filename"] not in sampled_set:
                sampled_set.add(sample["filename"])
                visualizations_generated.append(sample["filename"])
                
    print(f"Generated {len(sampled_set)} unique overlay visualization samples.")
    
    # --- METRICS CSV EXPORT ---
    print("\nWriting mask_statistics.csv...")
    csv_file = WORKSPACE_ROOT / "mask_statistics.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow([
            "filename", "crop", "leaf area %", "polygon vertices", 
            "mask area", "bounding box area", "convexity", 
            "solidity", "aspect ratio", "polygon perimeter"
        ])
        for r in results_db:
            writer.writerow([
                r["filename"], r["crop"], f"{r['leaf_area_pct']:.4f}", r["vertices_count"],
                r["mask_area"], r["bbox_area"], f"{r['convexity']:.4f}",
                f"{r['solidity']:.4f}", f"{r['aspect_ratio']:.4f}", f"{r['polygon_perimeter']:.4f}"
            ])
            
    # --- FINAL REPORTS ---
    print("\nGenerating statistical reports...")
    
    # coverage_statistics.json
    coverages = [r["leaf_area_pct"] for r in results_db]
    coverage_stats = {
        "mean": float(np.mean(coverages)) if coverages else 0.0,
        "median": float(np.median(coverages)) if coverages else 0.0,
        "min": float(np.min(coverages)) if coverages else 0.0,
        "max": float(np.max(coverages)) if coverages else 0.0,
        "std": float(np.std(coverages)) if coverages else 0.0,
    }
    with open(REPORTS_DIR / "coverage_statistics.json", "w", encoding="utf-8") as f:
        json.dump(coverage_stats, f, indent=4)
        
    # polygon_statistics.json
    vertices = [r["vertices_count"] for r in results_db]
    perimeters = [r["polygon_perimeter"] for r in results_db]
    poly_stats = {
        "vertices": {
            "mean": float(np.mean(vertices)) if vertices else 0.0,
            "median": float(np.median(vertices)) if vertices else 0.0,
            "min": int(np.min(vertices)) if vertices else 0,
            "max": int(np.max(vertices)) if vertices else 0,
        },
        "perimeter": {
            "mean": float(np.mean(perimeters)) if perimeters else 0.0,
            "median": float(np.median(perimeters)) if perimeters else 0.0,
            "min": float(np.min(perimeters)) if perimeters else 0.0,
            "max": float(np.max(perimeters)) if perimeters else 0.0,
        }
    }
    with open(REPORTS_DIR / "polygon_statistics.json", "w", encoding="utf-8") as f:
        json.dump(poly_stats, f, indent=4)
        
    # mask_statistics.json
    solidities = [r["solidity"] for r in results_db]
    convexities = [r["convexity"] for r in results_db]
    aspect_ratios = [r["aspect_ratio"] for r in results_db]
    mask_stats = {
        "solidity": {
            "mean": float(np.mean(solidities)) if solidities else 0.0,
            "min": float(np.min(solidities)) if solidities else 0.0,
            "max": float(np.max(solidities)) if solidities else 0.0,
        },
        "convexity": {
            "mean": float(np.mean(convexities)) if convexities else 0.0,
            "min": float(np.min(convexities)) if convexities else 0.0,
            "max": float(np.max(convexities)) if convexities else 0.0,
        },
        "aspect_ratio": {
            "mean": float(np.mean(aspect_ratios)) if aspect_ratios else 0.0,
            "min": float(np.min(aspect_ratios)) if aspect_ratios else 0.0,
            "max": float(np.max(aspect_ratios)) if aspect_ratios else 0.0,
        }
    }
    with open(REPORTS_DIR / "mask_statistics.json", "w", encoding="utf-8") as f:
        json.dump(mask_stats, f, indent=4)
        
    # split_statistics.json
    split_stats = {
        "total_successful": len(results_db),
        "split_totals": split_counts,
        "split_by_crop": split_details
    }
    with open(REPORTS_DIR / "split_statistics.json", "w", encoding="utf-8") as f:
        json.dump(split_stats, f, indent=4)
        
    # failed_images.json
    with open(REPORTS_DIR / "failed_images.json", "w", encoding="utf-8") as f:
        json.dump(failed_db, f, indent=4)
        
    # review_images.json
    with open(REPORTS_DIR / "review_images.json", "w", encoding="utf-8") as f:
        json.dump(review_db, f, indent=4)
        
    # Write dataset_summary.md
    summary_md = f"""# Leaf Segmentation Labeling Dataset Summary

## Pipeline Summary
* **Base Model**: SAM2 (`sam2_t.pt`)
* **Total Clean Images**: {len(all_images)}
* **Successfully Segmented**: {len(results_db)} ({len(results_db)/len(all_images)*100:.2f}%)
* **Sent to Review**: {len(review_db)} ({len(review_db)/len(all_images)*100:.2f}%)
* **Failed processing**: {len(failed_db)} ({len(failed_db)/len(all_images)*100:.2f}%)

## Dataset Split Statistics
| Split | Total Images | Tomato ({split_details['tomato']['train']+split_details['tomato']['val']+split_details['tomato']['test']}) | Cucumber ({split_details['cucumber']['train']+split_details['cucumber']['val']+split_details['cucumber']['test']}) | Grape ({split_details['grape']['train']+split_details['grape']['val']+split_details['grape']['test']}) |
| --- | --- | --- | --- | --- |
| **Train** | {split_counts['train']} | {split_details['tomato']['train']} | {split_details['cucumber']['train']} | {split_details['grape']['train']} |
| **Val** | {split_counts['val']} | {split_details['tomato']['val']} | {split_details['cucumber']['val']} | {split_details['grape']['val']} |
| **Test** | {split_counts['test']} | {split_details['tomato']['test']} | {split_details['cucumber']['test']} | {split_details['grape']['test']} |

## Morphological and Polygon Quality Metrics
* **Average Leaf Coverage**: {coverage_stats['mean']:.2f}% (Range: {coverage_stats['min']:.2f}% - {coverage_stats['max']:.2f}%)
* **Average Polygon Vertices**: {poly_stats['vertices']['mean']:.1f}
* **Average Solidity**: {mask_stats['solidity']['mean']:.3f}
* **Average Convexity**: {mask_stats['convexity']['mean']:.3f}
* **Average Aspect Ratio**: {mask_stats['aspect_ratio']['mean']:.2f}
"""
    with open(REPORTS_DIR / "dataset_summary.md", "w", encoding="utf-8") as f:
        f.write(summary_md)
        
    print(f"\n✓ Pipeline complete! Output dataset saved to {YOLO_DIR}.")
    print(f"Reports saved to {REPORTS_DIR}.")

if __name__ == '__main__':
    run_pipeline()
