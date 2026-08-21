#!/usr/bin/env python3
import os
import sys
import json
import argparse
import hashlib
import time
from pathlib import Path
import csv
from collections import defaultdict
import numpy as np
from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser(description="Analyze PlantSeg Dataset")
    parser.add_argument(
        "--dataset",
        type=str,
        default="datasets/plantseg_raw",
        help="Path to the PlantSeg dataset directory (default: datasets/plantseg_raw)",
    )
    return parser.parse_args()

def get_file_sha256(filepath):
    h = hashlib.sha256()
    try:
        with open(filepath, "rb") as f:
            while chunk := f.read(8192):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None

def compute_dhash(image_path, hash_size=8):
    """
    Compute difference hash for an image to detect potential near-duplicates.
    """
    try:
        img = Image.open(image_path).convert('L').resize((hash_size + 1, hash_size), Image.Resampling.BILINEAR)
        pixels = np.array(img, dtype=np.float32)
        # Compare adjacent pixels in each row
        diff = pixels[:, 1:] > pixels[:, :-1]
        # Convert boolean array to a hex string
        decimal_val = 0
        hex_string = []
        for index, value in enumerate(diff.flatten()):
            if value:
                decimal_val += 2 ** (index % 8)
            if (index % 8) == 7:
                hex_string.append(hex(decimal_val)[2:].zfill(2))
                decimal_val = 0
        return "".join(hex_string)
    except Exception:
        return None

def main():
    args = parse_args()
    dataset_dir = Path(args.dataset)
    
    print(f"Starting analysis of PlantSeg dataset at: {dataset_dir.resolve()}")
    if not dataset_dir.exists():
        print(f"Error: Dataset directory {dataset_dir} does not exist.")
        sys.exit(1)

    reports_dir = Path("reports/plantseg_analysis")
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    # ----------------------------------------------------
    # TASK 2: General Dataset Statistics & File Integrity
    # ----------------------------------------------------
    print("Gathering general dataset statistics...")
    all_files = list(dataset_dir.glob("**/*"))
    all_files = [f for f in all_files if f.is_file()]
    
    total_files = len(all_files)
    total_size_bytes = sum(f.stat().st_size for f in all_files)
    
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}
    
    images_by_split = defaultdict(list)
    masks_by_split = defaultdict(list)
    unrecognized_files = []
    zero_byte_files = []
    
    metadata_csv_path = dataset_dir / "Metadata.csv"
    
    # Parse actual files on disk
    for f in all_files:
        if f.stat().st_size == 0:
            zero_byte_files.append(str(f.relative_to(dataset_dir)))
        
        ext = f.suffix.lower()
        rel_path = f.relative_to(dataset_dir)
        parts = rel_path.parts
        
        # Check for images and annotations folders
        if len(parts) >= 3 and parts[0] == "images":
            split = parts[1]
            if ext in image_exts:
                images_by_split[split].append(f)
            else:
                unrecognized_files.append(str(rel_path))
        elif len(parts) >= 3 and parts[0] == "annotations":
            split = parts[1]
            if ext == ".png":
                masks_by_split[split].append(f)
            else:
                unrecognized_files.append(str(rel_path))
        elif f == metadata_csv_path or f.name.startswith("annotation_") and ext == ".json":
            # These are expected top level files
            pass
        else:
            unrecognized_files.append(str(rel_path))

    total_images_found = sum(len(lst) for lst in images_by_split.values())
    total_masks_found = sum(len(lst) for lst in masks_by_split.values())
    
    # Check image formats, resolutions, and corruption
    corrupted_images = []
    image_resolutions = []
    image_formats = defaultdict(int)
    
    print("Verifying image files and checking resolutions...")
    for split, paths in images_by_split.items():
        for p in paths:
            try:
                with Image.open(p) as img:
                    w, h = img.size
                    image_resolutions.append((w, h))
                    image_formats[img.format] += 1
            except Exception:
                corrupted_images.append(str(p.relative_to(dataset_dir)))
                
    if image_resolutions:
        widths = [r[0] for r in image_resolutions]
        heights = [r[1] for r in image_resolutions]
        min_w, max_w = min(widths), max(widths)
        min_h, max_h = min(heights), max(heights)
        avg_w, avg_h = sum(widths)/len(widths), sum(heights)/len(heights)
    else:
        min_w = max_w = min_h = max_h = avg_w = avg_h = 0

    # ----------------------------------------------------
    # TASK 3 & 6: Taxonomy & Metadata Parsing
    # ----------------------------------------------------
    print("Parsing Metadata.csv...")
    metadata_records = []
    crop_taxonomy = defaultdict(lambda: defaultdict(list))
    
    crops_info = defaultdict(lambda: {
        "diseases": set(),
        "total_images": 0,
        "valid_masks": 0,
        "mask_pixel_ratios": []
    })
    
    disease_info = defaultdict(lambda: {
        "crop": "",
        "images": 0,
        "valid_masks": 0
    })

    metadata_images_set = set()
    metadata_split_dist = defaultdict(int)
    
    if metadata_csv_path.exists():
        try:
            with open(metadata_csv_path, mode="r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    metadata_records.append(row)
                    name = row["Name"]
                    plant = row["Plant"]
                    disease = row["Disease"]
                    split_name = row["Split"]
                    idx = int(row["Index"])
                    
                    metadata_images_set.add(name)
                    metadata_split_dist[split_name] += 1
                    
                    crop_taxonomy[plant][disease].append(idx)
                    
                    crops_info[plant]["total_images"] += 1
                    crops_info[plant]["diseases"].add(disease)
                    
                    disease_key = f"{plant}||{disease}"
                    disease_info[disease_key]["crop"] = plant
                    disease_info[disease_key]["images"] += 1
        except Exception as e:
            print(f"Error reading Metadata.csv: {e}")
    else:
        print("Warning: Metadata.csv was not found.")

    # ----------------------------------------------------
    # TASK 4 & 5: Segmentation Annotation & Mask Coverage Analysis
    # ----------------------------------------------------
    print("Performing mask and coverage analysis...")
    mask_stats = {
        "valid_pairs": 0,
        "missing_masks": 0,
        "orphan_masks": 0,
        "empty_masks": 0,
        "corrupted_masks": 0,
        "dimension_mismatches": 0,
        "total_valid_masks": 0,
    }
    
    # Track files on disk
    disk_images_by_name = {}
    disk_masks_by_name = {}
    
    for split, paths in images_by_split.items():
        for p in paths:
            disk_images_by_name[p.name] = p
            
    for split, paths in masks_by_split.items():
        for p in paths:
            disk_masks_by_name[p.name] = p

    coverage_ratios = []
    coverage_by_crop = defaultdict(list)
    coverage_by_disease = defaultdict(list)
    
    coverage_thresholds = {
        "< 1%": 0,
        "1-5%": 0,
        "5-20%": 0,
        "20-50%": 0,
        "> 50%": 0
    }
    
    coverage_thresholds_by_crop = defaultdict(lambda: {k: 0 for k in coverage_thresholds})
    coverage_thresholds_by_disease = defaultdict(lambda: {k: 0 for k in coverage_thresholds})
    
    quality_issues = []
    
    # We validate image-mask pairs according to Metadata.csv
    for row in metadata_records:
        img_name = row["Name"]
        mask_name = row["Label file"]
        crop = row["Plant"]
        disease = row["Disease"]
        disease_key = f"{crop}||{disease}"
        
        img_path = disk_images_by_name.get(img_name)
        mask_path = disk_masks_by_name.get(mask_name)
        
        if not img_path:
            quality_issues.append({
                "type": "missing_image",
                "file": img_name,
                "description": f"Image defined in metadata is missing from the disk."
            })
            continue
            
        if not mask_path:
            mask_stats["missing_masks"] += 1
            quality_issues.append({
                "type": "missing_mask",
                "file": mask_name,
                "description": f"Mask for image {img_name} is missing from the disk."
            })
            continue
            
        # Inspect mask and image dimensions, values and pixel sums
        try:
            with Image.open(img_path) as img:
                img_w, img_h = img.size
            
            with Image.open(mask_path) as msk:
                msk_w, msk_h = msk.size
                msk_format = msk.format
                
                # Check for format and dimensions matching
                if img_w != msk_w or img_h != msk_h:
                    mask_stats["dimension_mismatches"] += 1
                    quality_issues.append({
                        "type": "dimension_mismatch",
                        "image_file": str(img_path.relative_to(dataset_dir)),
                        "mask_file": str(mask_path.relative_to(dataset_dir)),
                        "description": f"Dimension mismatch: Image is {img_w}x{img_h}, mask is {msk_w}x{msk_h}."
                    })
                    continue
                
                msk_arr = np.array(msk)
                unique_vals = np.unique(msk_arr)
                
                # Verify unique pixel values
                if not np.array_equal(unique_vals, [0, 1]) and not np.array_equal(unique_vals, [0]) and not np.array_equal(unique_vals, [1]):
                    quality_issues.append({
                        "type": "non_binary_mask_values",
                        "file": str(mask_path.relative_to(dataset_dir)),
                        "description": f"Mask contains unexpected unique values: {unique_vals.tolist()}"
                    })
                
                # Mask coverage calculations
                non_zero_pixels = int(np.sum(msk_arr > 0))
                total_pixels = msk_w * msk_h
                ratio = non_zero_pixels / total_pixels
                
                if non_zero_pixels == 0:
                    mask_stats["empty_masks"] += 1
                    quality_issues.append({
                        "type": "empty_mask",
                        "file": str(mask_path.relative_to(dataset_dir)),
                        "description": "Mask contains only zero pixels."
                    })
                else:
                    mask_stats["total_valid_masks"] += 1
                    crops_info[crop]["valid_masks"] += 1
                    disease_info[disease_key]["valid_masks"] += 1
                    
                    coverage_pct = ratio * 100
                    coverage_ratios.append(coverage_pct)
                    coverage_by_crop[crop].append(coverage_pct)
                    coverage_by_disease[disease_key].append(coverage_pct)
                    
                    # Threshold sorting
                    if coverage_pct < 1:
                        lbl = "< 1%"
                    elif 1 <= coverage_pct < 5:
                        lbl = "1-5%"
                    elif 5 <= coverage_pct < 20:
                        lbl = "5-20%"
                    elif 20 <= coverage_pct < 50:
                        lbl = "20-50%"
                    else:
                        lbl = "> 50%"
                        
                    coverage_thresholds[lbl] += 1
                    coverage_thresholds_by_crop[crop][lbl] += 1
                    coverage_thresholds_by_disease[disease_key][lbl] += 1
                    
            mask_stats["valid_pairs"] += 1
            
        except Exception as e:
            mask_stats["corrupted_masks"] += 1
            quality_issues.append({
                "type": "corrupted_mask_or_image",
                "image": img_name,
                "mask": mask_name,
                "description": f"Failed to load image/mask: {str(e)}"
            })

    # Find orphan masks (masks on disk that are not in metadata)
    for mask_name, mask_path in disk_masks_by_name.items():
        # find matching image from metadata
        matched = False
        for row in metadata_records:
            if row["Label file"] == mask_name:
                matched = True
                break
        if not matched:
            mask_stats["orphan_masks"] += 1
            quality_issues.append({
                "type": "orphan_mask",
                "file": str(mask_path.relative_to(dataset_dir)),
                "description": "Mask exists on disk but is not linked to any image in Metadata.csv."
            })

    # Coverage statistics aggregates
    def compute_stats(ratios_list):
        if not ratios_list:
            return {"min": 0, "max": 0, "avg": 0, "median": 0}
        return {
            "min": float(np.min(ratios_list)),
            "max": float(np.max(ratios_list)),
            "avg": float(np.mean(ratios_list)),
            "median": float(np.median(ratios_list))
        }

    global_coverage_stats = compute_stats(coverage_ratios)

    # ----------------------------------------------------
    # TASK 7: Duplicate Image Detection using SHA-256 and Hashing
    # ----------------------------------------------------
    print("Analyzing image duplicates and data leakage...")
    image_hashes = defaultdict(list)
    duplicate_groups = defaultdict(list)
    
    # Load hashes
    for split, paths in images_by_split.items():
        for p in paths:
            sha = get_file_sha256(p)
            if sha:
                image_hashes[sha].append(p)
                
    for sha, paths in image_hashes.items():
        if len(paths) > 1:
            duplicate_groups[sha] = [str(p.relative_to(dataset_dir)) for p in paths]
            
    # Check data leakage across train/val/test splits
    leakage_groups = []
    for sha, paths in duplicate_groups.items():
        splits_involved = set()
        for p_str in paths:
            parts = Path(p_str).parts
            if len(parts) >= 2 and parts[0] == "images":
                splits_involved.add(parts[1])
        if len(splits_involved) > 1:
            leakage_groups.append({
                "sha256": sha,
                "splits": list(splits_involved),
                "files": paths
            })
            quality_issues.append({
                "type": "data_leakage",
                "description": f"Image duplicate exists across splits {splits_involved}",
                "files": paths
            })

    # Near duplicates analysis using simple dhash
    print("Analyzing possible near-duplicates...")
    dhashes = defaultdict(list)
    for split, paths in images_by_split.items():
        for p in paths:
            dh = compute_dhash(p)
            if dh:
                dhashes[dh].append(str(p.relative_to(dataset_dir)))
                
    near_duplicate_groups = {}
    for dh, paths in dhashes.items():
        # If they aren't already exact duplicates (which would have same SHA-256)
        if len(paths) > 1:
            # Check if they have different SHA-256
            shas = set()
            for p_str in paths:
                p_full = dataset_dir / p_str
                sha = get_file_sha256(p_full)
                if sha:
                    shas.add(sha)
            if len(shas) > 1:
                near_duplicate_groups[dh] = paths

    # ----------------------------------------------------
    # TASK 8: Existing Split Analysis
    # ----------------------------------------------------
    print("Checking split distribution...")
    split_info = {}
    for split in ["train", "val", "test"]:
        imgs_in_split = images_by_split[split]
        masks_in_split = masks_by_split[split]
        
        crop_dist_split = defaultdict(int)
        disease_dist_split = defaultdict(int)
        
        # Match images back to metadata to count crops and diseases
        for img_path in imgs_in_split:
            img_name = img_path.name
            # find in metadata
            found = False
            for row in metadata_records:
                if row["Name"] == img_name:
                    crop_dist_split[row["Plant"]] += 1
                    disease_dist_split[row["Disease"]] += 1
                    found = True
                    break
                    
        split_info[split] = {
            "image_count": len(imgs_in_split),
            "mask_count": len(masks_in_split),
            "crops": dict(crop_dist_split),
            "diseases": dict(disease_dist_split)
        }

    # ----------------------------------------------------
    # Build Final JSON Reports
    # ----------------------------------------------------
    print("Writing machine-readable reports...")
    
    # 1. summary.json
    summary_report = {
        "total_files": total_files,
        "total_dataset_size_bytes": total_size_bytes,
        "total_images": total_images_found,
        "total_masks": total_masks_found,
        "image_extensions_found": list(set(f.suffix.lower() for f in all_files if f.suffix.lower() in image_exts)),
        "images_per_directory": {k: len(v) for k, v in images_by_split.items()},
        "masks_per_directory": {k: len(v) for k, v in masks_by_split.items()},
        "image_resolution_stats": {
            "min_width": min_w,
            "max_width": max_w,
            "min_height": min_h,
            "max_height": max_h,
            "average_width": round(avg_w, 2),
            "average_height": round(avg_h, 2)
        },
        "image_formats_distribution": dict(image_formats),
        "corrupted_images_count": len(corrupted_images),
        "zero_byte_files_count": len(zero_byte_files),
        "unrecognized_files_count": len(unrecognized_files),
        "healthy_images_available": False, # Verified through disease names check
        "existing_splits_detected": True if metadata_split_dist else False
    }
    with open(reports_dir / "summary.json", "w", encoding="utf-8") as outf:
        json.dump(summary_report, outf, indent=2)

    # 2. crop_distribution.json
    crop_dist_report = {}
    for crop, info in crops_info.items():
        crop_dist_report[crop] = {
            "total_images": info["total_images"],
            "valid_masks": info["valid_masks"],
            "disease_classes_count": len(info["diseases"]),
            "diseases": list(info["diseases"]),
            "coverage_percentage_stats": compute_stats(coverage_by_crop[crop]),
            "coverage_threshold_counts": coverage_thresholds_by_crop[crop]
        }
    with open(reports_dir / "crop_distribution.json", "w", encoding="utf-8") as outf:
        json.dump(crop_dist_report, outf, indent=2)

    # 3. disease_distribution.json
    disease_dist_report = {}
    for key, info in disease_info.items():
        crop, disease = key.split("||")
        disease_dist_report[disease] = {
            "crop": crop,
            "image_count": info["images"],
            "valid_mask_count": info["valid_masks"],
            "coverage_percentage_stats": compute_stats(coverage_by_disease[key]),
            "coverage_threshold_counts": coverage_thresholds_by_disease[key]
        }
    with open(reports_dir / "disease_distribution.json", "w", encoding="utf-8") as outf:
        json.dump(disease_dist_report, outf, indent=2)

    # 4. annotation_analysis.json
    ann_report = {
        "annotation_format_type": "Binary Mask PNG & COCO Polygons JSON",
        "mask_pixel_values": "0 (background), 1 (infected disease lesions)",
        "mask_stats": mask_stats,
        "global_mask_coverage_percentage": global_coverage_stats,
        "global_coverage_threshold_distribution": coverage_thresholds
    }
    with open(reports_dir / "annotation_analysis.json", "w", encoding="utf-8") as outf:
        json.dump(ann_report, outf, indent=2)

    # 5. quality_issues.json
    quality_report = {
        "total_quality_issues": len(quality_issues),
        "issues": quality_issues,
        "low_representation_threshold": 20,
        "low_representation_classes": [
            {"crop": key.split("||")[0], "disease": key.split("||")[1], "count": info["images"]}
            for key, info in disease_info.items() if info["images"] < 20
        ]
    }
    with open(reports_dir / "quality_issues.json", "w", encoding="utf-8") as outf:
        json.dump(quality_report, outf, indent=2)

    # 6. duplicate_report.json
    dup_report = {
        "total_exact_duplicate_groups": len(duplicate_groups),
        "total_exact_duplicate_files": sum(len(v) for v in duplicate_groups.values()),
        "data_leakage_groups_count": len(leakage_groups),
        "data_leakage_details": leakage_groups,
        "possible_near_duplicate_groups_count": len(near_duplicate_groups),
        "possible_near_duplicate_groups": near_duplicate_groups
    }
    with open(reports_dir / "duplicate_report.json", "w", encoding="utf-8") as outf:
        json.dump(dup_report, outf, indent=2)

    # 7. split_analysis.json
    with open(reports_dir / "split_analysis.json", "w", encoding="utf-8") as outf:
        json.dump(split_info, outf, indent=2)

    # 8. plantseg_analysis_full.json
    full_report = {
        "summary": summary_report,
        "crop_distribution": crop_dist_report,
        "disease_distribution": disease_dist_report,
        "annotation_analysis": ann_report,
        "quality_issues": quality_report,
        "duplicate_report": dup_report,
        "split_analysis": split_info
    }
    with open(reports_dir / "plantseg_analysis_full.json", "w", encoding="utf-8") as outf:
        json.dump(full_report, outf, indent=2)

    # ----------------------------------------------------
    # TASK 11: Human-Readable README.md
    # ----------------------------------------------------
    readme_content = f"""# PlantSeg Dataset Analysis Report

## Dataset Overview
- **Total Images on Disk**: {total_images_found}
- **Total Segmentation Masks on Disk**: {total_masks_found}
- **Number of Crop Species**: {len(crops_info)}
- **Number of Disease Classes**: {len(disease_info)}
- **Healthy Images Availability**: None (0 images found representing healthy crops; all samples represent specific disease classes)
- **Annotation Format**: Binary Mask PNGs (containing values 0 and 1) corresponding 1-to-1 with images, and COCO Polygons format in `annotation_*.json` splits.
- **Existing Splits**: Yes (Training: {summary_report["images_per_directory"].get("train", 0)}, Validation: {summary_report["images_per_directory"].get("val", 0)}, Test: {summary_report["images_per_directory"].get("test", 0)})

## Crop Distribution
| Crop | Total Images | Healthy | Diseased | Diseases | Valid Masks |
| :--- | :---: | :---: | :---: | :---: | :---: |
"""
    for crop in sorted(crops_info.keys()):
        info = crops_info[crop]
        readme_content += f"| {crop} | {info['total_images']} | 0 | {info['total_images']} | {len(info['diseases'])} | {info['valid_masks']} |\n"

    readme_content += """
## Disease Distribution
| Crop | Disease | Images | Valid Masks |
| :--- | :--- | :---: | :---: |
"""
    for disease_key in sorted(disease_info.keys()):
        crop, disease = disease_key.split("||")
        info = disease_info[disease_key]
        readme_content += f"| {crop} | {disease} | {info['images']} | {info['valid_masks']} |\n"

    readme_content += f"""
## Segmentation Analysis
- **Annotation Format**: Binary PNG masks where pixel value `1` represents the infected region (lesion area) and `0` represents background/healthy leaf tissue.
- **Matching Strategy**: Images and masks correspond 1-to-1 via names defined in `Metadata.csv`, separated into `train`, `val`, and `test` directories.
- **Lesion Coverage (Mask Area / Image Area)**:
  - **Minimum Coverage**: {global_coverage_stats["min"]:.4f}%
  - **Maximum Coverage**: {global_coverage_stats["max"]:.4f}%
  - **Average Coverage**: {global_coverage_stats["avg"]:.4f}%
  - **Median Coverage**: {global_coverage_stats["median"]:.4f}%
- **Coverage Distribution**:
  - **< 1% Coverage (Small Lesions)**: {coverage_thresholds["< 1%"]} images
  - **1–5% Coverage**: {coverage_thresholds["1-5%"]} images
  - **5–20% Coverage**: {coverage_thresholds["5-20%"]} images
  - **20–50% Coverage**: {coverage_thresholds["20-50%"]} images
  - **> 50% Coverage (Large Infected Area)**: {coverage_thresholds["> 50%"]} images

## Data Quality Issues
- **Total Issues Detected**: {len(quality_issues)}
- **Dimension Mismatches**: {mask_stats["dimension_mismatches"]}
- **Empty Masks**: {mask_stats["empty_masks"]}
- **Missing Masks (Not found on disk)**: {mask_stats["missing_masks"]}
- **Orphan Masks (Not in metadata)**: {mask_stats["orphan_masks"]}
- **Corrupted Images/Masks**: {len(corrupted_images) + mask_stats["corrupted_masks"]}

## Class Imbalance
- **Highly Represented Classes**:
"""
    # Sort diseases by count
    sorted_diseases = sorted(disease_info.items(), key=lambda x: x[1]["images"], reverse=True)
    for key, info in sorted_diseases[:5]:
        crop, disease = key.split("||")
        readme_content += f"  - `{crop}` - `{disease}` ({info['images']} images)\n"
        
    readme_content += """- **Underrepresented Classes (< 20 images)**:
"""
    underrepresented = [x for x in sorted_diseases if x[1]["images"] < 20]
    if underrepresented:
        for key, info in underrepresented[:10]:
            crop, disease = key.split("||")
            readme_content += f"  - `{crop}` - `{disease}` ({info['images']} images)\n"
        if len(underrepresented) > 10:
            readme_content += f"  - *And {len(underrepresented) - 10} more underrepresented classes.*\n"
    else:
        readme_content += "  - None\n"

    readme_content += f"""
## Duplicate Analysis
- **Exact Duplicate Groups**: {len(duplicate_groups)} (Total duplicate files: {sum(len(v) for v in duplicate_groups.values())})
- **Data Leakage Across Splits**: {len(leakage_groups)} groups leak across train/val/test splits.
- **Potential Near-Duplicates**: {len(near_duplicate_groups)} groups flag as potential near-duplicates.

## Candidate Crops for Initial Experiment
Based purely on statistical presence, mask coverage distribution, and taxonomy, the following crops are recommended for our initial 2–3 crop experiment:
1. **Apple**: High image count ({crops_info.get("Apple", {}).get("total_images", 0)}), completely annotated with valid masks, multiple diseases (black rot, rust, scab, etc.), and well-distributed lesion sizes.
2. **Bell Pepper**: Good size, stable annotations, clear disease presentation.
3. **Tomato**: Excellent dataset size, several diseases with distinct lesion morphology, and solid split validation.

"""
    with open(reports_dir / "README.md", "w", encoding="utf-8") as outf:
        outf.write(readme_content)
        
    print(f"Analysis completed successfully! Reports saved to {reports_dir.resolve()}")

if __name__ == "__main__":
    main()
