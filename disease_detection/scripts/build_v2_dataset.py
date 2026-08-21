#!/usr/bin/env python3
"""
build_v2_dataset.py
-------------------
Expands the PlantSeg dataset from V1 to V2.
Includes Tomato, Cucumber, Grape, Banana, Corn, Soybean.
Uses continuous class IDs (0-29).
"""

import os
import sys
import csv
import json
import shutil
import hashlib
import argparse
from pathlib import Path
from collections import defaultdict

try:
    from PIL import Image
    import numpy as np
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# Class mappings: 0-29
DISEASE_CLASSES = [
    # Tomato (0-6)
    "tomato_bacterial_leaf_spot",
    "tomato_early_blight",
    "tomato_late_blight",
    "tomato_leaf_mold",
    "tomato_mosaic_virus",
    "tomato_septoria_leaf_spot",
    "tomato_yellow_leaf_curl_virus",
    # Cucumber (7-9)
    "cucumber_angular_leaf_spot",
    "cucumber_bacterial_wilt",
    "cucumber_powdery_mildew",
    # Grape (10-13)
    "grape_black_rot",
    "grape_downy_mildew",
    "grape_leaf_spot",
    "grapevine_leafroll_disease",
    # Banana (14-19)
    "banana_anthracnose",
    "banana_black_leaf_streak",
    "banana_bunchy_top",
    "banana_cigar_end_rot",
    "banana_cordana_leaf_spot",
    "banana_panama_disease",
    # Corn (20-23)
    "corn_northern_leaf_blight",
    "corn_gray_leaf_spot",
    "corn_rust",
    "corn_smut",
    # Soybean (24-29)
    "soybean_bacterial_blight",
    "soybean_brown_spot",
    "soybean_downy_mildew",
    "soybean_frog_eye_leaf_spot",
    "soybean_mosaic",
    "soybean_rust"
]

DISEASE_TO_ID = {d: i for i, d in enumerate(DISEASE_CLASSES)}

DISEASE_TO_CROP = {
    d: "Tomato" if d.startswith("tomato") else (
        "Cucumber" if d.startswith("cucumber") else (
            "Grape" if d.startswith("grape") else (
                "Banana" if d.startswith("banana") else (
                    "Corn" if d.startswith("corn") else "Soybean"
                )
            )
        )
    )
    for d in DISEASE_CLASSES
}

TARGET_CROPS_DISPLAY = {"Tomato", "Cucumber", "Grape", "Banana", "Corn", "Soybean"}

SPLIT_MAP = {
    "training": "train",
    "train": "train",
    "validation": "val",
    "val": "val",
    "test": "test",
    "testing": "test",
}

def get_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

def normalize_disease_name(raw_name: str) -> str:
    # lowercase, snake_case, no spaces, hyphens, double underscores
    name = raw_name.lower().strip()
    name = name.replace("-", "_").replace(" ", "_")
    while "__" in name:
        name = name.replace("__", "_")
    
    # Map raw names that have slightly different syntax to expected classes
    if name == "banana_panama":
        name = "banana_panama_disease"
    elif name == "grape_leafroll_disease":
        name = "grapevine_leafroll_disease"
    return name

def build_v2(raw_dir: Path, coco_v2_dir: Path, yolo_v2_dir: Path, verbose: bool):
    print("Step 1: Filtering Metadata.csv...")
    metadata_csv = raw_dir / "Metadata.csv"
    if not metadata_csv.exists():
        print(f"Error: Metadata.csv not found at {metadata_csv}")
        sys.exit(1)
        
    v2_records = []
    with open(metadata_csv, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            plant = row.get("Plant", "").strip()
            if plant in TARGET_CROPS_DISPLAY:
                v2_records.append(row)
                
    print(f"Found {len(v2_records)} target crop records in Metadata.csv.")
    
    # Deduplication using SHA256 of image files
    print("Deduplicating image files...")
    image_hashes = {}
    cleaned_records = []
    duplicates_removed = 0
    
    # Group by split to prioritize train > val > test
    split_priority = {"train": 0, "val": 1, "test": 2}
    v2_records = sorted(
        v2_records,
        key=lambda r: split_priority.get(SPLIT_MAP.get(r.get("Split", "").strip().lower(), ""), 99)
    )
    
    for r in v2_records:
        filename = r.get("Name", "").strip()
        label_file = r.get("Label file", "").strip()
        split_raw = r.get("Split", "").strip().lower()
        split = SPLIT_MAP.get(split_raw, split_raw)
        
        if split not in {"train", "val", "test"}:
            continue
            
        img_path = raw_dir / "images" / split / filename
        if not img_path.exists():
            continue
            
        h = get_sha256(img_path)
        if h in image_hashes:
            duplicates_removed += 1
            continue
            
        image_hashes[h] = (split, filename)
        cleaned_records.append(r)
        
    print(f"Deduplication complete. Removed {duplicates_removed} duplicate images. Kept {len(cleaned_records)} records.")
    
    # Create output directories
    coco_v2_dir.mkdir(parents=True, exist_ok=True)
    yolo_v2_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy images and mask PNGs
    print("Copying images and masks...")
    final_records = []
    for r in cleaned_records:
        filename = r.get("Name", "").strip()
        label_file = r.get("Label file", "").strip()
        split_raw = r.get("Split", "").strip().lower()
        split = SPLIT_MAP.get(split_raw, split_raw)
        
        src_img = raw_dir / "images" / split / filename
        src_mask = raw_dir / "annotations" / split / label_file
        
        if not src_img.exists() or not src_mask.exists():
            continue
            
        dst_img = coco_v2_dir / "images" / split / filename
        dst_mask = coco_v2_dir / "annotations" / split / label_file
        
        dst_img.parent.mkdir(parents=True, exist_ok=True)
        dst_mask.parent.mkdir(parents=True, exist_ok=True)
        
        shutil.copy2(src_img, dst_img)
        shutil.copy2(src_mask, dst_mask)
        
        # Dimensions Check and repair
        if PIL_AVAILABLE:
            try:
                with Image.open(dst_img) as img, Image.open(dst_mask) as msk:
                    if img.size != msk.size:
                        if img.size == (msk.size[1], msk.size[0]):
                            msk = msk.rotate(-90, expand=True)
                            msk.save(dst_mask)
                        else:
                            msk = msk.resize(img.size, Image.NEAREST)
                            msk.save(dst_mask)
            except Exception as e:
                print(f"Warning: Failed to verify/repair sizes for {filename}: {e}")
                
        final_records.append(r)
        
    # Build metadata csv for COCO v2
    meta_dir = coco_v2_dir / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    subset_meta_csv = meta_dir / "subset_metadata.csv"
    
    with open(subset_meta_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "label_file", "split", "crop", "disease"])
        for r in final_records:
            filename = r.get("Name", "").strip()
            label_file = r.get("Label file", "").strip()
            split_raw = r.get("Split", "").strip().lower()
            split = SPLIT_MAP.get(split_raw, split_raw)
            crop = r.get("Plant", "").strip()
            disease_raw = r.get("Disease", "").strip()
            disease = normalize_disease_name(disease_raw)
            writer.writerow([filename, label_file, split, crop, disease])
            
    print(f"Copied {len(final_records)} images and masks successfully.")
    
    # Filter and Convert COCO Annotation files to YOLO V2
    print("Converting COCO JSON annotations to YOLO format...")
    for split in ["train", "val", "test"]:
        coco_json_src = raw_dir / f"annotation_{split}.json"
        if not coco_json_src.exists():
            continue
            
        with open(coco_json_src, encoding="utf-8") as f:
            coco = json.load(f)
            
        # Filter images belonging to target crops
        v2_image_ids = set()
        v2_images = []
        filename_to_disease = {}
        
        # Load from final records
        for r in final_records:
            split_raw = r.get("Split", "").strip().lower()
            s = SPLIT_MAP.get(split_raw, split_raw)
            if s == split:
                filename_to_disease[r.get("Name", "").strip()] = normalize_disease_name(r.get("Disease", "").strip())
                
        for img in coco.get("images", []):
            fname = img["file_name"]
            if fname in filename_to_disease:
                v2_images.append(img)
                v2_image_ids.add(img["id"])
                
        # Filter annotations
        v2_annotations = []
        for ann in coco.get("annotations", []):
            if ann["image_id"] in v2_image_ids:
                v2_annotations.append(ann)
                
        # Save COCO json subset
        coco_dst_dir = coco_v2_dir / "coco"
        coco_dst_dir.mkdir(parents=True, exist_ok=True)
        with open(coco_dst_dir / f"annotation_{split}.json", "w", encoding="utf-8") as f:
            json.dump({
                "info": coco.get("info", {}),
                "licenses": coco.get("licenses", []),
                "images": v2_images,
                "annotations": v2_annotations,
                "categories": coco.get("categories", [])
            }, f, indent=2)
            
        # Copy to YOLO annotations/
        yolo_ann_dir = yolo_v2_dir / "annotations"
        yolo_ann_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(coco_dst_dir / f"annotation_{split}.json", yolo_ann_dir / f"{split}.json")
        
        # Generate TXT labels for YOLO
        yolo_img_dir = yolo_v2_dir / "images" / split
        yolo_lbl_dir = yolo_v2_dir / "labels" / split
        yolo_img_dir.mkdir(parents=True, exist_ok=True)
        yolo_lbl_dir.mkdir(parents=True, exist_ok=True)
        
        # Map annotations
        anns_by_img = defaultdict(list)
        for ann in v2_annotations:
            anns_by_img[ann["image_id"]].append(ann)
            
        for img_info in v2_images:
            fname = img_info["file_name"]
            w = img_info["width"]
            h = img_info["height"]
            
            src_img_path = coco_v2_dir / "images" / split / fname
            if not src_img_path.exists():
                continue
                
            # Copy to YOLO images
            shutil.copy2(src_img_path, yolo_img_dir / fname)
            
            disease = filename_to_disease[fname]
            class_id = DISEASE_TO_ID.get(disease, -1)
            if class_id == -1:
                continue
                
            label_lines = []
            for ann in anns_by_img[img_info["id"]]:
                seg = ann.get("segmentation")
                if not seg or not isinstance(seg, list):
                    continue
                for poly in seg:
                    if len(poly) < 6:
                        continue
                    normalized_coords = []
                    for idx in range(0, len(poly), 2):
                        px = max(0.0, min(1.0, poly[idx] / w))
                        py = max(0.0, min(1.0, poly[idx+1] / h))
                        normalized_coords.append(f"{px:.6f} {py:.6f}")
                    label_lines.append(f"{class_id} " + " ".join(normalized_coords))
                    
            txt_name = Path(fname).stem + ".txt"
            with open(yolo_lbl_dir / txt_name, "w", encoding="utf-8") as f_out:
                f_out.write("\n".join(label_lines) + "\n")
                
    # Write data.yaml
    names_dict = {i: d for i, d in enumerate(DISEASE_CLASSES)}
    yaml_content = [
        "path: plantseg_tcg_yolo_v2",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "",
        "names:",
    ]
    for idx, name in names_dict.items():
        yaml_content.append(f"  {idx}: {name}")
        
    with open(yolo_v2_dir / "data.yaml", "w", encoding="utf-8") as f:
        f.write("\n".join(yaml_content) + "\n")
        
    # Write classes.txt
    lines = ["class_id,class_name,crop"]
    for i, name in enumerate(DISEASE_CLASSES):
        crop = DISEASE_TO_CROP[name]
        lines.append(f"{i},{name},{crop}")
        
    with open(yolo_v2_dir / "classes.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
        
    print("V2 Dataset generation and YOLO conversion finished successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default="datasets/plantseg_raw")
    parser.add_argument("--coco_v2", default="datasets/plantseg_v2")
    parser.add_argument("--yolo_v2", default="datasets/plantseg_tcg_yolo_v2")
    args = parser.parse_args()
    build_v2(Path(args.raw), Path(args.coco_v2), Path(args.yolo_v2), True)
