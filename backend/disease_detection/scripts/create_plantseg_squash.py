#!/usr/bin/env python3
"""
create_plantseg_squash.py
-------------------------
Extracts Squash Powdery Mildew images and COCO segmentations from plantseg_raw,
converts them to YOLO segmentation format (normalized polygon coordinates),
and builds the plant_seg_squash dataset structure:

plant_seg_squash/
    ├── images/
    │   ├── train/
    │   ├── val/
    │   └── test/
    ├── labels/
    │   ├── train/
    │   ├── val/
    │   └── test/
    └── data.yaml
"""

import json
import shutil
import sys
from pathlib import Path
import pandas as pd
import yaml

def main():
    raw_dir = Path(r"e:\AI_Service\disease_detection\datasets\plantseg_raw")
    dst_dir = Path(r"e:\AI_Service\disease_detection\datasets\plantseg_squash")
    
    if not raw_dir.exists():
        print(f"Error: Raw dataset directory not found at {raw_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Reading metadata from {raw_dir / 'Metadata.csv'}...")
    meta_df = pd.read_csv(raw_dir / "Metadata.csv")
    
    # Filter for squash powdery mildew
    squash_df = meta_df[
        (meta_df["Plant"].str.lower() == "squash") & 
        (meta_df["Disease"].str.lower() == "squash powdery mildew")
    ]
    print(f"Found {len(squash_df)} squash powdery mildew entries.")
    
    # Create target directories
    for split in ["train", "val", "test"]:
        (dst_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (dst_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
        
    split_map = {
        "train": "train",
        "training": "train",
        "val": "val",
        "validation": "val",
        "test": "test"
    }
    
    total_images = 0
    total_labels = 0
    total_polygons = 0
    
    for split_key in ["train", "val", "test"]:
        ann_file = raw_dir / f"annotation_{split_key}.json"
        if not ann_file.exists():
            print(f"Warning: Annotation file {ann_file} not found.", file=sys.stderr)
            continue
            
        with open(ann_file, encoding="utf-8") as f:
            coco = json.load(f)
            
        images_by_name = {img["file_name"]: img for img in coco.get("images", [])}
        anns_by_img_id = {}
        for ann in coco.get("annotations", []):
            img_id = ann["image_id"]
            if img_id not in anns_by_img_id:
                anns_by_img_id[img_id] = []
            anns_by_img_id[img_id].append(ann)
            
        split_rows = squash_df[
            squash_df["Split"].str.lower().map(lambda s: split_map.get(s, s)) == split_key
        ]
        
        print(f"\nProcessing split '{split_key}' ({len(split_rows)} items)...")
        
        split_img_cnt = 0
        split_lbl_cnt = 0
        split_poly_cnt = 0
        
        for idx, row in split_rows.iterrows():
            fname = row["Name"]
            src_img_path = raw_dir / "images" / split_key / fname
            
            if not src_img_path.exists():
                print(f"  Warning: Image {src_img_path} does not exist. Skipping.")
                continue
                
            dst_img_path = dst_dir / "images" / split_key / fname
            shutil.copy2(src_img_path, dst_img_path)
            split_img_cnt += 1
            
            if fname not in images_by_name:
                print(f"  Warning: {fname} not found in {ann_file.name}. Creating empty label.")
                dst_lbl_path = dst_dir / "labels" / split_key / f"{Path(fname).stem}.txt"
                dst_lbl_path.write_text("", encoding="utf-8")
                split_lbl_cnt += 1
                continue
                
            img_info = images_by_name[fname]
            img_id = img_info["id"]
            w = img_info["width"]
            h = img_info["height"]
            
            anns = anns_by_img_id.get(img_id, [])
            label_lines = []
            
            for ann in anns:
                seg = ann.get("segmentation")
                if not seg or not isinstance(seg, list):
                    continue
                for poly in seg:
                    if len(poly) < 6:
                        continue
                    norm_coords = []
                    for i in range(0, len(poly), 2):
                        px = poly[i] / w
                        py = poly[i+1] / h
                        px = max(0.0, min(1.0, px))
                        py = max(0.0, min(1.0, py))
                        norm_coords.append(f"{px:.6f} {py:.6f}")
                    label_lines.append(f"0 {' '.join(norm_coords)}")
                    split_poly_cnt += 1
                    
            txt_name = f"{Path(fname).stem}.txt"
            dst_lbl_path = dst_dir / "labels" / split_key / txt_name
            with open(dst_lbl_path, "w", encoding="utf-8") as f_out:
                f_out.write("\n".join(label_lines) + ("\n" if label_lines else ""))
            split_lbl_cnt += 1
            
        print(f"  [Split {split_key}] Copied {split_img_cnt} images, generated {split_lbl_cnt} label files, {split_poly_cnt} polygons.")
        total_images += split_img_cnt
        total_labels += split_lbl_cnt
        total_polygons += split_poly_cnt

    # Create data.yaml
    data_yaml = {
        "path": f"{dst_dir.as_posix()}",
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": 1,
        "names": {
            0: "squash_powdery_mildew"
        }
    }
    
    with open(dst_dir / "data.yaml", "w", encoding="utf-8") as f_yaml:
        yaml.dump(data_yaml, f_yaml, default_flow_style=False, sort_keys=False)
        
    print("\n==========================================")
    print("Dataset Creation Complete!")
    print(f"Target directory: {dst_dir}")
    print(f"Total Images: {total_images}")
    print(f"Total Labels: {total_labels}")
    print(f"Total Polygons: {total_polygons}")
    print(f"data.yaml created at: {dst_dir / 'data.yaml'}")
    print("==========================================")

if __name__ == "__main__":
    main()
