#!/usr/bin/env python3
"""
convert_to_yolo.py
------------------
Converts COCO polygons from plantseg_tcg_v1 into YOLO11 Segmentation format.
Generates:
  - images/<split>/ (copied)
  - labels/<split>/ (TXT labels with normalized polygon coordinates)
  - annotations/ (copied COCO files)
  - data.yaml
  - classes.txt

Usage:
    python scripts/convert_to_yolo.py \
        --input datasets/plantseg_tcg_v1 \
        --output datasets/plantseg_tcg_yolo_v1 \
        --verbose
"""

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

# ── Disease Class mapping (0 to 13) ──────────────────────────────────────────
DISEASE_CLASSES = [
    "tomato bacterial leaf spot",
    "tomato early blight",
    "tomato late blight",
    "tomato leaf mold",
    "tomato mosaic virus",
    "tomato septoria leaf spot",
    "tomato yellow leaf curl virus",
    "cucumber angular leaf spot",
    "cucumber bacterial wilt",
    "cucumber powdery mildew",
    "grape black rot",
    "grape downy mildew",
    "grape leaf spot",
    "grapevine leafroll disease",
]

DISEASE_TO_ID = {disease: i for i, disease in enumerate(DISEASE_CLASSES)}

DISEASE_TO_CROP = {
    d: "Tomato" if d.startswith("tomato") else ("Cucumber" if d.startswith("cucumber") else "Grape")
    for d in DISEASE_CLASSES
}


def load_metadata_lookup(meta_path: Path) -> dict:
    """Load subset_metadata.csv and return dictionary filename -> disease."""
    if not meta_path.exists():
        print(f"WARNING: Metadata not found at {meta_path}. Falling back to filename parsing.", file=sys.stderr)
        return {}
    lookup = {}
    with open(meta_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lookup[row["filename"]] = row["disease"].strip().lower()
    return lookup


def infer_disease_from_filename(filename: str) -> str:
    """Fallback if metadata does not match: parse filename stem."""
    stem = Path(filename).stem.lower()
    for d in DISEASE_CLASSES:
        # Match standard patterns, replacing underscores
        d_und = d.replace(" ", "_")
        if stem.startswith(d_und):
            return d
    return ""


def convert_split(
    split: str,
    src_root: Path,
    dst_root: Path,
    meta_lookup: dict,
    verbose: bool
) -> dict:
    """Convert one split (train / val / test)."""
    coco_path = src_root / "coco" / f"annotation_{split}.json"
    if not coco_path.exists():
        # Check standard path
        coco_path = src_root / "coco" / f"annotation_{split}.json"
        if not coco_path.exists():
            print(f"ERROR: COCO file not found for split {split} at {coco_path}", file=sys.stderr)
            return {"images": 0, "labels": 0, "failed": 0}

    with open(coco_path, encoding="utf-8") as f:
        coco = json.load(f)

    img_dir_dst = dst_root / "images" / split
    label_dir_dst = dst_root / "labels" / split

    img_dir_dst.mkdir(parents=True, exist_ok=True)
    label_dir_dst.mkdir(parents=True, exist_ok=True)

    # Copy COCO files to annotations/
    ann_dir_dst = dst_root / "annotations"
    ann_dir_dst.mkdir(parents=True, exist_ok=True)
    shutil.copy2(coco_path, ann_dir_dst / f"{split}.json")

    # Map image ID -> file metadata
    images_by_id = {img["id"]: img for img in coco.get("images", [])}

    # Group annotations by image_id
    anns_by_img = {}
    for ann in coco.get("annotations", []):
        img_id = ann["image_id"]
        if img_id not in anns_by_img:
            anns_by_img[img_id] = []
        anns_by_img[img_id].append(ann)

    converted_polygons = 0
    failed_polygons = 0
    labels_count = 0
    images_count = 0

    for img_id, img_info in images_by_id.items():
        fname = img_info["file_name"]
        w = img_info["width"]
        h = img_info["height"]

        src_img = src_root / "images" / split / fname
        if not src_img.exists():
            if verbose:
                print(f"    ⚠ Source image not found: {src_img}")
            continue

        # Copy image to destination
        dst_img = img_dir_dst / fname
        if not dst_img.exists():
            shutil.copy2(src_img, dst_img)
        images_count += 1

        # Determine class ID
        disease = meta_lookup.get(fname, "")
        if not disease:
            disease = infer_disease_from_filename(fname)
        
        class_id = DISEASE_TO_ID.get(disease, -1)
        if class_id == -1:
            if verbose:
                print(f"    ⚠ Mapped file {fname} (disease: '{disease}') is not a target class. Skipping.")
            continue

        # Convert annotations to polygons
        label_lines = []
        anns = anns_by_img.get(img_id, [])

        for ann in anns:
            seg = ann.get("segmentation")
            if not seg or not isinstance(seg, list):
                failed_polygons += 1
                continue

            for poly in seg:
                # poly is flat list of [x1, y1, x2, y2, ...]
                if len(poly) < 6:  # Need at least 3 points (6 coords)
                    failed_polygons += 1
                    continue

                normalized_coords = []
                for idx in range(0, len(poly), 2):
                    px = poly[idx] / w
                    py = poly[idx+1] / h
                    # Clamping coordinates to [0, 1] to ensure validation passes
                    px = max(0.0, min(1.0, px))
                    py = max(0.0, min(1.0, py))
                    normalized_coords.append(f"{px:.6f} {py:.6f}")

                coord_str = " ".join(normalized_coords)
                label_lines.append(f"{class_id} {coord_str}")
                converted_polygons += 1

        # Save TXT label file
        txt_name = Path(fname).stem + ".txt"
        dst_txt = label_dir_dst / txt_name
        
        # Write even if empty (YOLO segment train allows empty labels, but requirements say check for empty. 
        # We will write all annotations, empty files represent background or unannotated images if any exist)
        with open(dst_txt, "w", encoding="utf-8") as f_out:
            f_out.write("\n".join(label_lines) + "\n")
        labels_count += 1

    if verbose:
        print(f"  [Split {split}] Converted {images_count} images, {labels_count} labels, {converted_polygons} polygons")

    return {
        "images": images_count,
        "labels": labels_count,
        "converted_polygons": converted_polygons,
        "failed_polygons": failed_polygons,
    }


def write_data_yaml(dst_root: Path):
    """Write standard Ultralytics data.yaml file."""
    yaml_path = dst_root / "data.yaml"
    
    names_dict = {i: d.replace(" ", "_") for i, d in enumerate(DISEASE_CLASSES)}
    
    content = [
        f"path: {dst_root.name}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "",
        "names:",
    ]
    for idx, name in names_dict.items():
        content.append(f"  {idx}: {name}")

    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write("\n".join(content) + "\n")


def write_classes_txt(dst_root: Path):
    """Write classes.txt detailing class_id, class_name, crop."""
    txt_path = dst_root / "classes.txt"
    lines = ["class_id,class_name,crop"]
    for i, name in enumerate(DISEASE_CLASSES):
        crop = DISEASE_TO_CROP[name]
        lines.append(f"{i},{name.replace(' ', '_')},{crop}")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def run(args):
    src_root = Path(args.input)
    dst_root = Path(args.output)
    dst_root.mkdir(parents=True, exist_ok=True)

    if args.verbose:
        print("\n══ YOLO Dataset Converter ══")

    # Load metadata lookup
    meta_path = src_root / "metadata" / "subset_metadata.csv"
    meta_lookup = load_metadata_lookup(meta_path)

    stats = {}
    for split in ["train", "val", "test"]:
        stats[split] = convert_split(split, src_root, dst_root, meta_lookup, args.verbose)

    write_data_yaml(dst_root)
    write_classes_txt(dst_root)

    total_images = sum(stats[s]["images"] for s in stats)
    total_labels = sum(stats[s]["labels"] for s in stats)
    total_polygons = sum(stats[s]["converted_polygons"] for s in stats)
    total_failed = sum(stats[s]["failed_polygons"] for s in stats)

    # Output simple summary JSON in the output root for the orchestrator/stats scripts
    summary = {
        "images": total_images,
        "labels": total_labels,
        "converted_polygons": total_polygons,
        "failed_conversions": total_failed,
        "train": stats["train"],
        "val": stats["val"],
        "test": stats["test"],
    }
    
    with open(dst_root / "conversion_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if args.verbose:
        print(f"\n  ✓ Conversion complete!")
        print(f"    Total Images       : {total_images}")
        print(f"    Total Labels       : {total_labels}")
        print(f"    Converted Polygons : {total_polygons}")
        print(f"    Failed Polygons    : {total_failed}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Convert PlantSeg TCG to YOLO11 Segmentation format")
    parser.add_argument("--input",   required=True, help="Path to validated plantseg_tcg_v1 dataset")
    parser.add_argument("--output",  required=True, help="Path to destination plantseg_tcg_yolo_v1 dataset")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
