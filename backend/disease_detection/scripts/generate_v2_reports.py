#!/usr/bin/env python3
"""
generate_v2_reports.py
-----------------------
Generates all reports for the PlantSeg V2 YOLO11 Segmentation dataset.

Outputs:
  reports/
    dataset_summary.json
    class_distribution.json
    crop_distribution.json
    split_distribution.json
    dataset_statistics.json
    validation_report.json   (copied from validation step)
    dataset_summary.md
    class_distribution.md
    crop_distribution.md
    dataset_statistics.md

Usage:
    python scripts/generate_v2_reports.py \
        --input  datasets/plantseg_tcg_yolo_v2 \
        --output datasets/plantseg_tcg_yolo_v2/reports \
        --verbose
"""

import argparse
import json
import csv
from pathlib import Path
from collections import defaultdict

DISEASE_CLASSES = [
    "tomato_bacterial_leaf_spot", "tomato_early_blight", "tomato_late_blight",
    "tomato_leaf_mold", "tomato_mosaic_virus", "tomato_septoria_leaf_spot",
    "tomato_yellow_leaf_curl_virus",
    "cucumber_angular_leaf_spot", "cucumber_bacterial_wilt", "cucumber_powdery_mildew",
    "grape_black_rot", "grape_downy_mildew", "grape_leaf_spot", "grapevine_leafroll_disease",
    "banana_anthracnose", "banana_black_leaf_streak", "banana_bunchy_top",
    "banana_cigar_end_rot", "banana_cordana_leaf_spot", "banana_panama_disease",
    "corn_northern_leaf_blight", "corn_gray_leaf_spot", "corn_rust", "corn_smut",
    "soybean_bacterial_blight", "soybean_brown_spot", "soybean_downy_mildew",
    "soybean_frog_eye_leaf_spot", "soybean_mosaic", "soybean_rust",
]

CLASS_TO_CROP = {
    "tomato_bacterial_leaf_spot": "Tomato", "tomato_early_blight": "Tomato",
    "tomato_late_blight": "Tomato", "tomato_leaf_mold": "Tomato",
    "tomato_mosaic_virus": "Tomato", "tomato_septoria_leaf_spot": "Tomato",
    "tomato_yellow_leaf_curl_virus": "Tomato",
    "cucumber_angular_leaf_spot": "Cucumber", "cucumber_bacterial_wilt": "Cucumber",
    "cucumber_powdery_mildew": "Cucumber",
    "grape_black_rot": "Grape", "grape_downy_mildew": "Grape",
    "grape_leaf_spot": "Grape", "grapevine_leafroll_disease": "Grape",
    "banana_anthracnose": "Banana", "banana_black_leaf_streak": "Banana",
    "banana_bunchy_top": "Banana", "banana_cigar_end_rot": "Banana",
    "banana_cordana_leaf_spot": "Banana", "banana_panama_disease": "Banana",
    "corn_northern_leaf_blight": "Corn", "corn_gray_leaf_spot": "Corn",
    "corn_rust": "Corn", "corn_smut": "Corn",
    "soybean_bacterial_blight": "Soybean", "soybean_brown_spot": "Soybean",
    "soybean_downy_mildew": "Soybean", "soybean_frog_eye_leaf_spot": "Soybean",
    "soybean_mosaic": "Soybean", "soybean_rust": "Soybean",
}

V1_CLASSES = DISEASE_CLASSES[:14]
NEW_CLASSES = DISEASE_CLASSES[14:]


def collect_stats(dataset_root: Path):
    split_image_counts = {}
    split_label_counts = {}
    split_polygon_counts = {}
    class_dist_per_split = {}
    all_class_counts = defaultdict(int)
    polygon_lengths = []
    polygons_per_image = []

    for split in ["train", "val", "test"]:
        img_dir = dataset_root / "images" / split
        lbl_dir = dataset_root / "labels" / split

        images = list(img_dir.glob("*")) if img_dir.exists() else []
        labels = list(lbl_dir.glob("*.txt")) if lbl_dir.exists() else []

        split_image_counts[split] = len(images)
        split_label_counts[split] = len(labels)
        split_polygon_counts[split] = 0
        class_dist_per_split[split] = defaultdict(int)

        for txt_file in labels:
            try:
                with open(txt_file, encoding="utf-8") as f:
                    lines = [l.strip() for l in f if l.strip()]
            except Exception:
                continue

            if not lines:
                polygons_per_image.append(0)
                continue

            polygons_per_image.append(len(lines))
            split_polygon_counts[split] += len(lines)

            for line in lines:
                parts = line.split()
                if len(parts) < 7:
                    continue
                try:
                    cid = int(parts[0])
                    if 0 <= cid < len(DISEASE_CLASSES):
                        name = DISEASE_CLASSES[cid]
                        class_dist_per_split[split][name] += 1
                        all_class_counts[name] += 1
                    pts = (len(parts) - 1) // 2
                    polygon_lengths.append(pts)
                except ValueError:
                    continue

    return (split_image_counts, split_label_counts, split_polygon_counts,
            class_dist_per_split, dict(all_class_counts), polygon_lengths, polygons_per_image)


def run(args):
    dataset_root = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.verbose:
        print("\n== V2 Report Generator ==")

    (split_image_counts, split_label_counts, split_polygon_counts,
     class_dist_per_split, all_class_counts, polygon_lengths, polygons_per_image) = collect_stats(dataset_root)

    total_images = sum(split_image_counts.values())
    total_labels = sum(split_label_counts.values())
    total_polygons = sum(split_polygon_counts.values())

    # ── 1. dataset_summary.json ──────────────────────────────────────────────
    summary = {
        "version": "v2",
        "dataset_root": str(dataset_root),
        "num_classes": len(DISEASE_CLASSES),
        "num_crops": 6,
        "crops": ["Tomato", "Cucumber", "Grape", "Banana", "Corn", "Soybean"],
        "num_new_crops": 3,
        "new_crops": ["Banana", "Corn", "Soybean"],
        "total_images": total_images,
        "total_labels": total_labels,
        "total_polygons": total_polygons,
        "splits": {
            "train": split_image_counts["train"],
            "val":   split_image_counts["val"],
            "test":  split_image_counts["test"],
        },
    }
    with open(output_dir / "dataset_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # ── 2. class_distribution.json ───────────────────────────────────────────
    class_dist_final = {
        "splits": {
            s: {c: class_dist_per_split[s].get(c, 0) for c in DISEASE_CLASSES}
            for s in ["train", "val", "test"]
        },
        "totals": {c: all_class_counts.get(c, 0) for c in DISEASE_CLASSES}
    }
    with open(output_dir / "class_distribution.json", "w", encoding="utf-8") as f:
        json.dump(class_dist_final, f, indent=2)

    # ── 3. crop_distribution.json ────────────────────────────────────────────
    crops = ["Tomato", "Cucumber", "Grape", "Banana", "Corn", "Soybean"]
    crop_dist = {}
    for crop in crops:
        crop_classes = [c for c in DISEASE_CLASSES if CLASS_TO_CROP[c] == crop]
        crop_total = sum(all_class_counts.get(c, 0) for c in crop_classes)
        is_new = crop in ["Banana", "Corn", "Soybean"]
        crop_dist[crop] = {
            "is_new_in_v2": is_new,
            "disease_classes": crop_classes,
            "num_diseases": len(crop_classes),
            "total_polygons": crop_total,
            "per_split": {
                s: sum(class_dist_per_split[s].get(c, 0) for c in crop_classes)
                for s in ["train", "val", "test"]
            }
        }
    with open(output_dir / "crop_distribution.json", "w", encoding="utf-8") as f:
        json.dump(crop_dist, f, indent=2)

    # ── 4. split_distribution.json ───────────────────────────────────────────
    split_dist = {
        "train": {
            "images": split_image_counts["train"],
            "labels": split_label_counts["train"],
            "polygons": split_polygon_counts["train"],
        },
        "val": {
            "images": split_image_counts["val"],
            "labels": split_label_counts["val"],
            "polygons": split_polygon_counts["val"],
        },
        "test": {
            "images": split_image_counts["test"],
            "labels": split_label_counts["test"],
            "polygons": split_polygon_counts["test"],
        },
        "total": {
            "images": total_images,
            "labels": total_labels,
            "polygons": total_polygons,
        }
    }
    with open(output_dir / "split_distribution.json", "w", encoding="utf-8") as f:
        json.dump(split_dist, f, indent=2)

    # ── 5. dataset_statistics.json ────────────────────────────────────────────
    dataset_stats = {
        "polygon_statistics": {
            "total_polygons": total_polygons,
            "points_per_polygon": {
                "min": min(polygon_lengths) if polygon_lengths else 0,
                "max": max(polygon_lengths) if polygon_lengths else 0,
                "avg": round(sum(polygon_lengths) / len(polygon_lengths), 2) if polygon_lengths else 0,
            },
            "polygons_per_image": {
                "min": min(polygons_per_image) if polygons_per_image else 0,
                "max": max(polygons_per_image) if polygons_per_image else 0,
                "avg": round(sum(polygons_per_image) / len(polygons_per_image), 2) if polygons_per_image else 0,
            },
        },
        "class_id_mapping": {
            str(i): DISEASE_CLASSES[i] for i in range(len(DISEASE_CLASSES))
        },
        "v1_classes": V1_CLASSES,
        "v2_new_classes": NEW_CLASSES,
    }
    with open(output_dir / "dataset_statistics.json", "w", encoding="utf-8") as f:
        json.dump(dataset_stats, f, indent=2)

    # ── 6. Markdown: dataset_summary.md ──────────────────────────────────────
    md_summary = f"""# PlantSeg V2 — Dataset Summary

## Overview
| Property | Value |
|---|---|
| **Dataset Version** | V2 |
| **Total Crops** | 6 |
| **Total Classes** | {len(DISEASE_CLASSES)} |
| **Total Images** | {total_images} |
| **Total Labels** | {total_labels} |
| **Total Polygons** | {total_polygons} |

## Crops Included
| Crop | Status | Diseases |
|---|---|---|
| Tomato | ✅ V1 (unchanged) | 7 |
| Cucumber | ✅ V1 (unchanged) | 3 |
| Grape | ✅ V1 (unchanged) | 4 |
| Banana | 🆕 New in V2 | 6 |
| Corn | 🆕 New in V2 | 4 |
| Soybean | 🆕 New in V2 | 6 |

## Split Distribution
| Split | Images | Labels | Polygons |
|---|---|---|---|
| train | {split_image_counts['train']} | {split_label_counts['train']} | {split_polygon_counts['train']} |
| val | {split_image_counts['val']} | {split_label_counts['val']} | {split_polygon_counts['val']} |
| test | {split_image_counts['test']} | {split_label_counts['test']} | {split_polygon_counts['test']} |
| **Total** | **{total_images}** | **{total_labels}** | **{total_polygons}** |

## Backward Compatibility
- V1 classes (0–13) are **unchanged**
- V1 training scripts, FastAPI service, and inference pipeline are **fully compatible**
- New classes (14–29) extend V2 continuously
"""
    with open(output_dir / "dataset_summary.md", "w", encoding="utf-8") as f:
        f.write(md_summary)

    # ── 7. Markdown: class_distribution.md ──────────────────────────────────
    md_class = "# PlantSeg V2 — Class Distribution\n\n"
    md_class += "| Class ID | Class Name | Crop | Train | Val | Test | Total | Status |\n"
    md_class += "|---|---|---|---|---|---|---|---|\n"
    for i, cls in enumerate(DISEASE_CLASSES):
        crop = CLASS_TO_CROP[cls]
        train_cnt = class_dist_per_split["train"].get(cls, 0)
        val_cnt = class_dist_per_split["val"].get(cls, 0)
        test_cnt = class_dist_per_split["test"].get(cls, 0)
        total_cnt = all_class_counts.get(cls, 0)
        status = "🆕 New" if i >= 14 else "✅ V1"
        md_class += f"| {i} | `{cls}` | {crop} | {train_cnt} | {val_cnt} | {test_cnt} | {total_cnt} | {status} |\n"
    with open(output_dir / "class_distribution.md", "w", encoding="utf-8") as f:
        f.write(md_class)

    # ── 8. Markdown: crop_distribution.md ───────────────────────────────────
    md_crop = "# PlantSeg V2 — Crop Distribution\n\n"
    for crop, data in crop_dist.items():
        status = "🆕 New in V2" if data["is_new_in_v2"] else "✅ V1 (unchanged)"
        md_crop += f"## {crop} ({status})\n"
        md_crop += f"- **Diseases**: {data['num_diseases']}\n"
        md_crop += f"- **Total Polygons**: {data['total_polygons']}\n"
        md_crop += f"- **Train / Val / Test**: {data['per_split']['train']} / {data['per_split']['val']} / {data['per_split']['test']}\n"
        md_crop += "- **Disease Classes**:\n"
        for d in data["disease_classes"]:
            md_crop += f"  - `{d}`\n"
        md_crop += "\n"
    with open(output_dir / "crop_distribution.md", "w", encoding="utf-8") as f:
        f.write(md_crop)

    # ── 9. Markdown: dataset_statistics.md ──────────────────────────────────
    ps = dataset_stats["polygon_statistics"]
    md_stats = f"""# PlantSeg V2 — Dataset Statistics

## Polygon Statistics
| Metric | Value |
|---|---|
| Total Polygons | {ps['total_polygons']} |
| Points per Polygon (min) | {ps['points_per_polygon']['min']} |
| Points per Polygon (max) | {ps['points_per_polygon']['max']} |
| Points per Polygon (avg) | {ps['points_per_polygon']['avg']} |
| Polygons per Image (min) | {ps['polygons_per_image']['min']} |
| Polygons per Image (max) | {ps['polygons_per_image']['max']} |
| Polygons per Image (avg) | {ps['polygons_per_image']['avg']} |

## Class ID Mapping
| ID | Class Name |
|---|---|
"""
    for i, cls in enumerate(DISEASE_CLASSES):
        md_stats += f"| {i} | `{cls}` |\n"
    with open(output_dir / "dataset_statistics.md", "w", encoding="utf-8") as f:
        f.write(md_stats)

    if args.verbose:
        print(f"  [OK] All reports written to {output_dir}")
        print(f"    Total images  : {total_images}")
        print(f"    Total polygons: {total_polygons}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Generate V2 dataset reports")
    parser.add_argument("--input",   required=True, help="YOLO V2 dataset root")
    parser.add_argument("--output",  required=True, help="Report output directory")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
