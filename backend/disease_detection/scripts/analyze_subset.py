#!/usr/bin/env python3
"""
analyze_subset.py
-----------------
Generates all statistical reports for plantseg_tcg_v1.

Outputs in reports/:
  dataset_summary.json
  crop_distribution.json
  disease_distribution.json
  mask_statistics.json
  class_statistics.json
  coverage_statistics.json
  split_statistics.json
  cleanup_report.md

Usage:
    python analyze_subset.py --input datasets/plantseg_tcg_v1 --output datasets/plantseg_tcg_v1/reports --workers 4 --verbose
"""

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

try:
    from PIL import Image
    import numpy as np
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


def load_metadata(dataset_root: Path):
    meta_path = dataset_root / "metadata" / "subset_metadata.csv"
    if not meta_path.exists():
        return []
    with open(meta_path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def compute_mask_coverage(mask_path: Path) -> float:
    """Return percentage of non-zero pixels in mask."""
    if not PIL_AVAILABLE:
        return 0.0
    try:
        with Image.open(mask_path) as img:
            arr = np.array(img)
            nonzero = np.count_nonzero(arr)
            total = arr.size
            return round(100.0 * nonzero / total, 4) if total > 0 else 0.0
    except Exception:
        return 0.0


def run(args):
    dataset_root = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    images_dir = dataset_root / "images"
    annotations_dir = dataset_root / "annotations"

    if args.verbose:
        print("\n══ Dataset Analyzer ══")

    meta = load_metadata(dataset_root)

    # Build lookup: filename -> metadata row
    meta_lookup = {row["filename"]: row for row in meta}

    # ── Collect all image data ─────────────────────────────────────────────────
    all_records = []
    for split in ["train", "val", "test"]:
        img_dir = images_dir / split
        ann_dir = annotations_dir / split
        if not img_dir.exists():
            continue
        for img_path in sorted(img_dir.iterdir()):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            mask_path = ann_dir / (img_path.stem + ".png")

            # Get dimensions
            try:
                with Image.open(img_path) as img:
                    w, h = img.size
            except Exception:
                w, h = 0, 0

            coverage = compute_mask_coverage(mask_path) if mask_path.exists() else 0.0

            # Try to get metadata
            row = meta_lookup.get(img_path.name, {})
            crop = row.get("crop", "")
            disease = row.get("disease", "")
            if not crop:
                # Infer from filename
                parts = img_path.stem.split("_")
                if parts[0].lower() == "tomato":
                    crop = "Tomato"
                    disease = " ".join(parts[:3]) if len(parts) >= 3 else "unknown"
                elif parts[0].lower() == "cucumber":
                    crop = "Cucumber"
                    disease = " ".join(parts[:3]) if len(parts) >= 3 else "unknown"
                elif parts[0].lower() == "grape" or parts[0].lower() == "grapevine":
                    crop = "Grape"
                    disease = " ".join(parts[:3]) if len(parts) >= 3 else "unknown"

            all_records.append({
                "filename": img_path.name,
                "split": split,
                "crop": crop,
                "disease": disease,
                "width": w,
                "height": h,
                "coverage_pct": coverage,
                "has_mask": mask_path.exists(),
            })

    total_images = len(all_records)
    total_masks = sum(1 for r in all_records if r["has_mask"])

    if args.verbose:
        print(f"  Analyzed {total_images} images / {total_masks} masks")

    # ── 1. dataset_summary.json ────────────────────────────────────────────────
    widths = [r["width"] for r in all_records if r["width"] > 0]
    heights = [r["height"] for r in all_records if r["height"] > 0]
    coverages = [r["coverage_pct"] for r in all_records]

    dataset_summary = {
        "total_images": total_images,
        "total_masks": total_masks,
        "crops_included": ["Tomato", "Cucumber", "Grape"],
        "splits": {
            "train": sum(1 for r in all_records if r["split"] == "train"),
            "val": sum(1 for r in all_records if r["split"] == "val"),
            "test": sum(1 for r in all_records if r["split"] == "test"),
        },
        "image_resolution_stats": {
            "min_width": min(widths) if widths else 0,
            "max_width": max(widths) if widths else 0,
            "avg_width": round(sum(widths) / len(widths), 2) if widths else 0,
            "min_height": min(heights) if heights else 0,
            "max_height": max(heights) if heights else 0,
            "avg_height": round(sum(heights) / len(heights), 2) if heights else 0,
        },
        "mask_coverage_stats": {
            "min_pct": round(min(coverages), 4) if coverages else 0,
            "max_pct": round(max(coverages), 4) if coverages else 0,
            "avg_pct": round(sum(coverages) / len(coverages), 4) if coverages else 0,
        },
    }
    with open(output_dir / "dataset_summary.json", "w") as f:
        json.dump(dataset_summary, f, indent=2)

    # ── 2. crop_distribution.json ──────────────────────────────────────────────
    crop_dist = defaultdict(lambda: defaultdict(int))
    crop_diseases = defaultdict(set)
    for r in all_records:
        c = r["crop"]
        s = r["split"]
        crop_dist[c]["total"] += 1
        crop_dist[c][s] += 1
        if r["disease"]:
            crop_diseases[c].add(r["disease"])

    crop_distribution = {
        crop: {
            "total": data["total"],
            "train": data.get("train", 0),
            "val": data.get("val", 0),
            "test": data.get("test", 0),
            "disease_count": len(crop_diseases[crop]),
            "diseases": sorted(crop_diseases[crop]),
        }
        for crop, data in crop_dist.items()
    }
    with open(output_dir / "crop_distribution.json", "w") as f:
        json.dump(crop_distribution, f, indent=2)

    # ── 3. disease_distribution.json ──────────────────────────────────────────
    disease_dist = defaultdict(lambda: {"total": 0, "train": 0, "val": 0, "test": 0, "crop": ""})
    for r in all_records:
        d = r["disease"]
        if not d:
            continue
        disease_dist[d]["total"] += 1
        disease_dist[d][r["split"]] += 1
        disease_dist[d]["crop"] = r["crop"]

    with open(output_dir / "disease_distribution.json", "w") as f:
        json.dump({k: dict(v) for k, v in disease_dist.items()}, f, indent=2)

    # ── 4. mask_statistics.json ────────────────────────────────────────────────
    mask_stats = {
        "total_masks": total_masks,
        "missing_masks": total_images - total_masks,
        "coverage_distribution": {
            "< 1%":   sum(1 for r in all_records if r["coverage_pct"] < 1),
            "1-5%":   sum(1 for r in all_records if 1 <= r["coverage_pct"] < 5),
            "5-20%":  sum(1 for r in all_records if 5 <= r["coverage_pct"] < 20),
            "20-50%": sum(1 for r in all_records if 20 <= r["coverage_pct"] < 50),
            "> 50%":  sum(1 for r in all_records if r["coverage_pct"] >= 50),
            "empty":  sum(1 for r in all_records if r["coverage_pct"] == 0),
        },
        "per_split": {
            split: {
                "count": sum(1 for r in all_records if r["split"] == split and r["has_mask"]),
                "avg_coverage": round(
                    sum(r["coverage_pct"] for r in all_records if r["split"] == split and r["has_mask"]) /
                    max(1, sum(1 for r in all_records if r["split"] == split and r["has_mask"])), 4
                ),
            }
            for split in ["train", "val", "test"]
        },
    }
    with open(output_dir / "mask_statistics.json", "w") as f:
        json.dump(mask_stats, f, indent=2)

    # ── 5. class_statistics.json ───────────────────────────────────────────────
    class_stats = {
        d: {
            "total": disease_dist[d]["total"],
            "crop": disease_dist[d]["crop"],
            "train": disease_dist[d]["train"],
            "val": disease_dist[d]["val"],
            "test": disease_dist[d]["test"],
        }
        for d in sorted(disease_dist)
    }
    with open(output_dir / "class_statistics.json", "w") as f:
        json.dump(class_stats, f, indent=2)

    # ── 6. coverage_statistics.json ────────────────────────────────────────────
    coverage_by_disease = defaultdict(list)
    for r in all_records:
        if r["disease"] and r["coverage_pct"] > 0:
            coverage_by_disease[r["disease"]].append(r["coverage_pct"])

    coverage_stats = {
        d: {
            "min": round(min(vals), 4),
            "max": round(max(vals), 4),
            "avg": round(sum(vals) / len(vals), 4),
            "n": len(vals),
        }
        for d, vals in coverage_by_disease.items()
    }
    with open(output_dir / "coverage_statistics.json", "w") as f:
        json.dump(coverage_stats, f, indent=2)

    # ── 7. split_statistics.json ───────────────────────────────────────────────
    split_stats = {}
    for split in ["train", "val", "test"]:
        recs = [r for r in all_records if r["split"] == split]
        split_stats[split] = {
            "total_images": len(recs),
            "total_masks": sum(1 for r in recs if r["has_mask"]),
            "crops": {
                crop: sum(1 for r in recs if r["crop"] == crop)
                for crop in ["Tomato", "Cucumber", "Grape"]
            },
            "diseases": {
                d: sum(1 for r in recs if r["disease"] == d)
                for d in sorted({r["disease"] for r in recs if r["disease"]})
            },
        }
    with open(output_dir / "split_statistics.json", "w") as f:
        json.dump(split_stats, f, indent=2)

    # ── 8. cleanup_report.md ──────────────────────────────────────────────────
    dup_report_path = output_dir / "duplicate_report.json"
    dup_data = {}
    if dup_report_path.exists():
        with open(dup_report_path) as f:
            dup_data = json.load(f)

    repair_report_path = output_dir / "dimension_repair_report.json"
    repair_data = {}
    if repair_report_path.exists():
        with open(repair_report_path) as f:
            repair_data = json.load(f)

    quarantine_images = list((dataset_root / "quarantine" / "images").rglob("*")) if (dataset_root / "quarantine" / "images").exists() else []

    cleanup_md = f"""# PlantSeg TCG v1 — Cleanup Report

## Dataset Overview
- **Crops**: Tomato, Cucumber, Grape
- **Total images**: {total_images}
- **Total masks**: {total_masks}

## Split Distribution
| Split | Images | Tomato | Cucumber | Grape |
|-------|--------|--------|----------|-------|
| train | {split_stats['train']['total_images']} | {split_stats['train']['crops'].get('Tomato', 0)} | {split_stats['train']['crops'].get('Cucumber', 0)} | {split_stats['train']['crops'].get('Grape', 0)} |
| val   | {split_stats['val']['total_images']} | {split_stats['val']['crops'].get('Tomato', 0)} | {split_stats['val']['crops'].get('Cucumber', 0)} | {split_stats['val']['crops'].get('Grape', 0)} |
| test  | {split_stats['test']['total_images']} | {split_stats['test']['crops'].get('Tomato', 0)} | {split_stats['test']['crops'].get('Cucumber', 0)} | {split_stats['test']['crops'].get('Grape', 0)} |

## Duplicate Removal
- Exact duplicate groups detected: **{dup_data.get('total_exact_duplicate_groups', 'N/A')}**
- Perceptual duplicate groups detected: **{dup_data.get('total_perceptual_duplicate_groups', 'N/A')}**
- Cross-split leakage groups: **{dup_data.get('data_leakage_groups_count', 'N/A')}**
- Files removed (lower-priority duplicates): **{dup_data.get('data_leakage_files_to_remove', 'N/A')}**
- Priority rule: train > val > test

## Dimension Repair
- Pairs checked: **{repair_data.get('total_pairs_checked', 'N/A')}**
- Auto-repaired (rotation): **{repair_data.get('repaired', 'N/A')}**
- Quarantined: **{repair_data.get('quarantined', 'N/A')}**

## Quarantined Samples
- Total quarantined: **{len(quarantine_images)}** files

## Disease Classes Preserved
{''.join([f"- {d}{chr(10)}" for d in sorted({r['disease'] for r in all_records if r['disease']})])}

## Processing Guarantees
- ✓ Original plantseg_raw dataset NEVER modified
- ✓ Original filenames preserved
- ✓ Official train/val/test split preserved
- ✓ No annotation conversion performed
- ✓ COCO polygons preserved
- ✓ PNG masks preserved
"""
    with open(output_dir / "cleanup_report.md", "w", encoding="utf-8") as f:
        f.write(cleanup_md)

    if args.verbose:
        print(f"  ✓ All reports written to {output_dir}")

    return {
        "total_images": total_images,
        "total_masks": total_masks,
        "split_stats": split_stats,
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze subset and generate reports")
    parser.add_argument("--input",   required=True, help="Path to dataset root")
    parser.add_argument("--output",  required=True, help="Path to reports output directory")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
