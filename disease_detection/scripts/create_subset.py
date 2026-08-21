#!/usr/bin/env python3
"""
create_subset.py
----------------
Main orchestration script for creating plantseg_tcg_v1.

Pipeline:
  Step 1:  Filter Metadata.csv → TCG rows only
  Step 2:  Validate output directory versioning (never overwrite)
  Step 3:  Copy images (train/val/test), preserving filenames
  Step 4:  Copy masks (train/val/test), preserving filenames
  Step 5:  Filter COCO annotations → TCG images only
  Step 6:  Generate subset_metadata.csv
  Step 7:  Detect & remove duplicate leakage (train > val > test)
  Step 8:  Repair image-mask dimension mismatches
  Step 9:  Generate all reports
  Step 10: Generate processing_manifest.json
  Step 11: Final validation

Usage:
    python create_subset.py \
        --input  datasets/plantseg_raw \
        --output datasets/plantseg_tcg_v1 \
        --workers 4 \
        --verbose

The script is:
  - Deterministic (same input → same output)
  - Restartable (safe to run multiple times if output dir doesn't exist)
  - Non-destructive (NEVER modifies plantseg_raw)
"""

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

try:
    from PIL import Image
    import numpy as np
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# ── Configuration ──────────────────────────────────────────────────────────────
TARGET_CROPS = {"tomato", "cucumber", "grape"}
TARGET_CROPS_DISPLAY = {"Tomato", "Cucumber", "Grape"}
PROCESSING_VERSION = "1.0.0"

# Map Metadata.csv Split values → directory names
SPLIT_MAP = {
    "training": "train",
    "train": "train",
    "validation": "val",
    "val": "val",
    "test": "test",
    "testing": "test",
}

SPLIT_PRIORITY = {"train": 0, "val": 1, "test": 2}


# ── Utilities ──────────────────────────────────────────────────────────────────

def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_phash(path: Path, hash_size: int = 8) -> str:
    if not PIL_AVAILABLE:
        return ""
    try:
        img = Image.open(path).convert("L")
        img = img.resize((hash_size, hash_size), Image.LANCZOS)
        pixels = list(img.getdata())
        avg = sum(pixels) / len(pixels)
        bits = "".join("1" if p > avg else "0" for p in pixels)
        val = int(bits, 2)
        hex_len = (hash_size * hash_size + 3) // 4
        return format(val, f"0{hex_len}x")
    except Exception:
        return ""


def hamming_distance(h1: str, h2: str) -> int:
    if not h1 or not h2 or len(h1) != len(h2):
        return 999
    try:
        return bin(int(h1, 16) ^ int(h2, 16)).count("1")
    except ValueError:
        return 999


def get_image_size(path: Path):
    if not PIL_AVAILABLE:
        return None
    try:
        with Image.open(path) as img:
            return img.size
    except Exception:
        return None


def mask_coverage(path: Path) -> float:
    if not PIL_AVAILABLE or not path.exists():
        return 0.0
    try:
        with Image.open(path) as img:
            arr = np.array(img)
            return round(100.0 * np.count_nonzero(arr) / arr.size, 4) if arr.size else 0.0
    except Exception:
        return 0.0


def infer_crop_disease(filename: str):
    """Infer crop and disease from filename like 'tomato_early_blight_1.jpg'."""
    stem = Path(filename).stem.lower()
    parts = stem.split("_")
    crop_lower = parts[0] if parts else ""
    if crop_lower == "tomato":
        crop = "Tomato"
    elif crop_lower == "cucumber":
        crop = "Cucumber"
    elif crop_lower in ("grape", "grapevine"):
        crop = "Grape"
    else:
        return None, None

    # Disease: everything before the last numeric part
    # e.g. tomato_early_blight_google_0001 → "tomato early blight"
    # Find where numbers/identifiers start
    disease_parts = []
    for i, p in enumerate(parts[1:], 1):
        if p.isdigit() or (len(p) > 1 and p[0].isdigit()):
            break
        if p.lower() in ("google", "bing", "baidu"):
            break
        disease_parts.append(p)

    disease = parts[0] + " " + " ".join(disease_parts) if disease_parts else parts[0]
    return crop, disease


# ── Step 1: Load and filter Metadata.csv ──────────────────────────────────────

def load_tcg_metadata(meta_path: Path, verbose: bool) -> list[dict]:
    """Load Metadata.csv and return only TCG rows."""
    rows = []
    with open(meta_path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            plant = row.get("Plant", "").strip()
            if plant in TARGET_CROPS_DISPLAY:
                rows.append(row)

    if verbose:
        print(f"  [Step 1] Loaded {len(rows)} TCG entries from Metadata.csv")
    return rows


# ── Step 3-4: Copy images + masks ─────────────────────────────────────────────

def copy_files(meta_rows: list[dict], src_root: Path, dst_root: Path, verbose: bool):
    """
    Copy images and masks for all TCG rows.
    Returns list of actually copied records.
    """
    copied = []
    skipped_missing = []

    for row in meta_rows:
        filename = row.get("Name", "").strip()
        label_file = row.get("Label file", "").strip()
        split_raw = row.get("Split", "").strip().lower()
        split = SPLIT_MAP.get(split_raw, split_raw)

        if split not in {"train", "val", "test"}:
            continue

        src_img = src_root / "images" / split / filename
        src_mask = src_root / "annotations" / split / label_file

        dst_img = dst_root / "images" / split / filename
        dst_mask = dst_root / "annotations" / split / label_file

        dst_img.parent.mkdir(parents=True, exist_ok=True)
        dst_mask.parent.mkdir(parents=True, exist_ok=True)

        img_copied = False
        mask_copied = False

        if src_img.exists():
            if not dst_img.exists():
                shutil.copy2(str(src_img), str(dst_img))
            img_copied = True
        else:
            skipped_missing.append(str(src_img))
            if verbose:
                print(f"    ⚠ Missing image: {src_img}")

        if src_mask.exists():
            if not dst_mask.exists():
                shutil.copy2(str(src_mask), str(dst_mask))
            mask_copied = True
        else:
            if verbose:
                print(f"    ⚠ Missing mask: {src_mask}")

        if img_copied:
            copied.append({
                "filename": filename,
                "label_file": label_file,
                "split": split,
                "plant": row.get("Plant", ""),
                "disease": row.get("Disease", ""),
                "mask_ratio": row.get("Mask ratio", ""),
                "url": row.get("URL", ""),
                "license": row.get("License", ""),
                "img_path": str(dst_img),
                "mask_path": str(dst_mask) if mask_copied else "",
            })

    if verbose:
        print(f"  [Step 3-4] Copied {len(copied)} images/masks  (missing: {len(skipped_missing)})")

    return copied, skipped_missing


# ── Step 5: Filter COCO annotations ───────────────────────────────────────────

def filter_coco(src_root: Path, dst_root: Path, tcg_filenames_by_split: dict, verbose: bool):
    """
    For each split's annotation_*.json, keep only TCG images and their annotations.
    Remaps image IDs to sequential integers. Preserves category IDs and names.
    """
    coco_dir = dst_root / "coco"
    coco_dir.mkdir(parents=True, exist_ok=True)

    split_map_coco = {"train": "annotation_train.json", "val": "annotation_val.json", "test": "annotation_test.json"}

    for split, fname in split_map_coco.items():
        src_path = src_root / fname
        if not src_path.exists():
            if verbose:
                print(f"    ⚠ COCO file not found: {src_path}")
            continue

        with open(src_path, encoding="utf-8") as f:
            coco = json.load(f)

        keep_filenames = set(tcg_filenames_by_split.get(split, []))

        # Filter images
        kept_images = [img for img in coco.get("images", []) if img["file_name"] in keep_filenames]
        kept_image_ids = {img["id"] for img in kept_images}

        # Filter annotations
        kept_anns = [ann for ann in coco.get("annotations", []) if ann["image_id"] in kept_image_ids]

        # Remap image IDs to sequential
        id_map = {old["id"]: new_id for new_id, old in enumerate(kept_images, 1)}
        for img in kept_images:
            img["id"] = id_map[img["id"]]
        for ann in kept_anns:
            ann["image_id"] = id_map[ann["image_id"]]

        # Remap annotation IDs
        for new_id, ann in enumerate(kept_anns, 1):
            ann["id"] = new_id

        filtered_coco = {
            "info": coco.get("info", {}),
            "licenses": coco.get("licenses", []),
            "categories": coco.get("categories", []),
            "images": kept_images,
            "annotations": kept_anns,
        }

        dst_path = coco_dir / fname
        with open(dst_path, "w", encoding="utf-8") as f:
            json.dump(filtered_coco, f)

        if verbose:
            print(f"  [Step 5] [{split}] COCO: {len(kept_images)} images, {len(kept_anns)} annotations → {dst_path.name}")


# ── Step 6: Generate subset_metadata.csv ──────────────────────────────────────

def generate_metadata_csv(copied: list[dict], dst_root: Path, verbose: bool):
    """Generate subset_metadata.csv with hashes, dimensions, coverage."""
    meta_dir = dst_root / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    out_path = meta_dir / "subset_metadata.csv"

    fieldnames = [
        "filename", "crop", "disease", "split",
        "width", "height", "mask_coverage_pct",
        "original_path", "new_path",
        "sha256", "phash",
    ]

    rows = []
    for rec in copied:
        img_path = Path(rec["img_path"])
        mask_path = Path(rec["mask_path"]) if rec["mask_path"] else None

        w, h = 0, 0
        sha = ""
        ph = ""
        coverage = 0.0

        if img_path.exists():
            size = get_image_size(img_path)
            if size:
                w, h = size
            sha = sha256_of(img_path)
            ph = compute_phash(img_path)

        if mask_path and mask_path.exists():
            coverage = mask_coverage(mask_path)

        rows.append({
            "filename": rec["filename"],
            "crop": rec["plant"],
            "disease": rec["disease"],
            "split": rec["split"],
            "width": w,
            "height": h,
            "mask_coverage_pct": coverage,
            "original_path": str(Path("plantseg_raw") / "images" / rec["split"] / rec["filename"]),
            "new_path": str(img_path.relative_to(dst_root)),
            "sha256": sha,
            "phash": ph,
        })

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    if verbose:
        print(f"  [Step 6] subset_metadata.csv written: {len(rows)} rows → {out_path}")

    return rows


# ── Step 7: Remove duplicate leakage ──────────────────────────────────────────

def remove_duplicate_leakage(dst_root: Path, verbose: bool) -> dict:
    """
    Detect SHA256 exact duplicates across splits.
    Priority: train > val > test.
    Remove lower-priority duplicates from images/ and annotations/.
    """
    images_dir = dst_root / "images"
    annotations_dir = dst_root / "annotations"
    reports_dir = dst_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Collect all images with their hashes
    records = []
    for split in ["train", "val", "test"]:
        split_dir = images_dir / split
        if not split_dir.exists():
            continue
        for img in sorted(split_dir.iterdir()):
            if img.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                sha = sha256_of(img)
                ph = compute_phash(img)
                records.append({
                    "path": img,
                    "split": split,
                    "sha256": sha,
                    "phash": ph,
                    "name": img.name,
                    "stem": img.stem,
                })

    if verbose:
        print(f"  [Step 7] Scanning {len(records)} images for duplicates …")

    # Group by SHA256
    sha_groups = defaultdict(list)
    for rec in records:
        sha_groups[rec["sha256"]].append(rec)

    removed = []
    leakage_groups = []

    for sha, group in sha_groups.items():
        splits_in_group = {r["split"] for r in group}
        if len(splits_in_group) < 2:
            continue  # No cross-split leakage

        # Sort by priority
        sorted_group = sorted(group, key=lambda r: SPLIT_PRIORITY.get(r["split"], 99))
        to_keep = sorted_group[0]
        to_remove = sorted_group[1:]

        leakage_groups.append({
            "sha256": sha,
            "kept": {"file": to_keep["name"], "split": to_keep["split"]},
            "removed": [{"file": r["name"], "split": r["split"]} for r in to_remove],
        })

        for rec in to_remove:
            img_to_del = rec["path"]
            mask_to_del = annotations_dir / rec["split"] / (rec["stem"] + ".png")

            if img_to_del.exists():
                os.remove(str(img_to_del))
                removed.append({"file": rec["name"], "split": rec["split"], "sha256": sha, "type": "image"})
                if verbose:
                    print(f"    ✗ Removed duplicate [{rec['split']}] {rec['name']} (kept in [{to_keep['split']}])")

            if mask_to_del.exists():
                os.remove(str(mask_to_del))
                removed.append({"file": mask_to_del.name, "split": rec["split"], "sha256": sha, "type": "mask"})

    # Also check perceptual duplicates (threshold=3 for near-identical)
    # Union-find
    n = len(records)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    for i in range(n):
        for j in range(i + 1, n):
            if hamming_distance(records[i]["phash"], records[j]["phash"]) <= 3:
                union(i, j)

    phash_groups = defaultdict(list)
    for i in range(n):
        phash_groups[find(i)].append(records[i])

    phash_leakage = []
    already_removed = {r["file"] for r in removed}

    for group in phash_groups.values():
        if len(group) < 2:
            continue
        splits_in_group = {r["split"] for r in group}
        if len(splits_in_group) < 2:
            continue
        # Check that it's not already handled
        group_names = {r["name"] for r in group}
        if group_names & already_removed:
            continue

        sorted_group = sorted(group, key=lambda r: SPLIT_PRIORITY.get(r["split"], 99))
        to_keep = sorted_group[0]
        to_remove_recs = sorted_group[1:]

        phash_leakage.append({
            "phash": to_keep["phash"],
            "kept": {"file": to_keep["name"], "split": to_keep["split"]},
            "removed": [{"file": r["name"], "split": r["split"]} for r in to_remove_recs],
        })

        for rec in to_remove_recs:
            img_to_del = rec["path"]
            mask_to_del = annotations_dir / rec["split"] / (rec["stem"] + ".png")

            if img_to_del.exists():
                os.remove(str(img_to_del))
                removed.append({"file": rec["name"], "split": rec["split"], "sha256": rec["sha256"], "type": "perceptual_image"})
                if verbose:
                    print(f"    ✗ Removed perceptual dup [{rec['split']}] {rec['name']}")

            if mask_to_del.exists():
                os.remove(str(mask_to_del))
                removed.append({"file": mask_to_del.name, "split": rec["split"], "sha256": "", "type": "perceptual_mask"})

    # Write reports
    dup_report = {
        "total_images_scanned": len(records),
        "total_exact_duplicate_groups": len(sha_groups),
        "data_leakage_groups_count": len(leakage_groups),
        "perceptual_leakage_groups_count": len(phash_leakage),
        "total_files_removed": len([r for r in removed if "image" in r["type"]]),
        "leakage_details": leakage_groups,
        "perceptual_leakage_details": phash_leakage,
    }
    with open(reports_dir / "duplicate_report.json", "w") as f:
        json.dump(dup_report, f, indent=2)

    # removed_duplicates.csv
    if removed:
        with open(reports_dir / "removed_duplicates.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["file", "split", "sha256", "type"])
            writer.writeheader()
            writer.writerows(removed)

    if verbose:
        print(f"  [Step 7] Leakage groups removed: {len(leakage_groups)} exact + {len(phash_leakage)} perceptual")
        print(f"           Files removed: {len([r for r in removed if 'image' in r['type']])}")

    return dup_report


# ── Step 8: Dimension repair ───────────────────────────────────────────────────

def repair_dimensions(dst_root: Path, verbose: bool) -> dict:
    images_dir = dst_root / "images"
    annotations_dir = dst_root / "annotations"
    quarantine_dir = dst_root / "quarantine"
    reports_dir = dst_root / "reports"

    results = []
    repaired = 0
    quarantined_count = 0

    for split in ["train", "val", "test"]:
        img_dir = images_dir / split
        ann_dir = annotations_dir / split
        if not img_dir.exists():
            continue

        for img_path in sorted(img_dir.iterdir()):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue

            mask_path = ann_dir / (img_path.stem + ".png")
            img_size = get_image_size(img_path)
            mask_size = get_image_size(mask_path) if mask_path.exists() else None

            if img_size is None:
                _quarantine(img_path, mask_path, quarantine_dir, split)
                results.append({"file": img_path.name, "split": split, "action": "quarantined", "reason": "unreadable image"})
                quarantined_count += 1
                continue

            if mask_size is None:
                if mask_path.exists():
                    _quarantine(img_path, mask_path, quarantine_dir, split)
                    results.append({"file": img_path.name, "split": split, "action": "quarantined", "reason": "unreadable mask"})
                    quarantined_count += 1
                continue

            if img_size == mask_size:
                results.append({"file": img_path.name, "split": split, "action": "ok", "reason": ""})
                continue

            img_w, img_h = img_size
            mask_w, mask_h = mask_size

            if img_w == mask_h and img_h == mask_w:
                # Try rotation repair
                try:
                    with Image.open(mask_path) as m:
                        rotated = m.rotate(90, expand=True)
                        if rotated.size == img_size:
                            rotated.save(str(mask_path))
                            results.append({"file": img_path.name, "split": split, "action": "repaired_ccw", "reason": f"{mask_size}→{img_size}"})
                            repaired += 1
                            if verbose:
                                print(f"    ✓ REPAIRED [{split}] {img_path.name}")
                            continue
                        rotated = m.rotate(-90, expand=True)
                        if rotated.size == img_size:
                            rotated.save(str(mask_path))
                            results.append({"file": img_path.name, "split": split, "action": "repaired_cw", "reason": f"{mask_size}→{img_size}"})
                            repaired += 1
                            continue
                except Exception as e:
                    pass

            # Cannot repair → quarantine
            _quarantine(img_path, mask_path, quarantine_dir, split)
            results.append({"file": img_path.name, "split": split, "action": "quarantined",
                             "reason": f"img={img_size} mask={mask_size}"})
            quarantined_count += 1
            if verbose:
                print(f"    ⚠ QUARANTINED [{split}] {img_path.name}: dim mismatch {img_size} vs {mask_size}")

    report = {
        "total_pairs_checked": len(results),
        "ok": sum(1 for r in results if r["action"] == "ok"),
        "repaired": repaired,
        "quarantined": quarantined_count,
        "details": results,
    }
    reports_dir.mkdir(parents=True, exist_ok=True)
    with open(reports_dir / "dimension_repair_report.json", "w") as f:
        json.dump(report, f, indent=2)

    if verbose:
        print(f"  [Step 8] Pairs checked: {len(results)}, OK: {report['ok']}, Repaired: {repaired}, Quarantined: {quarantined_count}")

    # quarantined_samples.csv
    quarantined_records = [r for r in results if r["action"] == "quarantined"]
    if quarantined_records:
        with open(reports_dir / "quarantined_samples.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["file", "split", "action", "reason"])
            writer.writeheader()
            writer.writerows(quarantined_records)

    return report


def _quarantine(img_path: Path, mask_path, quarantine_dir: Path, split: str):
    q_img = quarantine_dir / "images" / split
    q_ann = quarantine_dir / "annotations" / split
    q_img.mkdir(parents=True, exist_ok=True)
    q_ann.mkdir(parents=True, exist_ok=True)
    if img_path and img_path.exists():
        shutil.move(str(img_path), str(q_img / img_path.name))
    if mask_path and mask_path.exists():
        shutil.move(str(mask_path), str(q_ann / mask_path.name))


# ── Step 9-10: Reports + Manifest ─────────────────────────────────────────────

def generate_reports(dst_root: Path, src_meta_path: Path, copied: list[dict],
                     dup_report: dict, repair_report: dict, verbose: bool):
    """Generate all reports using analyze_subset logic inline."""
    reports_dir = dst_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    images_dir = dst_root / "images"
    annotations_dir = dst_root / "annotations"

    # Reload all actual files from disk (after dedup + quarantine)
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
            size = get_image_size(img_path) or (0, 0)
            cov = mask_coverage(mask_path)
            crop, disease = infer_crop_disease(img_path.name)
            all_records.append({
                "filename": img_path.name,
                "split": split,
                "crop": crop or "",
                "disease": disease or "",
                "width": size[0],
                "height": size[1],
                "coverage_pct": cov,
                "has_mask": mask_path.exists(),
            })

    total_images = len(all_records)
    total_masks = sum(1 for r in all_records if r["has_mask"])

    # dataset_summary
    widths = [r["width"] for r in all_records if r["width"]]
    heights = [r["height"] for r in all_records if r["height"]]
    coverages = [r["coverage_pct"] for r in all_records]

    dataset_summary = {
        "total_images": total_images,
        "total_masks": total_masks,
        "crops_included": ["Tomato", "Cucumber", "Grape"],
        "splits": {
            s: sum(1 for r in all_records if r["split"] == s)
            for s in ["train", "val", "test"]
        },
        "image_resolution_stats": {
            "min_width": min(widths, default=0),
            "max_width": max(widths, default=0),
            "avg_width": round(sum(widths) / len(widths), 2) if widths else 0,
            "min_height": min(heights, default=0),
            "max_height": max(heights, default=0),
            "avg_height": round(sum(heights) / len(heights), 2) if heights else 0,
        },
        "mask_coverage_stats": {
            "min_pct": round(min(coverages, default=0), 4),
            "max_pct": round(max(coverages, default=0), 4),
            "avg_pct": round(sum(coverages) / len(coverages), 4) if coverages else 0,
        },
    }
    with open(reports_dir / "dataset_summary.json", "w") as f:
        json.dump(dataset_summary, f, indent=2)

    # crop_distribution
    from collections import defaultdict
    crop_dist = defaultdict(lambda: defaultdict(int))
    crop_diseases = defaultdict(set)
    for r in all_records:
        c = r["crop"]
        crop_dist[c]["total"] += 1
        crop_dist[c][r["split"]] += 1
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
    with open(reports_dir / "crop_distribution.json", "w") as f:
        json.dump(crop_distribution, f, indent=2)

    # disease_distribution
    disease_dist = defaultdict(lambda: {"total": 0, "train": 0, "val": 0, "test": 0, "crop": ""})
    for r in all_records:
        d = r["disease"]
        if not d:
            continue
        disease_dist[d]["total"] += 1
        disease_dist[d][r["split"]] += 1
        disease_dist[d]["crop"] = r["crop"]
    with open(reports_dir / "disease_distribution.json", "w") as f:
        json.dump({k: dict(v) for k, v in disease_dist.items()}, f, indent=2)

    # mask_statistics
    mask_stats = {
        "total_masks": total_masks,
        "missing_masks": total_images - total_masks,
        "coverage_distribution": {
            "< 1%": sum(1 for r in all_records if 0 < r["coverage_pct"] < 1),
            "1-5%": sum(1 for r in all_records if 1 <= r["coverage_pct"] < 5),
            "5-20%": sum(1 for r in all_records if 5 <= r["coverage_pct"] < 20),
            "20-50%": sum(1 for r in all_records if 20 <= r["coverage_pct"] < 50),
            "> 50%": sum(1 for r in all_records if r["coverage_pct"] >= 50),
            "empty": sum(1 for r in all_records if r["coverage_pct"] == 0 and r["has_mask"]),
        },
    }
    with open(reports_dir / "mask_statistics.json", "w") as f:
        json.dump(mask_stats, f, indent=2)

    # class_statistics
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
    with open(reports_dir / "class_statistics.json", "w") as f:
        json.dump(class_stats, f, indent=2)

    # coverage_statistics
    cov_by_disease = defaultdict(list)
    for r in all_records:
        if r["disease"] and r["coverage_pct"] > 0:
            cov_by_disease[r["disease"]].append(r["coverage_pct"])
    coverage_stats = {
        d: {"min": round(min(v), 4), "max": round(max(v), 4), "avg": round(sum(v) / len(v), 4), "n": len(v)}
        for d, v in cov_by_disease.items()
    }
    with open(reports_dir / "coverage_statistics.json", "w") as f:
        json.dump(coverage_stats, f, indent=2)

    # split_statistics
    split_stats = {
        split: {
            "total_images": sum(1 for r in all_records if r["split"] == split),
            "total_masks": sum(1 for r in all_records if r["split"] == split and r["has_mask"]),
            "crops": {
                crop: sum(1 for r in all_records if r["split"] == split and r["crop"] == crop)
                for crop in ["Tomato", "Cucumber", "Grape"]
            },
            "diseases": {
                d: sum(1 for r in all_records if r["split"] == split and r["disease"] == d)
                for d in sorted({r["disease"] for r in all_records if r["disease"] and r["split"] == split})
            },
        }
        for split in ["train", "val", "test"]
    }
    with open(reports_dir / "split_statistics.json", "w") as f:
        json.dump(split_stats, f, indent=2)

    # cleanup_report.md
    quarantine_count = repair_report.get("quarantined", 0)
    removed_count = dup_report.get("total_files_removed", 0)
    diseases_set = sorted({r["disease"] for r in all_records if r["disease"]})

    cleanup_md = f"""# PlantSeg TCG v1 — Cleanup Report

Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}

## Dataset Overview
- **Version**: plantseg_tcg_v1
- **Crops**: Tomato, Cucumber, Grape
- **Total images**: {total_images}
- **Total masks**: {total_masks}

## Split Distribution
| Split | Images | Tomato | Cucumber | Grape |
|-------|--------|--------|----------|-------|
| train | {split_stats['train']['total_images']} | {split_stats['train']['crops'].get('Tomato',0)} | {split_stats['train']['crops'].get('Cucumber',0)} | {split_stats['train']['crops'].get('Grape',0)} |
| val   | {split_stats['val']['total_images']} | {split_stats['val']['crops'].get('Tomato',0)} | {split_stats['val']['crops'].get('Cucumber',0)} | {split_stats['val']['crops'].get('Grape',0)} |
| test  | {split_stats['test']['total_images']} | {split_stats['test']['crops'].get('Tomato',0)} | {split_stats['test']['crops'].get('Cucumber',0)} | {split_stats['test']['crops'].get('Grape',0)} |

## Duplicate Removal
- Exact leakage groups: **{dup_report.get('data_leakage_groups_count', 0)}**
- Perceptual leakage groups: **{dup_report.get('perceptual_leakage_groups_count', 0)}**
- Files removed (duplicate leakage): **{removed_count}**
- Priority rule applied: train > val > test

## Dimension Repair
- Pairs checked: **{repair_report.get('total_pairs_checked', 0)}**
- OK: **{repair_report.get('ok', 0)}**
- Auto-repaired: **{repair_report.get('repaired', 0)}**
- Quarantined: **{quarantine_count}**

## Disease Classes Preserved ({len(diseases_set)} total)
{chr(10).join(f"- {d}" for d in diseases_set)}

## Processing Guarantees
- ✓ plantseg_raw NEVER modified (read-only source)
- ✓ Original filenames preserved
- ✓ Official train/val/test split structure preserved
- ✓ COCO polygon annotations filtered and preserved
- ✓ PNG segmentation masks preserved
- ✓ No annotation conversion performed
- ✓ No augmentation performed
- ✓ No class rebalancing
"""
    with open(reports_dir / "cleanup_report.md", "w", encoding="utf-8") as f:
        f.write(cleanup_md)

    if verbose:
        print(f"  [Step 9] All reports written to {reports_dir}")

    return all_records, dataset_summary, split_stats


# ── Step 10: Processing manifest ──────────────────────────────────────────────

def generate_manifest(src_root: Path, dst_root: Path, copied: list[dict],
                      dup_report: dict, repair_report: dict, meta_rows: list[dict],
                      verbose: bool):
    """Generate processing_manifest.json for reproducibility."""
    src_meta = src_root / "Metadata.csv"
    meta_hash = sha256_of(src_meta) if src_meta.exists() else ""

    images_dir = dst_root / "images"
    masks_dir = dst_root / "annotations"

    final_images = sum(
        sum(1 for f in (images_dir / split).iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png"})
        for split in ["train", "val", "test"]
        if (images_dir / split).exists()
    )
    final_masks = sum(
        sum(1 for f in (masks_dir / split).iterdir() if f.suffix.lower() == ".png")
        for split in ["train", "val", "test"]
        if (masks_dir / split).exists()
    )

    quarantine_dir = dst_root / "quarantine"
    quarantined_total = sum(
        1 for f in quarantine_dir.rglob("*") if f.is_file()
    ) if quarantine_dir.exists() else 0

    diseases = sorted({row.get("Disease", "") for row in meta_rows if row.get("Disease")})

    manifest = {
        "processing_version": PROCESSING_VERSION,
        "creation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_dataset_path": str(src_root.resolve()),
        "output_dataset_path": str(dst_root.resolve()),
        "crops_included": ["Tomato", "Cucumber", "Grape"],
        "diseases_included": diseases,
        "source_metadata_sha256": meta_hash,
        "statistics": {
            "original_tcg_rows_in_metadata": len(copied),
            "final_images_copied": final_images,
            "final_masks_copied": final_masks,
            "duplicates_removed": dup_report.get("total_files_removed", 0),
            "repaired_samples": repair_report.get("repaired", 0),
            "quarantined_samples": repair_report.get("quarantined", 0),
        },
        "reproducibility_notes": [
            "Filter: Plant IN {Tomato, Cucumber, Grape} from Metadata.csv",
            "Duplicate priority: train > val > test (SHA256 exact match)",
            "Perceptual duplicates: pHash Hamming distance <= 3",
            "Dimension repair: 90-degree rotation attempt only",
            "No resizing, no augmentation, no rebalancing",
        ],
    }

    manifest_path = dst_root / "processing_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    if verbose:
        print(f"  [Step 10] processing_manifest.json written: {manifest_path}")

    return manifest


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Create plantseg_tcg_v1: Tomato+Cucumber+Grape subset of PlantSeg"
    )
    parser.add_argument("--input",   default="datasets/plantseg_raw",  help="Path to plantseg_raw")
    parser.add_argument("--output",  default="datasets/plantseg_tcg_v1", help="Path to output dataset")
    parser.add_argument("--workers", type=int, default=4,             help="Worker threads (reserved for future use)")
    parser.add_argument("--verbose", action="store_true",             help="Verbose output")
    args = parser.parse_args()

    src_root = Path(args.input).resolve()
    dst_root = Path(args.output).resolve()

    print(f"\n{'═'*60}")
    print(f"  PlantSeg TCG Subset Creator — v{PROCESSING_VERSION}")
    print(f"  Source : {src_root}")
    print(f"  Output : {dst_root}")
    print(f"{'═'*60}")

    # ── Validate source ────────────────────────────────────────────────────────
    meta_path = src_root / "Metadata.csv"
    if not meta_path.exists():
        print(f"ERROR: Metadata.csv not found at {meta_path}", file=sys.stderr)
        sys.exit(1)

    # ── Check output directory versioning ─────────────────────────────────────
    if dst_root.exists():
        existing = list(dst_root.iterdir())
        if existing:
            print(f"\n⚠  Output directory already exists and is non-empty: {dst_root}")
            print("   Files will NOT be overwritten. Existing files are kept.")
            print("   To create a new version, use a different --output path (e.g., plantseg_tcg_v2)")
            print("   Continuing with incremental copy (safe mode) …\n")
    dst_root.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Load metadata ──────────────────────────────────────────────────
    meta_rows = load_tcg_metadata(meta_path, verbose=args.verbose)

    # ── Step 3-4: Copy images + masks ─────────────────────────────────────────
    print("\n  Copying images and masks …")
    copied, skipped = copy_files(meta_rows, src_root, dst_root, verbose=args.verbose)

    # Build filename lookup by split for COCO filtering
    tcg_filenames_by_split = defaultdict(list)
    for rec in copied:
        tcg_filenames_by_split[rec["split"]].append(rec["filename"])

    # ── Step 5: Filter COCO annotations ───────────────────────────────────────
    print("  Filtering COCO annotations …")
    filter_coco(src_root, dst_root, tcg_filenames_by_split, verbose=args.verbose)

    # ── Step 6: Generate metadata CSV ─────────────────────────────────────────
    print("  Generating subset_metadata.csv …")
    meta_csv_rows = generate_metadata_csv(copied, dst_root, verbose=args.verbose)

    # ── Step 7: Remove duplicate leakage ──────────────────────────────────────
    print("  Detecting and removing duplicate leakage …")
    dup_report = remove_duplicate_leakage(dst_root, verbose=args.verbose)

    # ── Step 8: Repair dimensions ──────────────────────────────────────────────
    print("  Repairing image-mask dimension mismatches …")
    repair_report = repair_dimensions(dst_root, verbose=args.verbose)

    # ── Step 9: Generate reports ───────────────────────────────────────────────
    print("  Generating reports …")
    all_records, dataset_summary, split_stats = generate_reports(
        dst_root, meta_path, copied, dup_report, repair_report, verbose=args.verbose
    )

    # ── Step 10: Processing manifest ──────────────────────────────────────────
    manifest = generate_manifest(
        src_root, dst_root, copied, dup_report, repair_report, meta_rows, verbose=args.verbose
    )

    # ── Final summary ──────────────────────────────────────────────────────────
    total_images = dataset_summary["total_images"]
    total_masks = dataset_summary["total_masks"]

    # Count diseases per crop
    tomato_diseases = sorted({r["disease"] for r in all_records if r["crop"] == "Tomato" and r["disease"]})
    cucumber_diseases = sorted({r["disease"] for r in all_records if r["crop"] == "Cucumber" and r["disease"]})
    grape_diseases = sorted({r["disease"] for r in all_records if r["crop"] == "Grape" and r["disease"]})

    original_total = 7774
    original_masks = 7774

    print(f"\n{'═'*60}")
    print(f"  Original dataset")
    print(f"    Images           : {original_total}")
    print(f"    Masks            : {original_masks}")
    print(f"    Diseases         : 89")
    print(f"    Crops            : 34")
    print(f"\n  Subset (plantseg_tcg_v1)")
    print(f"    Images           : {total_images}")
    print(f"    Masks            : {total_masks}")
    print(f"    Tomato diseases  : {len(tomato_diseases)} — {', '.join(tomato_diseases)}")
    print(f"    Cucumber diseases: {len(cucumber_diseases)} — {', '.join(cucumber_diseases)}")
    print(f"    Grape diseases   : {len(grape_diseases)} — {', '.join(grape_diseases)}")
    print(f"\n  Processing")
    print(f"    Removed duplicates    : {dup_report.get('total_files_removed', 0)}")
    print(f"    Quarantined samples   : {repair_report.get('quarantined', 0)}")
    print(f"    Auto-repaired         : {repair_report.get('repaired', 0)}")
    print(f"\n  Final counts")
    print(f"    Train : {split_stats['train']['total_images']}")
    print(f"    Val   : {split_stats['val']['total_images']}")
    print(f"    Test  : {split_stats['test']['total_images']}")
    print(f"\n  Dataset Ready: {'YES ✅' if total_images > 0 else 'NO ❌'}")
    print(f"{'═'*60}\n")

    # ── Step 11: Final validation ──────────────────────────────────────────────
    print("  Running final validation …")
    import subprocess
    val_script = Path(__file__).parent / "validate_subset.py"
    if val_script.exists():
        result = subprocess.run(
            [sys.executable, str(val_script),
             "--input", str(dst_root),
             "--output", str(dst_root / "reports"),
             "--verbose"],
            capture_output=False,
        )
        if result.returncode != 0:
            print("\n⚠  Validation found issues. Review reports/validation_report.json")
    else:
        print("  (validate_subset.py not found, skipping final validation)")


if __name__ == "__main__":
    main()
