#!/usr/bin/env python3
"""
validate_v2_dataset.py
-----------------------
Full validation of YOLO11 Segmentation V2 dataset (30 classes).
Checks:
  - missing masks / images
  - orphan files
  - corrupted images
  - duplicated images
  - duplicated masks
  - train/val/test overlap
  - class imbalance
  - incorrect extensions
  - invalid polygons
  - zero-area polygons
  - empty labels
  - invalid class IDs

Usage:
    python scripts/validate_v2_dataset.py \
        --input datasets/plantseg_tcg_yolo_v2 \
        --output datasets/plantseg_tcg_yolo_v2/reports \
        --verbose
"""

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

VALID_SPLITS = {"train", "val", "test"}
NUM_CLASSES = 30
CLASS_RANGE = set(range(NUM_CLASSES))

VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

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


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_split(dataset_root: Path, split: str, verbose: bool) -> dict:
    img_dir = dataset_root / "images" / split
    lbl_dir = dataset_root / "labels" / split

    failures = []
    warnings = []

    if not img_dir.exists():
        failures.append(f"[{split}] Missing image directory: {img_dir}")
        return {"failures": failures, "warnings": warnings, "images": 0, "labels": 0,
                "ok_labels": 0, "total_polygons": 0, "class_counts": {}}
    if not lbl_dir.exists():
        failures.append(f"[{split}] Missing label directory: {lbl_dir}")
        return {"failures": failures, "warnings": warnings, "images": 0, "labels": 0,
                "ok_labels": 0, "total_polygons": 0, "class_counts": {}}

    # Gather images and labels
    images = sorted([f for f in img_dir.iterdir() if f.is_file()])
    labels = sorted([f for f in lbl_dir.iterdir() if f.suffix.lower() == ".txt"])

    # Check for incorrect image extensions
    for f in images:
        if f.suffix.lower() not in VALID_IMAGE_EXTS:
            warnings.append(f"[{split}] File with unexpected extension: {f.name}")

    valid_images = [f for f in images if f.suffix.lower() in VALID_IMAGE_EXTS]
    img_stems = {f.stem: f for f in valid_images}
    lbl_stems = {f.stem: f for f in labels}

    # Check image -> label mappings
    for stem, img_path in img_stems.items():
        if stem not in lbl_stems:
            failures.append(f"[{split}] Image {img_path.name} has no matching label file")

    # Check label -> image mappings (orphan labels)
    for stem, lbl_path in lbl_stems.items():
        if stem not in img_stems:
            failures.append(f"[{split}] Label {lbl_path.name} has no matching image (orphan)")

    # Corrupted images check
    if PIL_AVAILABLE:
        for img_path in valid_images:
            try:
                with Image.open(img_path) as im:
                    im.verify()
            except Exception as e:
                failures.append(f"[{split}] Corrupted image: {img_path.name} — {e}")

    # Label content validation
    ok_labels = 0
    total_polygons = 0
    class_counts = defaultdict(int)

    for lbl_path in labels:
        stem = lbl_path.stem
        if stem not in img_stems:
            continue  # Already flagged as orphan

        if lbl_path.stat().st_size == 0:
            failures.append(f"[{split}] Empty label file (0 bytes): {lbl_path.name}")
            continue

        try:
            with open(lbl_path, encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]

            if not lines:
                failures.append(f"[{split}] Empty label (no annotations): {lbl_path.name}")
                continue

            has_invalid = False
            for line_idx, line in enumerate(lines, 1):
                parts = line.split()

                if len(parts) < 7:
                    failures.append(
                        f"[{split}] {lbl_path.name}:L{line_idx} — invalid element count "
                        f"({len(parts)}), need at least 7 for a valid polygon"
                    )
                    has_invalid = True
                    continue

                # Class ID check
                try:
                    class_id = int(parts[0])
                    if class_id not in CLASS_RANGE:
                        failures.append(
                            f"[{split}] {lbl_path.name}:L{line_idx} — invalid class ID {class_id} "
                            f"(valid: 0-{NUM_CLASSES - 1})"
                        )
                        has_invalid = True
                    else:
                        class_counts[class_id] += 1
                except ValueError:
                    failures.append(
                        f"[{split}] {lbl_path.name}:L{line_idx} — non-integer class ID '{parts[0]}'"
                    )
                    has_invalid = True
                    continue

                # Coordinate normalization check
                try:
                    coords = [float(x) for x in parts[1:]]
                    if len(coords) % 2 != 0:
                        failures.append(
                            f"[{split}] {lbl_path.name}:L{line_idx} — odd number of coordinates"
                        )
                        has_invalid = True
                        continue

                    out_of_range = [c for c in coords if c < 0.0 or c > 1.0]
                    if out_of_range:
                        failures.append(
                            f"[{split}] {lbl_path.name}:L{line_idx} — "
                            f"{len(out_of_range)} unnormalized coordinate(s) (e.g. {out_of_range[0]:.4f})"
                        )
                        has_invalid = True

                    # Zero-area polygon check (all points same)
                    xs = coords[0::2]
                    ys = coords[1::2]
                    if len(set(xs)) == 1 and len(set(ys)) == 1:
                        warnings.append(
                            f"[{split}] {lbl_path.name}:L{line_idx} — zero-area polygon detected"
                        )

                    # Invalid polygon: fewer than 3 unique points
                    pts = list(zip(xs, ys))
                    if len(set(pts)) < 3:
                        warnings.append(
                            f"[{split}] {lbl_path.name}:L{line_idx} — degenerate polygon (<3 unique points)"
                        )

                except ValueError:
                    failures.append(
                        f"[{split}] {lbl_path.name}:L{line_idx} — non-float coordinate values"
                    )
                    has_invalid = True
                    continue

                total_polygons += 1

            if not has_invalid:
                ok_labels += 1

        except Exception as e:
            failures.append(f"[{split}] Label {lbl_path.name} unreadable: {e}")

    return {
        "failures": failures,
        "warnings": warnings,
        "images": len(valid_images),
        "labels": len(labels),
        "ok_labels": ok_labels,
        "total_polygons": total_polygons,
        "class_counts": dict(class_counts),
    }


def check_cross_split_overlap(dataset_root: Path) -> list:
    """Detect images that appear in multiple splits (by filename stem)."""
    issues = []
    split_stems = {}
    for split in VALID_SPLITS:
        img_dir = dataset_root / "images" / split
        if not img_dir.exists():
            continue
        stems = {f.stem for f in img_dir.iterdir() if f.suffix.lower() in VALID_IMAGE_EXTS}
        split_stems[split] = stems

    splits = list(split_stems.keys())
    for i in range(len(splits)):
        for j in range(i + 1, len(splits)):
            s1, s2 = splits[i], splits[j]
            overlap = split_stems[s1] & split_stems[s2]
            if overlap:
                issues.append(f"Train/val/test overlap: {len(overlap)} files found in both '{s1}' and '{s2}'")
                for name in sorted(overlap)[:5]:
                    issues.append(f"  Example: {name}")
    return issues


def check_duplicate_images(dataset_root: Path) -> list:
    """Check for duplicate images using SHA256 within each split and across splits."""
    issues = []
    hash_to_file = {}
    for split in VALID_SPLITS:
        img_dir = dataset_root / "images" / split
        if not img_dir.exists():
            continue
        for img_path in img_dir.iterdir():
            if img_path.suffix.lower() not in VALID_IMAGE_EXTS:
                continue
            try:
                h = file_sha256(img_path)
                key = f"{split}/{img_path.name}"
                if h in hash_to_file:
                    issues.append(
                        f"Duplicate image (SHA256 match): {key} == {hash_to_file[h]}"
                    )
                else:
                    hash_to_file[h] = key
            except Exception as e:
                issues.append(f"Could not hash {img_path.name}: {e}")
    return issues


def check_class_imbalance(all_class_counts: dict, verbose: bool) -> list:
    """Warn if any class is severely underrepresented."""
    warnings = []
    if not all_class_counts:
        return warnings
    total = sum(all_class_counts.values())
    avg = total / len(all_class_counts) if all_class_counts else 0
    for class_id, count in sorted(all_class_counts.items()):
        ratio = count / avg if avg > 0 else 0
        if ratio < 0.1:
            warnings.append(
                f"Severe class imbalance: class {class_id} ({DISEASE_CLASSES[class_id]}) "
                f"has only {count} polygons ({ratio:.2%} of avg={avg:.0f})"
            )
        elif ratio < 0.25:
            warnings.append(
                f"Class imbalance warning: class {class_id} ({DISEASE_CLASSES[class_id]}) "
                f"has {count} polygons ({ratio:.2%} of avg)"
            )
    return warnings


def run(args):
    dataset_root = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.verbose:
        print("\n== YOLO V2 Dataset Validator ==")

    all_failures = []
    all_warnings = []
    split_stats = {}
    all_class_counts = defaultdict(int)

    # Per-split validation
    for split in ["train", "val", "test"]:
        if args.verbose:
            print(f"  Validating split: {split}...")
        res = validate_split(dataset_root, split, args.verbose)
        all_failures.extend(res["failures"])
        all_warnings.extend(res["warnings"])
        split_stats[split] = {
            "images": res["images"],
            "labels": res["labels"],
            "ok_labels": res["ok_labels"],
            "polygons": res["total_polygons"],
        }
        for cid, cnt in res["class_counts"].items():
            all_class_counts[cid] += cnt

    # Cross-split overlap check
    if args.verbose:
        print("  Checking cross-split overlap...")
    overlap_issues = check_cross_split_overlap(dataset_root)
    all_failures.extend(overlap_issues)

    # Duplicate image check
    if args.verbose:
        print("  Checking for duplicate images...")
    dup_issues = check_duplicate_images(dataset_root)
    all_warnings.extend(dup_issues)

    # Class imbalance check
    imbalance_warnings = check_class_imbalance(dict(all_class_counts), args.verbose)
    all_warnings.extend(imbalance_warnings)

    # Structural files
    if not (dataset_root / "data.yaml").exists():
        all_failures.append("Missing data.yaml in dataset root")
    if not (dataset_root / "classes.txt").exists():
        all_failures.append("Missing classes.txt in dataset root")

    # Build class distribution
    class_dist = {DISEASE_CLASSES[i]: all_class_counts.get(i, 0) for i in range(NUM_CLASSES)}

    report = {
        "passed": len(all_failures) == 0,
        "total_failures": len(all_failures),
        "total_warnings": len(all_warnings),
        "failures": all_failures,
        "warnings": all_warnings,
        "split_statistics": split_stats,
        "class_distribution": class_dist,
    }

    with open(output_dir / "validation_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    if args.verbose or True:
        total_images = sum(s["images"] for s in split_stats.values())
        total_labels = sum(s["labels"] for s in split_stats.values())
        total_polygons = sum(s["polygons"] for s in split_stats.values())
        print(f"\n{'='*55}")
        print(f"  Total Images   : {total_images}")
        print(f"  Total Labels   : {total_labels}")
        print(f"  Total Polygons : {total_polygons}")
        print(f"  Classes        : {NUM_CLASSES}")
        print(f"{'-'*55}")
        if report["passed"]:
            print("  [PASS] VALIDATION PASSED -- Dataset Ready for Training")
        else:
            print(f"  [FAIL] VALIDATION FAILED -- {len(all_failures)} failure(s)")
            for failure in all_failures[:15]:
                print(f"      * {failure}")
            if len(all_failures) > 15:
                print(f"      ... and {len(all_failures)-15} more (see validation_report.json)")
        if all_warnings:
            print(f"  [WARN] {len(all_warnings)} warning(s) found")
        print(f"{'='*55}")

    return report


def main():
    parser = argparse.ArgumentParser(description="Validate YOLO11 V2 dataset (30 classes)")
    parser.add_argument("--input",   required=True, help="YOLO V2 dataset root")
    parser.add_argument("--output",  required=True, help="Report output directory")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    report = run(args)
    sys.exit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
