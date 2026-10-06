#!/usr/bin/env python3
"""
validate_yolo_dataset.py
------------------------
Strict validation of YOLO11 Segmentation format dataset:
  ✓ Every image has a TXT label
  ✓ Every TXT label has an image
  ✓ Class IDs valid (0-13)
  ✓ Polygon coordinates normalized (0.0 to 1.0)
  ✓ No empty annotations
  ✓ No corrupted labels
  ✓ Split integrity preserved
  ✓ Image count equals label count

Usage:
    python scripts/validate_yolo_dataset.py \
        --input datasets/plantseg_tcg_yolo_v1 \
        --output datasets/plantseg_tcg_yolo_v1/reports \
        --verbose
"""

import argparse
import json
import sys
from pathlib import Path

VALID_SPLITS = {"train", "val", "test"}
CLASS_RANGE = set(range(14))


def validate_split_integrity(dataset_root: Path, split: str, verbose: bool) -> dict:
    """Validate image-label pairs and polygon coordinates for one split."""
    img_dir = dataset_root / "images" / split
    lbl_dir = dataset_root / "labels" / split

    failures = []
    warnings = []

    if not img_dir.exists():
        failures.append(f"Missing image split directory: {img_dir}")
        return {"failures": failures, "warnings": warnings, "images": 0, "labels": 0}
    if not lbl_dir.exists():
        failures.append(f"Missing label split directory: {lbl_dir}")
        return {"failures": failures, "warnings": warnings, "images": 0, "labels": 0}

    # Gather images and labels
    images = sorted([f for f in img_dir.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    labels = sorted([f for f in lbl_dir.iterdir() if f.suffix.lower() == ".txt"])

    img_stems = {f.stem: f for f in images}
    lbl_stems = {f.stem: f for f in labels}

    # 1. Check mapping: Image -> Label
    for stem, img_path in img_stems.items():
        if stem not in lbl_stems:
            failures.append(f"[{split}] Image {img_path.name} has no matching TXT label file")

    # 2. Check mapping: Label -> Image
    for stem, lbl_path in lbl_stems.items():
        if stem not in img_stems:
            failures.append(f"[{split}] Label {lbl_path.name} has no matching image file")

    # 3. Content checking for each label
    ok_labels = 0
    total_polygons = 0

    for lbl_path in labels:
        stem = lbl_path.stem
        if stem not in img_stems:
            continue  # Already flagged as orphan label

        # Check size / empty file
        if lbl_path.stat().st_size == 0:
            failures.append(f"[{split}] Label file {lbl_path.name} is empty (0 bytes)")
            continue

        try:
            with open(lbl_path, encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]
            
            if not lines:
                failures.append(f"[{split}] Label file {lbl_path.name} has no valid non-empty annotation lines")
                continue

            has_invalid_line = False
            for line_idx, line in enumerate(lines, 1):
                parts = line.split()
                if not parts:
                    continue

                # Format check: class_id, x1, y1, x2, y2, x3, y3... (Must have at least 1 class ID + 6 coordinates = 7 elements)
                if len(parts) < 7:
                    failures.append(f"[{split}] {lbl_path.name}: Line {line_idx} has invalid element count ({len(parts)}), not a valid polygon")
                    has_invalid_line = True
                    continue

                # Class ID check
                try:
                    class_id = int(parts[0])
                    if class_id not in CLASS_RANGE:
                        failures.append(f"[{split}] {lbl_path.name}: Line {line_idx} has invalid class ID {class_id} (must be 0-13)")
                        has_invalid_line = True
                except ValueError:
                    failures.append(f"[{split}] {lbl_path.name}: Line {line_idx} has non-integer class ID '{parts[0]}'")
                    has_invalid_line = True
                    continue

                # Coordinate normalization check
                try:
                    coords = [float(x) for x in parts[1:]]
                    for idx, c in enumerate(coords):
                        if c < 0.0 or c > 1.0:
                            failures.append(f"[{split}] {lbl_path.name}: Line {line_idx} coordinate {idx} has unnormalized value {c}")
                            has_invalid_line = True
                except ValueError:
                    failures.append(f"[{split}] {lbl_path.name}: Line {line_idx} contains non-float coordinate values")
                    has_invalid_line = True
                    continue

                total_polygons += 1

            if not has_invalid_line:
                ok_labels += 1

        except Exception as e:
            failures.append(f"[{split}] Label file {lbl_path.name} is corrupted / unreadable: {e}")

    return {
        "failures": failures,
        "warnings": warnings,
        "images": len(images),
        "labels": len(labels),
        "ok_labels": ok_labels,
        "total_polygons": total_polygons,
    }


def run(args):
    dataset_root = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.verbose:
        print("\n══ YOLO Dataset Validator ══")

    failures = []
    warnings = []
    split_stats = {}

    for split in ["train", "val", "test"]:
        res = validate_split_integrity(dataset_root, split, args.verbose)
        failures.extend(res["failures"])
        warnings.extend(res["warnings"])
        split_stats[split] = {
            "images": res["images"],
            "labels": res["labels"],
            "ok_labels": res["ok_labels"],
            "polygons": res["total_polygons"],
        }

    # Check structural files
    if not (dataset_root / "data.yaml").exists():
        failures.append("Missing data.yaml file in root")
    if not (dataset_root / "classes.txt").exists():
        failures.append("Missing classes.txt file in root")

    report = {
        "passed": len(failures) == 0,
        "total_failures": len(failures),
        "failures": failures,
        "warnings": warnings,
        "split_statistics": split_stats,
    }

    # Write report
    with open(output_dir / "validation_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    if args.verbose or True:
        print(f"\n{'═'*50}")
        if report["passed"]:
            print("  ✅  VALIDATION PASSED — Dataset Ready for Training: YES")
        else:
            print(f"  ❌  VALIDATION FAILED — {len(failures)} issue(s) detected")
            for f in failures[:15]:
                print(f"      • {f}")
            if len(failures) > 15:
                print(f"      … and {len(failures) - 15} more (see validation_report.json)")
        print(f"{'═'*50}")

    sys.exit(0 if report["passed"] else 1)


def main():
    parser = argparse.ArgumentParser(description="Validate Ultralytics YOLO11 Segmentation dataset")
    parser.add_argument("--input",   required=True, help="Path to YOLO dataset root")
    parser.add_argument("--output",  required=True, help="Path to validation report destination directory")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
