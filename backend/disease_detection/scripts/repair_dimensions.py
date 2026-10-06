#!/usr/bin/env python3
"""
repair_dimensions.py
--------------------
Validates image-mask dimension matching.
- Detects dimension mismatches
- Auto-repairs rotated pairs (transposes mask if W/H are swapped)
- Quarantines unresolvable mismatches
- Generates a repair report

Usage:
    python repair_dimensions.py --input datasets/plantseg_tcg_v1 --output datasets/plantseg_tcg_v1 --workers 4 --verbose
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("ERROR: Pillow is required. Install with: pip install Pillow", file=sys.stderr)
    sys.exit(1)


def get_image_size(path: Path):
    """Return (width, height) or None if unreadable."""
    try:
        with Image.open(path) as img:
            return img.size  # (width, height)
    except Exception as e:
        return None


def repair_or_quarantine(
    img_path: Path,
    mask_path: Path,
    quarantine_images: Path,
    quarantine_annotations: Path,
    split: str,
    verbose: bool = False,
) -> dict:
    """
    Check a single image/mask pair.
    Returns a result dict describing the outcome.
    """
    result = {
        "image": str(img_path.name),
        "split": split,
        "status": "ok",
        "action": "none",
        "image_size": None,
        "mask_size": None,
        "notes": "",
    }

    # Check image readability
    img_size = get_image_size(img_path)
    if img_size is None:
        result["status"] = "error"
        result["action"] = "quarantined"
        result["notes"] = "Image unreadable"
        _move_to_quarantine(img_path, mask_path, quarantine_images, quarantine_annotations, split)
        return result

    # Check mask readability
    if not mask_path.exists():
        result["status"] = "error"
        result["action"] = "quarantined"
        result["notes"] = "Mask missing"
        _move_to_quarantine(img_path, None, quarantine_images, quarantine_annotations, split)
        return result

    mask_size = get_image_size(mask_path)
    if mask_size is None:
        result["status"] = "error"
        result["action"] = "quarantined"
        result["notes"] = "Mask unreadable"
        _move_to_quarantine(img_path, mask_path, quarantine_images, quarantine_annotations, split)
        return result

    result["image_size"] = list(img_size)
    result["mask_size"] = list(mask_size)

    if img_size == mask_size:
        result["status"] = "ok"
        result["action"] = "none"
        return result

    img_w, img_h = img_size
    mask_w, mask_h = mask_size

    # Check if mask is just rotated 90° (W/H swapped)
    if img_w == mask_h and img_h == mask_w:
        # Auto-repair: rotate mask 90° counterclockwise
        try:
            with Image.open(mask_path) as mask_img:
                # Determine correct rotation direction
                # Rotate 90 CW (ROTATE_270) or CCW (ROTATE_90) to match image
                rotated = mask_img.rotate(90, expand=True)
                if rotated.size == img_size:
                    rotated.save(mask_path)
                    result["status"] = "repaired"
                    result["action"] = "mask_rotated_90ccw"
                    result["notes"] = f"Mask rotated from {mask_size} to {rotated.size}"
                    if verbose:
                        print(f"    ✓ REPAIRED {img_path.name}: mask rotated 90° CCW")
                    return result
                # Try 90 CW
                rotated = mask_img.rotate(-90, expand=True)
                if rotated.size == img_size:
                    rotated.save(mask_path)
                    result["status"] = "repaired"
                    result["action"] = "mask_rotated_90cw"
                    result["notes"] = f"Mask rotated from {mask_size} to {rotated.size}"
                    if verbose:
                        print(f"    ✓ REPAIRED {img_path.name}: mask rotated 90° CW")
                    return result
        except Exception as e:
            result["notes"] = f"Rotation repair failed: {e}"

    # Cannot auto-repair → quarantine
    result["status"] = "mismatch"
    result["action"] = "quarantined"
    result["notes"] = f"Dimensions mismatch: image={img_size} mask={mask_size}, cannot auto-repair"
    if verbose:
        print(f"    ⚠ QUARANTINED {img_path.name}: {result['notes']}")
    _move_to_quarantine(img_path, mask_path, quarantine_images, quarantine_annotations, split)
    return result


def _move_to_quarantine(img_path, mask_path, q_images, q_annotations, split):
    """Move image and/or mask to quarantine directories."""
    q_img_split = q_images / split
    q_ann_split = q_annotations / split
    q_img_split.mkdir(parents=True, exist_ok=True)
    q_ann_split.mkdir(parents=True, exist_ok=True)

    if img_path and img_path.exists():
        shutil.move(str(img_path), str(q_img_split / img_path.name))

    if mask_path and mask_path.exists():
        shutil.move(str(mask_path), str(q_ann_split / mask_path.name))


def run(args):
    input_dir = Path(args.input)
    output_dir = Path(args.output)

    images_dir = input_dir / "images"
    annotations_dir = input_dir / "annotations"
    quarantine_dir = output_dir / "quarantine"
    quarantine_images = quarantine_dir / "images"
    quarantine_annotations = quarantine_dir / "annotations"
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    if args.verbose:
        print("\n══ Dimension Repair ══")

    results = []
    total = 0
    repaired = 0
    quarantined = 0
    mismatched = 0

    for split in ["train", "val", "test"]:
        img_split_dir = images_dir / split
        ann_split_dir = annotations_dir / split

        if not img_split_dir.exists():
            continue

        img_files = sorted([
            p for p in img_split_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ])

        if args.verbose:
            print(f"\n  Checking {len(img_files)} images in [{split}] …")

        for img_path in img_files:
            # Find corresponding mask (same stem, .png)
            mask_path = ann_split_dir / (img_path.stem + ".png")

            total += 1
            res = repair_or_quarantine(
                img_path, mask_path,
                quarantine_images, quarantine_annotations,
                split, verbose=args.verbose
            )
            results.append(res)

            if res["action"] in ("mask_rotated_90cw", "mask_rotated_90ccw"):
                repaired += 1
            elif res["action"] == "quarantined":
                quarantined += 1
            if res["status"] == "mismatch":
                mismatched += 1

    # Generate report
    report = {
        "total_pairs_checked": total,
        "ok": total - repaired - quarantined,
        "repaired": repaired,
        "quarantined": quarantined,
        "dimension_mismatches_found": mismatched + repaired,
        "details": results,
    }

    report_path = reports_dir / "dimension_repair_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    if args.verbose:
        print(f"\n  ✓ Dimension repair complete")
        print(f"    Total pairs checked : {total}")
        print(f"    OK                  : {report['ok']}")
        print(f"    Auto-repaired       : {repaired}")
        print(f"    Quarantined         : {quarantined}")
        print(f"    Report saved        : {report_path}")

    return report


def main():
    parser = argparse.ArgumentParser(description="Repair image-mask dimension mismatches")
    parser.add_argument("--input",   required=True, help="Path to dataset root")
    parser.add_argument("--output",  required=True, help="Path to output root (quarantine goes here)")
    parser.add_argument("--workers", type=int, default=4, help="Worker threads (reserved)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
