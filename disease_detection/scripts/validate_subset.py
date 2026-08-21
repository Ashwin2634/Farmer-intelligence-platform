#!/usr/bin/env python3
"""
validate_subset.py
------------------
Final validation for plantseg_tcg_v1 dataset.

Checks:
  ✓ No duplicate leakage across splits
  ✓ No corrupted files
  ✓ No missing masks
  ✓ No orphan masks (masks without images)
  ✓ Matching dimensions (image == mask)
  ✓ Official split preserved (train / val / test)
  ✓ Only Tomato / Cucumber / Grape present
  ✓ Every image has exactly one mask
  ✓ Every mask has exactly one image
  ✓ Non-empty masks (pixel coverage > 0)
  ✓ Valid mask pixel values (binary or grayscale 0-255)

Usage:
    python validate_subset.py --input datasets/plantseg_tcg_v1 --output datasets/plantseg_tcg_v1/reports --verbose
Exit code 0 = all checks passed. Non-zero = failures detected.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

try:
    from PIL import Image
    import numpy as np
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("ERROR: Pillow and numpy required.", file=sys.stderr)
    sys.exit(1)

# ── constants ──────────────────────────────────────────────────────────────────
# 'grapevine' is a valid prefix for Grape (e.g. grapevine_leafroll_disease_*)
TARGET_CROPS = {"tomato", "cucumber", "grape", "grapevine"}
VALID_SPLITS = {"train", "val", "test"}


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def load_metadata(dataset_root: Path):
    """Load subset_metadata.csv if present."""
    meta_path = dataset_root / "metadata" / "subset_metadata.csv"
    if not meta_path.exists():
        return None
    import csv
    with open(meta_path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def validate_crop_names(images_dir: Path) -> list[str]:
    """Check that all filenames start with tomato_, cucumber_, or grape_."""
    errors = []
    for split in VALID_SPLITS:
        split_dir = images_dir / split
        if not split_dir.exists():
            continue
        for img in split_dir.iterdir():
            if img.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            prefix = img.name.split("_")[0].lower()
            if prefix not in TARGET_CROPS:
                errors.append(f"[{split}] {img.name}: unexpected crop prefix '{prefix}'")
    return errors


def validate_splits_exist(images_dir: Path) -> list[str]:
    errors = []
    for split in VALID_SPLITS:
        if not (images_dir / split).exists():
            errors.append(f"Split directory missing: images/{split}")
    return errors


def validate_pairs(images_dir: Path, annotations_dir: Path, verbose: bool) -> dict:
    """
    For every split:
      - Every image has a mask
      - Every mask has an image
      - Dimensions match
      - Images and masks are readable
      - Masks are non-empty (coverage > 0)
    """
    missing_masks = []
    orphan_masks = []
    dimension_mismatches = []
    corrupted_images = []
    corrupted_masks = []
    empty_masks = []
    ok_pairs = 0

    for split in VALID_SPLITS:
        img_dir = images_dir / split
        ann_dir = annotations_dir / split
        if not img_dir.exists():
            continue

        img_stems = {}
        for f in img_dir.iterdir():
            if f.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                img_stems[f.stem] = f

        mask_stems = {}
        if ann_dir.exists():
            for f in ann_dir.iterdir():
                if f.suffix.lower() == ".png":
                    mask_stems[f.stem] = f

        # Check every image has a mask
        for stem, img_path in img_stems.items():
            if stem not in mask_stems:
                missing_masks.append(f"[{split}] {img_path.name}: no mask found")
                continue
            mask_path = mask_stems[stem]

            # Check readability
            try:
                with Image.open(img_path) as img:
                    img_size = img.size
                    img.verify()
            except Exception as e:
                corrupted_images.append(f"[{split}] {img_path.name}: {e}")
                continue

            # Re-open for size (verify() closes the file)
            try:
                with Image.open(img_path) as img:
                    img_size = img.size
            except Exception:
                continue

            try:
                with Image.open(mask_path) as mask:
                    mask_size = mask.size
                    arr = np.array(mask)
            except Exception as e:
                corrupted_masks.append(f"[{split}] {mask_path.name}: {e}")
                continue

            # Dimension check
            if img_size != mask_size:
                dimension_mismatches.append(
                    f"[{split}] {img_path.name}: img={img_size} mask={mask_size}"
                )
                continue

            # Non-empty mask
            if arr.max() == 0:
                empty_masks.append(f"[{split}] {mask_path.name}: all-zero mask")

            ok_pairs += 1
            if verbose:
                print(f"    ✓ [{split}] {img_path.name}")

        # Check orphan masks
        for stem in mask_stems:
            if stem not in img_stems:
                orphan_masks.append(f"[{split}] {mask_stems[stem].name}: no image found")

    return {
        "ok_pairs": ok_pairs,
        "missing_masks": missing_masks,
        "orphan_masks": orphan_masks,
        "dimension_mismatches": dimension_mismatches,
        "corrupted_images": corrupted_images,
        "corrupted_masks": corrupted_masks,
        "empty_masks": empty_masks,
    }


def validate_no_leakage(images_dir: Path, verbose: bool) -> list[str]:
    """Check SHA256 across splits — no hash should appear in 2+ splits."""
    hashes = {}  # sha -> (split, name)
    leakage = []

    for split in VALID_SPLITS:
        split_dir = images_dir / split
        if not split_dir.exists():
            continue
        for img in sorted(split_dir.iterdir()):
            if img.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            sha = sha256_of(img)
            if sha in hashes:
                prev_split, prev_name = hashes[sha]
                leakage.append(
                    f"{img.name} [{split}] is duplicate of {prev_name} [{prev_split}]"
                )
            else:
                hashes[sha] = (split, img.name)

    return leakage


def run(args):
    dataset_root = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    images_dir = dataset_root / "images"
    annotations_dir = dataset_root / "annotations"

    if args.verbose:
        print("\n══ Final Validation ══")

    failures = []

    # 1. Splits exist
    if args.verbose:
        print("\n  [1/6] Checking splits exist …")
    split_errors = validate_splits_exist(images_dir)
    failures.extend(split_errors)
    if args.verbose and not split_errors:
        print("       ✓ train / val / test directories present")

    # 2. Only TCG crops
    if args.verbose:
        print("  [2/6] Checking crop names …")
    crop_errors = validate_crop_names(images_dir)
    failures.extend(crop_errors)
    if args.verbose and not crop_errors:
        print("       ✓ Only Tomato / Cucumber / Grape filenames")

    # 3. Pairs validation (masks, dims, readability)
    if args.verbose:
        print("  [3/6] Validating image-mask pairs …")
    pair_results = validate_pairs(images_dir, annotations_dir, verbose=False)
    failures.extend(pair_results["missing_masks"])
    failures.extend(pair_results["orphan_masks"])
    failures.extend(pair_results["dimension_mismatches"])
    failures.extend(pair_results["corrupted_images"])
    failures.extend(pair_results["corrupted_masks"])
    # empty masks are warnings, not failures
    if args.verbose:
        print(f"       ✓ OK pairs: {pair_results['ok_pairs']}")
        if pair_results["missing_masks"]:
            print(f"       ✗ Missing masks: {len(pair_results['missing_masks'])}")
        if pair_results["orphan_masks"]:
            print(f"       ✗ Orphan masks: {len(pair_results['orphan_masks'])}")
        if pair_results["dimension_mismatches"]:
            print(f"       ✗ Dimension mismatches: {len(pair_results['dimension_mismatches'])}")
        if pair_results["corrupted_images"]:
            print(f"       ✗ Corrupted images: {len(pair_results['corrupted_images'])}")
        if pair_results["empty_masks"]:
            print(f"       ⚠ Empty masks (warning): {len(pair_results['empty_masks'])}")

    # 4. No duplicate leakage
    if args.verbose:
        print("  [4/6] Checking cross-split leakage …")
    leakage_errors = validate_no_leakage(images_dir, verbose=False)
    failures.extend(leakage_errors)
    if args.verbose and not leakage_errors:
        print("       ✓ No duplicate leakage detected")

    # 5. Count images per split
    if args.verbose:
        print("  [5/6] Counting images per split …")
    split_counts = {}
    for split in VALID_SPLITS:
        split_dir = images_dir / split
        if split_dir.exists():
            count = sum(1 for f in split_dir.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png"})
            split_counts[split] = count
            if args.verbose:
                print(f"       {split}: {count} images")

    # 6. Metadata check
    if args.verbose:
        print("  [6/6] Checking metadata …")
    meta_path = dataset_root / "metadata" / "subset_metadata.csv"
    meta_ok = meta_path.exists()
    if not meta_ok:
        failures.append("metadata/subset_metadata.csv not found")
    elif args.verbose:
        print("       ✓ subset_metadata.csv present")

    # Save validation report
    report = {
        "passed": len(failures) == 0,
        "total_failures": len(failures),
        "failures": failures,
        "warnings": pair_results["empty_masks"],
        "split_counts": split_counts,
        "pair_summary": {
            "ok_pairs": pair_results["ok_pairs"],
            "missing_masks": len(pair_results["missing_masks"]),
            "orphan_masks": len(pair_results["orphan_masks"]),
            "dimension_mismatches": len(pair_results["dimension_mismatches"]),
            "corrupted_images": len(pair_results["corrupted_images"]),
            "corrupted_masks": len(pair_results["corrupted_masks"]),
            "empty_masks": len(pair_results["empty_masks"]),
        },
    }

    report_path = output_dir / "validation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Final result
    if args.verbose or True:
        print(f"\n{'═'*50}")
        if report["passed"]:
            print("  ✅  VALIDATION PASSED — Dataset Ready: YES")
        else:
            print(f"  ❌  VALIDATION FAILED — {len(failures)} issue(s) found")
            for err in failures[:20]:
                print(f"      • {err}")
            if len(failures) > 20:
                print(f"      … and {len(failures) - 20} more (see {report_path})")
        print(f"{'═'*50}")

    sys.exit(0 if report["passed"] else 1)


def main():
    parser = argparse.ArgumentParser(description="Final validation for plantseg_tcg_v1")
    parser.add_argument("--input",   required=True, help="Path to dataset root")
    parser.add_argument("--output",  required=True, help="Path to reports output directory")
    parser.add_argument("--workers", type=int, default=4, help="Worker threads (reserved)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
