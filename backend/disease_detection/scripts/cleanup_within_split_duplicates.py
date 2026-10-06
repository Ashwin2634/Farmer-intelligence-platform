#!/usr/bin/env python3
"""
cleanup_within_split_duplicates.py
-----------------------------------
Removes within-split exact duplicates (same SHA256 hash inside the same split).
For same-split duplicates, keeps the first alphabetically, removes the rest.
Updates duplicate_report.json and removed_duplicates.csv.

These represent mislabeled or cross-disease duplicate images in the original dataset.
"""

import csv
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def run():
    dataset_root = Path("datasets/plantseg_tcg_v1")
    images_dir = dataset_root / "images"
    annotations_dir = dataset_root / "annotations"
    reports_dir = dataset_root / "reports"

    newly_removed = []

    for split in ["train", "val", "test"]:
        img_dir = images_dir / split
        ann_dir = annotations_dir / split
        if not img_dir.exists():
            continue

        img_files = sorted(
            f for f in img_dir.iterdir()
            if f.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )

        seen_hashes = {}  # sha256 -> first filename (kept)
        to_remove = []

        for img in img_files:
            sha = sha256_of(img)
            if sha in seen_hashes:
                # This is a within-split duplicate — remove it
                to_remove.append((img, sha, seen_hashes[sha]))
            else:
                seen_hashes[sha] = img.name

        for img_path, sha, kept_name in to_remove:
            mask_path = ann_dir / (img_path.stem + ".png")
            print(f"  Removing [{split}] {img_path.name} (duplicate of {kept_name})")

            if img_path.exists():
                os.remove(str(img_path))
                newly_removed.append({
                    "file": img_path.name,
                    "split": split,
                    "sha256": sha,
                    "type": "within_split_exact_duplicate",
                    "kept": kept_name,
                })

            if mask_path.exists():
                os.remove(str(mask_path))
                newly_removed.append({
                    "file": mask_path.name,
                    "split": split,
                    "sha256": sha,
                    "type": "within_split_exact_duplicate_mask",
                    "kept": kept_name,
                })

    print(f"\n  Removed {len([r for r in newly_removed if 'mask' not in r['type']])} duplicate images")

    # Update removed_duplicates.csv
    existing = []
    csv_path = reports_dir / "removed_duplicates.csv"
    if csv_path.exists():
        with open(csv_path, encoding="utf-8") as f:
            existing = list(csv.DictReader(f))

    all_removed = existing + newly_removed
    fieldnames = ["file", "split", "sha256", "type", "kept"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_removed:
            # Ensure all fields present
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    # Update duplicate_report.json
    dup_path = reports_dir / "duplicate_report.json"
    dup_data = {}
    if dup_path.exists():
        with open(dup_path) as f:
            dup_data = json.load(f)

    new_within_split_groups = len([r for r in newly_removed if "mask" not in r["type"]])
    dup_data["within_split_duplicate_groups_removed"] = new_within_split_groups
    dup_data["total_files_removed"] = dup_data.get("total_files_removed", 0) + new_within_split_groups

    with open(dup_path, "w") as f:
        json.dump(dup_data, f, indent=2)

    print(f"  Updated: {csv_path}")
    print(f"  Updated: {dup_path}")


if __name__ == "__main__":
    run()
