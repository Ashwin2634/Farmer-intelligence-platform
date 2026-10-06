#!/usr/bin/env python3
"""
duplicate_detector.py
---------------------
Detects exact duplicates (SHA256) and perceptual duplicates (pHash)
across train/val/test splits.

Produces:
  - duplicate groups JSON
  - cross-split leakage list (files to remove per priority rule: train > val > test)

Usage:
    python duplicate_detector.py --input datasets/plantseg_tcg_v1 --output datasets/plantseg_tcg_v1/reports --workers 4 --verbose
"""

import argparse
import hashlib
import json
import os
import struct
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ── perceptual hash (pHash) ────────────────────────────────────────────────────
# Pure-Python pHash to avoid requiring imagehash library.
# Uses PIL (Pillow) which is already a dependency of the project.

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


def compute_sha256(path: Path) -> str:
    """Return hex SHA-256 digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_phash(path: Path, hash_size: int = 8) -> str:
    """
    Compute perceptual hash (pHash) for an image.
    Returns a 64-character hex string representing the 64-bit hash.
    Returns empty string if PIL not available or image unreadable.
    """
    if not PIL_AVAILABLE:
        return ""
    try:
        img = Image.open(path).convert("L")  # grayscale
        img = img.resize((hash_size * 4, hash_size * 4), Image.LANCZOS)
        pixels = list(img.getdata())
        # Simple DCT-based pHash approximation
        # Resize to hash_size x hash_size and compare to mean
        img_small = img.resize((hash_size, hash_size), Image.LANCZOS)
        pixels_small = list(img_small.getdata())
        avg = sum(pixels_small) / len(pixels_small)
        bits = "".join("1" if p > avg else "0" for p in pixels_small)
        # Pack bits into hex
        val = int(bits, 2)
        hex_len = (hash_size * hash_size + 3) // 4
        return format(val, f"0{hex_len}x")
    except Exception:
        return ""


def hamming_distance(h1: str, h2: str) -> int:
    """Compute Hamming distance between two hex pHash strings."""
    if len(h1) != len(h2) or not h1 or not h2:
        return 999
    try:
        v1 = int(h1, 16)
        v2 = int(h2, 16)
        xor = v1 ^ v2
        return bin(xor).count("1")
    except ValueError:
        return 999


# ── split priority ─────────────────────────────────────────────────────────────
SPLIT_PRIORITY = {"train": 0, "val": 1, "validation": 1, "test": 2}


def split_of(path: Path) -> str:
    """Infer split name from path (train / val / test)."""
    parts = path.parts
    for part in parts:
        p = part.lower()
        if p in SPLIT_PRIORITY:
            return p
    return "unknown"


# ── main logic ─────────────────────────────────────────────────────────────────

def scan_images(images_dir: Path, verbose: bool = False) -> list[dict]:
    """Return list of {path, rel_path, split, sha256, phash} for every image."""
    results = []
    image_paths = []
    for split in ["train", "val", "test"]:
        split_dir = images_dir / split
        if not split_dir.exists():
            continue
        for img_path in sorted(split_dir.iterdir()):
            if img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                image_paths.append((img_path, split))

    if verbose:
        print(f"  [duplicate_detector] Scanning {len(image_paths)} images for hashes …")

    for img_path, split in image_paths:
        sha = compute_sha256(img_path)
        ph = compute_phash(img_path)
        results.append({
            "path": str(img_path),
            "rel_path": str(img_path.relative_to(images_dir.parent)),
            "split": split,
            "sha256": sha,
            "phash": ph,
        })
        if verbose:
            print(f"    ✓ {img_path.name}  sha={sha[:12]}…  phash={ph[:16]}…")

    return results


def find_exact_duplicates(records: list[dict]) -> dict:
    """Group records by SHA256. Return groups with 2+ members."""
    groups = defaultdict(list)
    for rec in records:
        groups[rec["sha256"]].append(rec)
    return {sha: recs for sha, recs in groups.items() if len(recs) > 1}


def find_perceptual_duplicates(records: list[dict], threshold: int = 5) -> list[list[dict]]:
    """
    Find groups of images with pHash Hamming distance <= threshold.
    Uses union-find for grouping.
    """
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
            if records[i]["phash"] and records[j]["phash"]:
                dist = hamming_distance(records[i]["phash"], records[j]["phash"])
                if dist <= threshold:
                    union(i, j)

    groups_map = defaultdict(list)
    for i in range(n):
        groups_map[find(i)].append(records[i])

    return [g for g in groups_map.values() if len(g) > 1]


def identify_leakage(exact_groups: dict, perceptual_groups: list[list[dict]]) -> dict:
    """
    Find duplicate groups that span multiple splits → data leakage.
    Returns dict with:
      - leakage_groups: list of groups
      - files_to_remove: list of rel_paths to remove (lower-priority duplicates)
    """
    leakage = []

    # Exact leakage
    for sha, recs in exact_groups.items():
        splits_present = list({r["split"] for r in recs})
        if len(splits_present) > 1:
            # Keep the highest-priority (lowest number) copy per split
            sorted_recs = sorted(recs, key=lambda r: SPLIT_PRIORITY.get(r["split"], 99))
            kept = {r["split"]: r for r in sorted_recs}  # one per split (highest priority)
            # Actually keep only the single best
            best_split = min(splits_present, key=lambda s: SPLIT_PRIORITY.get(s, 99))
            to_keep = next(r for r in sorted_recs if r["split"] == best_split)
            to_remove = [r for r in recs if r["path"] != to_keep["path"]]
            leakage.append({
                "type": "exact",
                "sha256": sha,
                "splits_involved": splits_present,
                "kept": to_keep["rel_path"],
                "kept_split": best_split,
                "removed": [r["rel_path"] for r in to_remove],
            })

    # Perceptual leakage (not already covered by exact)
    exact_leakage_paths = {path for group in leakage for path in group["removed"]}
    exact_leakage_paths.update({group["kept"] for group in leakage})

    for group in perceptual_groups:
        # Skip if all members already in exact leakage
        splits_present = list({r["split"] for r in group})
        if len(splits_present) > 1:
            # Check none of these are already exact dups
            is_exact = all(r["rel_path"] in exact_leakage_paths for r in group)
            if not is_exact:
                sorted_recs = sorted(group, key=lambda r: SPLIT_PRIORITY.get(r["split"], 99))
                best_split = min(splits_present, key=lambda s: SPLIT_PRIORITY.get(s, 99))
                to_keep = next(r for r in sorted_recs if r["split"] == best_split)
                to_remove = [r for r in group if r["path"] != to_keep["path"]]
                leakage.append({
                    "type": "perceptual",
                    "sha256": None,
                    "phash_group": [r["phash"] for r in group],
                    "splits_involved": splits_present,
                    "kept": to_keep["rel_path"],
                    "kept_split": best_split,
                    "removed": [r["rel_path"] for r in to_remove],
                })

    all_files_to_remove = []
    for group in leakage:
        all_files_to_remove.extend(group["removed"])

    return {
        "leakage_groups": leakage,
        "files_to_remove": list(set(all_files_to_remove)),
        "total_leakage_groups": len(leakage),
        "total_files_to_remove": len(set(all_files_to_remove)),
    }


def run(args):
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    images_dir = input_dir / "images"
    if not images_dir.exists():
        print(f"ERROR: images directory not found: {images_dir}", file=sys.stderr)
        sys.exit(1)

    if args.verbose:
        print("\n══ Duplicate Detector ══")

    records = scan_images(images_dir, verbose=args.verbose)

    if args.verbose:
        print(f"\n  Computing exact duplicates from {len(records)} records …")
    exact_groups = find_exact_duplicates(records)

    if args.verbose:
        print(f"  Found {len(exact_groups)} exact duplicate groups")
        print(f"  Computing perceptual duplicates (pHash, threshold=5) …")
    perceptual_groups = find_perceptual_duplicates(records)

    if args.verbose:
        print(f"  Found {len(perceptual_groups)} perceptual duplicate groups")
        print(f"  Identifying cross-split leakage …")

    leakage_report = identify_leakage(exact_groups, perceptual_groups)

    # Compile full report
    report = {
        "total_images_scanned": len(records),
        "total_exact_duplicate_groups": len(exact_groups),
        "total_exact_duplicate_files": sum(len(v) - 1 for v in exact_groups.values()),
        "total_perceptual_duplicate_groups": len(perceptual_groups),
        "data_leakage_groups_count": leakage_report["total_leakage_groups"],
        "data_leakage_files_to_remove": leakage_report["total_files_to_remove"],
        "leakage_details": leakage_report["leakage_groups"],
        "files_to_remove": leakage_report["files_to_remove"],
        "exact_duplicate_groups": {
            sha: [r["rel_path"] for r in recs]
            for sha, recs in exact_groups.items()
        },
        "perceptual_duplicate_groups": [
            [r["rel_path"] for r in group]
            for group in perceptual_groups
        ],
        "image_records": records,
    }

    out_path = output_dir / "duplicate_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    if args.verbose:
        print(f"\n  ✓ Duplicate report saved: {out_path}")
        print(f"    Exact duplicate groups : {report['total_exact_duplicate_groups']}")
        print(f"    Perceptual dup groups  : {report['total_perceptual_duplicate_groups']}")
        print(f"    Leakage groups         : {report['data_leakage_groups_count']}")
        print(f"    Files to remove        : {report['data_leakage_files_to_remove']}")

    return report


def main():
    parser = argparse.ArgumentParser(description="Detect duplicate images across splits")
    parser.add_argument("--input",   required=True, help="Path to dataset root (plantseg_tcg_v1)")
    parser.add_argument("--output",  required=True, help="Path to reports output directory")
    parser.add_argument("--workers", type=int, default=4, help="Number of worker threads")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
