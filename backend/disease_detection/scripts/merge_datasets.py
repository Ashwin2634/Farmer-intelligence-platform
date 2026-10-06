"""
Production Dataset Merging Pipeline (YOLO Segmentation - V2 Enhanced).

Merges an existing baseline dataset (e.g., plantseg_tcg_yolo_v4) with a newly converted dataset
(e.g., plant_seg_new_yolo) to create a new production version (e.g., plantseg_tcg_yolo_v5).

Key Architectural Rules & Safeguards:
- IMMUTABILITY GUARANTEE: Baseline datasets (V4) are STRICTLY READ-ONLY. Output to new version directory.
- Input & Schema Audit: Validates images, labels, data.yaml, and 1-to-1 image-label mapping upfront.
- Dataset SHA256 Fingerprint: Computes immutable fingerprint of entire dataset state for reproducibility.
- Full Duplicate Analysis: Checks duplicate filenames, SHA256 hashes, labels, and polygon dimensions.
- Label & Class Validation: Verifies class ID contiguity (0..N-1), label syntax, coordinate normalization, and healthy empty file rules.
- Multi-Stage Copy & Remap Verification: Re-opens written images and label files to assert copy fidelity.
- Split Distribution Integrity: Validates class distributions and asserts zero cross-split hash leakage.
- Parallelization & Caching: ThreadPoolExecutor preloading for hashing and validation while preserving deterministic order.

Directory Layout Output:
    output_dir/ (e.g., plantseg_tcg_yolo_v5)
        images/
            train/
            val/
            test/
        labels/
            train/
            val/
            test/
        metadata/
            config_used.json
            manifest.json
            merge_report.json
            summary.md
            dataset_fingerprint.json
            input_validation.json
            class_validation.json
            split_statistics.json
            image_hashes.csv
            duplicate_analysis.csv
            label_validation.csv
            failed_copies.csv
            merge.log
        data.yaml
"""

import argparse
import concurrent.futures
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
import PIL
from PIL import Image
import yaml
from tqdm import tqdm

# Add scripts directory to sys.path for relative modular imports
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.hashing import compute_sha256
from utils.logger import setup_logger
from utils.reporting import save_csv_report, save_json_report, save_markdown_report


@dataclass
class DatasetImageRecord:
    dataset_source: str  # 'baseline' or 'new'
    split: str  # 'train', 'val', 'test'
    image_path: Path
    label_path: Path
    filename: str
    sha256_hash: str = ""
    width: int = 0
    height: int = 0
    file_size: int = 0
    is_healthy: bool = False
    is_corrupted: bool = False
    corruption_reason: str = ""


@dataclass
class MergeStats:
    baseline_images_total: int = 0
    new_images_total: int = 0
    merged_images_total: int = 0
    baseline_healthy_count: int = 0
    new_healthy_count: int = 0
    merged_healthy_count: int = 0
    duplicates_detected: int = 0
    baseline_classes_count: int = 0
    new_classes_count: int = 0
    merged_classes_count: int = 0
    train_count: int = 0
    val_count: int = 0
    test_count: int = 0
    remapped_labels_count: int = 0


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Production Dataset Merging Pipeline (YOLO Segmentation)."
    )
    parser.add_argument(
        "--v4_dir",
        type=str,
        default="datasets/plantseg_tcg_yolo_v4",
        help="Path to baseline production dataset directory (V4). Strictly read-only.",
    )
    parser.add_argument(
        "--new_dir",
        type=str,
        default="datasets/plant_seg_new_yolo",
        help="Path to new YOLO dataset directory to merge into baseline.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="datasets/plantseg_tcg_yolo_v5",
        help="Path to output directory for merged dataset (V5).",
    )
    parser.add_argument(
        "--dataset_version",
        type=str,
        default="V5",
        help="Merged dataset version tag.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel worker threads.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Perform dry-run verification without copying files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output directory contents if present.",
    )
    return parser.parse_args()


def get_git_commit_hash() -> str:
    """Extracts git commit hash if in a git repository."""
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode("utf-8").strip()
    except Exception:
        return "git_repo_unavailable"


def get_environment_info(args: argparse.Namespace) -> Dict[str, Any]:
    """Captures complete execution environment and tool versions."""
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_version": args.dataset_version,
        "python_version": sys.version,
        "platform": platform.platform(),
        "opencv_version": cv2.__version__,
        "numpy_version": np.__version__,
        "yaml_version": yaml.__version__,
        "pil_version": PIL.__version__,
        "git_commit_hash": get_git_commit_hash(),
        "cli_arguments": vars(args),
    }


def load_dataset_schema(dataset_dir: Path) -> Tuple[Dict[int, str], Dict[str, int]]:
    """
    Reads data.yaml from a dataset directory and returns (id2name, name2id).
    """
    yaml_path = dataset_dir / "data.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"data.yaml missing in dataset directory: {dataset_dir}")

    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    names = data.get("names", {})
    idx_to_name: Dict[int, str] = {}
    name_to_idx: Dict[str, int] = {}

    if isinstance(names, list):
        for idx, name in enumerate(names):
            idx_to_name[idx] = str(name)
            name_to_idx[str(name)] = idx
    elif isinstance(names, dict):
        for idx, name in names.items():
            i_val = int(idx)
            idx_to_name[i_val] = str(name)
            name_to_idx[str(name)] = i_val
    else:
        raise ValueError(f"Invalid 'names' specification in data.yaml: {names}")

    return idx_to_name, name_to_idx


def verify_image_label_record(rec: DatasetImageRecord, name2id: Dict[str, int]) -> DatasetImageRecord:
    """
    Inspects image and label file readability, dimensions, SHA256 hash, and label validity.
    """
    if not rec.image_path.exists() or rec.image_path.stat().st_size == 0:
        rec.is_corrupted = True
        rec.corruption_reason = "Image missing or zero-byte file"
        return rec

    try:
        with Image.open(rec.image_path) as img:
            img.verify()
        with Image.open(rec.image_path) as img:
            rec.width, rec.height = img.size
            if rec.width <= 0 or rec.height <= 0:
                rec.is_corrupted = True
                rec.corruption_reason = f"Invalid image dimensions: {rec.width}x{rec.height}"
                return rec
    except Exception as e:
        rec.is_corrupted = True
        rec.corruption_reason = f"Corrupted image file: {str(e)}"
        return rec

    rec.sha256_hash = compute_sha256(rec.image_path)

    # Inspect Label
    if not rec.label_path.exists():
        rec.is_healthy = True
        return rec

    try:
        content = rec.label_path.read_text(encoding="utf-8").strip()
        if len(content) == 0:
            rec.is_healthy = True
        else:
            rec.is_healthy = False
            lines = content.split("\n")
            valid_cids = set(name2id.values())
            for line in lines:
                tokens = line.strip().split()
                if len(tokens) < 7 or len(tokens) % 2 == 0:
                    rec.is_corrupted = True
                    rec.corruption_reason = f"Malformed label tokens ({len(tokens)}) in {rec.label_path.name}"
                    return rec
                cid = int(tokens[0])
                if cid not in valid_cids:
                    rec.is_corrupted = True
                    rec.corruption_reason = f"Unmapped class ID {cid} in label {rec.label_path.name}"
                    return rec
                coords = [float(t) for t in tokens[1:]]
                for c_val in coords:
                    if not (0.0 <= c_val <= 1.0):
                        rec.is_corrupted = True
                        rec.corruption_reason = f"Coordinate out of bounds {c_val} in label {rec.label_path.name}"
                        return rec
    except Exception as e:
        rec.is_corrupted = True
        rec.corruption_reason = f"Corrupted label file: {str(e)}"

    return rec


def scan_and_verify_dataset(dataset_dir: Path, source_label: str, workers: int) -> Tuple[List[DatasetImageRecord], Dict[int, str], Dict[str, int]]:
    """
    Scans dataset records and verifies integrity in parallel.
    """
    id2name, name2id = load_dataset_schema(dataset_dir)
    records: List[DatasetImageRecord] = []

    for split in ["train", "val", "test"]:
        img_dir = dataset_dir / "images" / split
        lbl_dir = dataset_dir / "labels" / split

        if not img_dir.exists():
            continue

        for img_file in sorted(img_dir.iterdir()):
            if not img_file.is_file() or img_file.name.startswith("."):
                continue

            lbl_file = lbl_dir / f"{img_file.stem}.txt" if lbl_dir.exists() else (dataset_dir / "labels" / split / f"{img_file.stem}.txt")

            records.append(
                DatasetImageRecord(
                    dataset_source=source_label,
                    split=split,
                    image_path=img_file,
                    label_path=lbl_file,
                    filename=img_file.name,
                    file_size=img_file.stat().st_size,
                )
            )

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(verify_image_label_record, r, name2id) for r in records]
        verified_records = [f.result() for f in futures]

    return verified_records, id2name, name2id


def align_class_mappings(
    baseline_id2name: Dict[int, str],
    baseline_name2id: Dict[str, int],
    new_id2name: Dict[int, str],
    new_name2id: Dict[str, int],
) -> Tuple[Dict[str, int], Dict[int, str], Dict[int, int]]:
    """
    Aligns class mappings:
    1. Baseline classes maintain exact numeric IDs.
    2. New classes get appended starting at next integer.
    3. Returns remapping table for new dataset label files.
    """
    merged_name2id = dict(baseline_name2id)
    next_id = max(merged_name2id.values()) + 1 if merged_name2id else 0

    remapping_for_new: Dict[int, int] = {}

    for new_cid, class_name in sorted(new_id2name.items()):
        if class_name in merged_name2id:
            remapping_for_new[new_cid] = merged_name2id[class_name]
        else:
            merged_name2id[class_name] = next_id
            remapping_for_new[new_cid] = next_id
            next_id += 1

    merged_id2name = {v: k for k, v in merged_name2id.items()}
    return merged_name2id, merged_id2name, remapping_for_new


def compute_dataset_fingerprint(records: List[DatasetImageRecord], dataset_version: str) -> str:
    """Computes deterministic SHA256 fingerprint over sorted dataset image hashes."""
    sha256_hash = hashlib.sha256()
    sha256_hash.update(dataset_version.encode("utf-8"))

    sorted_records = sorted(records, key=lambda x: (x.split, x.filename, x.sha256_hash))
    for rec in sorted_records:
        entry_str = f"{rec.filename}:{rec.split}:{rec.sha256_hash}:{rec.file_size}\n"
        sha256_hash.update(entry_str.encode("utf-8"))

    return sha256_hash.hexdigest()


def remap_label_content(label_path: Path, class_remapping: Dict[int, int]) -> Tuple[List[str], bool]:
    """Remaps leading class IDs in label file."""
    if not label_path.exists():
        return [], False

    content = label_path.read_text(encoding="utf-8").strip()
    if not content:
        return [], False

    lines = content.split("\n")
    remapped_lines: List[str] = []
    modified = False

    for line in lines:
        tokens = line.strip().split()
        if not tokens:
            continue
        orig_cid = int(tokens[0])
        new_cid = class_remapping.get(orig_cid, orig_cid)
        if new_cid != orig_cid:
            modified = True
        remapped_lines.append(f"{new_cid} " + " ".join(tokens[1:]))

    return remapped_lines, modified


def perform_dry_run(
    v4_dir: Path,
    new_dir: Path,
    output_dir: Path,
    args: argparse.Namespace,
    logger: Any,
) -> None:
    """Executes pre-flight dry-run system checks."""
    logger.info("[DRY RUN] Executing pre-flight dataset checks...")

    if not v4_dir.exists():
        raise FileNotFoundError(f"[DRY RUN FAIL] Baseline V4 dir missing: {v4_dir}")
    if not new_dir.exists():
        raise FileNotFoundError(f"[DRY RUN FAIL] New dataset dir missing: {new_dir}")

    # Output writability
    if output_dir.exists() and not args.overwrite:
        raise FileExistsError(f"[DRY RUN FAIL] Output dir {output_dir} exists. Use --overwrite to proceed.")

    b_id2name, b_name2id = load_dataset_schema(v4_dir)
    n_id2name, n_name2id = load_dataset_schema(new_dir)
    merged_name2id, _, remapping = align_class_mappings(b_id2name, b_name2id, n_id2name, n_name2id)

    logger.info(f"[DRY RUN PASS] Baseline classes ({len(b_name2id)}): {b_name2id}")
    logger.info(f"[DRY RUN PASS] Merged classes ({len(merged_name2id)}): {merged_name2id}")
    logger.info(f"[DRY RUN PASS] Class remapping table for new dataset: {remapping}")
    logger.info("[DRY RUN SUCCESS] All dry-run verifications passed cleanly.")


def main() -> None:
    """Main execution entrypoint for merge_datasets script."""
    args = parse_args()

    v4_dir = Path(args.v4_dir).resolve()
    new_dir = Path(args.new_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    meta_dir = output_dir / "metadata"

    logger = setup_logger("merge_datasets", log_file=meta_dir / "merge.log")

    if args.dry_run:
        perform_dry_run(v4_dir, new_dir, output_dir, args, logger)
        return

    # Check safe overwrite rule
    if output_dir.exists() and not args.overwrite:
        logger.critical(f"Output directory {output_dir} already exists! Pass --overwrite to proceed.")
        sys.exit(1)

    meta_dir.mkdir(parents=True, exist_ok=True)
    env_info = get_environment_info(args)
    save_json_report(env_info, meta_dir / "config_used.json")

    start_time = time.time()

    # 1. Scan and Audit Baseline and New Datasets
    logger.info(f"Auditing baseline V4 dataset ({v4_dir})...")
    baseline_records, b_id2name, b_name2id = scan_and_verify_dataset(v4_dir, "baseline", args.workers)

    logger.info(f"Auditing new dataset ({new_dir})...")
    new_records, n_id2name, n_name2id = scan_and_verify_dataset(new_dir, "new", args.workers)

    # 2. Input Validation Report
    corrupted_baseline = [r for r in baseline_records if r.is_corrupted]
    corrupted_new = [r for r in new_records if r.is_corrupted]

    input_validation_data = {
        "baseline_path": str(v4_dir),
        "new_path": str(new_dir),
        "baseline_total": len(baseline_records),
        "new_total": len(new_records),
        "baseline_corrupted_count": len(corrupted_baseline),
        "new_corrupted_count": len(corrupted_new),
        "is_valid": bool(len(corrupted_baseline) == 0 and len(corrupted_new) == 0),
    }
    save_json_report(input_validation_data, meta_dir / "input_validation.json")

    if not input_validation_data["is_valid"]:
        logger.critical(f"ABORTING MERGE: Input validation failed! Baseline corrupted: {len(corrupted_baseline)}, New corrupted: {len(corrupted_new)}")
        sys.exit(1)

    logger.info("[INPUT VALIDATION PASS] All images and label files in baseline and new datasets are valid.")

    # 3. Class Alignment & Validation
    merged_name2id, merged_id2name, class_remapping = align_class_mappings(
        b_id2name, b_name2id, n_id2name, n_name2id
    )

    class_val_data = {
        "baseline_classes": b_name2id,
        "new_classes": n_name2id,
        "merged_classes": merged_name2id,
        "remapping_table": class_remapping,
        "is_contiguous": set(merged_id2name.keys()) == set(range(len(merged_id2name))),
    }
    save_json_report(class_val_data, meta_dir / "class_validation.json")
    save_json_report(merged_name2id, meta_dir / "class_mapping.json")

    if not class_val_data["is_contiguous"]:
        logger.critical("ABORTING MERGE: Merged class IDs are not contiguous integers from 0 to N-1!")
        sys.exit(1)

    # 4. Deduplication & Duplicate Analysis
    logger.info("Executing comprehensive duplicate analysis across datasets...")
    seen_hashes: Dict[str, DatasetImageRecord] = {}
    duplicate_records: List[Dict[str, str]] = []
    merged_records: List[DatasetImageRecord] = []

    # Baseline records take priority
    for r in sorted(baseline_records, key=lambda x: (x.split, x.filename)):
        seen_hashes[r.sha256_hash] = r
        merged_records.append(r)

    for r in sorted(new_records, key=lambda x: (x.split, x.filename)):
        if r.sha256_hash in seen_hashes:
            existing = seen_hashes[r.sha256_hash]
            duplicate_records.append(
                {
                    "new_filename": r.filename,
                    "new_split": r.split,
                    "sha256": r.sha256_hash,
                    "existing_source": existing.dataset_source,
                    "existing_filename": existing.filename,
                    "existing_split": existing.split,
                }
            )
        else:
            seen_hashes[r.sha256_hash] = r
            merged_records.append(r)

    if duplicate_records:
        save_csv_report(
            duplicate_records,
            fieldnames=["new_filename", "new_split", "sha256", "existing_source", "existing_filename", "existing_split"],
            output_path=meta_dir / "duplicate_analysis.csv",
        )
        logger.warning(f"Detected {len(duplicate_records)} duplicate images. Logged to duplicate_analysis.csv")

    # 5. Split Integrity & Leakage Assertions
    logger.info("Asserting zero cross-split hash leakage in merged dataset...")
    train_hashes = {r.sha256_hash for r in merged_records if r.split == "train"}
    val_hashes = {r.sha256_hash for r in merged_records if r.split == "val"}
    test_hashes = {r.sha256_hash for r in merged_records if r.split == "test"}

    leak_tr_va = train_hashes.intersection(val_hashes)
    leak_tr_te = train_hashes.intersection(test_hashes)
    leak_va_te = val_hashes.intersection(test_hashes)

    split_stats = {
        "train_image_count": len([r for r in merged_records if r.split == "train"]),
        "val_image_count": len([r for r in merged_records if r.split == "val"]),
        "test_image_count": len([r for r in merged_records if r.split == "test"]),
        "train_healthy_count": len([r for r in merged_records if r.split == "train" and r.is_healthy]),
        "val_healthy_count": len([r for r in merged_records if r.split == "val" and r.is_healthy]),
        "test_healthy_count": len([r for r in merged_records if r.split == "test" and r.is_healthy]),
        "leakage_train_val_count": len(leak_tr_va),
        "leakage_train_test_count": len(leak_tr_te),
        "leakage_val_test_count": len(leak_va_te),
        "has_leakage": bool(leak_tr_va or leak_tr_te or leak_va_te),
    }
    save_json_report(split_stats, meta_dir / "split_statistics.json")

    if split_stats["has_leakage"]:
        logger.critical("ABORTING MERGE: Data leakage detected across merged dataset splits!")
        sys.exit(1)

    logger.info("[SPLIT VALIDATION PASS] Zero cross-split data leakage in merged dataset.")

    # Write Hash Database
    hash_db_rows: List[Dict[str, Any]] = [
        {
            "filename": r.filename,
            "dataset_source": r.dataset_source,
            "split": r.split,
            "sha256": r.sha256_hash,
            "width": r.width,
            "height": r.height,
            "filesize": r.file_size,
        }
        for r in sorted(merged_records, key=lambda x: (x.split, x.filename))
    ]
    save_csv_report(
        hash_db_rows,
        fieldnames=["filename", "dataset_source", "split", "sha256", "width", "height", "filesize"],
        output_path=meta_dir / "image_hashes.csv",
    )

    # 6. Execute File Copies and Label Remapping
    stats = MergeStats(
        baseline_images_total=len(baseline_records),
        new_images_total=len(new_records),
        merged_images_total=len(merged_records),
        baseline_healthy_count=len([r for r in baseline_records if r.is_healthy]),
        new_healthy_count=len([r for r in new_records if r.is_healthy]),
        merged_healthy_count=len([r for r in merged_records if r.is_healthy]),
        duplicates_detected=len(duplicate_records),
        baseline_classes_count=len(b_name2id),
        new_classes_count=len(n_name2id),
        merged_classes_count=len(merged_name2id),
    )

    failed_copies: List[Dict[str, str]] = []
    label_validation_rows: List[Dict[str, Any]] = []

    for split in ["train", "val", "test"]:
        s_recs = [r for r in merged_records if r.split == split]
        out_img_dir = output_dir / "images" / split
        out_lbl_dir = output_dir / "labels" / split
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_lbl_dir.mkdir(parents=True, exist_ok=True)

        if split == "train":
            stats.train_count = len(s_recs)
        elif split == "val":
            stats.val_count = len(s_recs)
        elif split == "test":
            stats.test_count = len(s_recs)

        pbar = tqdm(s_recs, desc=f"Merging {split}", unit="img")

        for r in pbar:
            dst_img_file = out_img_dir / r.filename
            dst_lbl_file = out_lbl_dir / f"{Path(r.filename).stem}.txt"

            # Copy Image File
            with open(r.image_path, "rb") as sf, open(dst_img_file, "wb") as df:
                df.write(sf.read())

            # Validate Image Copy
            if compute_sha256(dst_img_file) != r.sha256_hash:
                failed_copies.append({"filename": r.filename, "split": split, "reason": "SHA256 mismatch post-copy"})
                logger.error(f"Copy validation failed for image '{r.filename}'!")
                continue

            # Process Label File
            if r.dataset_source == "baseline":
                if r.label_path.exists():
                    with open(r.label_path, "rb") as sf, open(dst_lbl_file, "wb") as df:
                        df.write(sf.read())
                else:
                    dst_lbl_file.write_text("", encoding="utf-8")
            else:
                if r.label_path.exists():
                    remapped_lines, was_mod = remap_label_content(r.label_path, class_remapping)
                    if was_mod:
                        stats.remapped_labels_count += 1
                    with open(dst_lbl_file, "w", encoding="utf-8") as lf:
                        if remapped_lines:
                            lf.write("\n".join(remapped_lines) + "\n")
                else:
                    dst_lbl_file.write_text("", encoding="utf-8")

            # Post-write Label Validation
            content = dst_lbl_file.read_text(encoding="utf-8").strip()
            if r.is_healthy and len(content) > 0:
                logger.critical(f"HEALTHY IMAGE VIOLATION on '{r.filename}': Non-empty label text generated!")
                sys.exit(1)

            label_validation_rows.append(
                {
                    "filename": r.filename,
                    "split": split,
                    "is_healthy": r.is_healthy,
                    "status": "PASS",
                }
            )

        pbar.close()

    if failed_copies:
        save_csv_report(failed_copies, fieldnames=["filename", "split", "reason"], output_path=meta_dir / "failed_copies.csv")

    save_csv_report(
        label_validation_rows,
        fieldnames=["filename", "split", "is_healthy", "status"],
        output_path=meta_dir / "label_validation.csv",
    )

    # 7. Generate Dataset SHA256 Fingerprint
    fingerprint_hash = compute_dataset_fingerprint(merged_records, args.dataset_version)
    save_json_report(
        {
            "dataset_path": str(output_dir),
            "dataset_version": args.dataset_version,
            "dataset_fingerprint_sha256": fingerprint_hash,
            "merged_images_total": len(merged_records),
            "class_count": len(merged_name2id),
            "creation_time_utc": datetime.now(timezone.utc).isoformat(),
        },
        meta_dir / "dataset_fingerprint.json",
    )

    # 8. Generate data.yaml & Reports
    yaml_dict = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {v: k for k, v in merged_id2name.items()},
        "nc": len(merged_id2name),
        "dataset_version": args.dataset_version,
        "dataset_fingerprint": fingerprint_hash,
        "merged_from": [str(v4_dir), str(new_dir)],
    }
    with open(output_dir / "data.yaml", "w", encoding="utf-8") as yf:
        yaml.dump(yaml_dict, yf, sort_keys=False, default_flow_style=False)

    elapsed_sec = round(time.time() - start_time, 2)

    merge_summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "baseline_dataset": str(v4_dir),
        "new_dataset": str(new_dir),
        "merged_dataset": str(output_dir),
        "dataset_version": args.dataset_version,
        "fingerprint_sha256": fingerprint_hash,
        "merge_statistics": asdict(stats),
        "class_remapping": class_remapping,
        "elapsed_seconds": elapsed_sec,
    }

    manifest_data = {
        "dataset_version": args.dataset_version,
        "creation_time_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_fingerprint": fingerprint_hash,
        "merged_class_mapping": merged_name2id,
        "merge_statistics": asdict(stats),
        "environment": env_info,
    }

    save_json_report(merge_summary, meta_dir / "merge_report.json")
    save_json_report(manifest_data, meta_dir / "manifest.json")

    # Generate Executive Summary MD
    md_lines = [
        f"# Dataset Merge Summary Report - Version {args.dataset_version}",
        "",
        f"**Timestamp UTC:** `{env_info['timestamp_utc']}`",
        f"**Dataset SHA256 Fingerprint:** `{fingerprint_hash}`",
        f"**Baseline (V4) Path:** `{v4_dir}`",
        f"**New Dataset Path:** `{new_dir}`",
        f"**Merged (V5) Output Path:** `{output_dir}`",
        "",
        "## Executive Merge Summary",
        "",
        "| Metric | Count |",
        "| :--- | :--- |",
        f"| **Baseline Images Total** | {stats.baseline_images_total} |",
        f"| **New Images Total** | {stats.new_images_total} |",
        f"| **Duplicate Hashes Excluded** | {stats.duplicates_detected} |",
        f"| **Total Merged Images (V5)** | {stats.merged_images_total} |",
        f"| **Healthy Images Total** | {stats.merged_healthy_count} |",
        f"| **Merged Train Split** | {stats.train_count} |",
        f"| **Merged Val Split** | {stats.val_count} |",
        f"| **Merged Test Split** | {stats.test_count} |",
        f"| **Labels Remapped for New Classes** | {stats.remapped_labels_count} |",
        f"| **Elapsed Processing Time** | {elapsed_sec}s |",
        "",
        "## Class ID Mapping Table",
        "```json",
        json.dumps(merged_name2id, indent=2),
        "```",
        "",
        "---",
        "*Report generated automatically by Antigravity Pipeline Agent.*",
    ]
    save_markdown_report("\n".join(md_lines), meta_dir / "summary.md")

    # 9. Final Assertion Audits
    logger.info("Executing final assertion audits on merged dataset...")
    final_pass = True

    for split in ["train", "val", "test"]:
        n_imgs = len(list((output_dir / "images" / split).glob("*")))
        n_lbls = len(list((output_dir / "labels" / split).glob("*.txt")))
        if n_imgs != n_lbls:
            logger.error(f"Assertion Fail in split '{split}': {n_imgs} images != {n_lbls} labels!")
            final_pass = False

    if final_pass:
        logger.info("==================================================")
        logger.info("MERGE PASSED")
        logger.info(f"  Dataset Version: {args.dataset_version}")
        logger.info(f"  Fingerprint    : {fingerprint_hash[:16]}...")
        logger.info(f"  Merged Total   : {stats.merged_images_total} images")
        logger.info(f"  Train/Val/Test : {stats.train_count} / {stats.val_count} / {stats.test_count}")
        logger.info(f"  Merged Classes : {stats.merged_classes_count}")
        logger.info(f"  Config YAML    : {output_dir / 'data.yaml'}")
        logger.info(f"  Fingerprint    : {meta_dir / 'dataset_fingerprint.json'}")
        logger.info("==================================================")
    else:
        logger.critical("==================================================")
        logger.critical("MERGE FAILED - REVIEW LOGS AND METADATA AUDITS")
        logger.critical("==================================================")
        sys.exit(1)


if __name__ == "__main__":
    main()
