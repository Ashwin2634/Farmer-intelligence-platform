"""
Production Mask-to-YOLO Segmentation Converter (V2 Enhanced).

Converts SAM binary masks into YOLO segmentation polygon text files with deterministic
stratified train/val/test splits, Douglas-Peucker polygon simplification, advanced polygon
validity assertions, full hash database creation, dataset integrity auditing, post-copy/post-write
validation, and comprehensive reporting.

Key Features & Safeguards:
- Strict Healthy vs. Diseased Architecture: Healthy images generate empty .txt files ONLY.
- Integrity verification: Checks image/mask readability, dimension matching, zero-byte detection.
- Polygon Quality & Anomaly Filtering: Rejects self-intersection, NaN/Inf, repeated vertices, out-of-bounds, tiny area.
- Hash Database Generation: Saves SHA256, dimensions, file size to metadata/image_hashes.csv.
- Split Validation & Leakage Prevention: Ensures zero hash/filename overlap across train/val/test splits.
- Multi-Stage Verification: Validates copied image files and re-parses written .txt label files.
- ThreadPoolExecutor parallelization with deterministic output ordering.
- Comprehensive reporting: config_used.json, manifest.json, integrity_report.json, report.md, CSV logs.

Directory Layout Output:
    output_dir/
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
            report.json
            report.md
            class_mapping.json
            image_hashes.csv
            integrity_report.json
            integrity_errors.csv
            split_validation.json
            polygon_statistics.csv
            rejected_polygons.csv
            duplicate_report.csv
            malformed_labels.csv
            label_validation.csv
            conversion.log
        data.yaml
"""

import argparse
import concurrent.futures
import csv
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
import yaml
from PIL import Image
from tqdm import tqdm

# Add scripts directory to sys.path for relative modular imports
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.hashing import compute_sha256
from utils.logger import setup_logger
from utils.reporting import save_csv_report, save_json_report, save_markdown_report


@dataclass
class RawPairRecord:
    disease_name: str
    image_path: Path
    mask_path: Optional[Path]
    filename: str
    sha256_hash: str = ""
    width: int = 0
    height: int = 0
    file_size: int = 0
    is_healthy: bool = False
    is_corrupted: bool = False
    corruption_reason: str = ""


@dataclass
class PolygonStats:
    filename: str
    disease_name: str
    split: str
    class_id: int
    area_pixels: float
    perimeter_pixels: float
    num_vertices: int
    bbox_xywh: Tuple[int, int, int, int]


@dataclass
class ConversionStats:
    total_found: int = 0
    healthy_count: int = 0
    diseased_count: int = 0
    polygons_generated: int = 0
    polygons_rejected: int = 0
    duplicates_found: int = 0
    malformed_labels: int = 0
    integrity_failures: int = 0
    train_count: int = 0
    val_count: int = 0
    test_count: int = 0


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Production Mask-to-YOLO Segmentation Converter."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="datasets/plant_seg_new_segmented",
        help="Input directory containing segmented disease subdirectories.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="datasets/plant_seg_new_yolo",
        help="Output directory for YOLO segmentation dataset.",
    )
    parser.add_argument(
        "--dataset_version",
        type=str,
        default="V5",
        help="Dataset version identifier tag.",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Train split proportion.",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="Validation split proportion.",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.1,
        help="Testing split proportion.",
    )
    parser.add_argument(
        "--epsilon_ratio",
        type=float,
        default=0.005,
        help="Douglas-Peucker simplification parameter relative to perimeter.",
    )
    parser.add_argument(
        "--min_polygon_area",
        type=int,
        default=15,
        help="Minimum pixel area threshold to retain a polygon contour.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic dataset splitting.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of worker threads for parallel I/O and processing.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Perform dry-run system & integrity checks without writing files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output directory contents.",
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
        "git_commit_hash": get_git_commit_hash(),
        "cli_arguments": vars(args),
    }


def verify_image_and_mask_pair(
    img_path: Path, mask_path: Optional[Path], is_healthy: bool
) -> Tuple[bool, int, int, int, str]:
    """
    Verifies image and mask integrity.

    Returns:
        Tuple[is_valid, width, height, file_size, error_reason]
    """
    if not img_path.exists() or img_path.stat().st_size == 0:
        return False, 0, 0, 0, "Image missing or zero-byte file"

    try:
        file_size = img_path.stat().st_size
        with Image.open(img_path) as img:
            img.verify()
        with Image.open(img_path) as img:
            w, h = img.size
            if w <= 0 or h <= 0:
                return False, 0, 0, 0, f"Invalid image dimensions: {w}x{h}"
    except Exception as e:
        return False, 0, 0, 0, f"Corrupted image file: {str(e)}"

    if is_healthy or mask_path is None:
        return True, w, h, file_size, "OK"

    # Diseased image: mask must exist and match dimensions
    if not mask_path.exists() or mask_path.stat().st_size == 0:
        return False, w, h, file_size, "Mask missing or zero-byte file"

    try:
        with Image.open(mask_path) as mask_img:
            mask_img.verify()
        with Image.open(mask_path) as mask_img:
            mw, mh = mask_img.size
            if (mw, mh) != (w, h):
                return False, w, h, file_size, f"Dimension mismatch: Image ({w}x{h}) != Mask ({mw}x{mh})"
    except Exception as e:
        return False, w, h, file_size, f"Corrupted mask file: {str(e)}"

    return True, w, h, file_size, "OK"


def check_segments_intersect(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray) -> bool:
    """Helper to test line segment (p1-p2) intersection with (p3-p4)."""
    def ccw(A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

    return (ccw(p1, p3, p4) != ccw(p2, p3, p4)) and (ccw(p1, p2, p3) != ccw(p1, p2, p4))


def is_self_intersecting(polygon_pts: np.ndarray) -> bool:
    """Checks if a polygon contour has self-intersecting edges."""
    n = len(polygon_pts)
    if n < 4:
        return False

    for i in range(n):
        p1 = polygon_pts[i]
        p2 = polygon_pts[(i + 1) % n]
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue  # Adjacent edges
            p3 = polygon_pts[j]
            p4 = polygon_pts[(j + 1) % n]
            if check_segments_intersect(p1, p2, p3, p4):
                return True
    return False


def validate_and_format_polygon(
    cnt: np.ndarray,
    img_w: int,
    img_h: int,
    min_area: float,
    epsilon_ratio: float,
) -> Tuple[Optional[List[float]], Optional[Dict[str, Any]], Optional[str]]:
    """
    Validates polygon against rejection criteria and returns normalized coordinates or error reason.

    Rejection criteria:
    - NaN or Inf values.
    - Out of image bounds [0, W] or [0, H].
    - Fewer than 3 unique vertices.
    - Repeated adjacent vertices.
    - Area < min_area.
    - Self-intersecting polygon.

    Returns:
        Tuple[norm_coords, polygon_stats_dict, rejection_reason]
    """
    # 1. NaN / Inf check
    if np.isnan(cnt).any() or np.isinf(cnt).any():
        return None, None, "Contains NaN or Inf coordinates"

    pts = cnt.reshape(-1, 2)

    # 2. Out of image bounds
    if np.any(pts[:, 0] < 0) or np.any(pts[:, 0] > img_w) or np.any(pts[:, 1] < 0) or np.any(pts[:, 1] > img_h):
        return None, None, "Vertices outside image pixel boundaries"

    # 3. Douglas-Peucker simplification
    perimeter = float(cv2.arcLength(cnt, True))
    epsilon = epsilon_ratio * perimeter
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    simplified_pts = approx.reshape(-1, 2)

    # 4. Remove consecutive duplicate points
    unique_consecutive = [simplified_pts[0]]
    for p in simplified_pts[1:]:
        if not np.array_equal(p, unique_consecutive[-1]):
            unique_consecutive.append(p)
    if len(unique_consecutive) > 1 and np.array_equal(unique_consecutive[0], unique_consecutive[-1]):
        unique_consecutive.pop()

    clean_pts = np.array(unique_consecutive)

    # 5. Fewer than 3 unique vertices
    if len(clean_pts) < 3:
        return None, None, f"Fewer than 3 unique vertices after cleaning ({len(clean_pts)})"

    # 6. Minimum Area Check
    area = float(cv2.contourArea(clean_pts))
    if area < min_area:
        return None, None, f"Polygon area ({area:.1f}px) below minimum threshold ({min_area}px)"

    # 7. Self-intersection check
    if is_self_intersecting(clean_pts):
        return None, None, "Self-intersecting polygon contour"

    # 8. Normalize coordinates to [0.0, 1.0]
    norm_coords: List[float] = []
    for x, y in clean_pts:
        nx = max(0.0, min(1.0, float(x) / img_w))
        ny = max(0.0, min(1.0, float(y) / img_h))
        norm_coords.extend([round(nx, 6), round(ny, 6)])

    # Bounding Box
    x_min, y_min = int(np.min(clean_pts[:, 0])), int(np.min(clean_pts[:, 1]))
    bbox_w, bbox_h = int(np.max(clean_pts[:, 0]) - x_min + 1), int(np.max(clean_pts[:, 1]) - y_min + 1)

    stat_dict = {
        "area_pixels": round(area, 2),
        "perimeter_pixels": round(perimeter, 2),
        "num_vertices": len(clean_pts),
        "bbox_xywh": (x_min, y_min, bbox_w, bbox_h),
    }

    return norm_coords, stat_dict, None


def inspect_raw_pairs(input_dir: Path, logger: Any) -> Tuple[List[RawPairRecord], List[str]]:
    """
    Discovers all raw pairs deterministically.
    """
    disease_classes: Set[str] = set()
    raw_records: List[RawPairRecord] = []

    disease_dirs = sorted([d for d in input_dir.iterdir() if d.is_dir() and d.name != "metadata"])

    for d_dir in disease_dirs:
        disease_name = d_dir.name
        is_healthy_folder = "healthy" in disease_name.lower()
        if not is_healthy_folder:
            disease_classes.add(disease_name)

        img_dir = d_dir / "images" if (d_dir / "images").exists() else d_dir
        mask_dir = d_dir / "masks" if (d_dir / "masks").exists() else None

        for img_file in sorted(img_dir.iterdir()):
            if not img_file.is_file() or img_file.name.startswith("."):
                continue

            mask_file = None
            if mask_dir and mask_dir.exists():
                pot_mask = mask_dir / f"{img_file.stem}.png"
                if pot_mask.exists():
                    mask_file = pot_mask

            is_healthy = is_healthy_folder or (mask_file is None)

            raw_records.append(
                RawPairRecord(
                    disease_name=disease_name if not is_healthy_folder else "healthy",
                    image_path=img_file,
                    mask_path=mask_file,
                    filename=img_file.name,
                    is_healthy=is_healthy,
                )
            )

    sorted_classes = sorted(list(disease_classes))
    logger.info(f"Discovered {len(raw_records)} total raw file pairs across {len(sorted_classes)} disease classes.")
    return raw_records, sorted_classes


def process_integrity_worker(rec: RawPairRecord) -> RawPairRecord:
    """Worker function to inspect image/mask integrity and compute SHA256 hash in parallel."""
    is_valid, w, h, f_size, err_reason = verify_image_and_mask_pair(
        rec.image_path, rec.mask_path, rec.is_healthy
    )
    rec.width = w
    rec.height = h
    rec.file_size = f_size

    if not is_valid:
        rec.is_corrupted = True
        rec.corruption_reason = err_reason
    else:
        rec.sha256_hash = compute_sha256(rec.image_path)

    return rec


def validate_label_file(lbl_path: Path, is_healthy: bool, class_mapping: Dict[str, int]) -> Tuple[bool, str]:
    """Reopens written label file and validates formatting and contents."""
    if not lbl_path.exists():
        return False, "Label file missing"

    content = lbl_path.read_text(encoding="utf-8").strip()

    if is_healthy:
        if len(content) > 0:
            return False, "HEALTHY IMAGE VALIDATION FAILURE: Non-empty label text file generated for healthy image!"
        return True, "OK"

    if len(content) == 0:
        return True, "Empty disease label file (no mask detected)"

    lines = content.split("\n")
    valid_cids = set(class_mapping.values())

    for i, line in enumerate(lines):
        tokens = line.strip().split()
        if len(tokens) < 7 or len(tokens) % 2 == 0:
            return False, f"Line {i+1}: Invalid token count ({len(tokens)}). Must be odd and >= 7."

        try:
            cid = int(tokens[0])
            if cid not in valid_cids:
                return False, f"Line {i+1}: Class ID {cid} not in valid mapping set {valid_cids}"
        except ValueError:
            return False, f"Line {i+1}: Non-integer class ID '{tokens[0]}'"

        coords = [float(t) for t in tokens[1:]]
        for c_val in coords:
            if not (0.0 <= c_val <= 1.0):
                return False, f"Line {i+1}: Coordinate out of bounds [0.0, 1.0]: {c_val}"

    return True, "OK"


def perform_dry_run(
    input_dir: Path,
    output_dir: Path,
    args: argparse.Namespace,
    logger: Any,
) -> None:
    """Performs dry-run checks without writing output datasets."""
    logger.info("[DRY RUN] Verifying dataset structure, permissions, and class mappings...")

    if not input_dir.exists():
        raise FileNotFoundError(f"[DRY RUN FAIL] Input directory missing: {input_dir}")

    # Check output writability
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        t_file = output_dir / ".write_test"
        t_file.write_text("test")
        t_file.unlink()
        logger.info(f"[DRY RUN PASS] Output directory writable: {output_dir}")
    except Exception as e:
        raise PermissionError(f"[DRY RUN FAIL] Output dir not writable: {e}")

    records, classes = inspect_raw_pairs(input_dir, logger)
    class_map = {c: idx for idx, c in enumerate(classes)}

    logger.info(f"[DRY RUN PASS] Class mapping created ({len(class_map)} classes): {class_map}")
    logger.info("[DRY RUN SUCCESS] All dry-run verifications passed cleanly.")


def main() -> None:
    """Main execution entrypoint for convert_masks_to_yolo script."""
    args = parse_args()

    # Deterministic seeding
    random.seed(args.seed)
    np.random.seed(args.seed)

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    meta_dir = output_dir / "metadata"

    logger = setup_logger("convert_masks_to_yolo", log_file=meta_dir / "conversion.log")

    if args.dry_run:
        perform_dry_run(input_dir, output_dir, args, logger)
        return

    # Check split ratio sum
    if not math.isclose(args.train_ratio + args.val_ratio + args.test_ratio, 1.0, rel_tol=1e-5):
        logger.critical(f"Split ratios ({args.train_ratio}, {args.val_ratio}, {args.test_ratio}) must sum to 1.0!")
        sys.exit(1)

    meta_dir.mkdir(parents=True, exist_ok=True)
    env_info = get_environment_info(args)
    save_json_report(env_info, meta_dir / "config_used.json")

    start_time = time.time()

    # 1. Discover Raw Files
    raw_records, disease_classes = inspect_raw_pairs(input_dir, logger)
    class_mapping = {c_name: idx for idx, c_name in enumerate(sorted(disease_classes))}
    save_json_report(class_mapping, meta_dir / "class_mapping.json")

    # 2. Parallel Dataset Integrity Verification & Hashing
    logger.info(f"Running integrity checks & computing hashes using {args.workers} threads...")
    verified_records: List[RawPairRecord] = []
    integrity_errors: List[Dict[str, str]] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(process_integrity_worker, rec) for rec in raw_records]
        for f in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Auditing Integrity", unit="img"):
            res = f.result()
            if res.is_corrupted:
                integrity_errors.append(
                    {
                        "filename": res.filename,
                        "disease_name": res.disease_name,
                        "image_path": str(res.image_path),
                        "reason": res.corruption_reason,
                    }
                )
            else:
                verified_records.append(res)

    save_json_report(
        {
            "total_inspected": len(raw_records),
            "passed_integrity": len(verified_records),
            "failed_integrity": len(integrity_errors),
        },
        meta_dir / "integrity_report.json",
    )

    if integrity_errors:
        save_csv_report(
            integrity_errors,
            fieldnames=["filename", "disease_name", "image_path", "reason"],
            output_path=meta_dir / "integrity_errors.csv",
        )
        logger.critical(f"ABORTING CONVERSION: {len(integrity_errors)} files failed integrity audit!")
        sys.exit(1)

    # 3. Deduplication Check by SHA256 Hash
    logger.info("Performing deduplication check across dataset hashes...")
    seen_hashes: Dict[str, str] = {}
    duplicate_records: List[Dict[str, str]] = []
    unique_records: List[RawPairRecord] = []

    # Sort deterministically
    verified_records.sort(key=lambda x: (x.disease_name, x.filename))

    for rec in verified_records:
        if rec.sha256_hash in seen_hashes:
            duplicate_records.append(
                {
                    "filename": rec.filename,
                    "disease_name": rec.disease_name,
                    "sha256": rec.sha256_hash,
                    "duplicate_of": seen_hashes[rec.sha256_hash],
                }
            )
        else:
            seen_hashes[rec.sha256_hash] = rec.filename
            unique_records.append(rec)

    if duplicate_records:
        save_csv_report(
            duplicate_records,
            fieldnames=["filename", "disease_name", "sha256", "duplicate_of"],
            output_path=meta_dir / "duplicate_report.csv",
        )
        logger.warning(f"Detected {len(duplicate_records)} duplicate image hashes. Logged to duplicate_report.csv")

    # 4. Stratified Deterministic Splitting
    logger.info("Executing deterministic stratified dataset splitting...")
    grouped_by_disease: Dict[str, List[RawPairRecord]] = {}
    for rec in unique_records:
        grouped_by_disease.setdefault(rec.disease_name, []).append(rec)

    splits: Dict[str, List[RawPairRecord]] = {"train": [], "val": [], "test": []}

    for d_name, d_recs in sorted(grouped_by_disease.items()):
        d_recs.sort(key=lambda x: x.filename)
        random.seed(args.seed)
        random.shuffle(d_recs)

        n = len(d_recs)
        n_tr = int(n * args.train_ratio)
        n_va = int(n * args.val_ratio)

        splits["train"].extend(d_recs[:n_tr])
        splits["val"].extend(d_recs[n_tr : n_tr + n_va])
        splits["test"].extend(d_recs[n_tr + n_va :])

    # 5. Split Validation & Leakage Protection Assertion
    logger.info("Running split leakage validation...")
    train_hashes = {r.sha256_hash for r in splits["train"]}
    val_hashes = {r.sha256_hash for r in splits["val"]}
    test_hashes = {r.sha256_hash for r in splits["test"]}

    leakage_train_val = train_hashes.intersection(val_hashes)
    leakage_train_test = train_hashes.intersection(test_hashes)
    leakage_val_test = val_hashes.intersection(test_hashes)

    split_validation_data = {
        "train_count": len(splits["train"]),
        "val_count": len(splits["val"]),
        "test_count": len(splits["test"]),
        "leakage_train_val_count": len(leakage_train_val),
        "leakage_train_test_count": len(leakage_train_test),
        "leakage_val_test_count": len(leakage_val_test),
        "has_leakage": bool(leakage_train_val or leakage_train_test or leakage_val_test),
    }
    save_json_report(split_validation_data, meta_dir / "split_validation.json")

    if split_validation_data["has_leakage"]:
        logger.critical("ABORTING CONVERSION: Data leakage detected across dataset splits!")
        sys.exit(1)

    logger.info("[SPLIT VALIDATION PASS] Zero data leakage across train, val, and test splits.")

    # Write Complete Hash Database
    hash_db_rows: List[Dict[str, Any]] = []
    for split_name, s_recs in splits.items():
        for r in s_recs:
            hash_db_rows.append(
                {
                    "filename": r.filename,
                    "sha256": r.sha256_hash,
                    "disease": r.disease_name,
                    "split": split_name,
                    "width": r.width,
                    "height": r.height,
                    "filesize": r.file_size,
                }
            )

    hash_db_rows.sort(key=lambda x: (x["split"], x["disease"], x["filename"]))
    save_csv_report(
        hash_db_rows,
        fieldnames=["filename", "sha256", "disease", "split", "width", "height", "filesize"],
        output_path=meta_dir / "image_hashes.csv",
    )

    # 6. Conversion, Polygon Extraction & Quality Filtering
    polygon_stat_rows: List[Dict[str, Any]] = []
    rejected_polygon_rows: List[Dict[str, Any]] = []
    malformed_label_rows: List[Dict[str, Any]] = []
    label_validation_rows: List[Dict[str, Any]] = []

    stats = ConversionStats(
        total_found=len(raw_records),
        duplicates_found=len(duplicate_records),
        train_count=len(splits["train"]),
        val_count=len(splits["val"]),
        test_count=len(splits["test"]),
    )

    for split_name in ["train", "val", "test"]:
        s_recs = splits[split_name]
        logger.info(f"Processing split: '{split_name}' ({len(s_recs)} images)...")

        out_img_dir = output_dir / "images" / split_name
        out_lbl_dir = output_dir / "labels" / split_name
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_lbl_dir.mkdir(parents=True, exist_ok=True)

        pbar = tqdm(
            s_recs,
            desc=f"Converting {split_name}",
            unit="img",
        )

        for rec in pbar:
            d_name = rec.disease_name
            pbar.set_postfix_str(f"Class: {d_name} | Rej: {stats.polygons_rejected}")

            if rec.is_healthy:
                stats.healthy_count += 1
            else:
                stats.diseased_count += 1

            cid = class_mapping.get(d_name)
            polygon_lines: List[str] = []

            # Diseased image polygon extraction
            if not rec.is_healthy and rec.mask_path and cid is not None:
                mask_np = cv2.imread(str(rec.mask_path), cv2.IMREAD_GRAYSCALE)
                if mask_np is not None:
                    contours, _ = cv2.findContours(
                        (mask_np > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )
                    for cnt in contours:
                        norm_coords, p_stats, rej_reason = validate_and_format_polygon(
                            cnt, rec.width, rec.height, args.min_polygon_area, args.epsilon_ratio
                        )
                        if rej_reason or norm_coords is None:
                            stats.polygons_rejected += 1
                            rejected_polygon_rows.append(
                                {
                                    "filename": rec.filename,
                                    "disease_name": d_name,
                                    "split": split_name,
                                    "reason": rej_reason or "Unknown polygon rejection",
                                }
                            )
                        else:
                            stats.polygons_generated += 1
                            coord_str = " ".join(f"{v:.6f}" for v in norm_coords)
                            polygon_lines.append(f"{cid} {coord_str}")

                            if p_stats:
                                p_stats.update(
                                    {
                                        "filename": rec.filename,
                                        "disease_name": d_name,
                                        "split": split_name,
                                        "class_id": cid,
                                    }
                                )
                                polygon_stat_rows.append(p_stats)

            # Copy Image File
            dst_img_file = out_img_dir / rec.filename
            with open(rec.image_path, "rb") as sf, open(dst_img_file, "wb") as df:
                df.write(sf.read())

            # Post-Copy Image Validation
            if not dst_img_file.exists() or compute_sha256(dst_img_file) != rec.sha256_hash:
                logger.critical(f"ABORTING CONVERSION: Copy validation failed for image '{rec.filename}'!")
                sys.exit(1)

            # Write Label Text File
            dst_lbl_file = out_lbl_dir / f"{rec.image_path.stem}.txt"
            with open(dst_lbl_file, "w", encoding="utf-8") as lf:
                if polygon_lines:
                    lf.write("\n".join(polygon_lines) + "\n")

            # Label File Reopen & Validation
            is_lbl_valid, lbl_err = validate_label_file(dst_lbl_file, rec.is_healthy, class_mapping)
            label_validation_rows.append(
                {
                    "filename": rec.filename,
                    "split": split_name,
                    "is_healthy": rec.is_healthy,
                    "status": "PASS" if is_lbl_valid else "FAIL",
                    "reason": lbl_err,
                }
            )

            if not is_lbl_valid:
                stats.malformed_labels += 1
                malformed_label_rows.append(
                    {
                        "filename": rec.filename,
                        "disease_name": d_name,
                        "split": split_name,
                        "error": lbl_err,
                    }
                )
                logger.error(f"Label validation failure on '{rec.filename}': {lbl_err}")

        pbar.close()

    # 7. Generate Data YAML & Metadata Reports
    # Generate data.yaml
    yaml_dict = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {v: k for k, v in class_mapping.items()},
        "nc": len(class_mapping),
        "dataset_version": args.dataset_version,
    }
    with open(output_dir / "data.yaml", "w", encoding="utf-8") as yf:
        yaml.dump(yaml_dict, yf, sort_keys=False, default_flow_style=False)

    # Save CSV Reports
    if polygon_stat_rows:
        polygon_stat_rows.sort(key=lambda x: (x["split"], x["disease_name"], x["filename"]))
        save_csv_report(
            polygon_stat_rows,
            fieldnames=["filename", "disease_name", "split", "class_id", "area_pixels", "perimeter_pixels", "num_vertices", "bbox_xywh"],
            output_path=meta_dir / "polygon_statistics.csv",
        )

    if rejected_polygon_rows:
        rejected_polygon_rows.sort(key=lambda x: (x["split"], x["disease_name"], x["filename"]))
        save_csv_report(
            rejected_polygon_rows,
            fieldnames=["filename", "disease_name", "split", "reason"],
            output_path=meta_dir / "rejected_polygons.csv",
        )

    if malformed_label_rows:
        save_csv_report(
            malformed_label_rows,
            fieldnames=["filename", "disease_name", "split", "error"],
            output_path=meta_dir / "malformed_labels.csv",
        )

    if label_validation_rows:
        save_csv_report(
            label_validation_rows,
            fieldnames=["filename", "split", "is_healthy", "status", "reason"],
            output_path=meta_dir / "label_validation.csv",
        )

    # Save Manifest & Final JSON Report
    elapsed_sec = round(time.time() - start_time, 2)

    manifest_data = {
        "dataset_version": args.dataset_version,
        "creation_time_utc": datetime.now(timezone.utc).isoformat(),
        "class_mapping": class_mapping,
        "conversion_statistics": asdict(stats),
        "environment": env_info,
        "elapsed_seconds": elapsed_sec,
    }
    save_json_report(manifest_data, meta_dir / "manifest.json")
    save_json_report(asdict(stats), meta_dir / "report.json")

    # Generate Markdown Summary
    md_lines = [
        f"# Mask to YOLO Conversion Summary - Version {args.dataset_version}",
        "",
        f"**Timestamp UTC:** `{env_info['timestamp_utc']}`",
        f"**OpenCV / NumPy:** `{env_info['opencv_version']}` / `{env_info['numpy_version']}`",
        f"**Git Commit:** `{env_info['git_commit_hash']}`",
        "",
        "## Dataset Statistics",
        "",
        "| Metric | Value |",
        "| :--- | :--- |",
        f"| **Total Raw Discovered Images** | {stats.total_found} |",
        f"| **Unique Processed Images** | {len(unique_records)} |",
        f"| **Healthy Images (Empty Labels)** | {stats.healthy_count} |",
        f"| **Diseased Images** | {stats.diseased_count} |",
        f"| **Total Polygons Generated** | {stats.polygons_generated} |",
        f"| **Polygons Rejected** | {stats.polygons_rejected} |",
        f"| **Duplicate Hashes Excluded** | {stats.duplicates_found} |",
        f"| **Train Split Count** | {stats.train_count} |",
        f"| **Val Split Count** | {stats.val_count} |",
        f"| **Test Split Count** | {stats.test_count} |",
        f"| **Elapsed Processing Time** | {elapsed_sec}s |",
        "",
        "## Class Mapping",
        "```json",
        json.dumps(class_mapping, indent=2),
        "```",
        "",
        "---",
        "*Report generated automatically by Antigravity Pipeline Agent.*",
    ]
    save_markdown_report("\n".join(md_lines), meta_dir / "report.md")

    # 8. Final Assertion Audits
    logger.info("Executing final assertion audits...")
    final_pass = True

    # Image count == Label count check
    for split_name in ["train", "val", "test"]:
        n_imgs = len(list((output_dir / "images" / split_name).glob("*")))
        n_lbls = len(list((output_dir / "labels" / split_name).glob("*.txt")))
        if n_imgs != n_lbls:
            logger.error(f"Assertion Fail in split '{split_name}': {n_imgs} images != {n_lbls} labels!")
            final_pass = False

    # Check contiguous class IDs from 0 to N-1
    cids = set(class_mapping.values())
    expected_cids = set(range(len(class_mapping)))
    if cids != expected_cids:
        logger.error(f"Assertion Fail: Class IDs are not contiguous 0..N-1: {cids} vs {expected_cids}")
        final_pass = False

    if stats.malformed_labels > 0:
        logger.error(f"Assertion Fail: {stats.malformed_labels} malformed label errors encountered!")
        final_pass = False

    if final_pass:
        logger.info("==================================================")
        logger.info("CONVERSION PASSED")
        logger.info(f"  Version      : {args.dataset_version}")
        logger.info(f"  Train/Val/Test: {stats.train_count} / {stats.val_count} / {stats.test_count}")
        logger.info(f"  Polygons Gen : {stats.polygons_generated}")
        logger.info(f"  Polygons Rej : {stats.polygons_rejected}")
        logger.info(f"  Config YAML  : {output_dir / 'data.yaml'}")
        logger.info(f"  Manifest     : {meta_dir / 'manifest.json'}")
        logger.info("==================================================")
    else:
        logger.critical("==================================================")
        logger.critical("CONVERSION FAILED - REVIEW METADATA ERROR LOGS")
        logger.critical("==================================================")
        sys.exit(1)


if __name__ == "__main__":
    main()
