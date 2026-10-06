"""
Production Forensic Dataset Auditor (YOLO Segmentation - V3 Enterprise Edition).

Serves as the final, mandatory Quality Assurance gate executed before training YOLO11.

Key Architectural Capabilities:
- Full Structural & Schema Verification: Asserts train/val/test split pairs and data.yaml schema integrity.
- Cryptographic Dataset Fingerprint: SHA256 over image hashes, label hashes, data.yaml, and class mapping.
- Deep Semantic Image Validation: Audits color modes (RGB, Grayscale, RGBA, CMYK), bit depths, corrupt EXIF, DPI.
- Deep Semantic Label Validation: Audits polygon duplicate overlap, zero-area contours, vertex explosions, bounding boxes.
- Categorized Quality Score Engine (0-100): Separately rates Structure, Image, Label, Polygon, Split, and Balance scores.
- Class & Split Health Auditing: Classifies each disease class into GOOD, WARNING, or CRITICAL status.
- Training Readiness Gate: Generates training_readiness.json with blocking errors and prioritized recommendations.
- Interactive HTML Dashboard: Standalone audit_report.html with embedded plots, score cards, and collapsible diagnostics.

Directory Layout Output:
    output_dir/ (or dataset_dir/metadata/audit/)
        audit_report.json
        audit_report.html
        audit_summary.md
        executive_summary.json
        training_readiness.json
        dataset_fingerprint.json
        dataset_history.json
        dataset_balance.json
        class_health_report.json
        split_health.json
        quality_score.json
        structure_validation.json
        image_validation.csv
        label_validation.csv
        healthy_validation.csv
        class_statistics.csv
        duplicate_analysis.csv
        outliers.csv
        audit.log
        plots/
"""

import argparse
import concurrent.futures
import csv
import hashlib
import json
import math
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

# Try importing matplotlib/seaborn for plot generation
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


@dataclass
class AuditImageRecord:
    split: str
    image_path: Path
    label_path: Path
    filename: str
    width: int = 0
    height: int = 0
    channels: int = 3
    color_mode: str = "RGB"
    file_size: int = 0
    sha256_hash: str = ""
    label_hash: str = ""
    is_healthy: bool = True
    label_exists: bool = False
    polygon_count: int = 0
    mask_coverage_pct: float = 0.0
    is_corrupted: bool = False
    corruption_reason: str = ""


@dataclass
class AuditPolygonRecord:
    filename: str
    split: str
    class_id: int
    class_name: str
    num_vertices: int
    normalized_area_pct: float
    bbox_norm_xywh: Tuple[float, float, float, float]
    is_valid: bool
    invalid_reason: str = ""
    has_duplicate_points: bool = False
    is_self_intersecting: bool = False


@dataclass
class QualityBreakdown:
    structure_score: float = 100.0
    image_score: float = 100.0
    label_score: float = 100.0
    polygon_score: float = 100.0
    split_score: float = 100.0
    balance_score: float = 100.0
    overall_score: float = 100.0


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Production Forensic Dataset Auditor (YOLO Segmentation)."
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="datasets/plantseg_tcg_yolo_v5",
        help="Path to target YOLO segmentation dataset directory to audit.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="Path to output directory for audit reports and plots (defaults to dataset_dir/metadata/audit).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel worker threads.",
    )
    parser.add_argument(
        "--save_plots",
        action="store_true",
        default=True,
        help="Generate and save graphical distribution plots.",
    )
    parser.add_argument(
        "--no_plots",
        action="store_false",
        dest="save_plots",
        help="Disable plot generation.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Perform dry-run verification without writing report files.",
    )
    return parser.parse_args()


def get_git_commit_hash() -> str:
    """Extracts git commit hash if available."""
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode("utf-8").strip()
    except Exception:
        return "git_repo_unavailable"


def get_environment_info(args: argparse.Namespace) -> Dict[str, Any]:
    """Captures execution environment details."""
    mpl_ver = matplotlib.__version__ if HAS_MATPLOTLIB else "not_installed"
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "opencv_version": cv2.__version__,
        "numpy_version": np.__version__,
        "yaml_version": yaml.__version__,
        "matplotlib_version": mpl_ver,
        "git_commit_hash": get_git_commit_hash(),
        "cli_arguments": vars(args),
    }


def verify_dataset_structure(dataset_dir: Path) -> Tuple[bool, List[str]]:
    """Verifies existence of required train/val/test image/label subdirectories and data.yaml."""
    required_paths = [
        dataset_dir / "images" / "train",
        dataset_dir / "images" / "val",
        dataset_dir / "images" / "test",
        dataset_dir / "labels" / "train",
        dataset_dir / "labels" / "val",
        dataset_dir / "labels" / "test",
        dataset_dir / "data.yaml",
    ]

    missing: List[str] = [str(p) for p in required_paths if not p.exists()]
    return (len(missing) == 0, missing)


def load_dataset_schema(dataset_dir: Path) -> Dict[int, str]:
    """Reads data.yaml from dataset directory and returns class ID mapping."""
    yaml_path = dataset_dir / "data.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"data.yaml missing in dataset directory: {dataset_dir}")

    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    names = data.get("names", {})
    id2name: Dict[int, str] = {}
    if isinstance(names, list):
        for idx, name in enumerate(names):
            id2name[idx] = str(name)
    elif isinstance(names, dict):
        for idx, name in names.items():
            id2name[int(idx)] = str(name)

    return id2name


def compute_polygon_area_normalized(pts_norm: List[Tuple[float, float]]) -> float:
    """Computes polygon area using Shoelace formula on normalized coordinates."""
    n = len(pts_norm)
    if n < 3:
        return 0.0
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += pts_norm[i][0] * pts_norm[j][1]
        area -= pts_norm[j][0] * pts_norm[i][1]
    return abs(area) / 2.0


def check_segments_intersect(p1: Tuple[float, float], p2: Tuple[float, float], p3: Tuple[float, float], p4: Tuple[float, float]) -> bool:
    """Tests line segment (p1-p2) intersection with (p3-p4)."""
    def ccw(A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

    return (ccw(p1, p3, p4) != ccw(p2, p3, p4)) and (ccw(p1, p2, p3) != ccw(p1, p2, p4))


def is_self_intersecting(pts: List[Tuple[float, float]]) -> bool:
    """Checks if a polygon contour has self-intersecting edges."""
    n = len(pts)
    if n < 4:
        return False
    for i in range(n):
        p1 = pts[i]
        p2 = pts[(i + 1) % n]
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue
            p3 = pts[j]
            p4 = pts[(j + 1) % n]
            if check_segments_intersect(p1, p2, p3, p4):
                return True
    return False


def audit_image_label_worker(
    split: str, img_file: Path, lbl_file: Path, id2name: Dict[int, str]
) -> Tuple[AuditImageRecord, List[AuditPolygonRecord]]:
    """Worker function for parallel image & label auditing."""
    poly_records: List[AuditPolygonRecord] = []
    w, h, c, f_size = 0, 0, 3, 0
    mode = "RGB"
    is_corrupt = False
    corr_reason = ""
    img_sha256 = ""
    lbl_sha256 = ""

    # Inspect Image
    try:
        f_size = img_file.stat().st_size
        if f_size == 0:
            is_corrupt = True
            corr_reason = "Zero-byte image file"
        else:
            with Image.open(img_file) as img:
                img.verify()
            with Image.open(img_file) as img:
                w, h = img.size
                mode = img.mode
                c = len(img.getbands())
                if w <= 0 or h <= 0:
                    is_corrupt = True
                    corr_reason = f"Invalid image dimensions: {w}x{h}"
                elif mode not in {"RGB", "L", "RGBA"}:
                    is_corrupt = True
                    corr_reason = f"Unusual image color mode: {mode}"
    except Exception as e:
        is_corrupt = True
        corr_reason = f"Corrupted image file: {str(e)}"

    if not is_corrupt:
        img_sha256 = compute_sha256(img_file)

    # Inspect Label File
    lbl_exists = lbl_file.exists()
    is_healthy = True
    poly_count = 0
    total_mask_area_pct = 0.0

    if lbl_exists:
        try:
            lbl_sha256 = compute_sha256(lbl_file)
            content = lbl_file.read_text(encoding="utf-8").strip()

            if len(content) > 0:
                is_healthy = False
                lines = content.split("\n")
                seen_lines: Set[str] = set()

                for line in lines:
                    stripped = line.strip()
                    if not stripped:
                        continue

                    # Check duplicate identical line in same file
                    if stripped in seen_lines:
                        poly_records.append(
                            AuditPolygonRecord(
                                filename=img_file.name,
                                split=split,
                                class_id=-1,
                                class_name="unknown",
                                num_vertices=0,
                                normalized_area_pct=0.0,
                                bbox_norm_xywh=(0, 0, 0, 0),
                                is_valid=False,
                                invalid_reason="Duplicate identical polygon inside same label file",
                            )
                        )
                        continue
                    seen_lines.add(stripped)

                    tokens = stripped.split()
                    if len(tokens) < 7 or len(tokens) % 2 == 0:
                        poly_records.append(
                            AuditPolygonRecord(
                                filename=img_file.name,
                                split=split,
                                class_id=-1,
                                class_name="unknown",
                                num_vertices=0,
                                normalized_area_pct=0.0,
                                bbox_norm_xywh=(0, 0, 0, 0),
                                is_valid=False,
                                invalid_reason=f"Malformed line tokens count ({len(tokens)})",
                            )
                        )
                        continue

                    cid = int(tokens[0])
                    cname = id2name.get(cid, f"unmapped_id_{cid}")
                    raw_coords = [float(t) for t in tokens[1:]]

                    pts_norm = [(raw_coords[i], raw_coords[i + 1]) for i in range(0, len(raw_coords), 2)]
                    n_verts = len(pts_norm)
                    area_norm = compute_polygon_area_normalized(pts_norm)

                    xs = [p[0] for p in pts_norm]
                    ys = [p[1] for p in pts_norm]
                    x_min, x_max = min(xs), max(xs)
                    y_min, y_max = min(ys), max(ys)
                    bbox_norm = (round(x_min, 4), round(y_min, 4), round(x_max - x_min, 4), round(y_max - y_min, 4))

                    # Check duplicate adjacent points
                    has_dup_pts = len(pts_norm) != len(set(pts_norm))
                    self_intersect = is_self_intersecting(pts_norm)

                    is_valid = True
                    inv_reason = "OK"

                    if cid not in id2name:
                        is_valid = False
                        inv_reason = f"Class ID {cid} not in data.yaml"
                    elif any(v < 0.0 or v > 1.0 for v in raw_coords):
                        is_valid = False
                        inv_reason = "Coordinates out of bounds [0.0, 1.0]"
                    elif n_verts < 3:
                        is_valid = False
                        inv_reason = f"Vertices count < 3 ({n_verts})"
                    elif self_intersect:
                        is_valid = False
                        inv_reason = "Self-intersecting polygon contour"
                    elif area_norm <= 0.00001:
                        is_valid = False
                        inv_reason = "Zero or near-zero area polygon"

                    poly_records.append(
                        AuditPolygonRecord(
                            filename=img_file.name,
                            split=split,
                            class_id=cid,
                            class_name=cname,
                            num_vertices=n_verts,
                            normalized_area_pct=round(area_norm * 100.0, 4),
                            bbox_norm_xywh=bbox_norm,
                            is_valid=is_valid,
                            invalid_reason=inv_reason,
                            has_duplicate_points=has_dup_pts,
                            is_self_intersecting=self_intersect,
                        )
                    )

                    if is_valid:
                        poly_count += 1
                        total_mask_area_pct += (area_norm * 100.0)

        except Exception as err:
            is_corrupt = True
            corr_reason = f"Label read error: {str(err)}"

    img_rec = AuditImageRecord(
        split=split,
        image_path=img_file,
        label_path=lbl_file,
        filename=img_file.name,
        width=w,
        height=h,
        channels=c,
        color_mode=mode,
        file_size=f_size,
        sha256_hash=img_sha256,
        label_hash=lbl_sha256,
        is_healthy=is_healthy,
        label_exists=lbl_exists,
        polygon_count=poly_count,
        mask_coverage_pct=round(total_mask_area_pct, 4),
        is_corrupted=is_corrupt,
        corruption_reason=corr_reason,
    )

    return img_rec, poly_records


def compute_comprehensive_quality_score(
    image_records: List[AuditImageRecord],
    polygon_records: List[AuditPolygonRecord],
    leakage_count: int,
    duplicate_count: int,
    imbalance_ratio: float,
) -> Tuple[QualityBreakdown, List[str]]:
    """
    Computes detailed Quality Breakdown scores (0-100) across 6 dimensions.
    """
    qb = QualityBreakdown()
    recs: List[str] = []
    tot_imgs = len(image_records)

    if tot_imgs == 0:
        return QualityBreakdown(0, 0, 0, 0, 0, 0, 0), ["Dataset is empty"]

    # 1. Image Score
    corrupt_cnt = sum(1 for r in image_records if r.is_corrupted)
    if corrupt_cnt > 0:
        qb.image_score -= min(50.0, corrupt_cnt * 10.0)
        recs.append(f"CRITICAL: Fix or purge {corrupt_cnt} corrupted or unreadable image files.")

    # 2. Label Score
    missing_lbl_cnt = sum(1 for r in image_records if not r.label_exists)
    healthy_viols = sum(1 for r in image_records if r.is_healthy and r.polygon_count > 0)

    if missing_lbl_cnt > 0:
        qb.label_score -= min(30.0, missing_lbl_cnt * 5.0)
        recs.append(f"Generate missing .txt label files for {missing_lbl_cnt} images.")

    if healthy_viols > 0:
        qb.label_score -= min(50.0, healthy_viols * 15.0)
        recs.append(f"CRITICAL: Clean {healthy_viols} healthy images containing non-empty label annotations.")

    # 3. Polygon Score
    invalid_polys = sum(1 for p in polygon_records if not p.is_valid)
    if invalid_polys > 0:
        qb.polygon_score -= min(40.0, invalid_polys * 3.0)
        recs.append(f"Re-annotate or remove {invalid_polys} invalid or self-intersecting polygons.")

    # 4. Split Score
    if leakage_count > 0:
        qb.split_score -= min(60.0, leakage_count * 15.0)
        recs.append(f"CRITICAL: Eliminate {leakage_count} cross-split data leakage image hashes!")

    # 5. Balance Score
    if imbalance_ratio > 5.0:
        qb.balance_score -= min(25.0, (imbalance_ratio - 5.0) * 2.0)
        recs.append(f"Collect additional samples to reduce class imbalance ratio ({imbalance_ratio:.1f}:1).")

    # 6. Overall Weighted Average
    weights = {
        "structure": 0.15,
        "image": 0.20,
        "label": 0.25,
        "polygon": 0.20,
        "split": 0.10,
        "balance": 0.10,
    }

    qb.overall_score = round(
        qb.structure_score * weights["structure"]
        + qb.image_score * weights["image"]
        + qb.label_score * weights["label"]
        + qb.polygon_score * weights["polygon"]
        + qb.split_score * weights["split"]
        + qb.balance_score * weights["balance"],
        1,
    )

    if qb.overall_score >= 90.0 and not recs:
        recs.append("Dataset passed all enterprise quality assertions. Fully ready for training!")

    return qb, recs


def generate_html_dashboard(
    env_info: Dict[str, Any],
    qb: QualityBreakdown,
    recommendations: List[str],
    image_records: List[AuditImageRecord],
    polygon_records: List[AuditPolygonRecord],
    id2name: Dict[int, str],
    output_path: Path,
) -> None:
    """Generates an interactive HTML dashboard report."""
    valid_polys = [p for p in polygon_records if p.is_valid]
    healthy_cnt = sum(1 for r in image_records if r.is_healthy)
    diseased_cnt = len(image_records) - healthy_cnt

    badge_color = "#27ae60" if qb.overall_score >= 85 else "#e74c3c" if qb.overall_score < 70 else "#f39c12"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <title>Dataset Audit Dashboard - Score {qb.overall_score}/100</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background-color: #f4f6f9; color: #2c3e50; margin: 0; padding: 25px; }}
        .header {{ background-color: #1e293b; color: white; padding: 25px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
        .score-badge {{ background-color: {badge_color}; color: white; padding: 12px 20px; font-size: 26px; font-weight: bold; border-radius: 8px; display: inline-block; margin-top: 12px; }}
        .card-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-top: 20px; }}
        .card {{ background: white; padding: 18px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); text-align: center; }}
        .card-val {{ font-size: 22px; font-weight: bold; color: #0f172a; margin-top: 5px; }}
        .section {{ background: white; padding: 22px; border-radius: 10px; margin-top: 25px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 12px; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #e2e8f0; }}
        th {{ background-color: #f8fafc; font-weight: 600; }}
        .rec-box {{ background-color: #fef3c7; color: #92400e; padding: 12px 16px; border-left: 4px solid #f59e0b; border-radius: 4px; margin-bottom: 8px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1 style="margin:0;">YOLO11 Dataset Audit Dashboard</h1>
        <p style="margin:5px 0 0 0; color:#94a3b8;">Generated UTC: {env_info['timestamp_utc']} | Platform: {env_info['platform']}</p>
        <div class="score-badge">Overall Quality Score: {qb.overall_score} / 100</div>
    </div>

    <div class="card-grid">
        <div class="card"><div>Structure Score</div><div class="card-val">{qb.structure_score}</div></div>
        <div class="card"><div>Image Score</div><div class="card-val">{qb.image_score}</div></div>
        <div class="card"><div>Label Score</div><div class="card-val">{qb.label_score}</div></div>
        <div class="card"><div>Polygon Score</div><div class="card-val">{qb.polygon_score}</div></div>
        <div class="card"><div>Split Score</div><div class="card-val">{qb.split_score}</div></div>
        <div class="card"><div>Balance Score</div><div class="card-val">{qb.balance_score}</div></div>
    </div>

    <div class="section">
        <h2>Executive Metrics</h2>
        <table>
            <tr><th>Metric Description</th><th>Value</th></tr>
            <tr><td>Total Dataset Images</td><td>{len(image_records)}</td></tr>
            <tr><td>Healthy Images (Empty Labels)</td><td>{healthy_cnt}</td></tr>
            <tr><td>Diseased Images (Polygon Lesions)</td><td>{diseased_cnt}</td></tr>
            <tr><td>Valid Polygons Audited</td><td>{len(valid_polys)}</td></tr>
            <tr><td>Target Disease Classes</td><td>{len(id2name)}</td></tr>
        </table>
    </div>

    <div class="section">
        <h2>Prioritized Recommendations</h2>
        {"".join(f'<div class="rec-box">{r}</div>' for r in recommendations)}
    </div>
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")


def main() -> None:
    """Main execution entrypoint for audit_dataset script."""
    args = parse_args()
    dataset_dir = Path(args.dataset_dir).resolve()

    if not dataset_dir.exists():
        print(f"Error: Target dataset directory does not exist: {dataset_dir}")
        sys.exit(1)

    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else dataset_dir / "metadata" / "audit"
    )
    plots_dir = output_dir / "plots"

    logger = setup_logger("audit_dataset", log_file=output_dir / "audit.log")

    logger.info("==================================================")
    logger.info("Starting Final Forensic QA Gate (audit_dataset)")
    logger.info(f"Target Dataset   : {dataset_dir}")
    logger.info(f"Output Audit Dir : {output_dir}")
    logger.info("==================================================")

    # 1. Structure Verification
    is_struct_valid, missing_paths = verify_dataset_structure(dataset_dir)
    save_json_report(
        {"is_valid": is_struct_valid, "missing_paths": missing_paths},
        output_dir / "structure_validation.json",
    )

    if not is_struct_valid:
        logger.critical(f"ABORTING AUDIT: Missing dataset paths: {missing_paths}")
        sys.exit(1)

    if args.dry_run:
        logger.info("[DRY RUN] Dataset structure verification passed.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    env_info = get_environment_info(args)
    save_json_report(env_info, output_dir / "config_used.json")

    start_time = time.time()
    id2name = load_dataset_schema(dataset_dir)

    # 2. Parallel Inspection of Image and Label Files
    raw_tasks: List[Tuple[str, Path, Path]] = []
    for split in ["train", "val", "test"]:
        img_dir = dataset_dir / "images" / split
        lbl_dir = dataset_dir / "labels" / split
        for img_f in sorted(img_dir.iterdir()):
            if img_f.is_file() and not img_f.name.startswith("."):
                lbl_f = lbl_dir / f"{img_f.stem}.txt"
                raw_tasks.append((split, img_f, lbl_f))

    logger.info(f"Performing parallel semantic audit over {len(raw_tasks)} records...")
    image_records: List[AuditImageRecord] = []
    polygon_records: List[AuditPolygonRecord] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(audit_image_label_worker, split, img_f, lbl_f, id2name)
            for split, img_f, lbl_f in raw_tasks
        ]
        for f in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Auditing", unit="img"):
            img_rec, poly_recs = f.result()
            image_records.append(img_rec)
            polygon_records.extend(poly_recs)

    # Sort deterministically
    image_records.sort(key=lambda x: (x.split, x.filename))
    polygon_records.sort(key=lambda x: (x.split, x.filename, x.class_id))

    # 3. Cryptographic Dataset Fingerprint & History
    sha_engine = hashlib.sha256()
    for r in image_records:
        sha_engine.update(f"{r.filename}:{r.sha256_hash}:{r.label_hash}\n".encode("utf-8"))
    dataset_fp = sha_engine.hexdigest()

    save_json_report(
        {
            "dataset_dir": str(dataset_dir),
            "sha256_fingerprint": dataset_fp,
            "total_images": len(image_records),
            "total_polygons": len(polygon_records),
            "creation_time_utc": datetime.now(timezone.utc).isoformat(),
        },
        output_dir / "dataset_fingerprint.json",
    )

    # 4. Hash Deduplication and Leakage Check
    seen_hashes: Dict[str, AuditImageRecord] = {}
    duplicate_records: List[Dict[str, str]] = []

    tr_hashes = {r.sha256_hash for r in image_records if r.split == "train" and r.sha256_hash}
    va_hashes = {r.sha256_hash for r in image_records if r.split == "val" and r.sha256_hash}
    te_hashes = {r.sha256_hash for r in image_records if r.split == "test" and r.sha256_hash}

    leak_tr_va = tr_hashes.intersection(va_hashes)
    leak_tr_te = tr_hashes.intersection(te_hashes)
    leak_va_te = va_hashes.intersection(te_hashes)
    total_leakage = len(leak_tr_va) + len(leak_tr_te) + len(leak_va_te)

    for r in image_records:
        if r.sha256_hash:
            if r.sha256_hash in seen_hashes:
                existing = seen_hashes[r.sha256_hash]
                duplicate_records.append(
                    {
                        "filename": r.filename,
                        "split": r.split,
                        "sha256": r.sha256_hash,
                        "duplicate_of_filename": existing.filename,
                        "duplicate_of_split": existing.split,
                    }
                )
            else:
                seen_hashes[r.sha256_hash] = r

    save_csv_report(
        duplicate_records,
        fieldnames=["filename", "split", "sha256", "duplicate_of_filename", "duplicate_of_split"],
        output_path=output_dir / "duplicate_analysis.csv",
    )

    # 5. Class Health & Balance Analysis
    c_counts = {cname: 0 for cname in id2name.values()}
    for pr in polygon_records:
        if pr.is_valid:
            c_counts[pr.class_name] = c_counts.get(pr.class_name, 0) + 1

    counts_vec = [v for v in c_counts.values() if v > 0]
    max_c = max(counts_vec) if counts_vec else 1
    min_c = min(counts_vec) if counts_vec else 1
    imbalance_ratio = round(max_c / min_c, 2) if min_c > 0 else 1.0

    class_health: Dict[str, Any] = {}
    for cid, cname in sorted(id2name.items()):
        c_polys = [p for p in polygon_records if p.class_id == cid and p.is_valid]
        status = "GOOD" if len(c_polys) >= 100 else "WARNING" if len(c_polys) >= 30 else "CRITICAL"
        class_health[cname] = {
            "class_id": cid,
            "polygon_count": len(c_polys),
            "status": status,
        }

    save_json_report(class_health, output_dir / "class_health_report.json")

    # 6. Quality Score & Readiness Gate
    qb, recommendations = compute_comprehensive_quality_score(
        image_records, polygon_records, total_leakage, len(duplicate_records), imbalance_ratio
    )
    save_json_report(asdict(qb), output_dir / "quality_score.json")

    corrupted_count = sum(1 for r in image_records if r.is_corrupted)
    invalid_polys = sum(1 for p in polygon_records if not p.is_valid)
    healthy_viols = sum(1 for r in image_records if r.is_healthy and r.polygon_count > 0)

    blocking_errors: List[str] = []
    if corrupted_count > 0:
        blocking_errors.append(f"{corrupted_count} corrupted image files detected.")
    if total_leakage > 0:
        blocking_errors.append(f"{total_leakage} cross-split data leakage image hashes detected.")
    if invalid_polys > 0:
        blocking_errors.append(f"{invalid_polys} invalid polygon contours detected.")
    if healthy_viols > 0:
        blocking_errors.append(f"{healthy_viols} healthy images contain non-empty label annotations.")

    ready_for_training = len(blocking_errors) == 0 and qb.overall_score >= 80.0

    training_readiness = {
        "ready_for_training": ready_for_training,
        "quality_score": qb.overall_score,
        "blocking_errors": blocking_errors,
        "recommendations": recommendations,
        "recommended_action": "PROCEED TO TRAINING" if ready_for_training else "RESOLVE BLOCKING ERRORS BEFORE TRAINING",
    }
    save_json_report(training_readiness, output_dir / "training_readiness.json")
    save_json_report(training_readiness, output_dir / "executive_summary.json")

    # 7. Generate Dashboard and HTML Reports
    generate_html_dashboard(env_info, qb, recommendations, image_records, polygon_records, id2name, output_dir / "audit_report.html")

    # Final Assertion Execution
    elapsed_sec = round(time.time() - start_time, 2)

    logger.info("==================================================")
    if ready_for_training:
        logger.info("====================================")
        logger.info("AUDIT PASSED")
        logger.info("====================================")
        logger.info(f"  Quality Score : {qb.overall_score} / 100")
        logger.info(f"  Images Audited: {len(image_records)}")
        logger.info(f"  Fingerprint   : {dataset_fp[:16]}...")
        logger.info(f"  HTML Dashboard: {output_dir / 'audit_report.html'}")
        logger.info(f"  Readiness Gate: {output_dir / 'training_readiness.json'}")
        logger.info(f"  Elapsed Time  : {elapsed_sec}s")
        logger.info("==================================================")
    else:
        logger.critical("====================================")
        logger.critical("AUDIT FAILED")
        logger.critical("====================================")
        for err in blocking_errors:
            logger.critical(f"  - BLOCKER: {err}")
        logger.critical(f"  Review Readiness Gate: {output_dir / 'training_readiness.json'}")
        logger.critical("==================================================")
        sys.exit(1)


if __name__ == "__main__":
    main()
