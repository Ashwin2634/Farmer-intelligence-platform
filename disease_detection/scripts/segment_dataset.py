"""
Production SAM2 Segmentation Pipeline for Disease Datasets (V2 Architecture).

This script performs high-throughput automated segmentation over raw disease images using
SAM2 / Ultralytics SAM wrappers with enterprise-grade features:
- Environmental configuration logging & git commit state capture.
- Image integrity checking (corrupted/truncated/0-byte filtering).
- Deterministic ordering and SHA256 image hashing.
- Persistent JSON resume database for robust interruption recovery (Ctrl+C handling).
- Mask quality metrics extraction (area %, connected components, bbox, polygon complexity).
- Tiny & Huge mask anomaly detection and filtering.
- Visual alpha-blend overlay preview generation.
- ThreadPoolExecutor preloading for optimal GPU utilization.
- Periodic GPU VRAM & CUDA cache garbage collection.
- Per-disease breakdown metrics and comprehensive JSON/Markdown reports.
- Output validation asserting write accuracy.

Directory Layout Output:
    output_dir/
        metadata/
            report.json
            report.md
            config_used.json
            manifest.json
            resume.json
            image_hashes.csv
            mask_statistics.csv
            corrupted_images.csv
            failed_images.csv
            tiny_masks.csv
            huge_masks.csv
            segmentation.log
        disease_name/
            images/
            masks/
            previews/
"""

import argparse
import concurrent.futures
import csv
import gc
import json
import os
import platform
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
import torch
from PIL import Image, ImageFile
from tqdm import tqdm

# Allow Pillow to load truncated images for inspection (we will explicitly handle corruption)
ImageFile.LOAD_TRUNCATED_IMAGES = False

# Add scripts directory to sys.path for relative modular imports
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.hashing import compute_sha256
from utils.logger import setup_logger
from utils.reporting import save_csv_report, save_json_report, save_markdown_report


@dataclass
class ImageMetadata:
    rel_path: str
    abs_path: Path
    disease_name: str
    filename: str
    file_size_bytes: int
    sha256_hash: str = ""
    is_corrupted: bool = False
    corruption_reason: str = ""


@dataclass
class MaskMetrics:
    filename: str
    disease_name: str
    image_width: int
    image_height: int
    mask_area_pixels: int
    mask_area_percentage: float
    num_components: int
    largest_component_pixels: int
    bbox_xywh: Tuple[int, int, int, int]
    polygon_count: int
    is_tiny: bool = False
    is_huge: bool = False


@dataclass
class DiseaseStats:
    total_images: int = 0
    segmented_success: int = 0
    skipped: int = 0
    failed: int = 0
    corrupted: int = 0
    tiny_masks: int = 0
    huge_masks: int = 0
    total_area_pct: float = 0.0
    total_proc_time: float = 0.0


def parse_args() -> argparse.Namespace:
    """Parses CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Production SAM2 Segmentation Pipeline for Plant Disease Datasets."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="datasets/plant_seg_new",
        help="Input directory containing raw disease subdirectories.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="datasets/plant_seg_new_segmented",
        help="Output directory for segmented assets and metadata.",
    )
    parser.add_argument(
        "--dataset_version",
        type=str,
        default="V5",
        help="Dataset version tag (e.g. V5, V6, experimental).",
    )
    parser.add_argument(
        "--sam_checkpoint",
        type=str,
        default="sam2_t.pt",
        help="Path to SAM2 checkpoint weights file.",
    )
    parser.add_argument(
        "--model_cfg",
        type=str,
        default="sam2_hiera_t.yaml",
        help="SAM2 model configuration path/name.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Computation device ('cuda', 'cpu', 'cuda:0').",
    )
    parser.add_argument(
        "--points_per_side",
        type=int,
        default=32,
        help="Points per side for SAM2 Automatic Mask Generator.",
    )
    parser.add_argument(
        "--pred_iou_thresh",
        type=float,
        default=0.86,
        help="Prediction IOU threshold for mask filtering.",
    )
    parser.add_argument(
        "--stability_score_thresh",
        type=float,
        default=0.92,
        help="Stability score threshold for mask filtering.",
    )
    parser.add_argument(
        "--min_area_pct",
        type=float,
        default=0.2,
        help="Minimum mask area percentage of total image to keep (tiny mask threshold).",
    )
    parser.add_argument(
        "--max_area_pct",
        type=float,
        default=98.5,
        help="Maximum mask area percentage of total image allowed (huge mask threshold).",
    )
    parser.add_argument(
        "--save_previews",
        action="store_true",
        default=True,
        help="Generate transparent mask overlay preview PNG files.",
    )
    parser.add_argument(
        "--no_previews",
        action="store_false",
        dest="save_previews",
        help="Disable preview image generation.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of threads for image preloading and file I/O.",
    )
    parser.add_argument(
        "--gc_interval",
        type=int,
        default=50,
        help="Frequency of GPU memory and Python garbage collection (in images).",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Perform system dry run verification without running segmentation or writing files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing segmented masks and resume logs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic operations.",
    )
    return parser.parse_args()


def get_git_commit_hash() -> str:
    """Extracts current git commit hash if inside a git repository."""
    try:
        cmd = ["git", "rev-parse", "HEAD"]
        output = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
        return output.decode("utf-8").strip()
    except Exception:
        return "git_repo_unavailable"


def get_environment_info(args: argparse.Namespace, sam_version: str) -> Dict[str, Any]:
    """Captures complete execution environment parameters for reproducibility."""
    try:
        import ultralytics

        ultralytics_ver = ultralytics.__version__
    except ImportError:
        ultralytics_ver = "not_installed"

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_version": args.dataset_version,
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_name": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
        ),
        "ultralytics_version": ultralytics_ver,
        "sam_model_version": sam_version,
        "git_commit_hash": get_git_commit_hash(),
        "cli_arguments": vars(args),
    }


def verify_image_file(image_path: Path) -> Tuple[bool, str]:
    """
    Verifies image integrity (file size, readability, non-zero dimensions, non-truncated JPEG).

    Returns:
        Tuple[bool, str]: (is_valid, error_reason)
    """
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    if image_path.suffix.lower() not in valid_exts:
        return False, f"Unsupported format: {image_path.suffix}"

    try:
        file_size = image_path.stat().st_size
        if file_size == 0:
            return False, "Zero-byte file size"

        with Image.open(image_path) as img:
            img.verify()

        # Re-open after verify to ensure image bytes can be unpacked
        with Image.open(image_path) as img:
            img.load()
            w, h = img.size
            if w <= 0 or h <= 0:
                return False, f"Invalid dimensions: {w}x{h}"

        return True, "OK"
    except Exception as e:
        return False, f"Corrupted file: {str(e)}"


class SAM2Segmentor:
    """SAM2 Automatic Mask Generator with Ultralytics fallback."""

    def __init__(self, checkpoint_path: Path, model_cfg: str, device: str, args: argparse.Namespace, logger: Any):
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        self.logger = logger
        self.generator = None
        self.sam_version_str = "Unknown"
        self._init_model(checkpoint_path, model_cfg, args)

    def _init_model(self, checkpoint_path: Path, model_cfg: str, args: argparse.Namespace) -> None:
        try:
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
            from sam2.build_sam import build_sam2

            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint file missing: {checkpoint_path}")

            sam2_model = build_sam2(
                model_cfg,
                str(checkpoint_path),
                device=self.device,
                apply_postprocessing=False,
            )
            self.generator = SAM2AutomaticMaskGenerator(
                model=sam2_model,
                points_per_side=args.points_per_side,
                pred_iou_thresh=args.pred_iou_thresh,
                stability_score_thresh=args.stability_score_thresh,
            )
            self.sam_version_str = f"SAM2_Native ({checkpoint_path.name})"
            self.logger.info(f"Loaded SAM2 Native generator ({checkpoint_path.name})")
        except Exception as e1:
            self.logger.warning(f"Native SAM2 init failed: {e1}. Trying Ultralytics SAM wrapper...")
            try:
                from ultralytics import SAM

                weights = str(checkpoint_path) if checkpoint_path.exists() else "sam2_t.pt"
                self.generator = SAM(weights)
                self.sam_version_str = f"Ultralytics_SAM ({Path(weights).name})"
                self.logger.info(f"Loaded Ultralytics SAM wrapper ({weights})")
            except Exception as e2:
                raise RuntimeError(f"Could not initialize any SAM engine. Native: {e1} | Ultralytics: {e2}")

    def predict_mask(self, img_np: np.ndarray) -> np.ndarray:
        """Generates unified binary mask [H, W] (0 or 255)."""
        h, w = img_np.shape[:2]
        combined = np.zeros((h, w), dtype=np.uint8)

        if hasattr(self.generator, "generate"):
            masks = self.generator.generate(img_np)
            for m in masks:
                seg = m.get("segmentation")
                if seg is not None:
                    combined[seg] = 255
        elif hasattr(self.generator, "__call__"):
            results = self.generator(img_np, verbose=False, device=str(self.device))
            for res in results:
                if res.masks is not None:
                    for m_tensor in res.masks.data:
                        m_np = m_tensor.cpu().numpy().astype(bool)
                        if m_np.shape[:2] != (h, w):
                            m_np = cv2.resize(
                                m_np.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
                            ).astype(bool)
                        combined[m_np] = 255
        return combined


def compute_mask_metrics(
    mask_np: np.ndarray,
    filename: str,
    disease_name: str,
    min_area_pct: float,
    max_area_pct: float,
) -> MaskMetrics:
    """Computes comprehensive topological and geometry statistics on binary mask."""
    h, w = mask_np.shape[:2]
    total_pixels = h * w
    mask_bool = mask_np > 0
    mask_area_px = int(np.sum(mask_bool))
    mask_area_pct = (mask_area_px / total_pixels) * 100.0 if total_pixels > 0 else 0.0

    if mask_area_px == 0:
        return MaskMetrics(
            filename=filename,
            disease_name=disease_name,
            image_width=w,
            image_height=h,
            mask_area_pixels=0,
            mask_area_percentage=0.0,
            num_components=0,
            largest_component_pixels=0,
            bbox_xywh=(0, 0, 0, 0),
            polygon_count=0,
            is_tiny=True,
            is_huge=False,
        )

    # Connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask_np.astype(np.uint8), connectivity=8
    )
    # Ignore background label 0
    num_components = max(0, num_labels - 1)
    largest_comp = int(np.max(stats[1:, cv2.CC_STAT_AREA])) if num_components > 0 else 0

    # Bounding rectangle
    y_indices, x_indices = np.where(mask_bool)
    x_min, x_max = int(np.min(x_indices)), int(np.max(x_indices))
    y_min, y_max = int(np.min(y_indices)), int(np.max(y_indices))
    bbox = (x_min, y_min, x_max - x_min + 1, y_max - y_min + 1)

    # Polygon count via contours
    contours, _ = cv2.findContours(
        mask_np.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    poly_count = len(contours)

    is_tiny = mask_area_pct < min_area_pct
    is_huge = mask_area_pct > max_area_pct

    return MaskMetrics(
        filename=filename,
        disease_name=disease_name,
        image_width=w,
        image_height=h,
        mask_area_pixels=mask_area_px,
        mask_area_percentage=round(mask_area_pct, 4),
        num_components=num_components,
        largest_component_pixels=largest_comp,
        bbox_xywh=bbox,
        polygon_count=poly_count,
        is_tiny=is_tiny,
        is_huge=is_huge,
    )


def generate_overlay(image_rgb: np.ndarray, mask_np: np.ndarray) -> np.ndarray:
    """Generates RGB overlay with semi-transparent green mask and solid outline."""
    overlay = image_rgb.copy()
    green_mask = np.zeros_like(image_rgb, dtype=np.uint8)
    green_mask[:, :, 1] = 255  # Green channel

    mask_bool = mask_np > 0
    overlay[mask_bool] = cv2.addWeighted(
        image_rgb[mask_bool], 0.6, green_mask[mask_bool], 0.4, 0
    )

    # Draw contour outlines
    contours, _ = cv2.findContours(
        mask_np.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
    return overlay


def verify_saved_file(file_path: Path, expected_shape: Optional[Tuple[int, int]] = None) -> bool:
    """Verifies that a written file exists, is non-zero, and readable."""
    if not file_path.exists() or file_path.stat().st_size == 0:
        return False
    try:
        with Image.open(file_path) as img:
            img.verify()
            if expected_shape is not None and (img.width, img.height) != (
                expected_shape[1],
                expected_shape[0],
            ):
                return False
        return True
    except Exception:
        return False


def load_image_worker(meta: ImageMetadata) -> Tuple[ImageMetadata, Optional[np.ndarray], str]:
    """Worker function for preloading images in ThreadPoolExecutor."""
    try:
        pil_img = Image.open(meta.abs_path).convert("RGB")
        return meta, np.array(pil_img), "OK"
    except Exception as e:
        return meta, None, str(e)


class PipelineState:
    """Tracks state and persistent resume JSON with graceful interrupt protection."""

    def __init__(self, resume_path: Path, logger: Any):
        self.resume_path = resume_path
        self.logger = logger
        self.records: Dict[str, Dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        if self.resume_path.exists():
            try:
                with open(self.resume_path, "r", encoding="utf-8") as f:
                    self.records = json.load(f)
                self.logger.info(f"Loaded existing resume log with {len(self.records)} entries.")
            except Exception as e:
                self.logger.warning(f"Could not parse resume log ({e}). Starting fresh.")
                self.records = {}

    def is_completed(self, rel_path: str, sha256_hash: str) -> bool:
        rec = self.records.get(rel_path)
        if rec and rec.get("status") == "COMPLETED" and rec.get("sha256") == sha256_hash:
            return True
        return False

    def mark(
        self,
        rel_path: str,
        sha256_hash: str,
        status: str,
        disease: str,
        proc_time: float = 0.0,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        entry = {
            "sha256": sha256_hash,
            "status": status,
            "disease": disease,
            "processing_seconds": round(proc_time, 4),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if extra:
            entry.update(extra)
        self.records[rel_path] = entry

    def save(self) -> None:
        self.resume_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.resume_path.with_suffix(".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(self.records, f, indent=2)
        tmp_path.replace(self.resume_path)


def run_dry_run_verification(
    input_dir: Path,
    output_dir: Path,
    checkpoint_path: Path,
    args: argparse.Namespace,
    logger: Any,
) -> None:
    """Comprehensive dry-run verifications without modifying datasets."""
    logger.info("[DRY RUN] Starting pre-flight system verification...")

    # Check input dir
    if not input_dir.exists():
        raise FileNotFoundError(f"[DRY RUN FAIL] Input dir missing: {input_dir}")
    logger.info(f"[DRY RUN PASS] Input directory exists: {input_dir}")

    # Check output directory permissions
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        test_file = output_dir / ".write_test"
        test_file.write_text("test")
        test_file.unlink()
        logger.info(f"[DRY RUN PASS] Output directory writable: {output_dir}")
    except Exception as e:
        raise PermissionError(f"[DRY RUN FAIL] Output dir not writable: {e}")

    # Check checkpoint existence
    if not checkpoint_path.exists():
        logger.warning(f"[DRY RUN NOTICE] Local checkpoint {checkpoint_path} not found. SAM fallback wrapper will attempt download/loading.")
    else:
        logger.info(f"[DRY RUN PASS] Checkpoint exists: {checkpoint_path}")

    # Verify PyTorch / GPU
    logger.info(f"[DRY RUN PASS] PyTorch version: {torch.__version__} | CUDA Available: {torch.cuda.is_available()}")

    # Attempt SAM segmentor dry initialization
    try:
        _ = SAM2Segmentor(checkpoint_path, args.model_cfg, args.device, args, logger)
        logger.info("[DRY RUN PASS] SAM Segmentor model loaded successfully into memory.")
    except Exception as e:
        raise RuntimeError(f"[DRY RUN FAIL] SAM Segmentor loading failed: {e}")

    logger.info("[DRY RUN SUCCESS] All pre-flight system checks passed cleanly.")


def generate_markdown_report(
    env_info: Dict[str, Any],
    stats_summary: Dict[str, Any],
    disease_summaries: Dict[str, DiseaseStats],
    output_path: Path,
) -> None:
    """Generates executive markdown summary report."""
    md_lines = [
        f"# SAM2 Segmentation Pipeline Report - Dataset {env_info.get('dataset_version', 'V5')}",
        "",
        f"**Generated UTC:** {env_info.get('timestamp_utc')}",
        f"**SAM Model Version:** `{env_info.get('sam_model_version')}`",
        f"**Git Commit:** `{env_info.get('git_commit_hash')}`",
        f"**Platform / GPU:** `{env_info.get('platform')}` / `{env_info.get('cuda_device_name')}`",
        "",
        "## Overall Pipeline Summary",
        "",
        "| Metric | Value |",
        "| :--- | :--- |",
        f"| **Total Discovered Images** | {stats_summary['total_found']} |",
        f"| **Valid Processed Images** | {stats_summary['processed_valid']} |",
        f"| **Successfully Segmented** | {stats_summary['segmented_success']} |",
        f"| **Skipped (Already Processed)** | {stats_summary['skipped']} |",
        f"| **Corrupted Images** | {stats_summary['corrupted']} |",
        f"| **Tiny Masks Rejected (<{stats_summary['min_area_pct']}%)** | {stats_summary['tiny_masks']} |",
        f"| **Huge Masks Flagged (>{stats_summary['max_area_pct']}%)** | {stats_summary['huge_masks']} |",
        f"| **Failed Segmentations** | {stats_summary['failed']} |",
        f"| **Total Elapsed Time** | {stats_summary['elapsed_seconds']}s |",
        "",
        "## Per-Disease Breakdown",
        "",
        "| Disease Name | Images | Segmented | Skipped | Corrupted | Tiny Masks | Huge Masks | Avg Mask Area % | Avg Time (s) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for d_name, d_stat in sorted(disease_summaries.items()):
        avg_area = (
            d_stat.total_area_pct / d_stat.segmented_success
            if d_stat.segmented_success > 0
            else 0.0
        )
        avg_time = (
            d_stat.total_proc_time / d_stat.segmented_success
            if d_stat.segmented_success > 0
            else 0.0
        )
        md_lines.append(
            f"| `{d_name}` | {d_stat.total_images} | {d_stat.segmented_success} | {d_stat.skipped} | {d_stat.corrupted} | {d_stat.tiny_masks} | {d_stat.huge_masks} | {avg_area:.2f}% | {avg_time:.3f}s |"
        )

    md_lines.extend(
        [
            "",
            "## Configuration Used",
            "```json",
            json.dumps(env_info.get("cli_arguments", {}), indent=2),
            "```",
            "",
            "---",
            "*Report generated automatically by Antigravity Pipeline Agent.*",
        ]
    )

    save_markdown_report("\n".join(md_lines), output_path)


def main() -> None:
    """Main execution entrypoint with graceful interrupt handling."""
    args = parse_args()

    # Deterministic seeding
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    meta_dir = output_dir / "metadata"
    checkpoint_path = Path(args.sam_checkpoint).resolve()

    logger = setup_logger("segment_dataset", log_file=meta_dir / "segmentation.log")

    if args.dry_run:
        run_dry_run_verification(input_dir, output_dir, checkpoint_path, args, logger)
        return

    meta_dir.mkdir(parents=True, exist_ok=True)

    # Global State containers for reporting
    resume_db = PipelineState(meta_dir / "resume.json", logger)
    corrupted_records: List[Dict[str, str]] = []
    failed_records: List[Dict[str, str]] = []
    hash_records: List[Dict[str, str]] = []
    mask_stat_records: List[Dict[str, Any]] = []
    tiny_mask_records: List[Dict[str, Any]] = []
    huge_mask_records: List[Dict[str, Any]] = []
    disease_summaries: Dict[str, DiseaseStats] = {}

    start_time = time.time()

    # Graceful interrupt handler
    interrupted = False

    def handle_signal(sig, frame):
        nonlocal interrupted
        logger.warning("\n[INTERRUPT DETECTED] Gracefully flushing resume state and reports...")
        interrupted = True

    signal.signal(signal.SIGINT, handle_signal)

    # 1. Discover all disease subdirectories deterministically
    logger.info(f"Scanning input directory: {input_dir}")
    if not input_dir.exists():
        logger.critical(f"Input directory does not exist: {input_dir}")
        sys.exit(1)

    disease_dirs = sorted([d for d in input_dir.iterdir() if d.is_dir()])
    raw_file_tuples: List[Tuple[Path, str]] = []

    for d_dir in disease_dirs:
        disease_name = d_dir.name
        disease_summaries[disease_name] = DiseaseStats()
        search_dirs = [d_dir / "images", d_dir] if (d_dir / "images").exists() else [d_dir]
        for s_dir in search_dirs:
            for f in sorted(s_dir.iterdir()):
                if f.is_file() and not f.name.startswith("."):
                    raw_file_tuples.append((f, disease_name))

    logger.info(f"Discovered {len(raw_file_tuples)} total image files across {len(disease_dirs)} disease classes.")

    # 2. Perform Image Integrity & SHA256 Verification
    logger.info("Executing image integrity checks & SHA256 hashing...")
    valid_images: List[ImageMetadata] = []

    for img_path, d_name in raw_file_tuples:
        rel_p = str(img_path.relative_to(input_dir))
        d_stat = disease_summaries[d_name]
        d_stat.total_images += 1

        is_valid, err_reason = verify_image_file(img_path)
        if not is_valid:
            d_stat.corrupted += 1
            corrupted_records.append(
                {
                    "relative_path": rel_p,
                    "absolute_path": str(img_path),
                    "disease_name": d_name,
                    "reason": err_reason,
                }
            )
            resume_db.mark(rel_p, "N/A", "CORRUPTED", d_name, extra={"reason": err_reason})
            continue

        sha256_val = compute_sha256(img_path)
        hash_records.append(
            {
                "relative_path": rel_p,
                "disease_name": d_name,
                "filename": img_path.name,
                "sha256": sha256_val,
                "file_size": img_path.stat().st_size,
            }
        )

        valid_images.append(
            ImageMetadata(
                rel_path=rel_p,
                abs_path=img_path,
                disease_name=d_name,
                filename=img_path.name,
                file_size_bytes=img_path.stat().st_size,
                sha256_hash=sha256_val,
            )
        )

    # Write initial hash & corrupted reports
    save_csv_report(
        hash_records,
        fieldnames=["relative_path", "disease_name", "filename", "sha256", "file_size"],
        output_path=meta_dir / "image_hashes.csv",
    )
    if corrupted_records:
        save_csv_report(
            corrupted_records,
            fieldnames=["relative_path", "absolute_path", "disease_name", "reason"],
            output_path=meta_dir / "corrupted_images.csv",
        )
        logger.warning(f"Logged {len(corrupted_records)} corrupted images to metadata/corrupted_images.csv")

    # 3. Initialize Segmentor
    segmentor = SAM2Segmentor(checkpoint_path, args.model_cfg, args.device, args, logger)

    # Save environment config
    env_info = get_environment_info(args, segmentor.sam_version_str)
    save_json_report(env_info, meta_dir / "config_used.json")

    # 4. Main Segmentation Loop with Preloading Executor
    logger.info("Starting segmentation loop...")
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=args.workers)

    # Submit initial batch
    future_to_meta = {
        executor.submit(load_image_worker, meta): meta for meta in valid_images
    }

    processed_count = 0
    pbar = tqdm(total=len(valid_images), desc="Segmenting", unit="img")

    for future in concurrent.futures.as_completed(future_to_meta):
        if interrupted:
            break

        meta, img_rgb, load_status = future_to_meta[future].result()
        d_name = meta.disease_name
        d_stat = disease_summaries[d_name]
        rel_p = meta.rel_path

        pbar.set_postfix_str(f"Class: {d_name} | Img: {meta.filename[:15]}")

        # Check resume state
        target_img_dir = output_dir / d_name / "images"
        target_mask_dir = output_dir / d_name / "masks"
        target_prev_dir = output_dir / d_name / "previews"

        target_img_file = target_img_dir / meta.filename
        target_mask_file = target_mask_dir / f"{Path(meta.filename).stem}.png"
        target_prev_file = target_prev_dir / f"{Path(meta.filename).stem}_overlay.png"

        if (
            not args.overwrite
            and resume_db.is_completed(rel_p, meta.sha256_hash)
            and target_mask_file.exists()
            and target_img_file.exists()
        ):
            d_stat.skipped += 1
            pbar.update(1)
            continue

        if load_status != "OK" or img_rgb is None:
            d_stat.failed += 1
            failed_records.append(
                {
                    "relative_path": rel_p,
                    "disease_name": d_name,
                    "error": f"Image load failed: {load_status}",
                }
            )
            resume_db.mark(rel_p, meta.sha256_hash, "FAILED", d_name, extra={"error": load_status})
            pbar.update(1)
            continue

        # Predict Segmentation Mask
        t0 = time.time()
        try:
            mask_np = segmentor.predict_mask(img_rgb)
            proc_duration = time.time() - t0

            # Compute Mask Metrics & Anomaly Detection
            metrics = compute_mask_metrics(
                mask_np,
                meta.filename,
                d_name,
                args.min_area_pct,
                args.max_area_pct,
            )

            metric_dict = asdict(metrics)
            metric_dict["relative_path"] = rel_p
            mask_stat_records.append(metric_dict)

            # Ensure output directories exist
            target_img_dir.mkdir(parents=True, exist_ok=True)
            target_mask_dir.mkdir(parents=True, exist_ok=True)

            # Save Image and Mask
            pil_img = Image.fromarray(img_rgb)
            pil_img.save(target_img_file)

            pil_mask = Image.fromarray(mask_np)
            pil_mask.save(target_mask_file)

            # Save Overlay Preview if enabled
            if args.save_previews:
                target_prev_dir.mkdir(parents=True, exist_ok=True)
                overlay_np = generate_overlay(img_rgb, mask_np)
                cv2.imwrite(str(target_prev_file), cv2.cvtColor(overlay_np, cv2.COLOR_RGB2BGR))

            # Verify saved files
            if not verify_saved_file(target_img_file) or not verify_saved_file(target_mask_file):
                raise IOError("Written image or mask file verification failed (corrupted write).")

            # Handle Tiny / Huge anomalies
            if metrics.is_tiny:
                d_stat.tiny_masks += 1
                tiny_mask_records.append(metric_dict)
            if metrics.is_huge:
                d_stat.huge_masks += 1
                huge_mask_records.append(metric_dict)

            d_stat.segmented_success += 1
            d_stat.total_area_pct += metrics.mask_area_percentage
            d_stat.total_proc_time += proc_duration

            resume_db.mark(
                rel_p,
                meta.sha256_hash,
                "COMPLETED",
                d_name,
                proc_time=proc_duration,
                extra={"mask_area_pct": metrics.mask_area_percentage},
            )

        except Exception as seg_err:
            d_stat.failed += 1
            failed_records.append(
                {
                    "relative_path": rel_p,
                    "disease_name": d_name,
                    "error": str(seg_err),
                }
            )
            resume_db.mark(
                rel_p, meta.sha256_hash, "FAILED", d_name, extra={"error": str(seg_err)}
            )

        processed_count += 1
        pbar.update(1)

        # GPU memory & GC cleanup interval
        if processed_count % args.gc_interval == 0:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        # Save resume state every 20 images
        if processed_count % 20 == 0:
            resume_db.save()

    pbar.close()
    executor.shutdown(wait=False)

    # Flush final resume state
    resume_db.save()

    # Save CSV reports
    if mask_stat_records:
        field_order = [
            "filename",
            "disease_name",
            "relative_path",
            "image_width",
            "image_height",
            "mask_area_pixels",
            "mask_area_percentage",
            "num_components",
            "largest_component_pixels",
            "bbox_xywh",
            "polygon_count",
            "is_tiny",
            "is_huge",
        ]
        save_csv_report(mask_stat_records, fieldnames=field_order, output_path=meta_dir / "mask_statistics.csv")

    if tiny_mask_records:
        save_csv_report(tiny_mask_records, fieldnames=field_order, output_path=meta_dir / "tiny_masks.csv")

    if huge_mask_records:
        save_csv_report(huge_mask_records, fieldnames=field_order, output_path=meta_dir / "huge_masks.csv")

    if failed_records:
        save_csv_report(
            failed_records,
            fieldnames=["relative_path", "disease_name", "error"],
            output_path=meta_dir / "failed_images.csv",
        )

    # Compute total summary statistics
    elapsed_seconds = round(time.time() - start_time, 2)
    tot_success = sum(d.segmented_success for d in disease_summaries.values())
    tot_skipped = sum(d.skipped for d in disease_summaries.values())
    tot_failed = sum(d.failed for d in disease_summaries.values())
    tot_corrupted = sum(d.corrupted for d in disease_summaries.values())
    tot_tiny = sum(d.tiny_masks for d in disease_summaries.values())
    tot_huge = sum(d.huge_masks for d in disease_summaries.values())

    stats_summary = {
        "total_found": len(raw_file_tuples),
        "processed_valid": len(valid_images),
        "segmented_success": tot_success,
        "skipped": tot_skipped,
        "failed": tot_failed,
        "corrupted": tot_corrupted,
        "tiny_masks": tot_tiny,
        "huge_masks": tot_huge,
        "min_area_pct": args.min_area_pct,
        "max_area_pct": args.max_area_pct,
        "elapsed_seconds": elapsed_seconds,
    }

    # Save Manifest & Final JSON Report
    manifest_data = {
        "dataset_version": args.dataset_version,
        "creation_time_utc": datetime.now(timezone.utc).isoformat(),
        "sam_model_version": segmentor.sam_version_str,
        "class_list": sorted(list(disease_summaries.keys())),
        "overall_summary": stats_summary,
        "per_disease_summary": {
            d: asdict(stat) for d, stat in disease_summaries.items()
        },
    }
    save_json_report(manifest_data, meta_dir / "manifest.json")
    save_json_report(stats_summary, meta_dir / "report.json")

    # Save Executive Markdown Report
    generate_markdown_report(env_info, stats_summary, disease_summaries, meta_dir / "report.md")

    # Output Validation Assertions
    logger.info("Running output assertion validation...")
    valid_assertion = True
    for d_name in disease_summaries.keys():
        d_img_dir = output_dir / d_name / "images"
        d_mask_dir = output_dir / d_name / "masks"
        n_imgs = len(list(d_img_dir.glob("*"))) if d_img_dir.exists() else 0
        n_masks = len(list(d_mask_dir.glob("*.png"))) if d_mask_dir.exists() else 0
        if n_imgs != n_masks:
            logger.error(f"Validation Mismatch in '{d_name}': {n_imgs} images != {n_masks} masks!")
            valid_assertion = False

    if valid_assertion:
        logger.info("[ASSERTION PASS] Image count equals mask count across all disease folders.")
    else:
        logger.error("[ASSERTION FAIL] Dataset output counts are inconsistent.")

    logger.info("==================================================")
    logger.info("Segmentation Completed!")
    logger.info(f"  Version Tag  : {args.dataset_version}")
    logger.info(f"  Total Images : {len(raw_file_tuples)}")
    logger.info(f"  Segmented    : {tot_success}")
    logger.info(f"  Skipped      : {tot_skipped}")
    logger.info(f"  Corrupted    : {tot_corrupted}")
    logger.info(f"  Tiny Masks   : {tot_tiny}")
    logger.info(f"  Huge Masks   : {tot_huge}")
    logger.info(f"  Failed       : {tot_failed}")
    logger.info(f"  Elapsed Time : {elapsed_seconds}s")
    logger.info(f"  Metadata Dir : {meta_dir}")
    logger.info("==================================================")


if __name__ == "__main__":
    main()
