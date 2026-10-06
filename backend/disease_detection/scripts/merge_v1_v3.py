#!/usr/bin/env python3
"""
merge_v1_v3.py
--------------
Merges YOLO V1 disease dataset with healthy leaf dataset to create V3.
Performs class remapping, collision handling, validation, and report generation.
"""

import json
import logging
import shutil
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Tuple

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("merge_v1_v3")

# Constants
WORKSPACE_ROOT = Path("/content/drive/MyDrive/AI_Service")
V1_DIR = WORKSPACE_ROOT / "plantseg_tcg_yolo_v1"
HEALTHY_DIR = WORKSPACE_ROOT / "healthy_dataset_yolo"
V3_DIR = WORKSPACE_ROOT / "plantseg_tcg_yolo_v3"

CLASS_MAPPING_HEALTHY = {
    0: 14,  # tomato_healthy
    1: 15,  # cucumber_healthy
    2: 16   # grape_healthy
}

CLASS_NAMES = {
    0: "tomato_bacterial_leaf_spot",
    1: "tomato_early_blight",
    2: "tomato_late_blight",
    3: "tomato_leaf_mold",
    4: "tomato_mosaic_virus",
    5: "tomato_septoria_leaf_spot",
    6: "tomato_yellow_leaf_curl",
    7: "cucumber_angular_leaf_spot",
    8: "cucumber_bacterial_wilt",
    9: "cucumber_powdery_mildew",
    10: "grape_black_rot",
    11: "grape_downy_mildew",
    12: "grape_leaf_spot",
    13: "grapevine_leafroll_disease",
    14: "tomato_healthy",
    15: "cucumber_healthy",
    16: "grape_healthy"
}

def remap_label_file(src_path: Path, dst_path: Path) -> Tuple[int, List[str]]:
    """
    Reads a healthy label file, maps classes 0,1,2 to 14,15,16, and writes to dst_path.
    Returns the number of annotations successfully remapped and a list of warnings/errors.
    """
    warnings_list = []
    annotations_count = 0
    lines_to_write = []

    try:
        if not src_path.exists():
            return 0, [f"Label file does not exist: {src_path}"]
        
        if src_path.stat().st_size == 0:
            warnings_list.append(f"Empty label file: {src_path.name}")
            dst_path.touch()  # Create empty file
            return 0, warnings_list

        with open(src_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if not parts:
                    continue
                try:
                    class_id = int(parts[0])
                except ValueError:
                    warnings_list.append(f"Invalid non-integer class ID '{parts[0]}' in {src_path.name} line {line_idx}")
                    continue

                if class_id not in CLASS_MAPPING_HEALTHY:
                    warnings_list.append(f"Invalid class ID {class_id} for healthy dataset in {src_path.name} line {line_idx}")
                    continue

                new_class_id = CLASS_MAPPING_HEALTHY[class_id]
                new_line = f"{new_class_id} " + " ".join(parts[1:])
                lines_to_write.append(new_line)
                annotations_count += 1

        # Write mapped annotations
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dst_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines_to_write) + ("\n" if lines_to_write else ""))

    except Exception as e:
        warnings_list.append(f"Error processing label file {src_path.name}: {e}")

    return annotations_count, warnings_list


def copy_v1_dataset(v1_dir: Path, v3_dir: Path) -> Tuple[int, int, List[str], List[str]]:
    """
    Copies the V1 dataset to V3 destination (images and labels).
    Returns (images_copied, annotations_count, warnings, errors).
    """
    images_copied = 0
    annotations_count = 0
    warnings = []
    errors = []

    splits = ["train", "val", "test"]
    for split in splits:
        img_src_dir = v1_dir / "images" / split
        lbl_src_dir = v1_dir / "labels" / split

        img_dst_dir = v3_dir / "images" / split
        lbl_dst_dir = v3_dir / "labels" / split

        img_dst_dir.mkdir(parents=True, exist_ok=True)
        lbl_dst_dir.mkdir(parents=True, exist_ok=True)

        if not img_src_dir.exists():
            errors.append(f"V1 images folder does not exist: {img_src_dir}")
            continue
        if not lbl_src_dir.exists():
            errors.append(f"V1 labels folder does not exist: {lbl_src_dir}")
            continue

        for img_path in img_src_dir.iterdir():
            if img_path.is_file() and img_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
                dst_img_path = img_dst_dir / img_path.name
                try:
                    shutil.copy2(img_path, dst_img_path)
                    images_copied += 1
                except Exception as e:
                    errors.append(f"Failed to copy image {img_path}: {e}")
                    continue

                # Copy corresponding label if it exists
                lbl_name = img_path.stem + ".txt"
                lbl_path = lbl_src_dir / lbl_name
                dst_lbl_path = lbl_dst_dir / lbl_name

                if lbl_path.exists():
                    try:
                        shutil.copy2(lbl_path, dst_lbl_path)
                        # Count annotations in copied file
                        if dst_lbl_path.stat().st_size > 0:
                            with open(dst_lbl_path, "r", encoding="utf-8") as lf:
                                annotations_count += sum(1 for line in lf if line.strip())
                        else:
                            warnings.append(f"Empty V1 label file: {lbl_path.name}")
                    except Exception as e:
                        errors.append(f"Failed to copy/process V1 label {lbl_path}: {e}")
                else:
                    warnings.append(f"Missing label for V1 image: {img_path.name}")

    return images_copied, annotations_count, warnings, errors


def merge_healthy_split(
    split: str,
    healthy_dir: Path,
    v3_dir: Path,
    renamed_files: List[Tuple[str, str]]
) -> Tuple[int, int, List[str], List[str]]:
    """
    Merges healthy dataset's split (train, val, test) into V3, remapping labels and avoiding name collisions.
    Returns (images_merged, annotations_count, warnings, errors).
    """
    images_merged = 0
    annotations_count = 0
    warnings = []
    errors = []

    src_img_dir = healthy_dir / split / "images"
    src_lbl_dir = healthy_dir / split / "labels"

    dst_img_dir = v3_dir / "images" / split
    dst_lbl_dir = v3_dir / "labels" / split

    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)

    if not src_img_dir.exists():
        errors.append(f"Healthy split images folder does not exist: {src_img_dir}")
        return 0, 0, warnings, errors
    if not src_lbl_dir.exists():
        errors.append(f"Healthy split labels folder does not exist: {src_lbl_dir}")
        return 0, 0, warnings, errors

    for img_path in src_img_dir.iterdir():
        if not img_path.is_file() or img_path.suffix.lower() not in [".jpg", ".jpeg", ".png", ".bmp"]:
            continue

        base_name = img_path.stem
        suffix = img_path.suffix
        lbl_name = f"{base_name}.txt"
        lbl_path = src_lbl_dir / lbl_name

        # Resolve collisions
        dst_img_name = f"{base_name}{suffix}"
        dst_lbl_name = lbl_name

        collision = False
        counter = 1
        while (dst_img_dir / dst_img_name).exists() or (dst_lbl_dir / dst_lbl_name).exists():
            collision = True
            dst_img_name = f"{base_name}_healthy_{counter:03d}{suffix}"
            dst_lbl_name = f"{base_name}_healthy_{counter:03d}.txt"
            counter += 1

        if collision:
            renamed_files.append((img_path.name, dst_img_name))

        dst_img_path = dst_img_dir / dst_img_name
        dst_lbl_path = dst_lbl_dir / dst_lbl_name

        # Copy image
        try:
            shutil.copy2(img_path, dst_img_path)
            images_merged += 1
        except Exception as e:
            errors.append(f"Failed to copy healthy image {img_path.name}: {e}")
            continue

        # Remap and write label
        if lbl_path.exists():
            count, lbl_warnings = remap_label_file(lbl_path, dst_lbl_path)
            annotations_count += count
            warnings.extend(lbl_warnings)
        else:
            warnings.append(f"Missing label for healthy image: {img_path.name}")
            # Create an empty label file to avoid orphan images
            dst_lbl_path.touch()

    return images_merged, annotations_count, warnings, errors


def validate_dataset(v3_dir: Path) -> Tuple[Dict[str, int], List[str], List[str]]:
    """
    Validates the merged V3 dataset:
    - No orphan images or labels.
    - Class IDs are valid (0-16).
    - Checks for empty labels.
    Returns (split_counts, warnings, errors).
    """
    warnings = []
    errors = []
    split_counts = {"train": 0, "val": 0, "test": 0, "total": 0}
    class_distribution = defaultdict(int)

    splits = ["train", "val", "test"]
    for split in splits:
        img_dir = v3_dir / "images" / split
        lbl_dir = v3_dir / "labels" / split

        if not img_dir.exists() or not lbl_dir.exists():
            errors.append(f"Destination split directory is missing: {split}")
            continue

        img_files = {f.stem: f for f in img_dir.iterdir() if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]}
        lbl_files = {f.stem: f for f in lbl_dir.iterdir() if f.is_file() and f.suffix.lower() == ".txt"}

        split_counts[split] = len(img_files)
        split_counts["total"] += len(img_files)

        # Orphan images checking
        for stem, img_path in img_files.items():
            if stem not in lbl_files:
                errors.append(f"Orphan image: {img_path.relative_to(v3_dir)} has no matching label file.")

        # Orphan labels & validation checking
        for stem, lbl_path in lbl_files.items():
            if stem not in img_files:
                errors.append(f"Orphan label: {lbl_path.relative_to(v3_dir)} has no matching image file.")

            # Validate classes and check for empty labels
            if lbl_path.stat().st_size == 0:
                warnings.append(f"Empty label file: {lbl_path.relative_to(v3_dir)}")
                continue

            try:
                with open(lbl_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                    if not any(line.strip() for line in lines):
                        warnings.append(f"Empty label file (only whitespace): {lbl_path.relative_to(v3_dir)}")
                        continue
                    
                    for line_idx, line in enumerate(lines, 1):
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.split()
                        if not parts:
                            continue
                        try:
                            class_id = int(parts[0])
                        except ValueError:
                            errors.append(f"Invalid non-integer class ID in {lbl_path.relative_to(v3_dir)} at line {line_idx}")
                            continue

                        if not (0 <= class_id <= 16):
                            errors.append(f"Invalid class ID {class_id} in {lbl_path.relative_to(v3_dir)} at line {line_idx}")
                        else:
                            class_distribution[CLASS_NAMES[class_id]] += 1
            except Exception as e:
                errors.append(f"Failed to read/validate label {lbl_path.relative_to(v3_dir)}: {e}")

    # Convert class distribution to regular dict
    split_counts["class_distribution"] = dict(sorted(class_distribution.items()))
    return split_counts, warnings, errors


def write_yaml(v3_dir: Path) -> None:
    """
    Generates and writes data.yaml in the V3 directory.
    """
    yaml_content = f"""path: {v3_dir.as_posix()}
train: images/train
val: images/val
test: images/test

names:
"""
    for class_id, class_name in sorted(CLASS_NAMES.items()):
        yaml_content += f"  {class_id}: {class_name}\n"

    yaml_path = v3_dir / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)


def generate_report(
    v3_dir: Path,
    disease_copied: int,
    healthy_copied: int,
    disease_annotations: int,
    healthy_annotations: int,
    split_counts: Dict,
    renamed_files: List[Tuple[str, str]],
    warnings: List[str],
    errors: List[str]
) -> None:
    """
    Generates and saves reports/merge_report.json.
    """
    reports_dir = v3_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    report_data = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "dataset_name": "plantseg_tcg_yolo_v3",
        "statistics": {
            "disease_images_copied": disease_copied,
            "healthy_images_copied": healthy_copied,
            "train_images": split_counts.get("train", 0),
            "val_images": split_counts.get("val", 0),
            "test_images": split_counts.get("test", 0),
            "total_images": split_counts.get("total", 0),
            "disease_annotations": disease_annotations,
            "healthy_annotations": healthy_annotations,
            "total_annotations": disease_annotations + healthy_annotations,
            "class_distribution": split_counts.get("class_distribution", {})
        },
        "renamed_files": {
            "count": len(renamed_files),
            "mappings": [{"original": orig, "new": new} for orig, new in renamed_files]
        },
        "validation": {
            "missing_labels": len([w for w in warnings if "Missing label" in w]),
            "missing_images": len([e for e in errors if "Orphan label" in e]),
            "warnings_count": len(warnings),
            "errors_count": len(errors),
            "warnings": warnings,
            "errors": errors
        }
    }

    report_path = reports_dir / "merge_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=4)


def main() -> None:
    # 1. Setup paths
    logger.info("Initializing merge process...")

    # Clear destination if it exists (ensuring clean state)
    if V3_DIR.exists():
        logger.info(f"Removing pre-existing destination directory: {V3_DIR}")
        try:
            shutil.rmtree(V3_DIR)
        except Exception as e:
            logger.error(f"Failed to clear pre-existing V3 directory: {e}")
            sys.exit(1)

    V3_DIR.mkdir(parents=True, exist_ok=True)

    # 2. Copy V1
    logger.info("Copying V1...")
    disease_copied, disease_annotations, v1_warnings, v1_errors = copy_v1_dataset(V1_DIR, V3_DIR)

    # 3. Merge Healthy Splits
    healthy_copied = 0
    healthy_annotations = 0
    healthy_warnings = []
    healthy_errors = []
    renamed_files = []

    splits = ["train", "val", "test"]
    for split in splits:
        logger.info(f"Copying healthy {split}...")
        img_count, ann_count, sw, se = merge_healthy_split(split, HEALTHY_DIR, V3_DIR, renamed_files)
        healthy_copied += img_count
        healthy_annotations += ann_count
        healthy_warnings.extend(sw)
        healthy_errors.extend(se)

    # Combine errors/warnings from copying stage
    all_warnings = v1_warnings + healthy_warnings
    all_errors = v1_errors + healthy_errors

    # 4. Write data.yaml
    logger.info("Generating data.yaml...")
    try:
        write_yaml(V3_DIR)
    except Exception as e:
        all_errors.append(f"Failed to write data.yaml: {e}")

    # 5. Validate dataset
    logger.info("Validating...")
    split_counts, val_warnings, val_errors = validate_dataset(V3_DIR)
    all_warnings.extend(val_warnings)
    all_errors.extend(val_errors)

    # 6. Generate report
    logger.info("Generating report...")
    try:
        generate_report(
            V3_DIR,
            disease_copied=disease_copied,
            healthy_copied=healthy_copied,
            disease_annotations=disease_annotations,
            healthy_annotations=healthy_annotations,
            split_counts=split_counts,
            renamed_files=renamed_files,
            warnings=all_warnings,
            errors=all_errors
        )
    except Exception as e:
        logger.error(f"Failed to generate merge report: {e}")

    logger.info("Done.")


if __name__ == "__main__":
    main()
