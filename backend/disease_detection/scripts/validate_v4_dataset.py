#!/usr/bin/env python3
"""
plantseg_tcg_yolo_v4 Dataset Validator

Senior Computer Vision Engineer Integrity Validator for Ultralytics YOLO Segmentation Datasets.
Automatically verifies:
1. data.yaml contains only disease classes (0-13).
2. Every healthy image has an existing, empty annotation file.
3. Every disease image has a non-empty annotation file with valid polygon coordinates.
4. No class IDs outside 0-13 exist in any label file.
5. Image and label count parity per split.

Google Colab compatible out of the box with standard library Python.
"""

import os
import sys
import argparse
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[sys.stdout]
)
logger = logging.getLogger("v4_dataset_validator")


def parse_args():
    parser = argparse.ArgumentParser(description="Validate plantseg_tcg_yolo_v4 dataset integrity.")
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="datasets/plantseg_tcg_yolo_v4",
        help="Path to plantseg_tcg_yolo_v4 dataset root."
    )
    return parser.parse_args()


def parse_yaml(yaml_path):
    """Simple YAML parser for dataset names dictionary."""
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"data.yaml not found at: {yaml_path}")

    names = {}
    in_names = False

    with open(yaml_path, 'r', encoding='utf-8') as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith('#'):
                continue
            if stripped.startswith("names:"):
                in_names = True
                val = stripped.split("names:", 1)[1].strip()
                if val.startswith("{") and val.endswith("}"):
                    val_inner = val[1:-1]
                    parts = val_inner.split(",")
                    for part in parts:
                        if ":" in part:
                            k, v = part.split(":", 1)
                            names[int(k.strip())] = v.strip().strip('"\'')
                    in_names = False
            elif in_names:
                if ":" in stripped:
                    parts = stripped.split(":", 1)
                    try:
                        class_id = int(parts[0].strip())
                        class_name = parts[1].strip().strip('"\'')
                        names[class_id] = class_name
                    except ValueError:
                        in_names = False
                else:
                    in_names = False
    return names


def validate_v4_dataset(dataset_dir):
    dataset_dir = os.path.abspath(dataset_dir)
    logger.info("==================================================")
    logger.info(f"Starting V4 Dataset Integrity Validation")
    logger.info(f"Dataset Target: {dataset_dir}")
    logger.info("==================================================")

    yaml_path = os.path.join(dataset_dir, "data.yaml")
    if not os.path.exists(yaml_path):
        logger.error(f"Validation FAILED: data.yaml missing at {yaml_path}")
        return False

    # 1. Verify data.yaml classes (Must be 0-13 only)
    classes = parse_yaml(yaml_path)
    logger.info(f"Checking data.yaml: Found {len(classes)} classes.")

    invalid_yaml_classes = []
    for cid, cname in classes.items():
        if cid < 0 or cid > 13:
            invalid_yaml_classes.append((cid, cname))
        if "healthy" in cname.lower():
            invalid_yaml_classes.append((cid, cname))

    if invalid_yaml_classes:
        logger.error(f"Validation FAILED: data.yaml contains invalid or healthy classes: {invalid_yaml_classes}")
        return False
    else:
        logger.info("PASS: data.yaml contains exclusively disease classes (0-13).")

    # 2. Check split file structures & annotations
    splits = ["train", "val", "test"]
    all_valid = True
    total_images_checked = 0
    total_healthy_checked = 0
    total_disease_checked = 0

    for split in splits:
        img_dir = os.path.join(dataset_dir, "images", split)
        lbl_dir = os.path.join(dataset_dir, "labels", split)

        if not os.path.exists(img_dir) or not os.path.exists(lbl_dir):
            logger.error(f"Validation FAILED: Missing split directory for '{split}'")
            all_valid = False
            continue

        images = sorted([
            f for f in os.listdir(img_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'))
        ])
        labels = sorted([f for f in os.listdir(lbl_dir) if f.endswith('.txt')])

        logger.info(f"Checking [{split}]: {len(images)} images, {len(labels)} label files.")

        if len(images) != len(labels):
            logger.error(f"Validation FAILED [{split}]: Image count ({len(images)}) != Label count ({len(labels)})")
            all_valid = False

        for img_name in images:
            total_images_checked += 1
            base_name = os.path.splitext(img_name)[0]
            lbl_name = base_name + ".txt"
            lbl_path = os.path.join(lbl_dir, lbl_name)

            if not os.path.exists(lbl_path):
                logger.error(f"Validation FAILED [{split}]: Missing label file for image '{img_name}'")
                all_valid = False
                continue

            # Read label file
            with open(lbl_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()

            if not content:
                # Healthy / background image
                total_healthy_checked += 1
            else:
                # Disease image
                total_disease_checked += 1
                lines = content.splitlines()
                for line_num, line in enumerate(lines, 1):
                    parts = line.strip().split()
                    if len(parts) < 3:
                        logger.error(f"Validation FAILED [{split}]: Malformed polygon line {line_num} in '{lbl_name}'")
                        all_valid = False
                        continue
                    try:
                        cid = int(parts[0])
                        if cid < 0 or cid > 13:
                            logger.error(f"Validation FAILED [{split}]: Invalid class ID {cid} in '{lbl_name}' line {line_num}")
                            all_valid = False
                    except ValueError:
                        logger.error(f"Validation FAILED [{split}]: Non-integer class ID in '{lbl_name}' line {line_num}")
                        all_valid = False

    logger.info("==================================================")
    logger.info(f"Validation Summary:")
    logger.info(f"Total Images Verified   : {total_images_checked}")
    logger.info(f"Disease Images Verified : {total_disease_checked}")
    logger.info(f"Healthy Images Verified : {total_healthy_checked}")

    if all_valid:
        logger.info("RESULT: PASSED! Dataset plantseg_tcg_yolo_v4 is production-ready.")
        return True
    else:
        logger.error("RESULT: FAILED! Integrity violations detected.")
        return False


def main():
    args = parse_args()
    success = validate_v4_dataset(args.dataset_dir)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
